"""Extract ViT-Base-Patch16 class-token features from the largest tumor section."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import timm
import yaml
from PIL import Image
from skimage.exposure import equalize_adapthist


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def largest_slice(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    if mask.ndim == 4:
        mask = mask[0]
    areas = (mask > 0).reshape(mask.shape[0], -1).sum(axis=1)
    if areas.max() == 0:
        raise ValueError("Tumor mask contains no foreground voxels.")
    z = int(np.argmax(areas))
    ys, xs = np.where(mask[z] > 0)
    pad = 10
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[1], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[2], xs.max() + pad + 1)
    return image[z, y0:y1, x0:x1]


def to_tensor(crop: np.ndarray) -> torch.Tensor:
    """Match the study preprocessing: 224-pixel crop, CLAHE, then 256/224 input."""
    crop = crop - np.nanmin(crop)
    denom = np.nanpercentile(crop, 99) or 1.0
    crop = np.clip(crop / denom, 0, 1)
    crop = (crop * 255).astype(np.uint8)
    pil = Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR)
    crop = equalize_adapthist(np.asarray(pil, dtype=np.float32) / 255.0)
    pil = Image.fromarray((crop * 255).astype(np.uint8)).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    left = (256 - 224) // 2
    pil = pil.crop((left, left, left + 224, left + 224))
    x = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0
    # The three duplicated channels were normalized with mean and SD of 0.5.
    mean = torch.full((3, 1, 1), 0.5)
    std = torch.full((3, 1, 1), 0.5)
    return (x - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--image-suffix", default="_image.nii.gz")
    ap.add_argument("--mask-suffix", default="_mask.nii.gz")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output or cfg["data"]["vit_table"])
    output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).eval().to(device)
    rows = []
    with torch.no_grad():
        for image_path in sorted(Path(args.image_dir).glob(f"*{args.image_suffix}")):
            patient_id = image_path.name[: -len(args.image_suffix)]
            mask_path = Path(args.mask_dir) / f"{patient_id}{args.mask_suffix}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {patient_id}: {mask_path}")
            crop = largest_slice(read_array(image_path), read_array(mask_path))
            feat = model(to_tensor(crop)[None].to(device))
            feat = feat.detach().cpu().numpy().ravel()
            if feat.size != 768:
                raise RuntimeError(f"Expected 768 ViT features, got {feat.size}")
            # Preserve the feature naming convention used in Table S5.
            rows.append({"patient_id": patient_id, **{f"DL.feature_{i+1}": float(v) for i, v in enumerate(feat)}})
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} patients to {output}")


if __name__ == "__main__":
    main()
