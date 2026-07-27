"""Extract PyRadiomics features from preprocessed image and tumor-mask files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml
from radiomics import featureextractor


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
    output = Path(args.output or cfg["data"]["radiomics_table"])
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use the same preprocessing and PyRadiomics parameter file used for the study.
    params = cfg.get("radiomics", {}).get("params_file")
    extractor = featureextractor.RadiomicsFeatureExtractor(params) if params else featureextractor.RadiomicsFeatureExtractor()
    rows = []
    for image in sorted(Path(args.image_dir).glob(f"*{args.image_suffix}")):
        patient_id = image.name[: -len(args.image_suffix)]
        mask = Path(args.mask_dir) / f"{patient_id}{args.mask_suffix}"
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {patient_id}: {mask}")
        result = extractor.execute(str(image), str(mask))
        row = {"patient_id": patient_id}
        row.update({k: v for k, v in result.items() if not k.startswith("diagnostics_")})
        rows.append(row)
    if not rows:
        raise RuntimeError("No image files matched the requested suffix.")
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} patients to {output}")


if __name__ == "__main__":
    main()

