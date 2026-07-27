"""Extract PyRadiomics features from image and tumor-mask files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from radiomics import featureextractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-suffix", default="_image.nii.gz")
    parser.add_argument("--mask-suffix", default="_mask.nii.gz")
    args = parser.parse_args()

    config = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )
    output_path = Path(
        args.output or config["data"]["radiomics_table"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params_file = config.get("radiomics", {}).get("params_file")
    if params_file:
        extractor = featureextractor.RadiomicsFeatureExtractor(
            params_file
        )
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    rows = []

    for image_path in sorted(
        image_dir.glob(f"*{args.image_suffix}")
    ):
        patient_id = image_path.name[: -len(args.image_suffix)]
        mask_path = mask_dir / f"{patient_id}{args.mask_suffix}"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing mask for {patient_id}: {mask_path}"
            )

        result = extractor.execute(
            str(image_path),
            str(mask_path),
        )
        row = {"patient_id": patient_id}
        row.update(
            {
                key: value
                for key, value in result.items()
                if not key.startswith("diagnostics_")
            }
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No image files matched suffix: {args.image_suffix}"
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} patients to {output_path}")


if __name__ == "__main__":
    main()
