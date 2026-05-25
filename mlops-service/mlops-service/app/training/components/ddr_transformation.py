from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


CLASS_SUBDIRS = ("EX", "HE", "MA", "SE")


class DDRTransformation:
    def __init__(self, ddr_seg_dir: Path, output_dir: Path) -> None:
        self.ddr_seg_dir = Path(ddr_seg_dir)
        self.output_dir = Path(output_dir)

    def run(self) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifests: dict[str, Path] = {}
        for split in ("train", "val"):
            csv_path = self.output_dir / f"{split}.csv"
            self._build_split_manifest(split, csv_path)
            manifests[split] = csv_path

        return manifests

    def _build_split_manifest(self, split: str, csv_path: Path) -> None:
        image_dir = self.ddr_seg_dir / "images" / split
        ann_base = self.ddr_seg_dir / "annotations" / split / "label"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"DDR image directory not found: {image_dir}"
            )

        records: list[dict[str, str]] = []
        image_paths = sorted(image_dir.iterdir())

        for img_path in image_paths:
            stem = img_path.stem
            record: dict[str, str] = {
                "image_id": stem,
                "image_path": str(img_path),
            }
            for cls_name in CLASS_SUBDIRS:
                cls_dir = ann_base / cls_name
                mask_path = cls_dir / f"{stem}.png"
                record[f"{cls_name.lower()}_path"] = (
                    str(mask_path) if mask_path.exists() else ""
                )
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        logger.info(
            f"DDR {split} manifest: {len(df)} rows -> {csv_path}"
        )
