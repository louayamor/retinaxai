from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger


class DDRDataIngestion:
    HF_REPO: str = "ctmedtech/DDR-dataset"
    ALLOW_PATTERNS: list[str] = [
        "lesion_segmentation/images/train/**",
        "lesion_segmentation/images/val/**",
        "lesion_segmentation/annotations/train/**",
        "lesion_segmentation/annotations/val/**",
    ]

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = Path(target_dir)
        self._marker = self.target_dir / ".ddr_downloaded"

    def run(self) -> Path:
        if self._marker.exists():
            logger.info(f"DDR already downloaded at {self.target_dir}")
            return self.target_dir

        self.target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloading DDR lesion segmentation from {self.HF_REPO} to {self.target_dir}"
        )
        snapshot_download(
            repo_id=self.HF_REPO,
            repo_type="dataset",
            local_dir=str(self.target_dir),
            allow_patterns=self.ALLOW_PATTERNS,
        )

        self._validate_structure()

        self._marker.touch()
        logger.info(f"DDR download complete at {self.target_dir}")
        return self.target_dir

    def _validate_structure(self) -> None:
        base = self.target_dir / "lesion_segmentation"
        for split in ("train", "val"):
            img_dir = base / "images" / split
            if not img_dir.exists():
                raise FileNotFoundError(
                    f"DDR image directory missing: {img_dir}"
                )
            ann_dir = base / "annotations" / split
            if not ann_dir.exists():
                raise FileNotFoundError(
                    f"DDR annotation directory missing: {ann_dir}"
                )
            for cls_name in ("EX", "HE", "MA", "SE"):
                cls_dir = ann_dir / "label" / cls_name
                if not cls_dir.exists():
                    logger.warning(
                        f"DDR class annotation directory missing: {cls_dir}"
                    )

        n_train = len(list((base / "images" / "train").iterdir()))
        n_val = len(list((base / "images" / "val").iterdir()))
        logger.info(
            f"DDR structure valid: {n_train} train, {n_val} val images"
        )
