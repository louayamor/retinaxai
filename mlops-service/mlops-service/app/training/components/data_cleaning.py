from typing import Optional

from datasets import load_from_disk, Dataset
from loguru import logger

from app.config.config_entity import ImagingCleaningConfig


class ImagingDataCleaning:
    def __init__(self, config: ImagingCleaningConfig):
        self.config = config

    def run(self) -> Optional[Dataset]:  # type: ignore[type-var]
        import os

        force_regen = os.environ.get("FORCE_REGEN", "false").lower() == "true"
        clean_path = self.config.root_dir / "huggingface" / "train_clean"
        source_path = self.config.source_dir / "huggingface" / "train"
        logger.info(
            f"cleaning config: source_dir={self.config.source_dir}, root_dir={self.config.root_dir}, clean_path={clean_path}"
        )

        if not force_regen and clean_path.exists() and any(clean_path.iterdir()):
            logger.info(
                f"clean dataset already exists, skipping cleaning: {clean_path} (set FORCE_REGEN=true to force)"
            )
            return load_from_disk(str(clean_path))  # type: ignore[return-value]

        if force_regen:
            logger.info("FORCE_REGEN=true: regenerating clean dataset")

        if not source_path.exists() or not any(source_path.iterdir()):
            logger.warning(f"source dataset not found at {source_path}, cannot clean")
            return None

        logger.info(f"loading dataset from: {source_path}")

        ds = load_from_disk(str(source_path))  # type: ignore[return-value]
        original_size = len(ds)
        logger.info(f"loaded: {original_size} samples")

        before = len(ds)
        ds = ds.filter(lambda x: x["label_code"] is not None)  # type: ignore[return-value]
        logger.info(
            f"dropped null labels: removed={before - len(ds)}, remaining={len(ds)}"
        )

        before = len(ds)
        ds = ds.filter(lambda x: x["image"] is not None)  # type: ignore[return-value]
        logger.info(
            f"dropped null images: removed={before - len(ds)}, remaining={len(ds)}"
        )

        clean_path = self.config.root_dir / "huggingface" / "train_clean"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(clean_path))

        logger.info(f"imaging cleaning complete: {original_size} -> {len(ds)} samples")
        logger.info(f"clean dataset saved: {clean_path}")
        return ds  # type: ignore[return-value]
