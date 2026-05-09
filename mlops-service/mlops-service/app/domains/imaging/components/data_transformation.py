import io
import cv2
import numpy as np
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from app.entity.config_entity import ImagingTransformationConfig
from app.utils.common import read_yaml
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.domains.imaging.preprocessing import (
    circular_crop,
    apply_clahe_l_channel,
    CROP_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
)


def oversample_clinical(df: pd.DataFrame, ratio: int = 5) -> pd.DataFrame:
    """Oversample clinical data by repeating samples."""
    return pd.concat([df] * ratio, ignore_index=True)


def create_fine_tune_split(
    clinical_df: pd.DataFrame,
    eyepacs_df: pd.DataFrame,
    clinical_ratio: float = 0.7,
    no_dr_ratio: float = 0.25,
    oversample_ratio: int = 5,
) -> pd.DataFrame:
    """Create Phase 2 training split with healthy anchor (No DR samples)."""
    no_dr_count = int(len(eyepacs_df[eyepacs_df["label"] == 0]) * no_dr_ratio)
    eyepacs_no_dr = eyepacs_df[eyepacs_df["label"] == 0].sample(
        n=max(1, no_dr_count), random_state=42
    )
    dr_count = int(len(eyepacs_df[eyepacs_df["label"] != 0]) * (1 - no_dr_ratio))
    eyepacs_dr = eyepacs_df[eyepacs_df["label"] != 0].sample(
        n=max(1, dr_count), random_state=42
    )
    eyepacs_subset = pd.concat([eyepacs_no_dr, eyepacs_dr], ignore_index=True)

    clinical_oversampled = oversample_clinical(clinical_df, ratio=oversample_ratio)

    clinical_size = int(len(eyepacs_subset) * clinical_ratio / (1 - clinical_ratio))
    clinical_subset = clinical_oversampled.sample(
        n=min(clinical_size, len(clinical_oversampled)), random_state=42
    )

    return pd.concat([clinical_subset, eyepacs_subset], ignore_index=True)  # type: ignore[return-value]


class ImagingDataTransformation:
    def __init__(self, config: ImagingTransformationConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

    def _resize_and_save(self, img, path: Path) -> None:
        """
        Preprocess and save a fundus image.

        Pipeline:
        1. Convert to RGB
        2. Convert to BGR for OpenCV operations
        3. Circular crop (remove black borders)
        4. CLAHE on L channel (preserve color)
        5. Convert back to RGB
        6. Resize to target size
        7. Save

        NOTE: Histogram matching was REMOVED because:
        - Original implementation was BUGGY (inverted intensities, made images dark/black)
        - It copied grayscale to all channels (lost color information)
        - CLAHE alone is sufficient for contrast normalization
        """
        img = img.convert("RGB")
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_np = circular_crop(img_np)
        img_np = apply_clahe_l_channel(img_np)

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_np)

        img = img.resize(
            (self.config.image_size, self.config.image_size), Image.Resampling.LANCZOS
        )
        img.save(path, optimize=True)

    def _transform_eyepacs(self) -> None:
        source_path = self.config.source_dir / "huggingface" / "train_clean"
        logger.info(f"loading clean eyepacs dataset: {source_path}")
        ds = load_from_disk(str(source_path))
        logger.info(f"eyepacs source samples: {len(ds)}")

        split_cfg = self.schema.dl_dataset.train_test_split
        indices = list(range(len(ds)))

        # First split: separate test set (20%)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=split_cfg.test_size,
            random_state=split_cfg.seed,
            stratify=list(ds["label_code"]),
        )

        # Second split: separate validation from train (20% of remaining = 16% of total)
        val_size = split_cfg.get("val_size", 0.2)
        actual_val_size = val_size / (1 - split_cfg.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=actual_val_size,
            random_state=split_cfg.seed,
            stratify=[ds[i]["label_code"] for i in train_val_idx],
        )

        logger.info(
            f"stratified split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test"
        )

        for split_name, split_indices, csv_path in [
            ("train", train_idx, self.config.train_csv),
            ("val", val_idx, self.config.val_csv),
            ("test", test_idx, self.config.test_csv),
        ]:
            output_dir = self.config.root_dir / "images" / "eyepacs" / split_name
            output_dir.mkdir(parents=True, exist_ok=True)
            records = []
            total = len(split_indices)
            logger.info(
                f"eyepacs {split_name} split: total={total}, output_dir={output_dir}, csv={csv_path}"
            )

            for i, idx in enumerate(split_indices, 1):
                sample = ds[idx]
                img = sample["image"]
                if isinstance(img, dict):
                    raw_bytes = img.get("bytes")
                    if raw_bytes is not None:
                        img = Image.open(io.BytesIO(raw_bytes))
                elif not isinstance(img, Image.Image):
                    continue
                img_path = output_dir / f"{idx}.png"
                self._resize_and_save(img, img_path)
                records.append(
                    {
                        "image_path": str(img_path),
                        "label": sample["label_code"],
                        "source": "eyepacs",
                    }
                )
                if i % 500 == 0 or i == total:
                    logger.info(f"eyepacs {split_name}: {i}/{total} processed")

            pd.DataFrame(records).to_csv(csv_path, index=False)
            logger.info(
                f"eyepacs {split_name} CSV saved: {csv_path} ({len(records)} rows)"
            )

    def _transform_samaya(self) -> None:
        label_mapping = dict(self.schema.ml_dataset.label_mapping)
        source_csv = self.config.samaya_reports_csv

        if not source_csv.exists():
            logger.warning(f"samaya reports CSV not found: {source_csv}")
            return

        logger.info(f"loading samaya reports CSV: {source_csv}")
        df = pd.read_csv(source_csv)
        df = df.dropna(subset=["clinical_npdr_grade", "image_fundus_infrared_path"])
        df = df.drop_duplicates(subset=["image_fundus_infrared_path"], keep="first")
        df["label"] = df["clinical_npdr_grade"].replace(label_mapping)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        logger.info(f"samaya usable samples: {len(df)}")
        logger.info(f"samaya label mapping: {label_mapping}")
        logger.info(
            f"samaya label distribution: {df['label'].value_counts().sort_index().to_dict()}"
        )

        output_dir = self.config.root_dir / "images" / "samaya"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"samaya output_dir: {output_dir}, csv={self.config.samaya_csv}")

        records = []
        skipped = 0
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            src_path = Path(str(row["image_fundus_infrared_path"]))
            if not src_path.exists():
                skipped += 1
                continue
            img_path = output_dir / f"{i}.png"
            self._resize_and_save(Image.open(src_path), img_path)
            records.append(
                {
                    "image_path": str(img_path),
                    "label": row["label"],
                    "source": "samaya",
                }
            )
            if i % 10 == 0 or i == total:
                logger.info(f"samaya: {i}/{total} processed")

        if skipped:
            logger.warning(f"samaya: skipped {skipped} images not found on disk")

        pd.DataFrame(records).to_csv(self.config.samaya_csv, index=False)
        logger.info(f"samaya CSV saved: {self.config.samaya_csv} ({len(records)} rows)")

    def _validate_outputs(self) -> bool:
        """Validate that existing output images match the configured image_size."""
        expected_size = self.config.image_size

        # Check EyePACS train images
        eyepacs_train_dir = self.config.root_dir / "images" / "eyepacs" / "train"
        if eyepacs_train_dir.exists():
            sample_images = list(eyepacs_train_dir.glob("*.png"))[:1]
            if sample_images:
                try:
                    with Image.open(sample_images[0]) as img:
                        actual_size = img.size[0]
                        if actual_size != expected_size:
                            logger.warning(
                                f"image size mismatch: EyePACS images are {actual_size}x{actual_size}, "
                                f"expected {expected_size}x{expected_size}"
                            )
                            return False
                        logger.info(
                            f"output validation passed: EyePACS images are {expected_size}x{expected_size}"
                        )
                except Exception as e:
                    logger.warning(f"failed to validate EyePACS images: {e}")
                    return False
        else:
            logger.info("output validation: EyePACS train directory not found")
            return False

        # Check Samaya images
        samaya_dir = self.config.root_dir / "images" / "samaya"
        if samaya_dir.exists():
            sample_images = list(samaya_dir.glob("*.png"))[:1]
            if sample_images:
                try:
                    with Image.open(sample_images[0]) as img:
                        actual_size = img.size[0]
                        if actual_size != expected_size:
                            logger.warning(
                                f"image size mismatch: Samaya images are {actual_size}x{actual_size}, "
                                f"expected {expected_size}x{expected_size}"
                            )
                            return False
                        logger.info(
                            f"output validation passed: Samaya images are {expected_size}x{expected_size}"
                        )
                except Exception as e:
                    logger.warning(f"failed to validate Samaya images: {e}")
                    return False
            else:
                logger.info("output validation: Samaya directory is empty")
                return False
        else:
            logger.info("output validation: Samaya directory not found")
            return False

        return True

    def run(self) -> None:
        import os

        force_regen = os.environ.get("FORCE_REGEN", "false").lower() == "true"
        images_dir = self.config.root_dir / "images" / "eyepacs"

        # Validate output image sizes match config
        outputs_valid = self._validate_outputs()
        if not outputs_valid:
            logger.warning(
                f"output validation failed: regenerating images for image_size={self.config.image_size}"
            )
            force_regen = True

        if not force_regen and (
            self.config.train_csv.exists()
            and self.config.val_csv.exists()
            and self.config.test_csv.exists()
            and images_dir.exists()
            and any(images_dir.iterdir())
        ):
            logger.info(
                "transformation outputs already exist, skipping (set FORCE_REGEN=true to force)"
            )
            return

        if force_regen:
            logger.info("FORCE_REGEN=true: regenerating transformation outputs")

        logger.info("imaging data transformation started")
        logger.info(
            f"transformation config: source_dir={self.config.source_dir}, root_dir={self.config.root_dir}, image_size={self.config.image_size}, train_csv={self.config.train_csv}, val_csv={self.config.val_csv}, test_csv={self.config.test_csv}, samaya_csv={self.config.samaya_csv}"
        )
        self._transform_eyepacs()
        self._transform_samaya()
        logger.info("imaging data transformation complete")
