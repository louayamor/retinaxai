from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk
from loguru import logger
from PIL import Image
from torchvision import datasets, transforms

from app.config.settings import settings as _settings
from app.constants import PARAMS_FILE_PATH
from app.utils.common import read_yaml

logger.remove()
logger.add(sys.stdout, serialize=True)

ARTIFACTS_DIR = _settings.artifacts_root
RAW_DATA_DIR = _settings.artifacts_root / "data" / "raw" / "huggingface" / "train_clean"
PROCESSED_DATA_DIR = _settings.imaging_data_dir

FUNDUS_POSITIVE_SAMPLES = 10_000
FUNDUS_NEGATIVE_SAMPLES = 5_000


def _ensure_positives(raw_dir: Path, num_samples: int = FUNDUS_POSITIVE_SAMPLES) -> None:
    if raw_dir.exists():
        try:
            ds = load_from_disk(str(raw_dir))
            if len(ds) >= num_samples:
                logger.info(f"[FUNDUS] using {len(ds)} cached positive samples at {raw_dir}")
                return
        except (OSError, RuntimeError, ValueError):
            pass

    logger.info(f"[FUNDUS] downloading {num_samples} positive samples from EyePACS...")
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(Path(__file__).parent.parent.parent.parent / "config" / "config.yaml")
    dataset_name = cfg.get("data_ingestion", {}).get("huggingface", {}).get("dataset_name", "bumbledeep/eyepacs")

    ds = load_dataset(dataset_name, split="train", streaming=True)
    collected: list = []
    for _, sample in enumerate(ds):
        collected.append(sample)
        if len(collected) >= num_samples:
            break

    import datasets as hf_datasets
    hf_ds = hf_datasets.Dataset.from_list(collected)
    hf_ds.save_to_disk(str(raw_dir))
    logger.info(f"[FUNDUS] saved {len(hf_ds)} positive samples to {raw_dir}")


def download_imagenet_subset(output_dir: Path, num_samples: int = 5000) -> None:
    logger.info(f"[FUNDUS] downloading {num_samples} ImageNet images...")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = datasets.ImageNet(
            root=output_dir / "imagenet_raw",
            split="val",
            download=True,
        )

        indices = np.random.choice(
            len(dataset), min(num_samples, len(dataset)), replace=False
        )

        for i, idx in enumerate(indices):
            img, _label = dataset[idx]
            img.save(output_dir / f"imagenet_{i:05d}.jpg")

        logger.info(f"[FUNDUS] downloaded {len(indices)} ImageNet images")
        return
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"ImageNet download failed: {e}")

    logger.info("[FUNDUS] falling back to CIFAR-10 negatives")
    try:
        cifar_root = output_dir / "cifar10_raw"
        dataset = datasets.CIFAR10(root=cifar_root, train=False, download=True)
        indices = np.random.choice(
            len(dataset), min(num_samples, len(dataset)), replace=False
        )
        for i, idx in enumerate(indices):
            img, _label = dataset[idx]
            img.save(output_dir / f"cifar10_{i:05d}.jpg")
        logger.info(f"[FUNDUS] downloaded {len(indices)} CIFAR-10 images")
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"CIFAR-10 download failed: {e}")
        raise


def create_corrupted_fundus_from_raw(
    raw_dir: Path, output_dir: Path, num_samples: int = 5000
) -> None:
    logger.info(f"[FUNDUS] creating {num_samples} corrupted fundus images...")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        logger.warning(f"[FUNDUS] raw fundus directory not found: {raw_dir}")
        return

    ds = load_from_disk(str(raw_dir))
    if len(ds) == 0:
        logger.warning("[FUNDUS] raw fundus dataset is empty")
        return

    corruption_types = [
        "gaussian_noise",
        "blur",
        "invert",
        "grayscale",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "color_shift",
        "overexpose",
        "underexpose",
    ]

    for i in range(num_samples):
        try:
            sample = ds[np.random.randint(0, len(ds))]
            img = sample.get("image")
            if isinstance(img, dict) and img.get("bytes") is not None:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                continue

            corruption = np.random.choice(corruption_types)

            if corruption == "gaussian_noise":
                img_np = np.array(img).astype(np.float32)
                noise = np.random.normal(0, 50, img_np.shape)
                img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            elif corruption == "blur":
                from torchvision import transforms as T
                img = T.GaussianBlur(kernel_size=15, sigma=(5, 10))(img)
            elif corruption == "invert":
                img_np = 255 - np.array(img)
                img = Image.fromarray(img_np)
            elif corruption == "grayscale":
                img = img.convert("L").convert("RGB")
            elif corruption.startswith("rotate"):
                angle = int(corruption.split("_")[1])
                img = img.rotate(angle, expand=True)
            elif corruption == "color_shift":
                img_np = np.array(img)
                img_np[:, :, 0] = np.roll(img_np[:, :, 0], 50)
                img = Image.fromarray(img_np)
            elif corruption == "overexpose":
                img_np = np.clip(np.array(img) * 2.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            elif corruption == "underexpose":
                img_np = np.clip(np.array(img) * 0.2, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)

            img.save(output_dir / f"corrupted_{i:05d}.jpg")
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug(f"Failed to corrupt image: {e}")

    logger.info(f"[FUNDUS] created {num_samples} corrupted fundus images")


def _prepare_negatives(non_fundus_dir: Path) -> None:
    cached_marker = non_fundus_dir / ".cached"
    if cached_marker.exists():
        existing = len(list(non_fundus_dir.rglob("*.jpg"))) + len(
            list(non_fundus_dir.rglob("*.png"))
        )
        if existing >= FUNDUS_NEGATIVE_SAMPLES:
            logger.info(
                f"[FUNDUS] using {existing} cached negative samples at {non_fundus_dir}"
            )
            return
    logger.info("[FUNDUS] preparing negative samples...")
    download_imagenet_subset(non_fundus_dir, num_samples=FUNDUS_NEGATIVE_SAMPLES)
    create_corrupted_fundus_from_raw(RAW_DATA_DIR, non_fundus_dir, num_samples=FUNDUS_NEGATIVE_SAMPLES)
    cached_marker.touch()


def run() -> None:
    logger.info(">>> fundus stage 01: data ingestion started")

    _ensure_positives(RAW_DATA_DIR, FUNDUS_POSITIVE_SAMPLES)

    non_fundus_dir = PROCESSED_DATA_DIR / "non_fundus"
    _prepare_negatives(non_fundus_dir)

    logger.info(">>> fundus stage 01: data ingestion complete")
