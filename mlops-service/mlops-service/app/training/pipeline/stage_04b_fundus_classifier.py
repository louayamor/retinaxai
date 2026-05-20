from __future__ import annotations

import io
import sys
from pathlib import Path

import mlflow
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_from_disk
from PIL import Image
import numpy as np

from mlflow.exceptions import MlflowException

from app.training.preprocessing import preprocess_fundus_image
from app.utils.common import read_yaml, set_seed
from app.constants import PARAMS_FILE_PATH


logger.remove()
logger.add(sys.stdout, serialize=True)

ARTIFACTS_DIR = Path("artifacts/model/imaging")
PROCESSED_DATA_DIR = Path("artifacts/data/processed/imaging")
RAW_DATA_DIR = Path("artifacts/data/raw/huggingface/train")


class FundusValidationDataset(Dataset):
    """Binary dataset: fundus (1) vs non-fundus (0)."""

    def __init__(self, fundus_dir: Path, non_fundus_dir: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        pos_count = 0
        if fundus_dir.exists():
            for img_path in fundus_dir.rglob("*.jpg"):
                self.samples.append((img_path, 1))
                pos_count += 1
            for img_path in fundus_dir.rglob("*.png"):
                self.samples.append((img_path, 1))
                pos_count += 1

        neg_count = 0
        if non_fundus_dir.exists():
            for img_path in non_fundus_dir.rglob("*.jpg"):
                self.samples.append((img_path, 0))
                neg_count += 1
            for img_path in non_fundus_dir.rglob("*.png"):
                self.samples.append((img_path, 0))
                neg_count += 1

        logger.info(
            f"[FUNDUS] loaded {pos_count} positive samples, {neg_count} negative samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            img = self.transform(img)

        return img, label


class RawFundusParquetDataset(Dataset):
    """Raw fundus images from HuggingFace parquet/arrow cache (all positives)."""

    def __init__(self, raw_dir: Path, transform=None):
        self.raw_dir = raw_dir
        self.transform = transform

        if not raw_dir.exists():
            raise FileNotFoundError(f"raw fundus dataset not found: {raw_dir}")

        self._dataset = load_from_disk(str(raw_dir))
        self._length = len(self._dataset)

        logger.info(f"[FUNDUS] loaded raw fundus dataset: {self._length} samples")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        img = sample.get("image")

        if isinstance(img, dict) and img.get("bytes") is not None:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            raise ValueError("raw fundus image missing bytes")

        if self.transform:
            img = self.transform(img)

        return img, 1


def download_imagenet_subset(output_dir: Path, num_samples: int = 5000) -> None:
    """Download random ImageNet images as negative samples.

    If ImageNet is unavailable, fallback to CIFAR-10 download.
    """
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
    """Create corrupted fundus images as negative samples from raw parquet."""
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


def train_fundus_classifier(
    train_dir: Path,
    output_path: Path,
    model_name: str = "mobilenetv3_small_100",
    image_size: int = 384,
    num_classes: int = 2,
    dropout: float = 0.1,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    device: torch.device = None,
) -> Path:
    """Train the fundus classifier."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"[FUNDUS] training on {device}")

    # Transforms
    params = read_yaml(PARAMS_FILE_PATH)
    norm = params.augmentation.normalize

    train_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: preprocess_fundus_image(img, image_size=image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.mean, std=norm.std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: preprocess_fundus_image(img, image_size=image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.mean, std=norm.std),
        ]
    )

    # Dataset (raw fundus positives, downloaded negatives)
    non_fundus_dir = train_dir / "non_fundus"

    raw_dataset = RawFundusParquetDataset(RAW_DATA_DIR, transform=train_transform)
    neg_dataset = FundusValidationDataset(
        Path("/tmp/empty"), non_fundus_dir, train_transform
    )

    pos_count = len(raw_dataset)
    neg_count = len(neg_dataset)

    if pos_count == 0:
        raise ValueError(
            f"[FUNDUS] no positive fundus samples found at {RAW_DATA_DIR}."
        )
    if neg_count == 0:
        raise ValueError(f"[FUNDUS] no negative samples found at {non_fundus_dir}.")

    full_dataset = torch.utils.data.ConcatDataset([raw_dataset, neg_dataset])

    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply val transform to val set
    if hasattr(val_dataset, "dataset") and hasattr(val_dataset.dataset, "transform"):
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"[FUNDUS] train samples: {train_size}, val samples: {val_size}")

    # Model
    model = timm.create_model(
        model_name, pretrained=True, num_classes=num_classes, drop_rate=dropout
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        scheduler.step()

        mlflow.log_metrics(
            {
                "train_loss": float(round(train_loss, 4)),
                "train_acc": float(round(train_acc, 4)),
                "val_loss": float(round(val_loss, 4)),
                "val_acc": float(round(val_acc, 4)),
            },
            step=epoch,
        )

        logger.info(
            f"[FUNDUS] epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            logger.info(f"[FUNDUS] best model saved to {output_path}")

    mlflow.log_metric("best_val_acc", float(round(best_val_acc, 4)))
    logger.info(f"[FUNDUS] training complete: best_val_acc={best_val_acc:.4f}")
    return output_path


def run():
    """Main entry point for stage 04b."""
    logger.info(">>> stage 04b: fundus classifier training started")

    params = read_yaml(PARAMS_FILE_PATH)
    fc_cfg = params.get("fundus_classifier", {})

    model_name = fc_cfg.get("model_name", "mobilenetv3_small_100")
    image_size = fc_cfg.get("image_size", 384)
    num_classes = fc_cfg.get("num_classes", 2)
    threshold = fc_cfg.get("threshold", 0.3)

    from app.utils.mlflow_utils import configure_mlflow

    configure_mlflow()

    try:
        with mlflow.start_run(run_name="fundus_classifier"):
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "image_size": image_size,
                    "num_classes": num_classes,
                    "threshold": threshold,
                }
            )

            non_fundus_dir = PROCESSED_DATA_DIR / "non_fundus"

            logger.info("[FUNDUS] preparing negative samples...")
            download_imagenet_subset(non_fundus_dir, num_samples=5000)
            create_corrupted_fundus_from_raw(
                RAW_DATA_DIR, non_fundus_dir, num_samples=5000
            )

            output_path = ARTIFACTS_DIR / "fundus_classifier.pth"
            dropout_rate = fc_cfg.get("dropout", 0.1)
            train_fundus_classifier(
                train_dir=PROCESSED_DATA_DIR,
                output_path=output_path,
                model_name=model_name,
                image_size=image_size,
                num_classes=num_classes,
                dropout=dropout_rate,
                batch_size=32,
                num_epochs=5,
                learning_rate=0.001,
            )

            mlflow.log_artifact(str(output_path), artifact_path="fundus_classifier")
            logger.info(f"[FUNDUS] model logged to MLflow: {output_path}")
    except MlflowException as e:
        logger.warning(f"[FUNDUS] MLflow disabled: {e}")

        non_fundus_dir = PROCESSED_DATA_DIR / "non_fundus"

        logger.info("[FUNDUS] preparing negative samples...")
        download_imagenet_subset(non_fundus_dir, num_samples=5000)
        create_corrupted_fundus_from_raw(RAW_DATA_DIR, non_fundus_dir, num_samples=5000)

        output_path = ARTIFACTS_DIR / "fundus_classifier.pth"
        train_fundus_classifier(
            train_dir=PROCESSED_DATA_DIR,
            output_path=output_path,
            model_name=model_name,
            image_size=image_size,
            num_classes=num_classes,
            dropout=dropout_rate,
            batch_size=32,
            num_epochs=5,
            learning_rate=0.001,
        )

    logger.info(">>> stage 04b: fundus classifier training complete")


if __name__ == "__main__":
    run()
