from __future__ import annotations

import io
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_from_disk
from PIL import Image

from app.training.preprocessing import preprocess_fundus_image
from app.utils.common import read_yaml
from app.constants import PARAMS_FILE_PATH
from app.config.settings import settings as _settings

logger.remove()
logger.add(sys.stdout, serialize=True)

ARTIFACTS_DIR = _settings.artifacts_root
PROCESSED_DATA_DIR = _settings.imaging_data_dir
RAW_DATA_DIR = _settings.artifacts_root / "data" / "raw" / "huggingface" / "train_clean"


class FundusValidationDataset(Dataset):
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
    if device is None:
        from app.utils.common import require_cuda
        device = require_cuda()

    logger.info(f"[FUNDUS] training on {device}")

    params = read_yaml(PARAMS_FILE_PATH)
    aug = params.get("augmentation", {}) or {}
    norm = aug.get("normalize", {}) or {}

    train_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: preprocess_fundus_image(img, image_size=image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm.get("mean", [0.485, 0.456, 0.406]),
                std=norm.get("std", [0.229, 0.224, 0.225]),
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: preprocess_fundus_image(img, image_size=image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm.get("mean", [0.485, 0.456, 0.406]),
                std=norm.get("std", [0.229, 0.224, 0.225]),
            ),
        ]
    )

    non_fundus_dir = train_dir / "non_fundus"

    raw_dataset_train = RawFundusParquetDataset(RAW_DATA_DIR, transform=train_transform)
    neg_dataset_train = FundusValidationDataset(
        Path("/tmp/empty"), non_fundus_dir, train_transform
    )
    raw_dataset_val = RawFundusParquetDataset(RAW_DATA_DIR, transform=val_transform)
    neg_dataset_val = FundusValidationDataset(
        Path("/tmp/empty"), non_fundus_dir, val_transform
    )

    pos_count = len(raw_dataset_train)
    neg_count = len(neg_dataset_train)

    if pos_count == 0:
        raise ValueError(
            f"[FUNDUS] no positive fundus samples found at {RAW_DATA_DIR}."
        )
    if neg_count == 0:
        raise ValueError(f"[FUNDUS] no negative samples found at {non_fundus_dir}.")

    full_train = torch.utils.data.ConcatDataset([raw_dataset_train, neg_dataset_train])
    full_val = torch.utils.data.ConcatDataset([raw_dataset_val, neg_dataset_val])

    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    _, val_dataset = torch.utils.data.random_split(
        full_val, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"[FUNDUS] train samples: {train_size}, val samples: {val_size}")

    model = timm.create_model(
        model_name, pretrained=True, num_classes=num_classes, drop_rate=dropout
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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

    logger.info(f"[FUNDUS] training complete: best_val_acc={best_val_acc:.4f}")
    return output_path


def run() -> None:
    logger.info(">>> fundus stage 02: training started")

    params = read_yaml(PARAMS_FILE_PATH)
    fc_cfg = params.get("fundus_classifier", {})

    model_name = fc_cfg.get("model_name", "mobilenetv3_small_100")
    image_size = fc_cfg.get("image_size", 384)
    num_classes = fc_cfg.get("num_classes", 2)
    dropout_rate = fc_cfg.get("dropout", 0.1)
    batch_size = int(fc_cfg.get("batch_size", 32))
    num_epochs = int(fc_cfg.get("num_epochs", 5))
    learning_rate = float(fc_cfg.get("learning_rate", 0.001))

    output_path = ARTIFACTS_DIR / "fundus_classifier.pth"

    train_fundus_classifier(
        train_dir=PROCESSED_DATA_DIR,
        output_path=output_path,
        model_name=model_name,
        image_size=image_size,
        num_classes=num_classes,
        dropout=dropout_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    logger.info(">>> fundus stage 02: training complete")
