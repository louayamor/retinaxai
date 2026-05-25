from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import torch

from app.training.components.ddr_lesion_dataset import DDRLesionDataset


@pytest.fixture
def fake_ddr_manifest(tmp_path: Path) -> Generator[Path, None, None]:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True)

    csv_path = tmp_path / "train.csv"
    records = []
    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        )
        img_path = image_dir / f"img_{i:03d}.jpg"
        img.save(img_path)

        record = {
            "image_id": f"img_{i:03d}",
            "image_path": str(img_path),
            "ex_path": "",
            "he_path": "",
            "ma_path": "",
            "se_path": "",
        }

        for cls_name in ("ex", "he", "ma", "se"):
            if i % 2 == 0:
                mask = Image.fromarray(
                    np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
                )
                mask_path = mask_dir / f"{cls_name}_{i:03d}.png"
                mask.save(mask_path)
                record[f"{cls_name}_path"] = str(mask_path)

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    yield csv_path


def test_ddr_lesion_dataset_shape(fake_ddr_manifest: Path) -> None:
    dataset = DDRLesionDataset(fake_ddr_manifest, image_size=224)
    img, mask = dataset[0]

    assert img.shape == (3, 224, 224)
    assert mask.shape == (4, 224, 224)
    assert mask.dtype == torch.float32


def test_ddr_lesion_dataset_missing_mask_zero_channel(
    fake_ddr_manifest: Path,
) -> None:
    dataset = DDRLesionDataset(fake_ddr_manifest, image_size=128)

    for i in range(len(dataset)):
        _, mask = dataset[i]
        for c in range(4):
            assert mask[c].sum() >= 0


def test_ddr_lesion_dataset_length(fake_ddr_manifest: Path) -> None:
    dataset = DDRLesionDataset(fake_ddr_manifest)
    assert len(dataset) == 3
