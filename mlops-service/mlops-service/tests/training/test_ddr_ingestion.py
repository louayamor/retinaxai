from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.training.components.ddr_ingestion import DDRDataIngestion


@patch("app.training.components.ddr_ingestion.snapshot_download")
def test_ddr_ingestion_download(mock_snapshot: MagicMock, tmp_path: Path) -> None:
    target = tmp_path / "ddr"
    target.mkdir(parents=True)

    seg_dir = target / "lesion_segmentation"
    for split in ("train", "val"):
        (seg_dir / "images" / split).mkdir(parents=True)
        (seg_dir / "images" / split / f"img_{split}_001.jpg").touch()
        (seg_dir / "annotations" / split / "label" / "EX").mkdir(parents=True)
        (seg_dir / "annotations" / split / "label" / "HE").mkdir(parents=True)
        (seg_dir / "annotations" / split / "label" / "MA").mkdir(parents=True)
        (seg_dir / "annotations" / split / "label" / "SE").mkdir(parents=True)

    mock_snapshot.return_value = str(target)

    ingestion = DDRDataIngestion(target)
    result = ingestion.run()

    assert result == target
    assert (target / ".ddr_downloaded").exists()


def test_ddr_ingestion_missing_image_dir(tmp_path: Path) -> None:
    target = tmp_path / "ddr_invalid"
    target.mkdir(parents=True)
    (target / "lesion_segmentation").mkdir(parents=True)

    ingestion = DDRDataIngestion(target)

    with pytest.raises(FileNotFoundError, match="image directory missing"):
        ingestion._validate_structure()


@patch("app.training.components.ddr_ingestion.snapshot_download")
def test_ddr_ingestion_marker_skips_download(
    mock_snapshot: MagicMock, tmp_path: Path
) -> None:
    target = tmp_path / "ddr_cached"
    target.mkdir(parents=True)
    (target / ".ddr_downloaded").touch()

    ingestion = DDRDataIngestion(target)
    ingestion.run()

    mock_snapshot.assert_not_called()
