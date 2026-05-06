from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def tmp_target_dir(tmp_path: Path) -> Path:
    d = tmp_path / "samaya"
    d.mkdir()
    for i in range(5):
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(d / f"{i:03d}.png")
    return d


@pytest.fixture
def tmp_source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "eyepacs"
    d.mkdir()
    for i in range(10):
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(d / f"{i:03d}.png")
    return d


class TestFDAAugment:
    def test_cache_created(self, tmp_target_dir: Path, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        augment = FDAAugment(
            target_images_dir=tmp_target_dir, beta=0.15, cache_path=cache
        )
        amp = augment.target_amplitude
        assert cache.exists()
        assert amp.shape == (3, 224, 224)
        assert amp.dtype == torch.float32
        assert torch.isfinite(amp).all()

    def test_cache_reused(self, tmp_target_dir: Path, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        a1 = FDAAugment(tmp_target_dir, beta=0.15, cache_path=cache)
        _ = a1.target_amplitude
        mod_time = cache.stat().st_mtime

        a2 = FDAAugment(tmp_target_dir, beta=0.15, cache_path=cache)
        _ = a2.target_amplitude
        assert cache.stat().st_mtime == mod_time

    def test_beta_zero_is_identity(self, tmp_target_dir: Path, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        augment = FDAAugment(
            tmp_target_dir, beta=0.0, probability=1.0, cache_path=cache
        )
        x = torch.rand(1, 3, 224, 224)
        y = augment(x)
        assert torch.allclose(x, y, atol=1e-6)

    def test_output_in_valid_range(self, tmp_target_dir: Path, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        augment = FDAAugment(
            tmp_target_dir, beta=0.15, probability=1.0, cache_path=cache
        )
        x = torch.rand(1, 3, 224, 224)
        y = augment(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0
        assert torch.isfinite(y).all()

    def test_probability_respected(self, tmp_target_dir: Path, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        augment = FDAAugment(
            tmp_target_dir, beta=0.15, probability=1.0, cache_path=cache
        )
        x = torch.rand(1, 3, 224, 224)
        y = augment(x)
        assert not torch.allclose(x, y, atol=1e-4)

        augment_p0 = FDAAugment(
            tmp_target_dir, beta=0.15, probability=0.0, cache_path=cache
        )
        y2 = augment_p0(x)
        assert torch.allclose(x, y2, atol=1e-6)

    def test_inverse_without_source_returns_original(
        self, tmp_target_dir: Path, tmp_path: Path
    ) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        augment = FDAAugment(tmp_target_dir, beta=0.15, cache_path=cache)
        x = torch.rand(3, 224, 224)
        y = augment.inverse(x)
        assert torch.allclose(x, y)

    def test_inverse_with_source(
        self, tmp_target_dir: Path, tmp_source_dir: Path, tmp_path: Path
    ) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache_tgt = tmp_path / "amplitude_tgt.pt"
        cache_src = tmp_path / "amplitude_src.pt"
        augment = FDAAugment(
            tmp_target_dir,
            beta=0.15,
            cache_path=cache_tgt,
            source_amp_cache_path=cache_src,
        )
        augment.set_source_amplitude(tmp_source_dir, cache_src)
        assert augment.source_amplitude is not None
        x = torch.rand(3, 224, 224)
        y = augment.inverse(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0
        assert torch.isfinite(y).all()

    def test_beta_invalid_raises(self) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        with pytest.raises(ValueError, match="beta"):
            FDAAugment(Path("/tmp"), beta=-0.1)

        with pytest.raises(ValueError, match="beta"):
            FDAAugment(Path("/tmp"), beta=1.5)

    def test_probability_invalid_raises(self) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        with pytest.raises(ValueError, match="probability"):
            FDAAugment(Path("/tmp"), probability=-0.1)

        with pytest.raises(ValueError, match="probability"):
            FDAAugment(Path("/tmp"), probability=1.5)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        empty = tmp_path / "empty"
        empty.mkdir()
        augment = FDAAugment(empty, cache_path=tmp_path / "amp.pt")
        with pytest.raises(FileNotFoundError):
            _ = augment.target_amplitude

    def test_structure_preserved(self, tmp_target_dir: Path, tmp_path: Path) -> None:
        from app.domains.imaging.components.fda_augment import FDAAugment

        cache = tmp_path / "amplitude.pt"
        augment = FDAAugment(
            tmp_target_dir, beta=0.05, probability=1.0, cache_path=cache
        )

        x = torch.zeros(1, 3, 224, 224)
        x[:, :, 50:60, 50:60] = 1.0
        y = augment(x)
        edge_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        edge_y = y[:, :, :, 1:] - y[:, :, :, :-1]
        assert torch.any(edge_y.abs() > 0.001)
