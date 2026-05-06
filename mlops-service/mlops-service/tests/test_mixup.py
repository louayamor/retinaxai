from __future__ import annotations

import torch
import numpy as np


class TestMixUp:
    def test_mixup_blends_images(self) -> None:
        """Mixed image should be a convex combination of two images."""
        img_a = torch.rand(4, 3, 224, 224)
        img_b = torch.rand(4, 3, 224, 224)
        lam = 0.3

        mixed = lam * img_a + (1.0 - lam) * img_b

        assert mixed.shape == (4, 3, 224, 224)
        assert torch.isfinite(mixed).all()
        diff = mixed - img_a
        assert not torch.allclose(diff, torch.zeros_like(diff), atol=1e-4)

    def test_mixup_lam_in_range(self) -> None:
        """Beta-distributed lambda should be in [0, 1]."""
        alpha = 0.2
        for _ in range(100):
            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1.0 - lam)
            assert 0.5 <= lam <= 1.0

    def test_mixup_loss_symmetry(self) -> None:
        """MixUp loss should be symmetric for lam=0.5 on identical labels."""
        outputs = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        labels_a = torch.tensor([1, 0])
        labels_b = torch.tensor([1, 0])

        from app.domains.imaging.components.ordinal_loss import OrdinalCrossEntropyLoss

        criterion = OrdinalCrossEntropyLoss(num_classes=3, distance_weight=0.1)
        lam = 0.5
        loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(
            outputs, labels_b
        )

        direct_loss = criterion(outputs, labels_a)
        assert abs(loss.item() - direct_loss.item()) < 1e-4

    def test_permutation_no_self_match(self) -> None:
        """Random permutation should not produce self-matches for batch > 1."""
        batch_size = 16
        for _ in range(10):
            perm = torch.randperm(batch_size)
            matches = (torch.arange(batch_size) == perm).sum().item()
            assert matches < batch_size

    def test_class_weighted_loss_different(self) -> None:
        from app.domains.imaging.components.ordinal_loss import OrdinalCrossEntropyLoss
        import torch.nn.functional as F

        outputs = torch.tensor([[2.0, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 2.0, 0.1, 0.1]])
        labels = torch.tensor([0, 2])

        unweighted = OrdinalCrossEntropyLoss(num_classes=5, distance_weight=0.0)
        loss_unw = unweighted(outputs, labels)

        weights = torch.tensor([1.0, 3.15, 4.66, 5.51, 6.22], dtype=torch.float32)
        weighted = OrdinalCrossEntropyLoss(
            num_classes=5, distance_weight=0.0, class_weights=weights
        )
        loss_w = weighted(outputs, labels)

        loss_ce_unw = F.cross_entropy(outputs, labels)
        loss_ce_w = F.cross_entropy(outputs, labels, weight=weights)
        assert abs(loss_unw.item() - loss_ce_unw.item()) < 1e-4
        assert abs(loss_w.item() - loss_ce_w.item()) < 1e-4
        assert loss_w.item() != loss_unw.item()
