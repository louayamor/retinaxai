from __future__ import annotations

import torch

from app.training.components.dice_focal_loss import DiceFocalLoss


def test_dice_focal_loss_forward() -> None:
    loss_fn = DiceFocalLoss()
    logits = torch.randn(4, 4, 64, 64)
    targets = (torch.sigmoid(torch.randn(4, 4, 64, 64)) > 0.5).float()

    loss = loss_fn(logits, targets)

    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_dice_focal_loss_perfect_prediction() -> None:
    loss_fn = DiceFocalLoss()
    targets = torch.zeros(2, 4, 16, 16)
    targets[:, 0, :, :] = 1.0

    logits = torch.full((2, 4, 16, 16), -10.0)
    logits[:, 0, :, :] = 10.0

    loss = loss_fn(logits, targets)
    assert loss.item() < 0.1


def test_dice_focal_loss_all_zeros() -> None:
    loss_fn = DiceFocalLoss()
    logits = torch.zeros(2, 4, 32, 32)
    targets = torch.zeros(2, 4, 32, 32)

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_dice_focal_loss_symmetric() -> None:
    loss_fn = DiceFocalLoss()
    logits = torch.randn(2, 4, 32, 32)
    targets = (torch.sigmoid(torch.randn(2, 4, 32, 32)) > 0.5).float()

    loss1 = loss_fn(logits, targets)
    loss2 = loss_fn(-logits, 1.0 - targets)

    assert abs(loss1.item() - loss2.item()) < 0.01
