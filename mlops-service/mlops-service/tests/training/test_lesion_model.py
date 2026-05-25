from __future__ import annotations

import torch

from app.training.components.lesion_model import LesionSegmentationModel


def test_lesion_model_forward_shape() -> None:
    model = LesionSegmentationModel(
        encoder_name="timm-efficientnet-b3",
    )
    model.eval()

    batch = torch.randn(2, 3, 384, 384)
    with torch.inference_mode():
        output = model(batch)

    assert output.shape == (2, 4, 384, 384)


def test_lesion_model_decoder_gradients() -> None:
    model = LesionSegmentationModel()
    model.train()

    batch = torch.randn(1, 3, 128, 128)
    output = model(batch)

    loss = output.sum()
    loss.backward()

    decoder_params = [
        name for name, _ in model.model.decoder.named_parameters()
    ]
    for name, param in model.named_parameters():
        in_decoder = any(name.endswith(dp) or dp in name for dp in decoder_params)
        if param.requires_grad and in_decoder:
            assert param.grad is not None, f"{name} has no gradient"
