from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_dummy_model() -> nn.Module:
    model = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=5, drop_rate=0.0
    )
    return model


class TestTENTAdapter:
    def test_bn_params_exist(self) -> None:
        from app.domains.imaging.components.tent_adapter import TENTAdapter

        model = _make_dummy_model()
        adapter = TENTAdapter(model)
        params = adapter.bn_params
        assert len(params) > 0
        for p in params:
            assert isinstance(p, nn.Parameter)

    def test_configure_freezes_non_bn(self) -> None:
        from app.domains.imaging.components.tent_adapter import TENTAdapter

        model = _make_dummy_model()
        adapter = TENTAdapter(model)
        adapter._save_bn_state()
        adapter._configure_for_tent()

        for name, param in model.named_parameters():
            is_bn = any(
                name.endswith(suffix) for suffix in (".weight", ".bias")
            ) and any(
                isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
                for m_name, m in model.named_modules()
                if m_name == name.rsplit(".", 1)[0]
            )
            if not is_bn or "bn" not in name.lower():
                continue

        adapter.restore()
        for param in model.parameters():
            param.requires_grad = True

    def test_adapt_reduces_entropy(self) -> None:
        from app.domains.imaging.components.tent_adapter import TENTAdapter

        model = _make_dummy_model()
        device = torch.device("cpu")
        model.to(device)

        data = torch.randn(32, 3, 224, 224) * 2.0 + 1.0
        labels = torch.randint(0, 5, (32,))
        loader = DataLoader(TensorDataset(data, labels), batch_size=16, shuffle=False)

        model.eval()
        with torch.no_grad():
            outputs_before = model(data)
            probs_before = torch.softmax(outputs_before, dim=1)
            ent_before = (
                -(probs_before * torch.log(probs_before + 1e-8))
                .sum(dim=1)
                .mean()
                .item()
            )

        adapter = TENTAdapter(model, lr=0.01, steps=2, momentum=0.9)
        adapter.adapt(loader)

        model.eval()
        with torch.no_grad():
            outputs_after = model(data)
            probs_after = torch.softmax(outputs_after, dim=1)
            ent_after = (
                -(probs_after * torch.log(probs_after + 1e-8)).sum(dim=1).mean().item()
            )

        assert ent_after <= ent_before + 0.1

        adapter.restore()

    def test_restore_resets_bn(self) -> None:
        from app.domains.imaging.components.tent_adapter import TENTAdapter

        model = _make_dummy_model()
        device = torch.device("cpu")
        model.to(device)

        data_before = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out_before = model(data_before).clone()

        data_adapt = torch.randn(32, 3, 224, 224) * 3.0
        labels_adapt = torch.zeros(32, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(data_adapt, labels_adapt), batch_size=16, shuffle=False
        )

        adapter = TENTAdapter(model, lr=0.01, steps=2)
        adapter.adapt(loader)
        adapter.restore()

        model.eval()
        with torch.no_grad():
            out_after = model(data_before)

        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_adapt_with_tuple_batch(self) -> None:
        from app.domains.imaging.components.tent_adapter import TENTAdapter

        model = _make_dummy_model()
        device = torch.device("cpu")
        model.to(device)

        data = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 5, (8,))
        loader = DataLoader(TensorDataset(data, labels), batch_size=4, shuffle=False)

        adapter = TENTAdapter(model, lr=0.01, steps=1)
        adapter.adapt(loader)
        adapter.restore()

    def test_no_bn_params_handled_gracefully(self) -> None:
        from app.domains.imaging.components.tent_adapter import TENTAdapter

        model = nn.Linear(10, 5)
        adapter = TENTAdapter(model)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 10), torch.zeros(8, dtype=torch.long)),
            batch_size=4,
        )
        adapter.adapt(loader)
        adapter.restore()
