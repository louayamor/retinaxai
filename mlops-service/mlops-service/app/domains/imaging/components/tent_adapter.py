from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class TENTAdapter:
    """
    Test-time BN adaptation via entropy minimization.

    TENT (Test Entropy) — Wang et al., ICLR 2021.

    At test time: freezes all model parameters except BatchNorm
    affine parameters and running statistics, then minimizes prediction
    entropy over a target domain batch. BN layers are the only ones
    updated; all other weights remain frozen.

    After adaptation, BN statistics are reset to their pre-adaptation
    state so the model is not permanently altered for source domain use.

    Expected usage:
        adapter = TENTAdapter(model)
        adapter.adapt(dataloader)       # adapt BN to target domain
        predictions = inference(model)   # run inference
        adapter.restore()                # reset BN to original
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.0001,
        steps: int = 1,
        momentum: float = 0.9,
    ):
        self.model = model
        self.lr = lr
        self.steps = steps
        self.momentum = momentum

        self._optimizer: torch.optim.SGD | None = None
        self._original_bn_state: dict[str, torch.Tensor] | None = None
        self._original_bn_running: dict[str, torch.Tensor] | None = None
        self._configured = False

    @property
    def bn_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    params.append(m.weight)
                if m.bias is not None:
                    params.append(m.bias)
        return params

    def _save_bn_state(self) -> None:
        self._original_bn_state = {}
        self._original_bn_running = {}
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._original_bn_state[f"{name}.weight"] = (
                    m.weight.data.clone() if m.weight is not None else None
                )
                self._original_bn_state[f"{name}.bias"] = (
                    m.bias.data.clone() if m.bias is not None else None
                )
                self._original_bn_running[f"{name}.running_mean"] = (
                    m.running_mean.clone() if m.running_mean is not None else None
                )
                self._original_bn_running[f"{name}.running_var"] = (
                    m.running_var.clone() if m.running_var is not None else None
                )

    def _configure_for_tent(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.bn_params:
            param.requires_grad = True
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.track_running_stats = True
            else:
                m.eval()
        self._configured = True

    def adapt(self, dataloader: torch.utils.data.DataLoader) -> None:
        if not self._configured:
            self._save_bn_state()
            self._configure_for_tent()

        device = next(self.model.parameters()).device
        params = [p for p in self.bn_params if p.requires_grad]
        if not params:
            logger.warning("TENT: no BN params to adapt — skipping")
            return

        self._optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)

        for step in range(self.steps):
            total_entropy = 0.0
            total_samples = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                if isinstance(images, torch.Tensor):
                    images = images.to(device)

                self._optimizer.zero_grad()
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                entropy.backward()
                self._optimizer.step()

                total_entropy += entropy.item() * images.size(0)
                total_samples += images.size(0)

            avg_entropy = total_entropy / max(total_samples, 1)
            logger.info(f"TENT step {step + 1}/{self.steps}: entropy={avg_entropy:.4f}")

    def restore(self) -> None:
        if self._original_bn_state is None:
            return

        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                w_key = f"{name}.weight"
                b_key = f"{name}.bias"
                if (
                    w_key in self._original_bn_state
                    and self._original_bn_state[w_key] is not None
                ):
                    m.weight.data.copy_(self._original_bn_state[w_key])
                if (
                    b_key in self._original_bn_state
                    and self._original_bn_state[b_key] is not None
                ):
                    m.bias.data.copy_(self._original_bn_state[b_key])
                if f"{name}.running_mean" in self._original_bn_running:
                    m.running_mean.copy_(
                        self._original_bn_running[f"{name}.running_mean"]
                    )
                if f"{name}.running_var" in self._original_bn_running:
                    m.running_var.copy_(
                        self._original_bn_running[f"{name}.running_var"]
                    )

        self.model.eval()
        self._configured = False
        self._optimizer = None
        logger.info("TENT BN state restored to pre-adaptation values")
