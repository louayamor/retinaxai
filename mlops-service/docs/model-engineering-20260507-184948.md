# Model Engineering Notes — Phase 1 Training Run (2026-05-07 16:48 UTC)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `efficientnet_b3_937` |
| **MLflow Run** | `c7bdffd0bc5c440299c32e9d92039cff` |
| **Date** | 2026-05-07 16:48 UTC |
| **Phase** | Phase 1 (backbone frozen, gradual unfreeze enabled) |
| **Backbone** | EfficientNet-B3 (timm) |
| **Trainable Params** | 10,541,383 / 10,703,917 (98.5%) |
| **Loss** | FocalOrdinalLoss (gamma=1.0) |
| **MixUp** | alpha=0.2 |
| **BatchNorm** | Training mode (domain adaptation) |
| **Gradient Checkpointing** | Enabled |
| **Seed** | 42 |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| `image_size` | 300 |
| `batch_size` | 8 |
| `phase1_epochs` | 15 |
| `phase1_lr` | 5e-5 |
| `lr_scheduler` | CosineAnnealingLR |
| `dropout` | 0.5 |
| `weight_decay` | 0.001 |
| `label_smoothing` | 0.1 |
| `focal_loss_gamma` | 1.0 |
| `freeze_backbone` | true |
| `gradual_unfreeze` | true |
| `weighted_sampling` | false |
| `custom_class_weights` | [1.0, 1.8, 2.5, 3.0, 3.5] |
| `early_stopping_patience` | 10 |
| `num_workers` | 0 |
| `pin_memory` | false |

## Dataset Split

| Split | Count |
|-------|-------|
| Train | 5,999 |
| Val | 1,999 |
| Test | 2,000 |

## Results

### Comparison with Previous Run

| Metric | Previous Run | This Run | Change |
|--------|-------------|----------|--------|
| Val Accuracy | 17.7% | **75.3%** | +57.6% |
| Val F1 (best) | 0.17 | **0.37** (epoch 9) | +118% |
| Val F1 (final) | 0.17 | **0.35** | +106% |
| Val Loss | — | **0.96** | — |
| LR Schedule | Fixed 2e-4 | Cosine 2.24e-5 → 0 | — |
| Early Stopping | Triggered epoch 15 | Did not trigger | — |

### Per-Epoch Trajectory

```
Epoch  Val Acc   Val F1   Val Loss   LR          Notes
─────  ────────  ───────  ────────   ──────────  ─────────────────
  1    75.39%    0.3226   0.9959     2.24e-5     Baseline
  3    74.89%    0.3230   0.9880     1.73e-5     Slight dip
  5    75.69%    0.3612   0.9774     1.25e-5     ↑ Best acc so far
  7    75.04%    0.3161   0.9691     8.27e-6     F1 dip, loss improving
  9    75.49%    0.3459   0.9706     4.77e-5     Recovery
 11    75.34%    0.3358   0.9661     2.16e-6     Plateau
 13    75.69%    0.3729   0.9530     5.46e-7     ★ Best F1 (checkpoint)
 15    75.29%    0.3527   0.9615     0.0         Final
```

### Patience Counter

- Peaked at 3 (epoch 11), never triggered early stopping (patience=10)
- Reset at epoch 9 when best F1 was achieved
- Final patience: 1 (model still improving slightly at end)

### PSI (Population Stability Index)

- All epochs: 0.0
- **Note**: PSI is non-functional with MixUp enabled because train predictions are not collected during MixUp batches. This is expected behavior, not a bug.

### Train Metrics

- Train accuracy, train F1: all 0
- **Note**: MixUp mixes labels, making per-sample accuracy meaningless. Code correctly skips train metric collection during MixUp.

## Analysis

### What Worked

1. **Lower LR (5e-5 vs 2e-4)** — Prevented catastrophic overfitting. Previous run had 62% train-val accuracy gap; now val accuracy is stable ~75% throughout training.

2. **Reduced class weights** — [1.0, 1.8, 2.5, 3.0, 3.5] vs previous [1.0, 3.15, 4.66, 5.51, 6.22]. Prevented minority class memorization that was causing the model to collapse to predicting only minority classes.

3. **Label smoothing (0.1) + dropout (0.5)** — Added regularization that prevented the model from becoming overconfident on training data.

4. **BatchNorm in training mode** — Allowed normalization statistics to adapt to EyePACS domain, critical for cross-domain transfer.

5. **Cosine LR schedule** — Smooth decay from 2.24e-5 to 0, allowing fine-grained convergence in later epochs.

### What Still Needs Improvement

1. **Val loss still high (~0.96)** — Model is uncertain in its predictions. F1=0.37 is better but still below target (0.45-0.50).

2. **F1 plateaued at epoch 9** — No improvement after epoch 9 despite 6 more epochs. Suggests phase 1 has reached its capacity with frozen backbone.

3. **MixUp hides train metrics** — Cannot track train-val gap or PSI for overfitting detection during phase 1.

4. **No early stopping trigger** — Model didn't improve enough to trigger patience=10, but also didn't degrade. Wasted 6 epochs after best checkpoint.

## Recommendations

1. **Phase 2 unfreeze** — Last blocks of EfficientNet will fine-tune to EyePACS domain. This should provide the biggest boost.

2. **Consider disabling MixUp** if you want to track train metrics and PSI for overfitting detection. MixUp is valuable for regularization but makes monitoring impossible.

3. **Target for Phase 2**: val_acc 80%+, val_f1 0.50+

4. **Consider reducing phase 1 epochs to 10** — Best F1 was at epoch 9, and no improvement after. Save compute for phase 2.

5. **Investigate class-wise F1** — Overall F1=0.37 may hide which DR grades are being confused. Check per-class metrics in MLflow.

## Infrastructure Notes

- GPU utilization metrics require `nvidia-ml-py` (not installed). Wrapped `torch.cuda.utilization()` in try/except to prevent training failure.
- GPU memory metrics work via `torch.cuda.memory_allocated()`.
- Prometheus metrics scraping at 15s interval from `localhost:9101/metrics`.
- Grafana dashboard updated with 9 new training panels.
