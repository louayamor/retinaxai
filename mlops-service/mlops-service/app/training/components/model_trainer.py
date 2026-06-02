import time
import timm
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np

from app.config.config_entity import (
    ImagingModelTrainerConfig,
    ImagingTransformationConfig,
)
from app.utils.common import read_yaml, set_seed
from app.training.components.ordinal_loss import (
    OrdinalCrossEntropyLoss,
    FocalOrdinalLoss,
)
from app.training.components.fda_augment import FDAAugment
from app.training.components.epoch_metrics import EpochMetrics
from app.training.components.training_logger import TrainingLogger
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH


class RetinalDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"Failed to load image {row['image_path']}: {e}") from e
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


class ImagingModelTrainer:
    def __init__(
        self,
        config: ImagingModelTrainerConfig,
        transformation_config: ImagingTransformationConfig,
        phase: str = "phase1",
        load_checkpoint: Path | None = None,
        custom_train_csv: Path | None = None,
    ):
        self.config = config
        self.transformation_config = transformation_config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        from app.utils.common import require_cuda

        self.device = require_cuda(min_free_bytes=800_000_000)

        self.phase = phase
        self.load_checkpoint = load_checkpoint
        self.custom_train_csv = custom_train_csv

        training_cfg = self.params.get("training", {}) or {}
        global_cfg = self.params.get("global", {}) or {}

        self._global_num_classes = int(global_cfg.get("num_classes", 5))
        self._global_image_size = int(global_cfg.get("image_size", 384))
        self._global_seed = int(global_cfg.get("seed", 42))

        self._training_batch_size = int(training_cfg.get("batch_size", 8))
        self._training_num_workers = int(training_cfg.get("num_workers", 4))
        self._training_weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self._training_scheduler = training_cfg.get("scheduler", "reduce_on_plateau")
        self._training_lr_warmup_epochs = int(training_cfg.get("lr_warmup_epochs", 0))
        self._training_label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
        self._training_pin_memory = training_cfg.get("pin_memory", True) and torch.cuda.is_available()
        self._gradient_accumulation_steps = int(
            training_cfg.get("gradient_accumulation_steps", 1)
        )

        self.use_phase_based = phase in ("phase1", "phase2")

        if self.use_phase_based:
            phase_specific = training_cfg.get(phase, {}) or {}
            self.phase_epochs = int(phase_specific.get("epochs", 15))
            self.phase_lr = float(phase_specific.get("lr", 0.0001))

            self.freeze_backbone = phase_specific.get("freeze_backbone", True)
            self._freeze_blocks = phase_specific.get("freeze_blocks", 3)
            self.unfreeze_last_blocks = phase_specific.get(
                "unfreeze_last_blocks", False
            )

            self._focal_loss_gamma = phase_specific.get("focal_loss_gamma", 1.5)
            self._loss_type = phase_specific.get("loss", "ordinal_cross_entropy")

            self._training_dropout = float(phase_specific.get("dropout", 0.5))
            self._phase_dropout = self._training_dropout
            self._backbone_lr_ratio = float(
                phase_specific.get("backbone_lr_ratio", 1.0)
            )
        else:
            phase_specific = training_cfg.get("phase1", {}) or {}
            self.phase_epochs = int(phase_specific.get("epochs", 15))
            self.phase_lr = float(phase_specific.get("lr", 0.0001))

            self.freeze_backbone = False
            self._freeze_blocks = training_cfg.get("freeze_blocks", 3)
            self.unfreeze_last_blocks = False
            self._focal_loss_gamma = None
            self._loss_type = "ordinal_cross_entropy"

            self._training_dropout = float(phase_specific.get("dropout", 0.5))
            self._phase_dropout = self._training_dropout

        logger.info(
            f"phase={phase} epochs={self.phase_epochs} lr={self.phase_lr} "
            f"freeze_backbone={self.freeze_backbone} freeze_blocks={self._freeze_blocks} "
            f"loss={self._loss_type}"
        )

        self._class_weights_tensor: torch.Tensor | None = None
        use_weights = training_cfg.get("use_class_weights_in_loss", False)
        if use_weights:
            raw_weights = training_cfg.get("class_weights", [])
            if raw_weights:
                self._class_weights_tensor = torch.tensor(
                    raw_weights, dtype=torch.float32
                )
                logger.info(
                    f"class weights enabled: {self._class_weights_tensor.tolist()}"
                )

        self._fda_augment: FDAAugment | None = None
        fda_global = self.params.get("fda", {}) or {}

        if self.use_phase_based:
            phase_specific = training_cfg.get(self.phase, {}) or {}
            phase_fda = phase_specific.get("fda", {}) or {}
            fda_enabled = phase_fda.get("enabled", fda_global.get("enabled", False))
            fda_beta = phase_fda.get("beta", fda_global.get("beta", 0.15))
            fda_prob = phase_fda.get("probability", fda_global.get("probability", 0.5))
        else:
            fda_enabled = fda_global.get("enabled", False)
            fda_beta = fda_global.get("beta", 0.15)
            fda_prob = fda_global.get("probability", 0.5)

        if fda_enabled:
            target_dir = Path(fda_global.get("target_images_dir", ""))
            if str(target_dir):
                if not target_dir.is_absolute():
                    target_dir = Path.cwd() / target_dir
                cache_path_raw = fda_global.get("cache_path", "")
                cache_path = Path(cache_path_raw) if cache_path_raw else None
                if cache_path and not cache_path.is_absolute():
                    cache_path = Path.cwd() / cache_path

                self._fda_augment = FDAAugment(
                    target_images_dir=target_dir,
                    beta=float(fda_beta),
                    probability=float(fda_prob),
                    cache_path=cache_path,
                    expected_size=self.config.image_size,
                )
                _ = self._fda_augment.target_amplitude
                logger.info(
                    f"FDA augmentation loaded: beta={fda_beta}, prob={fda_prob}"
                )

        aug_cfg = self.params.get("augmentation", {}) or {}
        global_mixup = aug_cfg.get("mixup", {}) or {}

        if self.use_phase_based:
            phase_specific = training_cfg.get(self.phase, {}) or {}
            phase_mixup = phase_specific.get("mixup", {}) or {}
            self._use_mixup = phase_mixup.get(
                "enabled", global_mixup.get("enabled", False)
            )
            self._mixup_alpha = float(
                phase_mixup.get("alpha", global_mixup.get("alpha", 0.4))
                if self._use_mixup
                else 0.0
            )
        else:
            self._use_mixup = global_mixup.get("enabled", False)
            self._mixup_alpha = (
                float(global_mixup.get("alpha", 0.2)) if self._use_mixup else 0.0
            )

        if self._use_mixup:
            logger.info(f"MixUp enabled: alpha={self._mixup_alpha}")

    def _build_transforms(self):
        aug: dict = self.params.get("augmentation", {}) or {}
        norm = aug.get("normalize", {}) or {}
        dr: dict = aug.get("domain_robustness", {}) or {}
        image_size = int(self.config.image_size)
        cj: dict = aug.get("color_jitter", {}) or {}

        train_tf_list: list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(aug.get("random_rotation", 15)),
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.05),
            ),
        ]

        affine_translate = tuple(dr.get("random_affine_translate", [0.05, 0.05]))
        affine_scale = tuple(dr.get("random_affine_scale", [0.95, 1.05]))
        train_tf_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=affine_translate,
                scale=affine_scale,
            )
        )

        grayscale_prob = float(dr.get("random_grayscale_prob", 0.0))
        if grayscale_prob > 0:
            train_tf_list.append(transforms.RandomGrayscale(p=grayscale_prob))

        sharpness = float(dr.get("random_sharpness", 0.0))
        if sharpness > 0:
            train_tf_list.append(
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness, p=0.3)
            )

        train_tf_list.append(transforms.ToTensor())

        if self._fda_augment is not None:
            train_tf_list.append(transforms.Lambda(lambda t: self._fda_augment(t)))  # type: ignore[arg-type]

        train_tf_list.append(transforms.Normalize(mean=norm.get("mean", [0.485, 0.456, 0.406]), std=norm.get("std", [0.229, 0.224, 0.225])))

        train_tf = transforms.Compose(train_tf_list)

        val_tf_list: list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.get("mean", [0.485, 0.456, 0.406]), std=norm.get("std", [0.229, 0.224, 0.225])),
        ]
        val_tf = transforms.Compose(val_tf_list)
        return train_tf, val_tf

    def _build_model(self) -> nn.Module:
        dropout_rate = getattr(self, "_phase_dropout", self._training_dropout)

        model = timm.create_model(
            self.config.model_name,
            pretrained=self.config.pretrained,
            num_classes=self._global_num_classes,
            drop_rate=dropout_rate,
        )

        blocks: list[torch.nn.Module] = (
            list(model.blocks.children()) if hasattr(model, "blocks") else []  # type: ignore[attr-defined]
        )
        num_blocks = len(blocks)

        if self.use_phase_based:
            if self.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.classifier.parameters():
                    param.requires_grad = True

                if self.unfreeze_last_blocks and num_blocks > 0:
                    num_unfreeze = min(self._freeze_blocks, num_blocks)
                    unfreeze_from = num_blocks - num_unfreeze

                    for i in range(unfreeze_from, num_blocks):
                        for param in blocks[i].parameters():
                            param.requires_grad = True

                    logger.info(
                        f"Phase {self.phase}: backbone frozen, unfreezing last {num_unfreeze} blocks "
                        f"(indices {unfreeze_from} to {num_blocks - 1})"
                    )
                else:
                    logger.info(
                        f"Phase {self.phase}: backbone FULLY frozen (only classifier trainable), "
                        f"BatchNorm in training mode for domain adaptation"
                    )
            else:
                logger.info(
                    f"Phase {self.phase}: backbone UNFROZEN for full feature learning"
                )
        else:
            freeze_until = min(getattr(self, "_freeze_blocks", 3), num_blocks)
            for i, block in enumerate(blocks):
                if i < freeze_until:
                    for param in block.parameters():
                        param.requires_grad = False

            logger.info(f"Default mode: froze first {freeze_until} blocks")

        model.set_grad_checkpointing(enable=True)  # type: ignore[attr-defined]
        logger.info(f"gradient checkpointing enabled, dropout={dropout_rate}")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        trainable_pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info(
            f"trainable params: {trainable:,} / {total:,} ({trainable_pct:.2f}%)"
        )

        if self.freeze_backbone and not self.unfreeze_last_blocks:
            if trainable_pct > 10.0:
                logger.warning(
                    f">>> WARNING: freeze_backbone=True, unfreeze_last_blocks=False, "
                    f"but {trainable_pct:.2f}% params are trainable! "
                    f"This should be < 5% (classifier only)."
                )

        if self.load_checkpoint and self.load_checkpoint.exists():
            logger.info(f"loading checkpoint: {self.load_checkpoint}")
            state_dict = torch.load(self.load_checkpoint, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info("checkpoint loaded successfully")

        return model.to(self.device)

    def train(self) -> Path:
        set_seed(self._global_seed)
        train_tf, val_tf = self._build_transforms()

        train_csv_path = self.custom_train_csv or self.transformation_config.train_csv
        train_dataset = RetinalDataset(train_csv_path, train_tf)
        val_dataset = RetinalDataset(self.transformation_config.val_csv, val_tf)

        data_cfg = self.params.get("data", {}) or {}
        keep_no_dr_ratio = float(data_cfg.get("keep_no_dr_ratio", 1.0))
        if keep_no_dr_ratio < 1.0:
            df = train_dataset.df
            class0_mask = df["label"] == 0
            if class0_mask.any():
                class0 = df[class0_mask]
                n_orig = len(class0)
                n_keep = max(1, int(n_orig * keep_no_dr_ratio))
                kept_class0 = class0.sample(n=n_keep, random_state=self._global_seed)
                non_class0 = df[~class0_mask]
                train_dataset.df = pd.concat(
                    [kept_class0, non_class0], ignore_index=True
                )
                logger.info(
                    f"keep_no_dr_ratio={keep_no_dr_ratio}: class 0 {n_orig} -> {n_keep} "
                    f"(dataset now {len(train_dataset)} samples)"
                )

        training_cfg = self.params.get("training", {}) or {}

        use_weighted_sampling = training_cfg.get("weighted_sampling", False)
        sampler: WeightedRandomSampler | None = None
        if use_weighted_sampling:
            labels = train_dataset.df["label"].values
            class_counts = np.bincount(labels, minlength=5)
            class_weights = class_counts.max() / (class_counts + 1e-6)
            weight_cap = float(training_cfg.get("sampler_weight_cap", 8.0))
            class_weights = np.clip(class_weights, 0.0, weight_cap)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )
            logger.info(
                f"weighted sampler enabled: class_counts={class_counts.tolist()}, "
                f"inv_weights={class_weights.round(3).tolist()}"
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self._training_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self._training_num_workers,
            pin_memory=self._training_pin_memory,
            prefetch_factor=4,
            persistent_workers=self._training_num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._training_batch_size,
            shuffle=False,
            num_workers=self._training_num_workers,
            pin_memory=self._training_pin_memory,
            prefetch_factor=4,
            persistent_workers=self._training_num_workers > 0,
        )

        model = self._build_model()

        backbone_lr_ratio = getattr(self, "_backbone_lr_ratio", 1.0)
        apply_diff_lr = backbone_lr_ratio < 1.0 and any(
            p.requires_grad for p in model.parameters()
        )

        if apply_diff_lr:
            blocks = list(model.blocks.children()) if hasattr(model, "blocks") else []
            unfreeze_from = max(0, len(blocks) - 2)
            backbone_param_ids = set()
            for i, block in enumerate(blocks):
                if i < unfreeze_from:
                    for p in block.parameters():
                        backbone_param_ids.add(id(p))
            backbone_params = [
                p
                for p in model.parameters()
                if id(p) in backbone_param_ids and p.requires_grad
            ]
            head_params = [
                p
                for p in model.parameters()
                if id(p) not in backbone_param_ids and p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": backbone_params,
                        "lr": self.phase_lr * backbone_lr_ratio,
                    },
                    {"params": head_params, "lr": self.phase_lr},
                ],
                weight_decay=self._training_weight_decay,
            )
            logger.info(
                f"Differential LR: backbone={self.phase_lr * backbone_lr_ratio}, head={self.phase_lr}"
            )
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.phase_lr,
                weight_decay=self._training_weight_decay,
            )

        scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        if scaler:
            logger.info("AMP (mixed precision) enabled")

        warmup_epochs = self._training_lr_warmup_epochs
        if warmup_epochs > 0:
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.phase_epochs - warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    base_scheduler,
                ],
                milestones=[warmup_epochs],
            )
            logger.info(f"LR warmup enabled: {warmup_epochs} epochs")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.phase_epochs
            )

        criteria_weights = None
        if self._class_weights_tensor is not None:
            criteria_weights = self._class_weights_tensor.to(self.device)

        label_smoothing = self._training_label_smoothing

        loss_type = getattr(self, "_loss_type", "ordinal_cross_entropy")
        focal_gamma = getattr(self, "_focal_loss_gamma", None)
        dropout_rate = getattr(self, "_phase_dropout", self._training_dropout)

        use_focal = focal_gamma is not None and focal_gamma > 0
        if "focal" in loss_type.lower() or use_focal:
            actual_gamma = focal_gamma if focal_gamma is not None else 2.0
            criterion = FocalOrdinalLoss(
                num_classes=self._global_num_classes,
                distance_weight=0.1,
                gamma=float(actual_gamma),
                class_weights=criteria_weights,
                label_smoothing=label_smoothing,
            )
            logger.info(
                f"Using FocalOrdinalLoss (gamma={actual_gamma}, loss_type={loss_type}) "
                f"for ordinal DR grading"
            )
        else:
            criterion = OrdinalCrossEntropyLoss(
                num_classes=self._global_num_classes,
                distance_weight=0.1,
                class_weights=criteria_weights,
                label_smoothing=label_smoothing,
            )
            logger.info(
                f"Using OrdinalCrossEntropyLoss (loss_type={loss_type}) "
                f"for ordinal DR grading (preserves grade ordering)"
            )

        early_stopping_metric_name = training_cfg.get("early_stopping_metric", "qwk")
        best_val = -1.0
        best_val_acc = 0.0
        best_epoch_idx = -1
        patience_counter = 0
        checkpoint_path = self.config.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        accum_steps = self._gradient_accumulation_steps
        use_accum = accum_steps > 1
        num_batches = len(train_loader)

        with TrainingLogger(self) as run_logger:
            for epoch in range(self.phase_epochs):
                epoch_start = time.perf_counter()
                model.train()
                train_loss, train_total = 0.0, 0
                optimizer.zero_grad()

                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    if self._use_mixup:
                        lam = np.random.beta(self._mixup_alpha, self._mixup_alpha)
                        lam = max(lam, 1.0 - lam)
                        index = torch.randperm(images.size(0), device=self.device)
                        mixed_images = lam * images + (1.0 - lam) * images[index]
                        labels_a, labels_b = labels, labels[index]

                        if scaler:
                            with torch.amp.autocast("cuda"):
                                outputs = model(mixed_images)
                                loss = lam * criterion(outputs, labels_a) + (
                                    1.0 - lam
                                ) * criterion(outputs, labels_b)
                            scaler.scale(loss / accum_steps).backward()
                        else:
                            outputs = model(mixed_images)
                            loss = lam * criterion(outputs, labels_a) + (
                                1.0 - lam
                            ) * criterion(outputs, labels_b)
                            (loss / accum_steps).backward()

                        train_loss += loss.item() * images.size(0)
                        train_total += images.size(0)
                    else:
                        if scaler:
                            with torch.amp.autocast("cuda"):
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss / accum_steps).backward()
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            (loss / accum_steps).backward()

                        train_loss += loss.item() * images.size(0)
                        train_total += images.size(0)

                    if (i + 1) % 200 == 0:
                        elapsed = time.perf_counter() - epoch_start
                        lr_current = optimizer.param_groups[0]["lr"]
                        avg_loss = train_loss / max(train_total, 1)
                        logger.info(
                            f"epoch {epoch + 1}/{self.phase_epochs} "
                            f"batch {i + 1}/{num_batches} "
                            f"loss={avg_loss:.4f} "
                            f"lr={lr_current:.6f} "
                            f"elapsed={elapsed:.1f}s"
                        )

                    if (i + 1) % accum_steps == 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            optimizer.step()
                        optimizer.zero_grad()

                if use_accum and num_batches % accum_steps != 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()

                model.eval()
                val_correct, val_total = 0, 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        preds = outputs.argmax(1)
                        val_correct += (preds == labels).sum().item()
                        val_total += images.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                clean_train_preds, clean_train_labels = [], []
                with torch.no_grad():
                    for images, labels in train_loader:
                        images = images.to(self.device)
                        outputs = model(images)
                        clean_train_preds.extend(outputs.argmax(1).cpu().numpy())
                        clean_train_labels.extend(labels.cpu().numpy())

                model.train()

                epoch_duration = time.perf_counter() - epoch_start
                avg_loss = train_loss / train_total
                lr = float(scheduler.get_last_lr()[0])

                metrics = EpochMetrics.from_predictions(
                    epoch=epoch + 1,
                    phase=self.phase,
                    train_preds=clean_train_preds,
                    train_labels=clean_train_labels,
                    val_preds=all_preds,
                    val_labels=all_labels,
                    avg_loss=avg_loss,
                    lr=lr,
                    duration_s=epoch_duration,
                    num_classes=self._global_num_classes,
                )
                run_logger.log_epoch(metrics, patience_counter)
                run_logger.log_artifact(metrics)

                logger.info(
                    f"epoch={metrics.epoch}/{self.phase_epochs} "
                    f"loss={metrics.loss:.4f} "
                    f"train_acc={metrics.train_acc:.4f} "
                    f"val_acc={metrics.val_acc:.4f} "
                    f"train_f1={metrics.train_f1:.4f} "
                    f"val_f1={metrics.val_f1:.4f} "
                    f"qwk={metrics.qwk:.4f} "
                    f"mae={metrics.mae:.4f} "
                    f"rmse={metrics.rmse:.4f} "
                    f"recall=[{', '.join(f'{r:.3f}' for r in metrics.class_recall)}] "
                    f"lr={metrics.lr:.6f} "
                    f"duration={metrics.duration_s:.1f}s"
                )

                current_val = getattr(metrics, early_stopping_metric_name)
                if current_val > best_val:
                    best_val = current_val
                    best_val_acc = metrics.val_acc
                    best_epoch_idx = epoch
                    patience_counter = 0
                    try:
                        torch.save(model.state_dict(), checkpoint_path)
                    except (OSError, RuntimeError) as e:
                        raise RuntimeError(
                            f"Failed to save model checkpoint: {e}"
                        ) from e
                    run_logger.mark_checkpoint(metrics.qwk, metrics.val_f1)
                    logger.info(
                        f"checkpoint saved: "
                        f"{early_stopping_metric_name}={current_val:.4f} "
                        f"qwk={metrics.qwk:.4f} f1={metrics.val_f1:.4f}"
                    )
                else:
                    patience_counter += 1

                early_stopping_patience = training_cfg.get("early_stopping_patience", 8)
                if patience_counter >= early_stopping_patience:
                    logger.info(f"early stopping at epoch {metrics.epoch}")
                    break

            run_logger.summarize(best_epoch_idx, best_val_acc)
            if self.phase != "phase1":
                run_logger.log_model(
                    checkpoint_path,
                    self.config.model_name,
                    self._global_num_classes,
                    dropout_rate,
                )

        logger.info(f"training complete. best_val_acc={best_val_acc:.4f}")
        logger.info(f"model saved: {checkpoint_path}")
        return checkpoint_path
