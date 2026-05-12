import mlflow
import mlflow.pytorch
import time
import timm
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.entity.config_entity import (
    ImagingModelTrainerConfig,
    ImagingTransformationConfig,
)
from app.utils.common import read_yaml, set_seed
from app.domains.imaging.components.ordinal_loss import (
    OrdinalCrossEntropyLoss,
    FocalOrdinalLoss,
)
from app.domains.imaging.components.fda_augment import FDAAugment
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.services.monitoring.prometheus_metrics import (
    BEST_VAL_ACCURACY,
    EPOCH_TRAIN_LOSS,
    TRAINING_CURRENT_EPOCH,
    TRAINING_TOTAL_EPOCHS,
    TRAINING_EPOCH_ACCURACY,
    TRAINING_EPOCH_F1,
    TRAINING_LEARNING_RATE,
    TRAINING_EPOCH_DURATION,
    TRAINING_EPOCH_PSI,
    TRAINING_BEST_F1,
    TRAINING_PATIENCE_COUNTER,
    TRAINING_VAL_LOSS,
    TRAINING_PER_CLASS_F1,
    ACTIVE_TRAINING_JOBS,
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
    compute_psi,
)


def _parse_unfreeze_last_blocks(raw: bool | int) -> int:
    if isinstance(raw, bool):
        return 2 if raw else 0
    if isinstance(raw, (int, float)):
        return max(0, int(raw))
    return 0


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
        except Exception as e:
            raise RuntimeError(f"Failed to load image {row['image_path']}: {e}") from e
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"]), str(row["image_path"])


class ImagingModelTrainer:
    def __init__(
        self,
        config: ImagingModelTrainerConfig,
        transformation_config: ImagingTransformationConfig,
        phase: str = "phase1",
        load_checkpoint: Path | None = None,
        custom_train_csv: Path | None = None,
        register_model: bool = True,
    ):
        self.config = config
        self.transformation_config = transformation_config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        if torch.cuda.is_available():
            try:
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                free_memory = total_memory - allocated

                MIN_FREE_FOR_CUDA = 800_000_000  # 800MB minimum for training
                if free_memory > MIN_FREE_FOR_CUDA:
                    self.device = torch.device("cuda")
                    logger.info(
                        f"training device: cuda (total={total_memory / 1e9:.1f}GB, "
                        f"free={free_memory / 1e9:.1f}GB)"
                    )
                else:
                    self.device = torch.device("cpu")
                    logger.warning(
                        f"GPU memory low ({free_memory / 1e9:.1f}GB < 0.8GB required), "
                        f"using CPU for training. Stop MLOps/LLMOps services to free GPU."
                    )
            except Exception as e:
                self.device = torch.device("cpu")
                logger.warning(f"GPU check failed, using CPU: {e}")
        else:
            self.device = torch.device("cpu")
            logger.info(f"training device: {self.device}")

        self.phase = phase
        self.load_checkpoint = load_checkpoint
        self.custom_train_csv = custom_train_csv
        self.register_model = register_model

        training_cfg = self.params.get("training", {}) or {}
        global_cfg = self.params.get("global", {}) or {}

        self._global_num_classes = int(global_cfg.get("num_classes", 5))
        self._global_image_size = int(global_cfg.get("image_size", 300))
        self._global_seed = int(global_cfg.get("seed", 42))

        self._training_batch_size = int(training_cfg.get("batch_size", 8))
        self._training_num_workers = int(training_cfg.get("num_workers", 4))
        self._training_weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self._training_scheduler = training_cfg.get("scheduler", "reduce_on_plateau")
        self._training_lr_warmup_epochs = int(training_cfg.get("lr_warmup_epochs", 0))
        self._training_label_smoothing = float(training_cfg.get("label_smoothing", 0.0))

        self.use_phase_based = phase in ("phase1", "phase2")

        if self.use_phase_based:
            phase_specific = training_cfg.get(phase, {}) or {}
            self.phase_epochs = int(phase_specific.get("epochs", 15))
            self.phase_lr = float(phase_specific.get("lr", 0.0001))

            self.freeze_backbone = phase_specific.get("freeze_backbone", True)
            self._freeze_blocks = phase_specific.get("freeze_blocks", 3)
            self._unfreeze_last_count = _parse_unfreeze_last_blocks(
                phase_specific.get("unfreeze_last_blocks", False)
            )
            self._focal_loss_gamma = phase_specific.get("focal_loss_gamma", 1.5)
            self._loss_type = phase_specific.get("loss", "ordinal_cross_entropy")

            self._training_dropout = float(phase_specific.get("dropout", 0.5))
            self._phase_dropout = self._training_dropout
        else:
            phase_specific = training_cfg.get("phase1", {}) or {}
            self.phase_epochs = int(phase_specific.get("epochs", 15))
            self.phase_lr = float(phase_specific.get("lr", 0.0001))

            self.freeze_backbone = False
            self._freeze_blocks = training_cfg.get("freeze_blocks", 3)
            self._unfreeze_last_count = 0
            self._focal_loss_gamma = None
            self._loss_type = "ordinal_cross_entropy"

            self._training_dropout = float(phase_specific.get("dropout", 0.5))
            self._phase_dropout = self._training_dropout

        logger.info(
            f"phase={phase} epochs={self.phase_epochs} lr={self.phase_lr} "
            f"freeze_backbone={self.freeze_backbone} freeze_blocks={self._freeze_blocks} "
            f"unfreeze_last={self._unfreeze_last_count} "
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
        aug = self.params.augmentation
        norm = aug.normalize
        dr = aug.get("domain_robustness", {}) or {}
        image_size = int(self.config.image_size)

        train_tf_list: list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(aug.random_rotation),
            transforms.ColorJitter(
                brightness=aug.color_jitter.brightness,
                contrast=aug.color_jitter.contrast,
                saturation=aug.color_jitter.saturation,
                hue=aug.color_jitter.hue,
            ),
        ]

        # Gaussian blur
        gb = aug.get("gaussian_blur", {}) or {}
        if gb.get("enabled", False):
            train_tf_list.append(
                transforms.GaussianBlur(
                    kernel_size=gb.get("kernel_size", 3),
                    sigma=tuple(gb.get("sigma", [0.1, 1.5])),
                )
            )

        # Domain robustness affine
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

        # JPEG compression (tensor -> PIL -> tensor)
        jpg = dr.get("jpeg_compression", {}) or {}
        if jpg.get("enabled", False):

            def _jpeg_compress(tensor, quality_range):
                import io
                import random

                quality = random.randint(quality_range[0], quality_range[1])
                img = transforms.ToPILImage()(tensor)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                return transforms.ToTensor()(Image.open(buf))

            train_tf_list.append(
                transforms.Lambda(
                    lambda t, qr=tuple(jpg.get("quality_range", [60, 100])): (
                        _jpeg_compress(t, qr)
                    )
                )
            )

        # Illumination shift (gamma)
        ill = dr.get("illumination_shift", {}) or {}
        if ill.get("enabled", False):

            def _gamma_correct(tensor, gamma_range):
                import random

                gamma = random.uniform(gamma_range[0], gamma_range[1])
                out = tensor ** (1.0 / gamma)
                return torch.clamp(out, 0.0, 1.0)

            train_tf_list.append(
                transforms.Lambda(
                    lambda t, gr=tuple(ill.get("gamma_range", [0.7, 1.5])): (
                        _gamma_correct(t, gr)
                    )
                )
            )

        # Gaussian noise (tensor)
        noise_std = float(dr.get("gaussian_noise_std", 0.0))
        if noise_std > 0:
            train_tf_list.append(
                transforms.Lambda(
                    lambda t, s=noise_std: torch.clamp(
                        t + torch.randn_like(t) * s, 0.0, 1.0
                    )
                )
            )

        # Random erasing (tensor, before normalization)
        er = aug.get("random_erasing", {}) or {}
        if er.get("enabled", False):
            train_tf_list.append(
                transforms.RandomErasing(
                    p=er.get("probability", 0.15),
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                )
            )

        if self._fda_augment is not None:
            train_tf_list.append(transforms.Lambda(lambda t: self._fda_augment(t)))  # type: ignore[arg-type]

        train_tf_list.append(transforms.Normalize(mean=norm.mean, std=norm.std))

        train_tf = transforms.Compose(train_tf_list)

        val_tf_list: list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.mean, std=norm.std),
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

                unfreeze_count = self._unfreeze_last_count
                if unfreeze_count > 0 and num_blocks > 0:
                    num_unfreeze = min(unfreeze_count, num_blocks)
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

        if self.freeze_backbone and self._unfreeze_last_count == 0:
            if trainable_pct > 10.0:
                logger.warning(
                    f">>> WARNING: freeze_backbone=True, unfreeze_last_blocks=0, "
                    f"but {trainable_pct:.2f}% params are trainable! "
                    f"This should be < 5% (classifier only)."
                )

        if self.load_checkpoint and self.load_checkpoint.exists():
            logger.info(f"loading checkpoint: {self.load_checkpoint}")
            state_dict = torch.load(self.load_checkpoint, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info("checkpoint loaded successfully")

        return model.to(self.device)

    def _log_best_model_to_mlflow(
        self, checkpoint_path: Path, num_classes: int
    ) -> None:
        if not self.register_model:
            logger.info(
                f"model registration disabled for phase={self.phase}, skipping mlflow model log"
            )
            return

        if not checkpoint_path.exists():
            logger.warning("best checkpoint missing; skipping mlflow model log")
            return

        logger.info(f"logging best model to mlflow: {checkpoint_path}")

        mlflow.log_param("checkpoint_path", str(checkpoint_path))

        try:
            import timm

            model = timm.create_model(
                self.config.model_name,
                pretrained=False,
                num_classes=num_classes,
                drop_rate=self._training_dropout,
            )
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="imaging_model",
                registered_model_name="efficientnet_b3",
            )
            logger.info("model logged to mlflow model registry")
        except Exception as e:
            logger.warning(f"failed to log model to mlflow registry: {e}")

    def _save_misclassified_from_batch(
        self,
        preds: list,
        labels: list,
        probs: list,
        paths: list,
        phase: str,
        epoch: int,
        max_samples: int = 10,
    ) -> None:
        """Save misclassified images from a validation batch."""
        import shutil

        misclassified = []
        for i, (pred, label, prob) in enumerate(zip(preds, labels, probs)):
            if pred != label:
                confidence = float(np.max(prob))
                misclassified.append(
                    {
                        "path": paths[i] if i < len(paths) else f"unknown_{i}",
                        "true_label": int(label),
                        "pred_label": int(pred),
                        "confidence": confidence,
                    }
                )

        if not misclassified:
            logger.info(f"[epoch {epoch}] no misclassified samples")
            return

        misclassified.sort(key=lambda x: x["confidence"], reverse=True)
        top_misclassified = misclassified[:max_samples]

        output_dir = (
            self.config.checkpoint_path.parent
            / "misclassified"
            / f"phase{phase[-1]}"
            / f"epoch{epoch}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(top_misclassified):
            src = Path(sample["path"])
            if not src.exists():
                continue
            dst = (
                output_dir
                / f"mis_{i:03d}_true{sample['true_label']}_pred{sample['pred_label']}_conf{sample['confidence']:.2f}{src.suffix}"
            )
            shutil.copy2(src, dst)

        logger.info(
            f"[epoch {epoch}] saved {len(top_misclassified)} misclassified images to {output_dir}"
        )

    def train(self) -> Path:
        set_seed(self._global_seed)
        train_tf, val_tf = self._build_transforms()

        train_csv_path = self.custom_train_csv or self.transformation_config.train_csv
        train_dataset = RetinalDataset(train_csv_path, train_tf)
        val_dataset = RetinalDataset(self.transformation_config.val_csv, val_tf)

        training_cfg = self.params.get("training", {}) or {}

        use_weighted_sampling = training_cfg.get("weighted_sampling", False)
        sampler: WeightedRandomSampler | None = None
        if use_weighted_sampling:
            labels = train_dataset.df["label"].values
            class_counts = np.bincount(labels, minlength=5)
            class_weights = 1.0 / (class_counts + 1e-6)
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
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._training_batch_size,
            shuffle=False,
            num_workers=self._training_num_workers,
            pin_memory=False,
        )

        model = self._build_model()
        # Differential learning rate for Phase 2 (backbone 0.1x head LR)
        # Note: EfficientNet doesn't have backbone attribute, so this falls back to regular optimizer
        if self.phase == "phase2":
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
                    {"params": backbone_params, "lr": self.phase_lr * 0.1},
                    {"params": head_params, "lr": self.phase_lr},
                ],
                weight_decay=self._training_weight_decay,
            )
            logger.info(
                f"Phase 2 differential LR: backbone={self.phase_lr * 0.1}, head={self.phase_lr}"
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

        phase_specific = training_cfg.get(self.phase, {}) or {}
        gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
        early_stopping_min_delta = float(
            training_cfg.get("early_stopping_min_delta", 0.0)
        )
        save_misclassified = training_cfg.get("save_misclassified_images", True)
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

        best_macro_f1 = 0.0
        best_val_acc = 0.0
        patience_counter = 0
        checkpoint_path = self.config.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        run_suffix = f"_{int(time.time()) % 1000:03d}"
        with mlflow.start_run(
            run_name=self.params.get("mlflow", {}).get(
                "imaging_run_name", "efficientnet_b3"
            )
            + run_suffix
        ):
            mlflow.log_params(
                {
                    "model": self.config.model_name,
                    "pretrained": self.config.pretrained,
                    "epochs": self.phase_epochs,
                    "batch_size": self._training_batch_size,
                    "lr": self.phase_lr,
                    "weight_decay": self._training_weight_decay,
                    "scheduler": self._training_scheduler,
                    "num_classes": self._global_num_classes,
                    "dropout": dropout_rate,
                    "seed": self._global_seed,
                    "loss": loss_type,
                    "focal_gamma": focal_gamma,
                    "class_weights": (
                        self._class_weights_tensor.tolist()
                        if self._class_weights_tensor is not None
                        else None
                    ),
                    "freeze_backbone": self.freeze_backbone,
                    "freeze_blocks": getattr(self, "_freeze_blocks", 3),
                    "unfreeze_last_blocks": self._unfreeze_last_count,
                    "device": str(self.device),
                    "phase": self.phase,
                    "mixup_enabled": self._use_mixup,
                    "mixup_alpha": self._mixup_alpha if self._use_mixup else 0.0,
                    "fda_enabled": self._fda_augment is not None,
                }
            )

            TRAINING_TOTAL_EPOCHS.labels(pipeline="imaging").set(self.phase_epochs)
            ACTIVE_TRAINING_JOBS.inc()

            for epoch in range(self.phase_epochs):
                epoch_start = time.perf_counter()
                model.train()
                train_loss, train_total = 0.0, 0

                for images, labels, _ in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    if self._use_mixup:
                        lam = np.random.beta(self._mixup_alpha, self._mixup_alpha)
                        lam = max(lam, 1.0 - lam)
                        index = torch.randperm(images.size(0), device=self.device)
                        mixed_images = lam * images + (1.0 - lam) * images[index]
                        labels_a, labels_b = labels, labels[index]

                        optimizer.zero_grad()
                        if scaler:
                            with torch.amp.autocast("cuda"):
                                outputs = model(mixed_images)
                                loss = lam * criterion(outputs, labels_a) + (
                                    1.0 - lam
                                ) * criterion(outputs, labels_b)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=gradient_clip_norm
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(mixed_images)
                            loss = lam * criterion(outputs, labels_a) + (
                                1.0 - lam
                            ) * criterion(outputs, labels_b)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=gradient_clip_norm
                            )
                            optimizer.step()

                        train_loss += loss.item() * images.size(0)
                        train_total += images.size(0)
                    else:
                        optimizer.zero_grad()
                        if scaler:
                            with torch.amp.autocast("cuda"):
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=gradient_clip_norm
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=gradient_clip_norm
                            )
                            optimizer.step()

                        train_loss += loss.item() * images.size(0)
                        train_total += images.size(0)

                scheduler.step()

                model.eval()
                val_correct, val_total = 0, 0
                all_preds, all_labels = [], []
                all_probs, all_paths = [], []
                with torch.no_grad():
                    for images, labels, paths in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        preds = outputs.argmax(1)
                        val_correct += (preds == labels).sum().item()
                        val_total += images.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs)
                        all_paths.extend(
                            paths
                            if isinstance(paths, list)
                            else [str(p) for p in paths]
                        )

                # Save misclassified images from validation
                if save_misclassified and (
                    epoch == self.phase_epochs - 1 or (epoch + 1) % 5 == 0
                ):
                    self._save_misclassified_from_batch(
                        all_preds,
                        all_labels,
                        all_probs,
                        all_paths,
                        self.phase,
                        epoch + 1,
                    )

                clean_train_preds, clean_train_labels = [], []
                with torch.no_grad():
                    for images, labels, _ in train_loader:
                        images = images.to(self.device)
                        outputs = model(images)
                        clean_train_preds.extend(outputs.argmax(1).cpu().numpy())
                        clean_train_labels.extend(labels.cpu().numpy())

                model.train()

                epoch_duration = time.perf_counter() - epoch_start

                macro_f1 = float(
                    f1_score(
                        all_labels, all_preds, average="macro", zero_division="warn"
                    )
                )
                per_class_f1 = f1_score(
                    all_labels, all_preds, average=None, zero_division=0
                )
                cm = confusion_matrix(all_labels, all_preds)

                train_acc = float(
                    sum(
                        1
                        for p, lbl in zip(clean_train_preds, clean_train_labels)
                        if p == lbl
                    )
                    / max(len(clean_train_labels), 1)
                )
                train_f1 = float(
                    f1_score(
                        clean_train_labels,
                        clean_train_preds,
                        average="macro",
                        zero_division="warn",
                    )
                )
                val_acc = val_correct / val_total
                avg_loss = train_loss / train_total
                lr = float(scheduler.get_last_lr()[0])

                train_probs = np.array(clean_train_preds) / max(
                    self._global_num_classes - 1, 1
                )
                val_probs = np.array(all_preds) / max(self._global_num_classes - 1, 1)
                psi_score = compute_psi(val_probs, train_probs)

                # Update Prometheus metrics
                TRAINING_CURRENT_EPOCH.labels(pipeline="imaging").set(epoch + 1)
                TRAINING_EPOCH_ACCURACY.labels(pipeline="imaging", split="train").set(
                    train_acc
                )
                TRAINING_EPOCH_ACCURACY.labels(pipeline="imaging", split="val").set(
                    val_acc
                )
                TRAINING_EPOCH_F1.labels(pipeline="imaging", split="train").set(
                    train_f1
                )
                TRAINING_EPOCH_F1.labels(pipeline="imaging", split="val").set(macro_f1)
                TRAINING_LEARNING_RATE.labels(pipeline="imaging").set(lr)
                TRAINING_EPOCH_DURATION.labels(pipeline="imaging").set(epoch_duration)
                TRAINING_EPOCH_PSI.labels(pipeline="imaging").set(psi_score)
                TRAINING_PATIENCE_COUNTER.labels(pipeline="imaging").set(
                    patience_counter
                )
                TRAINING_VAL_LOSS.labels(pipeline="imaging").set(avg_loss)
                EPOCH_TRAIN_LOSS.labels(pipeline="imaging").observe(avg_loss)

                # Update per-class F1 gauges
                for cls_idx, cls_f1 in enumerate(per_class_f1):
                    TRAINING_PER_CLASS_F1.labels(
                        pipeline="imaging", dr_grade=str(cls_idx)
                    ).set(float(cls_f1))

                # Update GPU metrics
                if torch.cuda.is_available():
                    GPU_MEMORY_USED_BYTES.labels(device="0").set(
                        torch.cuda.memory_allocated(0)
                    )
                    try:
                        GPU_UTILIZATION_PERCENT.labels(device="0").set(
                            torch.cuda.utilization(0)
                        )
                    except Exception:
                        pass

                mlflow.log_metrics(
                    {
                        "train_loss": float(avg_loss),
                        "train_acc": float(train_acc),
                        "val_acc": float(val_acc),
                        "val_macro_f1": float(macro_f1),
                        "train_f1": float(train_f1),
                        "lr": float(lr),
                        "psi_score": float(psi_score),
                        "epoch_duration_s": float(epoch_duration),
                    },
                    step=epoch,
                )

                # Log per-class F1
                for cls_idx, cls_f1 in enumerate(per_class_f1):
                    mlflow.log_metric(
                        f"val_f1_class_{cls_idx}", float(cls_f1), step=epoch
                    )

                # Log confusion matrix as MLflow artifact
                dr_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    ax=ax,
                    xticklabels=dr_labels,
                    yticklabels=dr_labels,
                    cmap="Blues",
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(f"Confusion Matrix — Epoch {epoch + 1}")
                plt.tight_layout()
                cm_path = Path(f"/tmp/cm_epoch_{epoch + 1}.png")
                fig.savefig(cm_path)
                mlflow.log_artifact(str(cm_path), "confusion_matrices")
                plt.close(fig)

                drift_status = (
                    "DRIFT"
                    if psi_score > 0.25
                    else "MODERATE"
                    if psi_score > 0.1
                    else "STABLE"
                )
                logger.info(
                    f"epoch={epoch + 1}/{self.phase_epochs} "
                    f"loss={avg_loss:.4f} "
                    f"train_acc={train_acc:.4f} "
                    f"val_acc={val_acc:.4f} "
                    f"train_f1={train_f1:.4f} "
                    f"val_f1={macro_f1:.4f} "
                    f"lr={lr:.6f} "
                    f"psi={psi_score:.4f} [{drift_status}] "
                    f"duration={epoch_duration:.1f}s"
                )

                # Use macro-F1 for checkpointing (not accuracy)
                if macro_f1 > best_macro_f1 + early_stopping_min_delta:
                    best_macro_f1 = macro_f1
                    best_val_acc = val_acc
                    patience_counter = 0
                    try:
                        torch.save(model.state_dict(), checkpoint_path)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to save model checkpoint: {e}"
                        ) from e
                    BEST_VAL_ACCURACY.labels(pipeline="imaging").set(best_val_acc)
                    TRAINING_BEST_F1.labels(pipeline="imaging").set(best_macro_f1)
                    logger.info(f"checkpoint saved: macro_f1={macro_f1:.4f}")
                else:
                    patience_counter += 1
                    TRAINING_PATIENCE_COUNTER.labels(pipeline="imaging").set(
                        patience_counter
                    )
                    # Read phase-specific patience, fall back to global
                    phase_early_stopping = phase_specific.get(
                        "early_stopping_patience",
                        training_cfg.get("early_stopping_patience", 8),
                    )
                    if patience_counter >= phase_early_stopping:
                        logger.info(f"early stopping at epoch {epoch + 1}")
                        break

            ACTIVE_TRAINING_JOBS.dec()
            mlflow.log_metric("best_val_acc", float(best_val_acc))
            mlflow.log_metric("best_macro_f1", float(best_macro_f1))
            self._log_best_model_to_mlflow(checkpoint_path, self._global_num_classes)

        logger.info(f"training complete. best_val_acc={best_val_acc:.4f}")
        logger.info(f"model saved: {checkpoint_path}")
        return checkpoint_path
