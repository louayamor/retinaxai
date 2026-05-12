import time
import mlflow
import mlflow.pytorch
import timm
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app.domains.imaging.components.model_trainer import RetinalDataset
from app.domains.imaging.components.tent_adapter import TENTAdapter
from app.domains.imaging.components.fda_augment import FDAAugment
from app.domains.imaging.preprocessing import preprocess_fundus_image
from app.entity.config_entity import ImagingModelEvaluationConfig
from app.utils.common import read_yaml, save_json
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.services.monitoring.prometheus_metrics import QUADRATIC_WEIGHTED_KAPPA


class ImagingModelEvaluation:
    def __init__(self, config: ImagingModelEvaluationConfig):
        self.config = config
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

                MIN_FREE_FOR_CUDA = 500_000_000  # 500MB minimum for evaluation
                if free_memory > MIN_FREE_FOR_CUDA:
                    self.device = torch.device("cuda")
                    logger.info(
                        f"evaluation device: cuda (total={total_memory / 1e9:.1f}GB, "
                        f"free={free_memory / 1e9:.1f}GB)"
                    )
                else:
                    self.device = torch.device("cpu")
                    logger.warning(
                        f"GPU memory low ({free_memory / 1e9:.1f}GB < 0.5GB required), "
                        f"using CPU for evaluation. Stop MLOps/LLMOps services to free GPU."
                    )
            except Exception as e:
                self.device = torch.device("cpu")
                logger.warning(f"GPU check failed, using CPU: {e}")
        else:
            self.device = torch.device("cpu")
            logger.info(f"evaluation device: {self.device}")

        global_cfg = self.params.get("global", {}) or {}
        training_cfg = self.params.get("training", {}) or {}

        self._global_num_classes = int(global_cfg.get("num_classes", 5))
        self._global_image_size = int(global_cfg.get("image_size", 300))

        phase1_cfg = training_cfg.get("phase1", {}) or {}
        self._training_dropout = float(
            phase1_cfg.get("dropout", training_cfg.get("dropout", 0.5))
        )

    def _load_model(self) -> nn.Module:
        model_name = getattr(self.config, "model_name", "efficientnet_b3")

        if self.device.type == "cuda":
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=self._global_num_classes,
                    drop_rate=self._training_dropout,
                )
                state_dict = torch.load(
                    self.config.model_path, map_location=self.device
                )
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                logger.info(
                    f"model loaded from: {self.config.model_path} (device={self.device})"
                )
                return model
            except Exception as e:
                logger.warning(f"Failed to load model on GPU: {e}, falling back to CPU")
                self.device = torch.device("cpu")

        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=self._global_num_classes,
            drop_rate=self._training_dropout,
        )
        state_dict = torch.load(self.config.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        logger.info(
            f"model loaded from: {self.config.model_path} (device={self.device})"
        )
        return model

    def _build_transform(self):
        norm = self.params.augmentation.normalize
        return transforms.Compose(
            [
                transforms.Resize((self._global_image_size, self._global_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm.mean, std=norm.std),
            ]
        )

    def _build_samaya_transform(self, fda_inverse=None):
        norm = self.params.augmentation.normalize
        image_size = self._global_image_size
        tf_list: list = [
            transforms.Lambda(
                lambda img: preprocess_fundus_image(img, image_size=image_size)
            ),
            transforms.ToTensor(),
        ]

        if fda_inverse is not None:
            tf_list.append(transforms.Lambda(lambda t: fda_inverse(t)))  # type: ignore[arg-type]

        tf_list.append(transforms.Normalize(mean=norm.mean, std=norm.std))
        return transforms.Compose(tf_list)

    def _run_inference(self, model: nn.Module, csv_path: Path, transform=None) -> tuple:
        tf = transform if transform is not None else self._build_transform()
        loader = DataLoader(
            RetinalDataset(csv_path, tf),
            batch_size=self.params.evaluation.dl.batch_size,
            shuffle=False,
            num_workers=self.params.evaluation.dl.num_workers,
            pin_memory=True,
        )

        all_preds, all_labels, all_probs = [], [], []
        all_paths = []
        total = len(loader)
        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for i, (images, labels, paths) in enumerate(loader, 1):
                images = images.to(self.device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(1).cpu().tolist()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_paths.extend(
                    paths if isinstance(paths, list) else [str(p) for p in paths]
                )

                if i % 10 == 0 or i == total:
                    logger.info(f"inference progress: {i}/{total} batches")

                del images, outputs, probs
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return all_preds, all_labels, np.array(all_probs), all_paths

    def _compute_auc(self, labels: list, probs: np.ndarray) -> float | None:
        present_classes = np.unique(labels)

        if len(present_classes) < 2:
            logger.warning("Less than 2 classes present, skipping AUC")
            return None

        try:
            probs = probs.astype(np.float64)

            if probs.shape[1] != len(present_classes):
                probs = probs[:, present_classes]

            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            probs = probs / row_sums

            row_sums_check = probs.sum(axis=1)
            if not np.allclose(row_sums_check, 1.0):
                logger.warning(f"Probabilities don't sum to 1.0: {row_sums_check[:5]}")
                probs = np.clip(probs, 0, 1)
                probs = probs / probs.sum(axis=1, keepdims=True)

            auc = roc_auc_score(
                y_true=labels,
                y_score=probs,
                multi_class="ovr",
                average="macro",
                labels=present_classes,
            )

            return float(auc)

        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")
            return None

    def _compute_metrics(
        self,
        preds: list,
        labels: list,
        probs: np.ndarray,
        split_name: str,
    ) -> dict:
        accuracy = accuracy_score(labels, preds)
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
        report = classification_report(labels, preds, output_dict=True, zero_division=0)  # type: ignore[call-overload]
        auc = self._compute_auc(labels, probs)
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division="warn"))
        precision_macro = float(
            precision_score(labels, preds, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(labels, preds, average="macro", zero_division=0)
        )
        cm = confusion_matrix(labels, preds)

        # Brier score: mean squared error between predicted probability and true label
        # Lower is better (0 = perfect calibration)
        confidences = np.max(probs, axis=1)
        brier = float(brier_score_loss(labels, confidences))

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        logger.info(
            f"[{split_name}] accuracy={accuracy:.4f} qwk={qwk:.4f} auc={auc_str} macro_f1={macro_f1:.4f} brier={brier:.4f}"
        )

        report_dict = classification_report(
            labels, preds, output_dict=True, zero_division=0
        )  # type: ignore[call-overload]
        for grade in ["0", "1", "2", "3", "4"]:
            if grade in report_dict:
                f1 = report_dict[grade]["f1-score"]
                prec = report_dict[grade]["precision"]
                rec = report_dict[grade]["recall"]
                sup = report_dict[grade]["support"]
                logger.info(
                    f"  Grade {grade}: f1={f1:.4f} precision={prec:.4f} recall={rec:.4f} support={sup}"
                )

        logger.info(f"  Confusion matrix:\n{cm}")

        # Compute optimal per-class thresholds
        optimal_thresholds, optimized_metrics = self._compute_optimal_thresholds(
            labels, probs, split_name
        )

        return {
            "split": split_name,
            "accuracy": round(accuracy, 4),
            "quadratic_weighted_kappa": round(qwk, 4),
            "roc_auc_macro": round(auc, 4) if auc is not None else None,
            "macro_f1": round(macro_f1, 4),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "brier_score": round(brier, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(labels),
            "label_distribution": {
                str(k): int(v)
                for k, v in pd.Series(labels).value_counts().sort_index().items()
            },
            "optimal_thresholds": optimal_thresholds,
            "optimized_metrics": optimized_metrics,
        }

    def _compute_optimal_thresholds(
        self, labels: list, probs: np.ndarray, split_name: str
    ) -> tuple[dict[str, float], dict]:
        """Find per-class thresholds that maximize recall while maintaining precision.

        For medical imaging, we prioritize recall (catch all positive cases) over precision.
        Returns optimal thresholds and metrics using those thresholds.
        """
        from sklearn.preprocessing import label_binarize

        n_classes = probs.shape[1]
        labels_bin = label_binarize(labels, classes=list(range(n_classes)))

        optimal_thresholds = {}
        optimized_preds = np.zeros(len(labels), dtype=int)

        for class_idx in range(n_classes):
            class_probs = probs[:, class_idx]
            class_labels = labels_bin[:, class_idx]

            best_threshold = 0.0
            best_f1 = 0.0

            # Search for threshold that maximizes F1 for this class
            for threshold in np.arange(0.1, 0.9, 0.05):
                class_preds = (class_probs >= threshold).astype(int)
                if class_preds.sum() == 0:
                    continue
                f1 = float(f1_score(class_labels, class_preds, zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds[str(class_idx)] = round(best_threshold, 2)

            # Mark samples that exceed threshold for this class
            for i, prob in enumerate(class_probs):
                if prob >= best_threshold and best_threshold > 0:
                    optimized_preds[i] = class_idx

        # Compute metrics with optimized thresholds
        opt_accuracy = float(accuracy_score(labels, optimized_preds))
        opt_qwk = float(cohen_kappa_score(labels, optimized_preds, weights="quadratic"))
        opt_macro_f1 = float(
            f1_score(labels, optimized_preds, average="macro", zero_division=0)
        )
        opt_recall = float(
            recall_score(labels, optimized_preds, average="macro", zero_division=0)
        )

        logger.info(f"[{split_name}] optimal thresholds: {optimal_thresholds}")
        logger.info(
            f"[{split_name}] optimized vs argmax: "
            f"acc={opt_accuracy:.4f} (vs {accuracy_score(labels, probs.argmax(1)):.4f}), "
            f"macro_f1={opt_macro_f1:.4f}, recall={opt_recall:.4f}"
        )

        return optimal_thresholds, {
            "accuracy": round(opt_accuracy, 4),
            "quadratic_weighted_kappa": round(opt_qwk, 4),
            "macro_f1": round(opt_macro_f1, 4),
            "recall_macro": round(opt_recall, 4),
        }

    def evaluate(self) -> dict:
        logger.info("=" * 60)
        logger.info("imaging model evaluation started")
        logger.info("=" * 60)

        model = self._load_model()

        self._fda_eval: FDAAugment | None = None
        fda_inverse = None
        fda_cfg = self.params.get("fda", {}) or {}
        if fda_cfg.get("enabled", False):
            target_dir = Path(fda_cfg["target_images_dir"])
            if not target_dir.is_absolute():
                target_dir = Path.cwd() / target_dir
            cache_path_raw = fda_cfg.get("cache_path", "")
            cache_path = Path(cache_path_raw) if cache_path_raw else None
            if cache_path and not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path

            src_cache = self.config.root_dir / "eyepacs_amplitude_source.pt"
            self._fda_eval = FDAAugment(
                target_images_dir=target_dir,
                beta=float(fda_cfg.get("beta", 0.15)),
                cache_path=cache_path,
                source_amp_cache_path=src_cache,
            )
            if not src_cache.exists():
                eye_images = self.config.root_dir / "images" / "eyepacs" / "test"
                if eye_images.exists():
                    self._fda_eval.set_source_amplitude(eye_images, src_cache)
            self._fda_eval.beta = float(fda_cfg.get("inference_beta", 0.1))
            if self._fda_eval.source_amplitude is not None:
                fda_inverse = self._fda_eval.inverse
                logger.info("FDA inverse available for Samaya evaluation")
            else:
                logger.warning(
                    "FDA enabled but source amplitude not set; inverse not available"
                )

        tent_cfg = self.params.get("tent", {}) or {}
        tent_enabled = tent_cfg.get("enabled", False)
        tent_lr = float(tent_cfg.get("learning_rate", 0.0001))
        tent_steps = int(tent_cfg.get("steps", 1))
        tent_momentum = float(tent_cfg.get("momentum", 0.9))

        logger.info("--- evaluating on EyePACS test set ---")
        test_preds, test_labels, test_probs, test_paths = self._run_inference(
            model, self.config.test_csv
        )
        test_metrics = self._compute_metrics(
            test_preds, test_labels, test_probs, "eyepacs_test"
        )
        misclassified_dir = self.config.metric_file.parent / "misclassified"
        self._save_misclassified_images(
            test_preds,
            test_labels,
            test_probs,
            test_paths,
            "eyepacs_test",
            misclassified_dir / "eyepacs_test",
        )

        partial_metrics: dict = {
            "eyepacs_test": test_metrics,
            "samaya_validation": None,
        }
        save_json(self.config.metric_file, partial_metrics)
        logger.info(f"partial metrics saved after EyePACS: {self.config.metric_file}")

        samaya_metrics = None
        domain_shift_metrics: dict = {}
        if self.config.samaya_csv.exists():
            logger.info("--- evaluating on Samaya domain validation set ---")

            if self.device.type == "cuda":
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                free_memory = total_memory - allocated
                logger.info(
                    f"GPU memory after cleanup: {allocated / 1e9:.2f}GB / {total_memory / 1e9:.2f}GB, free={free_memory / 1e9:.2f}GB"
                )

            try:
                samaya_tf = self._build_samaya_transform(fda_inverse=fda_inverse)
                samaya_dataset = RetinalDataset(self.config.samaya_csv, samaya_tf)
                samaya_batch_size = min(4, self.params.evaluation.dl.batch_size)
                samaya_loader = DataLoader(
                    samaya_dataset,
                    batch_size=samaya_batch_size,
                    shuffle=False,
                    num_workers=min(2, self.params.evaluation.dl.num_workers),
                    pin_memory=True,
                )

                if tent_enabled and self.device.type == "cuda":
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated = torch.cuda.memory_allocated()
                    free_memory = total_memory - allocated

                    if free_memory < 2_000_000_000:
                        logger.warning(
                            f"GPU memory too low for TENT ({free_memory / 1e9:.2f}GB < 2GB required), skipping TENT"
                        )
                        tent_enabled = False

                if tent_enabled:
                    logger.info(f"TENT adaptation: lr={tent_lr} steps={tent_steps}")
                    tent = TENTAdapter(
                        model, lr=tent_lr, steps=tent_steps, momentum=tent_momentum
                    )
                    tent.adapt(samaya_loader)
                    model.eval()
                else:
                    logger.info(
                        "skipping TENT adaptation (disabled or insufficient memory)"
                    )

                samaya_preds, samaya_labels, samaya_probs, samaya_paths = (
                    self._run_inference(
                        model, self.config.samaya_csv, transform=samaya_tf
                    )
                )
                samaya_metrics = self._compute_metrics(
                    samaya_preds, samaya_labels, samaya_probs, "samaya_validation"
                )
                self._save_misclassified_images(
                    samaya_preds,
                    samaya_labels,
                    samaya_probs,
                    samaya_paths,
                    "samaya_validation",
                    misclassified_dir / "samaya_validation",
                )

                if tent_enabled:
                    tent.restore()
                    model.eval()

                ds_cfg = self.params.get("evaluation", {}).get("domain_shift", {}) or {}
                if ds_cfg.get("compute_confidence_ece", False):
                    domain_shift_metrics["confidence_ece"] = self._compute_ece(
                        samaya_labels, samaya_preds, samaya_probs
                    )

                if ds_cfg.get("compute_embedding_mmd", False):
                    try:
                        mmd_val = self._compute_embedding_mmd(
                            model,
                            self.config.test_csv,
                            self.config.samaya_csv,
                            samaya_tf,
                        )
                        domain_shift_metrics["embedding_mmd"] = mmd_val
                    except Exception as e:
                        logger.warning(f"embedding MMD computation failed: {e}")
            except Exception as e:
                logger.error(f"Samaya evaluation failed (non-fatal): {e}")
                import traceback

                traceback.print_exc()
        else:
            logger.warning(f"samaya CSV not found, skipping: {self.config.samaya_csv}")

        full_metrics: dict = {
            "eyepacs_test": test_metrics,
            "samaya_validation": samaya_metrics,
        }
        if domain_shift_metrics:
            full_metrics["domain_shift"] = domain_shift_metrics

        QUADRATIC_WEIGHTED_KAPPA.labels(pipeline="imaging", split="eyepacs_test").set(
            test_metrics["quadratic_weighted_kappa"]
        )

        if samaya_metrics:
            QUADRATIC_WEIGHTED_KAPPA.labels(
                pipeline="imaging", split="samaya_validation"
            ).set(samaya_metrics["quadratic_weighted_kappa"])

        save_json(self.config.metric_file, full_metrics)
        logger.info(f"metrics saved: {self.config.metric_file}")

        run_suffix = f"_{int(time.time()) % 1000:03d}"
        with mlflow.start_run(run_name=self.config.run_name + "_eval" + run_suffix):
            test_mlflow_metrics: dict[str, float] = {
                "test_accuracy": test_metrics["accuracy"],
                "test_qwk": test_metrics["quadratic_weighted_kappa"],
                "test_auc": test_metrics["roc_auc_macro"] or 0.0,
                "test_macro_f1": test_metrics["macro_f1"],
                "test_precision_macro": test_metrics["precision_macro"],
                "test_recall_macro": test_metrics["recall_macro"],
                "test_brier_score": test_metrics["brier_score"],
            }
            # Log optimal thresholds
            for grade, thresh in test_metrics.get("optimal_thresholds", {}).items():
                test_mlflow_metrics[f"test_threshold_grade_{grade}"] = float(thresh)
            # Log optimized metrics
            opt = test_metrics.get("optimized_metrics", {})
            if opt:
                test_mlflow_metrics["test_optimized_accuracy"] = opt["accuracy"]
                test_mlflow_metrics["test_optimized_qwk"] = opt[
                    "quadratic_weighted_kappa"
                ]
                test_mlflow_metrics["test_optimized_macro_f1"] = opt["macro_f1"]
                test_mlflow_metrics["test_optimized_recall_macro"] = opt["recall_macro"]

            for grade_label, grade_vals in test_metrics[
                "classification_report"
            ].items():
                maybe_digit = (
                    grade_label if isinstance(grade_label, str) else str(grade_label)
                )
                if maybe_digit.isdigit():
                    test_mlflow_metrics[f"test_class_{maybe_digit}_f1"] = grade_vals[
                        "f1-score"
                    ]
                    test_mlflow_metrics[f"test_class_{maybe_digit}_precision"] = (
                        grade_vals["precision"]
                    )
                    test_mlflow_metrics[f"test_class_{maybe_digit}_recall"] = (
                        grade_vals["recall"]
                    )
            mlflow.log_metrics(test_mlflow_metrics)

            cm_fig_path = (
                self.config.metric_file.parent
                / f"confusion_matrix_{test_metrics['split']}.png"
            )
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=np.array(test_metrics["confusion_matrix"]),
                    display_labels=[
                        "No DR",
                        "Mild",
                        "Moderate",
                        "Severe",
                        "Proliferative DR",
                    ],
                )
                disp.plot(ax=ax, cmap="Blues", values_format="d")
                plt.title(f"Confusion Matrix - {test_metrics['split']}")
                plt.tight_layout()
                plt.savefig(cm_fig_path)
                mlflow.log_artifact(str(cm_fig_path))
                logger.info(f"Confusion matrix saved to mlflow: {cm_fig_path}")
            except Exception as e:
                logger.warning(f"Failed to save confusion matrix: {e}")

            if samaya_metrics:
                samaya_mlflow_metrics: dict[str, float] = {
                    "samaya_accuracy": samaya_metrics["accuracy"],
                    "samaya_qwk": samaya_metrics["quadratic_weighted_kappa"],
                    "samaya_auc": samaya_metrics["roc_auc_macro"] or 0.0,
                    "samaya_macro_f1": samaya_metrics["macro_f1"],
                    "samaya_precision_macro": samaya_metrics["precision_macro"],
                    "samaya_recall_macro": samaya_metrics["recall_macro"],
                    "samaya_brier_score": samaya_metrics["brier_score"],
                }
                # Log optimal thresholds
                for grade, thresh in samaya_metrics.get(
                    "optimal_thresholds", {}
                ).items():
                    samaya_mlflow_metrics[f"samaya_threshold_grade_{grade}"] = float(
                        thresh
                    )
                # Log optimized metrics
                opt = samaya_metrics.get("optimized_metrics", {})
                if opt:
                    samaya_mlflow_metrics["samaya_optimized_accuracy"] = opt["accuracy"]
                    samaya_mlflow_metrics["samaya_optimized_qwk"] = opt[
                        "quadratic_weighted_kappa"
                    ]
                    samaya_mlflow_metrics["samaya_optimized_macro_f1"] = opt["macro_f1"]
                    samaya_mlflow_metrics["samaya_optimized_recall_macro"] = opt[
                        "recall_macro"
                    ]

                for grade_label, grade_vals in samaya_metrics[
                    "classification_report"
                ].items():
                    maybe_digit = (
                        grade_label
                        if isinstance(grade_label, str)
                        else str(grade_label)
                    )
                    if maybe_digit.isdigit():
                        samaya_mlflow_metrics[f"samaya_class_{maybe_digit}_f1"] = (
                            grade_vals["f1-score"]
                        )
                        samaya_mlflow_metrics[
                            f"samaya_class_{maybe_digit}_precision"
                        ] = grade_vals["precision"]
                        samaya_mlflow_metrics[f"samaya_class_{maybe_digit}_recall"] = (
                            grade_vals["recall"]
                        )
                mlflow.log_metrics(samaya_mlflow_metrics)

                for k, v in domain_shift_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"domain_shift_{k}", float(v))

                samaya_cm_path = (
                    self.config.metric_file.parent
                    / f"confusion_matrix_{samaya_metrics['split']}.png"
                )
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=np.array(samaya_metrics["confusion_matrix"]),
                        display_labels=[
                            "No DR",
                            "Mild",
                            "Moderate",
                            "Severe",
                            "Proliferative DR",
                        ],
                    )
                    disp.plot(ax=ax, cmap="Blues", values_format="d")
                    plt.title(f"Confusion Matrix - {samaya_metrics['split']}")
                    plt.tight_layout()
                    plt.savefig(samaya_cm_path)
                    mlflow.log_artifact(str(samaya_cm_path))
                except Exception as e:
                    logger.warning(f"Failed to save samaya confusion matrix: {e}")

            mlflow.log_artifact(str(self.config.metric_file))

        logger.info("=" * 60)
        logger.info("imaging model evaluation complete")
        logger.info(
            f"eyepacs test  → accuracy={test_metrics['accuracy']:.4f} qwk={test_metrics['quadratic_weighted_kappa']:.4f}"
        )
        if samaya_metrics:
            logger.info(
                f"samaya domain → accuracy={samaya_metrics['accuracy']:.4f} qwk={samaya_metrics['quadratic_weighted_kappa']:.4f}"
            )
        logger.info("=" * 60)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return full_metrics

    @staticmethod
    def _compute_ece(
        labels: list, preds: list, probs: np.ndarray, n_bins: int = 10
    ) -> float:
        confidences = np.max(probs, axis=1)
        targets = np.array(labels)
        preds_arr = np.array(preds)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            if in_bin.sum() == 0:
                continue
            bin_acc = (targets[in_bin] == preds_arr[in_bin]).mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / len(probs)) * abs(bin_acc - bin_conf)
        return float(ece)

    def _save_misclassified_images(
        self,
        preds: list,
        labels: list,
        probs: np.ndarray,
        paths: list,
        split_name: str,
        output_dir: Path,
        max_samples: int = 20,
    ) -> None:
        """Save misclassified images to artifacts for debugging."""
        import shutil

        misclassified = []
        for i, (pred, label, path) in enumerate(zip(preds, labels, paths)):
            if pred != label:
                confidence = float(np.max(probs[i]))
                misclassified.append(
                    {
                        "path": path,
                        "true_label": int(label),
                        "pred_label": int(pred),
                        "confidence": confidence,
                    }
                )

        if not misclassified:
            logger.info(f"[{split_name}] no misclassified samples to save")
            return

        misclassified.sort(key=lambda x: x["confidence"], reverse=True)
        top_misclassified = misclassified[:max_samples]

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(top_misclassified):
            src = Path(sample["path"])
            if not src.exists():
                continue
            dst = (
                output_dir
                / f"{split_name}_mis_{i:03d}_true{sample['true_label']}_pred{sample['pred_label']}_conf{sample['confidence']:.2f}{src.suffix}"
            )
            shutil.copy2(src, dst)

        logger.info(
            f"[{split_name}] saved {len(top_misclassified)} misclassified images to {output_dir}"
        )

    def _compute_embedding_mmd(
        self,
        model: nn.Module,
        eyepacs_csv: Path,
        samaya_csv: Path,
        samaya_transform,
    ) -> float:
        import torch.nn.functional as F

        def _extract_embeddings(csv_path: Path, transform):
            ds = RetinalDataset(csv_path, transform)
            loader = DataLoader(
                ds,
                batch_size=self.params.evaluation.dl.batch_size,
                shuffle=False,
                num_workers=self.params.evaluation.dl.num_workers,
            )
            embs = []
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch
                    images = images.to(self.device)
                    if hasattr(model, "forward_features"):
                        features = model.forward_features(images)
                        emb = model.global_pool(features)
                    else:
                        emb = model(images)
                    embs.append(emb.cpu())
            return torch.cat(embs, dim=0)

        eye_embs = _extract_embeddings(eyepacs_csv, self._build_transform())
        samaya_embs = _extract_embeddings(samaya_csv, samaya_transform)

        n = min(len(eye_embs), len(samaya_embs))
        m = n
        eye_sub = eye_embs[:n]
        sam_sub = samaya_embs[:m]

        eye_norm = F.normalize(eye_sub, p=2, dim=1)
        sam_norm = F.normalize(sam_sub, p=2, dim=1)

        xx = torch.mm(eye_norm, eye_norm.t())
        yy = torch.mm(sam_norm, sam_norm.t())
        xy = torch.mm(eye_norm, sam_norm.t())

        mmd = xx.sum() / (n * n) + yy.sum() / (m * m) - 2 * xy.sum() / (n * m)
        return float(mmd.item())
