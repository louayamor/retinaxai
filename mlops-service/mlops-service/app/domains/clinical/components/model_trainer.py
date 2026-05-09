import pickle
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

matplotlib.use("Agg")

from app.entity.config_entity import (  # noqa: E402
    ClinicalModelTrainerConfig,
    ClinicalTransformationConfig,
)
from app.utils.common import load_json, read_yaml, save_json
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.services.monitoring.prometheus_metrics import BEST_VAL_ACCURACY


class ClinicalModelTrainer:
    def __init__(
        self,
        config: ClinicalModelTrainerConfig,
        transformation_config: ClinicalTransformationConfig,
    ):
        self.config = config
        self.transformation_config = transformation_config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        try:
            train_df = pd.read_csv(self.transformation_config.train_csv)
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {e}") from e

        try:
            test_df = pd.read_csv(self.transformation_config.test_csv)
        except Exception as e:
            raise RuntimeError(f"Failed to load test data: {e}") from e

        logger.info(f"train samples: {len(train_df)}")
        logger.info(f"test samples: {len(test_df)}")
        logger.info(f"features: {[c for c in train_df.columns if c != 'label']}")

        min_label = int(min(train_df["label"].min(), test_df["label"].min()))
        if min_label != 0:
            logger.info(f"remapping labels: shifting by -{min_label} to start from 0")
            train_df["label"] = train_df["label"] - min_label
            test_df["label"] = test_df["label"] - min_label

        return train_df, test_df, min_label

    def _build_model(self) -> XGBClassifier:
        xgb_cfg = self.params.ml_training.xgboost
        p = self.params.ml_training
        return XGBClassifier(
            n_estimators=xgb_cfg.n_estimators,
            max_depth=xgb_cfg.max_depth,
            learning_rate=xgb_cfg.learning_rate,
            subsample=xgb_cfg.subsample,
            colsample_bytree=xgb_cfg.colsample_bytree,
            eval_metric=xgb_cfg.eval_metric,
            random_state=p.seed,
            n_jobs=-1,
            verbosity=0,
        )

    def train(self) -> Path:
        logger.info("=" * 60)
        logger.info("clinical model training started")
        logger.info("=" * 60)

        train_df, test_df, label_offset = self._load_data()

        feature_cols = [c for c in train_df.columns if c != "label"]
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_np)  # type: ignore[arg-type]
        logger.info(
            f"class distribution train: {dict(zip(*np.unique(y_train_np, return_counts=True)))}"
        )  # type: ignore[call-overload]
        logger.info(
            f"class distribution test: {dict(zip(*np.unique(y_test_np, return_counts=True)))}"
        )  # type: ignore[call-overload]

        model = self._build_model()
        xgb_cfg = self.params.ml_training.xgboost
        p = self.params.ml_training
        feature_meta = (
            load_json(self.config.feature_file)
            if self.config.feature_file.exists()
            else {}
        )
        categorical_encoders = (
            feature_meta.get("categorical_encoders", {}) if feature_meta else {}
        )

        run_suffix = f"_{int(time.time()) % 1000:03d}"
        with mlflow.start_run(
            run_name=self.params.get("mlflow", {}).get(
                "clinical_run_name", "xgboost_clinical"
            )
            + run_suffix
        ):
            mlflow.log_params(
                {
                    "model": self.config.model_name,
                    "n_estimators": xgb_cfg.n_estimators,
                    "max_depth": xgb_cfg.max_depth,
                    "learning_rate": xgb_cfg.learning_rate,
                    "subsample": xgb_cfg.subsample,
                    "colsample_bytree": xgb_cfg.colsample_bytree,
                    "eval_metric": xgb_cfg.eval_metric,
                    "seed": p.seed,
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "n_features": len(feature_cols),
                    "label_offset": label_offset,
                }
            )

            logger.info("fitting XGBoost model with balanced sample weights")
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            BEST_VAL_ACCURACY.labels(pipeline="clinical").set(float(test_acc))

            test_probs = model.predict_proba(X_test)
            test_qwk = cohen_kappa_score(y_test, test_preds, weights="quadratic")
            test_macro_f1 = float(
                f1_score(y_test, test_preds, average="macro", zero_division=0)
            )
            try:
                present = sorted(set(y_test))
                if len(present) >= 2:
                    test_auc = float(
                        roc_auc_score(
                            y_test,
                            test_probs[:, present]
                            if test_probs.shape[1] > len(present)
                            else test_probs,
                            multi_class="ovr",
                            average="macro",
                            labels=present,
                        )
                    )
                else:
                    test_auc = None
            except Exception:
                test_auc = None

            per_class_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)

            mlflow.log_metrics(
                {
                    "train_accuracy": float(round(train_acc, 4)),
                    "test_accuracy": float(round(test_acc, 4)),
                    "test_qwk": float(round(test_qwk, 4)),
                    "test_macro_f1": float(round(test_macro_f1, 4)),
                    "test_auc": float(round(test_auc, 4))
                    if test_auc is not None
                    else 0.0,
                }
            )

            for cls_idx, cls_f1 in enumerate(per_class_f1):
                mlflow.log_metric(f"test_f1_class_{cls_idx}", float(cls_f1))

            cm = confusion_matrix(y_test, test_preds)
            dr_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][
                : cm.shape[0]
            ]
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
            ax.set_title("Confusion Matrix — Clinical")
            plt.tight_layout()
            cm_path = Path("/tmp/cm_clinical_train.png")
            fig.savefig(cm_path)
            mlflow.log_artifact(str(cm_path), "confusion_matrices")
            plt.close(fig)

            auc_str = f"{test_auc:.4f}" if test_auc is not None else "N/A"
            logger.info(
                f"train_accuracy={train_acc:.4f} "
                f"test_accuracy={test_acc:.4f} "
                f"test_qwk={test_qwk:.4f} "
                f"test_macro_f1={test_macro_f1:.4f} "
                f"test_auc={auc_str}"
            )

            feature_importance = dict(
                zip(feature_cols, model.feature_importances_.tolist())
            )
            feature_importance_sorted = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            logger.info(f"top features: {list(feature_importance_sorted.keys())[:5]}")

            self.config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.checkpoint_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"model saved: {self.config.checkpoint_path}")

            feature_meta = {
                "feature_importance": feature_importance_sorted,
                "label_offset": label_offset,
                "feature_cols": feature_cols,
                "categorical_encoders": categorical_encoders,
                "numeric_medians": {
                    col: float(train_df[col].dropna().median())
                    for col in feature_cols
                    if col in train_df.columns
                    and pd.api.types.is_numeric_dtype(train_df[col])
                    and not train_df[col].dropna().empty
                },
            }
            save_json(self.config.feature_importance_path, feature_meta)
            logger.info(
                f"feature importance saved: {self.config.feature_importance_path}"
            )

            mlflow.log_artifact(str(self.config.feature_importance_path))
            mlflow.log_artifact(str(self.config.checkpoint_path))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="clinical_model",
                registered_model_name="xgboost_clinical",
            )

        logger.info("=" * 60)
        logger.info("clinical model training complete")
        logger.info("=" * 60)

        return self.config.checkpoint_path
