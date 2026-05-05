import pickle
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from app.entity.config_entity import (
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

            mlflow.log_metrics(
                {
                    "train_accuracy": float(round(train_acc, 4)),
                    "test_accuracy": float(round(test_acc, 4)),
                }
            )

            logger.info(f"train accuracy: {train_acc:.4f}")
            logger.info(f"test accuracy: {test_acc:.4f}")

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

        logger.info("=" * 60)
        logger.info("clinical model training complete")
        logger.info("=" * 60)

        return self.config.checkpoint_path
