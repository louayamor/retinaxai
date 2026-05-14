from pathlib import Path
from loguru import logger
from app.config.configuration import ConfigurationManager
from app.domains.imaging.components.model_trainer import ImagingModelTrainer
from app.domains.imaging.components.data_transformation import (
    create_fine_tune_split,
    oversample_clinical,
)
from app.utils.common import read_yaml
from app.constants import PARAMS_FILE_PATH
import pandas as pd


def _run_single_phase(
    phase: str,
    checkpoint_path: Path | None,
    cfg,
    transformation_cfg,
) -> Path:
    """Run a single training phase."""
    custom_train_csv = None

    if phase == "phase2" and checkpoint_path:
        logger.info("Creating Phase 2 fine-tuning dataset")
        params = read_yaml(PARAMS_FILE_PATH)
        data_cfg = params.get("data", {}) or {}
        ratios_cfg = data_cfg.get("ratios", {}) or {}

        oversample_ratio = data_cfg.get("oversample_ratio", 5)
        clinical_ratio = ratios_cfg.get("clinical", 0.7)
        no_dr_ratio = data_cfg.get("keep_no_dr_ratio", 0.55)

        try:
            samaya_csv = transformation_cfg.samaya_csv
            if samaya_csv.exists():
                samaya_df = pd.read_csv(samaya_csv)
                eyepacs_df = pd.read_csv(transformation_cfg.train_csv)
                fine_tune_df = create_fine_tune_split(
                    samaya_df,
                    eyepacs_df,
                    clinical_ratio=clinical_ratio,
                    no_dr_ratio=no_dr_ratio,
                    oversample_ratio=oversample_ratio,
                )
                temp_csv = transformation_cfg.train_csv.parent / "train_phase2.csv"
                fine_tune_df.to_csv(temp_csv, index=False)
                custom_train_csv = temp_csv
                logger.info(
                    f"Phase 2 training data: {len(fine_tune_df)} samples (oversample_ratio={oversample_ratio})"
                )
            else:
                logger.warning(
                    f"Samaya CSV not found: {samaya_csv}. "
                    f"Phase 2 (domain adaptation) is SKIPPED. "
                    f"Returning Phase 1 checkpoint unchanged."
                )
                return checkpoint_path
        except Exception as e:
            logger.warning(
                f"Failed to create Phase 2 dataset: {e}. "
                f"Phase 2 is SKIPPED. Returning Phase 1 checkpoint."
            )
            return checkpoint_path

    final_checkpoint = ImagingModelTrainer(
        cfg,
        transformation_cfg,
        phase=phase,
        load_checkpoint=checkpoint_path,
        custom_train_csv=custom_train_csv,
    ).train()

    logger.info(f">>> {phase} complete (checkpoint={final_checkpoint})")
    return final_checkpoint


def run(phase: str = "phase1", checkpoint_path: Path | None = None):
    """
    Run imaging model training.

    Default behavior (phase="phase1", checkpoint_path=None):
        - If checkpoint exists: Skip Phase 1, run Phase 2 (incremental fine-tuning)
        - If checkpoint missing: Run Phase 1 (EyePacs, frozen backbone) then Phase 2
        - If Phase 2 is skipped/fails: Return Phase 1 checkpoint

    Specific phase:
        - Run only the specified phase
    """
    logger.info(">>> stage 04: imaging model training started")

    manager = ConfigurationManager()
    cfg = manager.get_imaging_model_trainer_config()
    transformation_cfg = manager.get_imaging_transformation_config()
    logger.info(
        f"stage 04 config: model={cfg.model_name}, pretrained={cfg.pretrained}, "
        f"checkpoint={cfg.checkpoint_path}, train_csv={transformation_cfg.train_csv}"
    )

    if phase == "phase1" and checkpoint_path is None:
        existing_checkpoint = cfg.checkpoint_path
        if existing_checkpoint.exists():
            logger.info(
                f">>> Existing checkpoint found: {existing_checkpoint}. "
                f"Skipping Phase 1, running incremental fine-tuning (Phase 2 only)"
            )
            try:
                final_checkpoint = _run_single_phase(
                    "phase2", existing_checkpoint, cfg, transformation_cfg
                )
            except Exception as e:
                logger.warning(
                    f"Phase 2 incremental training failed: {e}. "
                    f"Returning existing checkpoint: {existing_checkpoint}"
                )
                final_checkpoint = existing_checkpoint

            logger.info(
                f">>> stage 04: imaging model training complete (final checkpoint={final_checkpoint})"
            )
            return final_checkpoint

        logger.info(">>> Phase 1: Full EyePacs training (frozen backbone)")
        phase1_checkpoint = _run_single_phase("phase1", None, cfg, transformation_cfg)

        logger.info(">>> Phase 2: Clinical fine-tuning with domain adaptation")
        try:
            final_checkpoint = _run_single_phase(
                "phase2", phase1_checkpoint, cfg, transformation_cfg
            )
        except Exception as e:
            logger.warning(
                f"Phase 2 training failed: {e}. "
                f"Returning Phase 1 checkpoint: {phase1_checkpoint}"
            )
            final_checkpoint = phase1_checkpoint

        logger.info(
            f">>> stage 04: imaging model training complete (final checkpoint={final_checkpoint})"
        )
        return final_checkpoint

    logger.info(f">>> Running specific phase: {phase}")
    final_checkpoint = _run_single_phase(
        phase, checkpoint_path, cfg, transformation_cfg
    )

    logger.info(
        f">>> stage 04: imaging model training complete (phase={phase}, checkpoint={final_checkpoint})"
    )
    return final_checkpoint


if __name__ == "__main__":
    run()
