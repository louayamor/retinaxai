from pathlib import Path
from loguru import logger
from app.config.configuration import ConfigurationManager
from app.training.components.model_trainer import ImagingModelTrainer


def _run_single_phase(
    phase: str,
    checkpoint_path: Path | None,
    cfg,
    transformation_cfg,
) -> Path:
    """Run a single training phase."""
    final_checkpoint = ImagingModelTrainer(
        cfg,
        transformation_cfg,
        phase=phase,
        load_checkpoint=checkpoint_path,
    ).train()

    logger.info(f">>> {phase} complete (checkpoint={final_checkpoint})")
    return final_checkpoint


def run(phase: str = "phase1", checkpoint_path: Path | None = None):
    """
    Run imaging model training.

    Default behavior (phase="phase1", checkpoint_path=None):
        - Run Phase 1 (EyePacs, frozen backbone)

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

    logger.info(f">>> Running phase: {phase}")
    final_checkpoint = _run_single_phase(
        phase, checkpoint_path, cfg, transformation_cfg
    )

    logger.info(
        f">>> stage 04: imaging model training complete (phase={phase}, checkpoint={final_checkpoint})"
    )
    return final_checkpoint


if __name__ == "__main__":
    run()
