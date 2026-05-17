from loguru import logger
from app.config.configuration import ConfigurationManager
from app.domains.imaging.evaluation.imaging_evaluation import ImagingModelEvaluation


def run():
    logger.info(">>> stage 05: imaging model evaluation started")
    manager = ConfigurationManager()
    cfg = manager.get_imaging_model_evaluation_config()
    logger.info(
        f"stage 05 config: model_path={cfg.model_path}, test_csv={cfg.test_csv}, samaya_csv={cfg.samaya_csv}, metric_file={cfg.metric_file}"
    )
    ImagingModelEvaluation(cfg).evaluate()
    logger.info(">>> stage 05: imaging model evaluation complete")


if __name__ == "__main__":
    run()
