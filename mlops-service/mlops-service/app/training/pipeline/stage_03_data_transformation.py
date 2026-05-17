from loguru import logger
from app.config.configuration import ConfigurationManager
from app.domains.imaging.components.data_transformation import ImagingDataTransformation


def run():
    logger.info(">>> stage 03: imaging data transformation started")
    manager = ConfigurationManager()
    cfg = manager.get_imaging_transformation_config()
    logger.info(
        "stage 03 config: "
        f"source={cfg.source_dir / 'huggingface' / 'train_clean'}, "
        f"train_csv={cfg.train_csv}, test_csv={cfg.test_csv}, samaya_csv={cfg.samaya_csv}, image_size={cfg.image_size}"
    )
    ImagingDataTransformation(cfg).run()
    logger.info(">>> stage 03: imaging data transformation complete")


if __name__ == "__main__":
    run()
