from loguru import logger
from app.config.configuration import ConfigurationManager
from app.domains.imaging.components.data_cleaning import ImagingDataCleaning


def run():
    logger.info(">>> stage 02: imaging data cleaning started")
    manager = ConfigurationManager()
    cfg = manager.get_imaging_cleaning_config()
    source_path = cfg.source_dir / "huggingface" / "train"
    clean_path = cfg.root_dir / "huggingface" / "train_clean"
    logger.info(f"stage 02 config: source={source_path}, output={clean_path}")
    ImagingDataCleaning(cfg).run()
    logger.info(">>> stage 02: imaging data cleaning complete")


if __name__ == "__main__":
    run()
