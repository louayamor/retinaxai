from loguru import logger
from app.config.configuration import ConfigurationManager
from app.domains.imaging.components.data_ingestion import ImagingDataIngestion


def run():
    logger.info(">>> stage 01: imaging data ingestion started")
    manager = ConfigurationManager()
    cfg = manager.get_imaging_ingestion_config()
    logger.info(
        f"stage 01 config: dataset={cfg.dataset_name}, split={cfg.train_split}, max_samples={cfg.max_samples}, root_dir={cfg.root_dir}"
    )
    ImagingDataIngestion(cfg).run()
    logger.info(">>> stage 01: imaging data ingestion complete")


if __name__ == "__main__":
    run()
