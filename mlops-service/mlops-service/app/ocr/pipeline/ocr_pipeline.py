import json
import pandas as pd
from pathlib import Path
from loguru import logger

from app.config.configuration import ConfigurationManager
from app.domains.ocr.components.preprocessor import (
    load_and_preprocess,
    load_color_image,
)
from app.domains.ocr.components.extractor import run_ocr
from app.domains.ocr.components.parser import parse_report
from app.domains.ocr.components.region_detector import extract_regions
from app.domains.ocr.components.image_exporter import export_regions
from app.entity.ocr_schema import OCTReport, RegionImage


def _to_flat_dict(report: OCTReport) -> dict:
    d = {"source_file": report.source_file}
    d.update({f"meta_{k}": v for k, v in report.metadata.model_dump().items()})
    d.update({f"patient_{k}": v for k, v in report.patient.model_dump().items()})
    d.update({f"thickness_{k}": v for k, v in report.thickness.model_dump().items()})
    d.update({f"clinical_{k}": v for k, v in report.clinical.model_dump().items()})
    for region_name, region_data in report.images.items():
        d[f"image_{region_name}_path"] = region_data.png_path
    d["warnings"] = "; ".join(report.extraction_warnings)
    return d


class OCRPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_ocr_pipeline_config()

    def _report_key(self, image_path: Path) -> str:
        return image_path.stem.replace(" ", "_").replace("__", "_")

    def _is_report_processed(self, patient_id: str, eye: str, report_key: str) -> bool:
        report_dir = self.config.images_dir / patient_id / eye / report_key
        return report_dir.exists() and any(report_dir.iterdir())

    def _process_single(self, image_path: Path):
        color = load_color_image(image_path)
        preprocessed = load_and_preprocess(image_path)
        text = run_ocr(preprocessed)
        report = parse_report(text, source_file=image_path.name)

        patient_id = image_path.parent.name
        eye = "OD" if "OD" in image_path.name else "OS"
        report_key = self._report_key(image_path)

        if self._is_report_processed(patient_id, eye, report_key):
            logger.info(
                f"SKIPPED: {image_path.name} | patient {patient_id} eye {eye} report {report_key} already processed"
            )
            return None

        crops = extract_regions(color, self.config.regions_config)
        region_data = export_regions(
            crops, self.config.images_dir, patient_id, report_key, eye
        )

        report.images = {
            name: RegionImage(png_path=d["png_path"], base64_png=d["base64_png"])
            for name, d in region_data.items()
        }
        return report

    def run(self) -> list[OCTReport]:
        input_dir = self.config.input_dir
        images = []

        for patient_dir in sorted(input_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            found = (
                list(patient_dir.glob("*.jpg"))
                + list(patient_dir.glob("*.JPG"))
                + list(patient_dir.glob("*.png"))
            )
            images.extend(found)

        if not images:
            logger.warning("No images to process")
            return []

        logger.info(f"OCR pipeline started | total={len(images)}")
        reports = []

        for img_path in images:
            try:
                report = self._process_single(img_path)
                if report is None:
                    continue
                reports.append(report)
                logger.info(f"OK: {img_path.name} | regions={len(report.images)}")
            except Exception as e:
                logger.error(f"FAILED: {img_path.name} | error={e}")

        self._export(reports)
        logger.info(f"OCR pipeline done | extracted={len(reports)}/{len(images)}")
        return reports

    def _export(self, reports: list[OCTReport]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        json_data = [r.model_dump(exclude={"raw_text"}) for r in reports]
        self.config.json_output.write_text(json.dumps(json_data, indent=2, default=str))
        logger.info(f"JSON exported: {self.config.json_output}")

        rows = [_to_flat_dict(r) for r in reports]
        pd.DataFrame(rows).to_csv(self.config.csv_output, index=False)
        logger.info(f"CSV exported: {self.config.csv_output}")
