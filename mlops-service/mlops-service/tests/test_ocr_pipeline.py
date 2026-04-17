from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from app.entity.ocr_schema import ClinicalFindings, OCTReport, PatientInfo, RegionImage, ReportMetadata, RetinalThickness
from app.domains.ocr.pipeline.ocr_pipeline import OCRPipeline


def test_ocr_pipeline_export_uses_configured_paths(monkeypatch, temp_input_dir, temp_output_dir, temp_images_dir):
    input_dir = temp_input_dir
    output_dir = temp_output_dir
    images_dir = temp_images_dir
    regions_config = {"fundus": {"rel": [0.0, 0.0, 1.0, 1.0]}}

    DummyConfig = SimpleNamespace(
        input_dir=input_dir,
        output_dir=output_dir,
        json_output=output_dir / "reports.json",
        csv_output=output_dir / "reports.csv",
        images_dir=images_dir,
        regions_config=regions_config,
    )

    report = OCTReport(
        source_file="scan.jpg",
        metadata=ReportMetadata(device="Triton"),
        patient=PatientInfo(patient_id="123"),
        thickness=RetinalThickness(center_thickness=241.0),
        clinical=ClinicalFindings(npdr_grade="mild"),
        images={"fundus": RegionImage(png_path="/tmp/fundus.png", base64_png="abc")},
        raw_text="raw text",
        extraction_warnings=[],
    )

    pipeline = cast(Any, OCRPipeline.__new__(OCRPipeline))
    pipeline.config = DummyConfig

    captured: dict[str, object] = {}

    def fake_write_text(_self, data: str) -> None:
        captured["json"] = json.loads(data)

    def fake_to_csv(self, path: Path, index: bool = False) -> None:
        captured["csv_path"] = str(path)
        captured["csv_index"] = index

    monkeypatch.setattr("app.domains.ocr.pipeline.ocr_pipeline.pd.DataFrame.to_csv", fake_to_csv)
    monkeypatch.setattr(Path, "write_text", fake_write_text, raising=False)

    pipeline._export([report])

    assert cast(list[dict[str, object]], captured["json"])[0]["source_file"] == "scan.jpg"
    assert captured["csv_path"] == str(DummyConfig.csv_output)
    assert captured["csv_index"] is False


def test_ocr_pipeline_skips_existing_report(monkeypatch, temp_input_dir, temp_output_dir, temp_images_dir):
    pipeline = cast(Any, OCRPipeline.__new__(OCRPipeline))
    pipeline.config = SimpleNamespace(
        images_dir=temp_images_dir,
        input_dir=temp_input_dir,
        regions_config={},
        output_dir=temp_output_dir,
        json_output=temp_output_dir / "reports.json",
        csv_output=temp_output_dir / "reports.csv",
    )

    report_dir = pipeline.config.images_dir / "patient" / "OD" / "scan"
    report_dir.mkdir(parents=True)
    (report_dir / "existing.png").write_text("x")

    assert pipeline._is_report_processed("patient", "OD", "scan") is True
    assert pipeline._is_report_processed("patient", "OS", "scan") is False
