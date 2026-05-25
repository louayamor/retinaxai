from __future__ import annotations
"""Import OCR reports into the backend database.

Usage:
    cd backend-service/backend-service
    source .venv/bin/activate
    python -m app.scripts.import_oct_reports
"""

import asyncio
import uuid
from pathlib import Path

import asyncpg
from app.core.config import settings

OCR_CSV = (
    Path(__file__).parent.parent.parent.parent
    / "mlops-service"
    / "mlops-service"
    / "artifacts"
    / "ocr"
    / "output"
    / "reports.csv"
)

THICKNESS_COLS = [
    "thickness_center_fovea",
    "thickness_average_thickness",
    "thickness_total_volume_mm3",
    "thickness_inner_superior",
    "thickness_inner_nasal",
    "thickness_inner_inferior",
    "thickness_inner_temporal",
    "thickness_outer_superior",
    "thickness_outer_nasal",
    "thickness_outer_inferior",
    "thickness_outer_temporal",
]


async def main():
    import pandas as pd  # type: ignore[reportMissingImports]

    if not OCR_CSV.exists():
        print(f"OCR CSV not found: {OCR_CSV}")
        return

    df = pd.read_csv(OCR_CSV)
    print(f"Loaded {len(df)} OCR reports from CSV")

    conn = await asyncpg.connect(settings.DATABASE_URL)
    imported = 0
    skipped = 0

    for _, row in df.iterrows():
        patient_id_str = str(row.get("patient_patient_id", "")).strip()
        if not patient_id_str:
            skipped += 1
            continue

        eye = str(row.get("meta_eye", "OD")).strip()
        source_file = str(row.get("source_file", "")).strip()
        if not source_file:
            skipped += 1
            continue

        # Find or create patient by ocr_patient_id
        patient = await conn.fetchrow(
            "SELECT id FROM patients WHERE ocr_patient_id = $1", patient_id_str
        )

        if not patient:
            # Create a minimal patient record
            patient_uuid = uuid.uuid4()
            gender = str(row.get("patient_gender", "M")).strip().upper()
            if gender not in ("M", "F"):
                gender = "M"
            age = int(row["patient_age"]) if pd.notna(row.get("patient_age")) else 30

            await conn.execute(
                """
                INSERT INTO patients (id, first_name, last_name, age, gender, medical_record_number, ocr_patient_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                patient_uuid,
                f"Patient_{patient_id_str}",
                patient_id_str,
                age,
                gender,
                f"MRN_{patient_id_str}",
                patient_id_str,
            )
            patient_id = patient_uuid
            print(f"  Created patient: {patient_id_str}")
        else:
            patient_id = patient["id"]

        dr_grade = row.get("clinical_npdr_grade")
        if pd.isna(dr_grade):
            dr_grade = None
        else:
            dr_grade = str(dr_grade).strip()

        edema = bool(row.get("clinical_edema", False))
        erm = row.get("clinical_erm_status")
        if pd.isna(erm):
            erm = None
        else:
            erm = str(erm).strip()

        image_quality = row.get("meta_image_quality")
        if pd.isna(image_quality):
            image_quality = None

        thickness = {}
        for col in THICKNESS_COLS:
            val = row.get(col)
            thickness[col] = float(val) if pd.notna(val) else None

        report_uuid = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO oct_reports (
                id, patient_id, eye, source_file, dr_grade, edema, erm_status,
                image_quality,
                thickness_center_fovea, thickness_average_thickness,
                thickness_total_volume_mm3,
                thickness_inner_superior, thickness_inner_nasal,
                thickness_inner_inferior, thickness_inner_temporal,
                thickness_outer_superior, thickness_outer_nasal,
                thickness_outer_inferior, thickness_outer_temporal
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            """,
            report_uuid,
            patient_id,
            eye,
            source_file,
            dr_grade,
            edema,
            erm,
            image_quality,
            thickness["thickness_center_fovea"],
            thickness["thickness_average_thickness"],
            thickness["thickness_total_volume_mm3"],
            thickness["thickness_inner_superior"],
            thickness["thickness_inner_nasal"],
            thickness["thickness_inner_inferior"],
            thickness["thickness_inner_temporal"],
            thickness["thickness_outer_superior"],
            thickness["thickness_outer_nasal"],
            thickness["thickness_outer_inferior"],
            thickness["thickness_outer_temporal"],
        )
        imported += 1

    await conn.close()
    print(f"\nImport complete: {imported} reports imported, {skipped} skipped")


if __name__ == "__main__":
    asyncio.run(main())
