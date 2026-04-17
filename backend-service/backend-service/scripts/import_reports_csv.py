#!/usr/bin/env python3
"""Import reports.csv into the RetinaXAI database.

Creates patients from OCR patient IDs, then inserts OCT reports.
"""

import asyncio
import csv
import uuid
from pathlib import Path

import asyncpg


DATABASE_URL = "postgresql://louay:louay@localhost:5432/retinaxai_db"
CSV_PATH = Path(__file__).resolve().parents[3] / "mlops-service" / "mlops-service" / "artifacts" / "ocr" / "output" / "reports.csv"


def parse_float(val: str) -> float | None:
    if not val or val.strip() in ("", "N/A", "nan"):
        return None
    try:
        f = float(val.strip())
        return None if f != f else f  # NaN check
    except (ValueError, TypeError):
        return None


def parse_bool(val: str) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("true", "1", "yes", "present")


def parse_gender(val: str) -> str:
    v = val.strip().upper()
    if v in ("M", "MALE", "M"):
        return "M"
    if v in ("F", "FEMALE", "F"):
        return "F"
    return "M"


def parse_eye(val: str) -> str:
    v = val.strip().upper()
    return "OD" if v == "OD" else "OS"


async def main():
    # Load CSV
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from CSV")

    conn = await asyncpg.connect(DATABASE_URL)

    # Build patient map: ocr_patient_id -> patient data
    patient_data: dict[str, dict] = {}
    for r in rows:
        pid = r["patient_patient_id"].strip()
        if pid not in patient_data:
            patient_data[pid] = {
                "ocr_patient_id": pid,
                "gender": parse_gender(r["patient_gender"]),
                "age": int(float(r["patient_age"])) if parse_float(r["patient_age"]) else None,
            }

    print(f"Found {len(patient_data)} unique patients")

    # Insert patients
    patient_ids: dict[str, uuid.UUID] = {}  # ocr_patient_id -> uuid
    inserted = 0
    skipped = 0

    for ocr_id, data in patient_data.items():
        # Check if already exists
        existing = await conn.fetchrow(
            "SELECT id FROM patients WHERE ocr_patient_id = $1", ocr_id
        )
        if existing:
            patient_ids[ocr_id] = existing["id"]
            skipped += 1
            continue

        patient_uuid = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO patients (id, first_name, last_name, ocr_patient_id, gender, age, medical_record_number, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
            """,
            patient_uuid,
            "Patient",
            data["ocr_patient_id"],
            data["ocr_patient_id"],
            data["gender"],
            data["age"] if data["age"] else 0,
            f"OCT-{data['ocr_patient_id']}",
        )
        patient_ids[ocr_id] = patient_uuid
        inserted += 1

    print(f"Patients: {inserted} inserted, {skipped} already existed")

    # Insert OCT reports
    report_cols = [
        "eye", "source_file", "image_quality",
        "thickness_center_fovea", "thickness_average_thickness", "thickness_total_volume_mm3",
        "thickness_inner_superior", "thickness_inner_nasal", "thickness_inner_inferior", "thickness_inner_temporal",
        "thickness_outer_superior", "thickness_outer_nasal", "thickness_outer_inferior", "thickness_outer_temporal",
        "edema", "erm_status",
    ]

    reports_inserted = 0
    reports_skipped = 0

    for r in rows:
        ocr_id = r["patient_patient_id"].strip()
        if ocr_id not in patient_ids:
            print(f"  WARNING: patient {ocr_id} not found, skipping report")
            continue

        patient_uuid = patient_ids[ocr_id]
        source_file = r["source_file"].strip()

        # Check duplicate
        existing = await conn.fetchrow(
            "SELECT id FROM oct_reports WHERE patient_id = $1 AND source_file = $2",
            patient_uuid, source_file,
        )
        if existing:
            reports_skipped += 1
            continue

        report_uuid = uuid.uuid4()
        await conn.execute(
            f"""
            INSERT INTO oct_reports (
                id, patient_id, eye, source_file, image_quality,
                thickness_center_fovea, thickness_average_thickness, thickness_total_volume_mm3,
                thickness_inner_superior, thickness_inner_nasal, thickness_inner_inferior, thickness_inner_temporal,
                thickness_outer_superior, thickness_outer_nasal, thickness_outer_inferior, thickness_outer_temporal,
                edema, erm_status, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16,
                $17, $18, NOW(), NOW()
            )
            """,
            report_uuid,
            patient_uuid,
            parse_eye(r["meta_eye"]),
            source_file,
            parse_float(r["meta_image_quality"]),
            parse_float(r["thickness_center_fovea"]),
            parse_float(r["thickness_average_thickness"]),
            parse_float(r["thickness_total_volume_mm3"]),
            parse_float(r["thickness_inner_superior"]),
            parse_float(r["thickness_inner_nasal"]),
            parse_float(r["thickness_inner_inferior"]),
            parse_float(r["thickness_inner_temporal"]),
            parse_float(r["thickness_outer_superior"]),
            parse_float(r["thickness_outer_nasal"]),
            parse_float(r["thickness_outer_inferior"]),
            parse_float(r["thickness_outer_temporal"]),
            parse_bool(r["clinical_edema"]),
            r["clinical_erm_status"].strip() or None,
        )
        reports_inserted += 1

    print(f"OCT Reports: {reports_inserted} inserted, {reports_skipped} duplicates skipped")

    # Verify counts
    total_patients = await conn.fetchval("SELECT COUNT(*) FROM patients")
    total_reports = await conn.fetchval("SELECT COUNT(*) FROM oct_reports")
    print(f"\nFinal: {total_patients} patients, {total_reports} oct_reports in DB")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
