from __future__ import annotations
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import DoctorUser
from app.db.session import get_db
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.report import Report, ReportType

router = APIRouter(prefix="/oct-stats", tags=["oct_stats"])


@router.get("/stats")
async def get_oct_stats(
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    # ==== OCT Reports Data (from reports table with report_type=OCT) ====
    total_oct = await db.scalar(
        select(func.count(Report.id)).where(Report.report_type == ReportType.OCT)
    )

    grade_result = await db.execute(
        select(Report.dr_grade, func.count(Report.id))
        .where(
            Report.report_type == ReportType.OCT,
            Report.dr_grade.isnot(None),
        )
        .group_by(Report.dr_grade)
    )
    grade_distribution = {row[0]: row[1] for row in grade_result.all()}

    eye_result = await db.execute(
        select(Report.eye, func.count(Report.id))
        .where(
            Report.report_type == ReportType.OCT,
            Report.eye.isnot(None),
        )
        .group_by(Report.eye)
    )
    eye_distribution = {row[0]: row[1] for row in eye_result.all()}

    edema_count = await db.scalar(
        select(func.count(Report.id)).where(
            Report.report_type == ReportType.OCT,
            Report.edema,
        )
    )
    no_edema_count = await db.scalar(
        select(func.count(Report.id)).where(
            Report.report_type == ReportType.OCT,
            ~Report.edema,
        )
    )

    erm_result = await db.execute(
        select(Report.erm_status, func.count(Report.id))
        .where(
            Report.report_type == ReportType.OCT,
            Report.erm_status.isnot(None),
        )
        .group_by(Report.erm_status)
    )
    erm_distribution = {row[0]: row[1] for row in erm_result.all()}

    thickness_avg = await db.execute(
        select(
            func.avg(Report.thickness_center_fovea),
            func.avg(Report.thickness_average_thickness),
            func.avg(Report.thickness_total_volume_mm3),
        ).where(Report.report_type == ReportType.OCT)
    )
    thickness_row = thickness_avg.first()
    thickness_averages = {
        "center_fovea": round(float(thickness_row[0]), 2)
        if thickness_row is not None and thickness_row[0] is not None
        else None,
        "average_thickness": round(float(thickness_row[1]), 2)
        if thickness_row is not None and thickness_row[1] is not None
        else None,
        "total_volume_mm3": round(float(thickness_row[2]), 2)
        if thickness_row is not None and thickness_row[2] is not None
        else None,
    }

    avg_quality = await db.scalar(
        select(func.avg(Report.image_quality)).where(
            Report.report_type == ReportType.OCT,
            Report.image_quality.isnot(None),
        )
    )

    # ==== Patients Data ====
    total_patients = await db.scalar(select(func.count(Patient.id)))
    new_patients_30d = await db.scalar(
        select(func.count(Patient.id)).where(
            Patient.created_at >= datetime.utcnow() - timedelta(days=30)
        )
    )

    gender_dist = await db.execute(
        select(Patient.gender, func.count(Patient.id)).group_by(Patient.gender)
    )
    gender_distribution = {row[0].value: row[1] for row in gender_dist.all()}

    # Patient age distribution - simplified without CASE
    age_distribution = {}
    all_patients = await db.execute(select(Patient.age))
    ages = [row[0] for row in all_patients.all() if row[0]]
    if ages:
        age_distribution = {
            "Under 18": sum(1 for a in ages if a < 18),
            "18-29": sum(1 for a in ages if 18 <= a < 30),
            "30-49": sum(1 for a in ages if 30 <= a < 50),
            "50-64": sum(1 for a in ages if 50 <= a < 65),
            "65+": sum(1 for a in ages if a >= 65),
        }

    # ==== Clinical Reports Data ====
    total_reports = await db.scalar(select(func.count(Report.id)))

    report_status_result = await db.execute(
        select(Report.status, func.count(Report.id)).group_by(Report.status)
    )
    report_status = {row[0]: row[1] for row in report_status_result.all()}

    # ==== Predictions Data ====
    total_predictions = await db.scalar(select(func.count(Prediction.id)))

    pred_status_result = await db.execute(
        select(Prediction.status, func.count(Prediction.id)).group_by(Prediction.status)
    )
    prediction_status = {row[0].value: row[1] for row in pred_status_result.all()}

    # Prediction severity from output_payload
    severity_result = await db.execute(
        select(Prediction.output_payload).where(Prediction.output_payload.isnot(None))
    )
    severity_counts = {}
    for row in severity_result.all():
        payload = row[0]
        if isinstance(payload, dict):
            combined_grade = payload.get("combined_grade")
            if combined_grade is not None:
                severity_counts[combined_grade] = (
                    severity_counts.get(combined_grade, 0) + 1
                )

    # Recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    new_preds_7d = await db.scalar(
        select(func.count(Prediction.id)).where(Prediction.created_at >= week_ago)
    )
    new_reports_7d = await db.scalar(
        select(func.count(Report.id)).where(Report.created_at >= week_ago)
    )
    new_patients_7d = await db.scalar(
        select(func.count(Patient.id)).where(Patient.created_at >= week_ago)
    )

    # ==== Build Combined Response ====
    return {
        # Summary stats
        "summary": {
            "total_patients": total_patients or 0,
            "total_oct_reports": total_oct or 0,
            "total_clinical_reports": total_reports or 0,
            "total_predictions": total_predictions or 0,
        },
        # Recent activity
        "recent_activity": {
            "patients_7d": new_patients_7d or 0,
            "patients_30d": new_patients_30d or 0,
            "predictions_7d": new_preds_7d or 0,
            "reports_7d": new_reports_7d or 0,
        },
        # Patient demographics
        "patient_demographics": {
            "gender": gender_distribution,
            "age_groups": age_distribution,
        },
        # Clinical reports
        "clinical_reports": {
            "total": total_reports or 0,
            "status": report_status,
        },
        # Predictions
        "predictions": {
            "total": total_predictions or 0,
            "status": prediction_status,
            "severity_distribution": severity_counts,
        },
        # OCT reports (keep existing)
        "oct_reports": {
            "total": total_oct or 0,
            "grade_distribution": grade_distribution,
            "eye_distribution": eye_distribution,
            "edema": {"present": edema_count or 0, "absent": no_edema_count or 0},
            "erm_distribution": erm_distribution,
            "thickness_averages": thickness_averages,
            "avg_image_quality": round(float(avg_quality), 2) if avg_quality else None,
        },
    }
