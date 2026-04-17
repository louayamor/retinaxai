from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.db.session import get_db
from app.models.patient import Patient
from app.models.prediction import Prediction, PredictionStatus
from app.models.report import Report
from app.models.oct_report import OCTReport

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(
    _: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get comprehensive dashboard statistics."""

    # Count totals
    total_patients = await db.scalar(select(func.count(Patient.id)))
    total_predictions = await db.scalar(select(func.count(Prediction.id)))
    total_reports = await db.scalar(select(func.count(Report.id)))
    total_scans = await db.scalar(select(func.count(OCTReport.id)))

    # Prediction status distribution
    prediction_status_result = await db.execute(
        select(Prediction.status, func.count(Prediction.id)).group_by(Prediction.status)
    )
    prediction_status_counts = {
        row[0].value: row[1] for row in prediction_status_result.all()
    }

    # Report status distribution
    report_status_result = await db.execute(
        select(Report.status, func.count(Report.id)).group_by(Report.status)
    )
    report_status_counts = {row[0].value: row[1] for row in report_status_result.all()}

    # Predictions by severity (from output_payload OR oct_reports dr_grade)
    severity_counts = {}
    
    # First try from predictions table
    severity_result = await db.execute(
        select(Prediction.output_payload).where(Prediction.output_payload.isnot(None))
    )
    for row in severity_result.all():
        payload = row[0]
        if isinstance(payload, dict):
            combined_grade = payload.get("combined_grade")
            if combined_grade is not None:
                severity_counts[combined_grade] = (
                    severity_counts.get(combined_grade, 0) + 1
                )
    
    # If no predictions, use oct_reports dr_grade as fallback
    if not severity_counts:
        dr_grade_result = await db.execute(
            select(OCTReport.dr_grade).where(OCTReport.dr_grade.isnot(None))
        )
        for (dr_grade,) in dr_grade_result.all():
            if dr_grade:
                # Map dr_grade strings to numeric (0=No DR, etc.)
                grade_map = {"No DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferative DR": 4}
                grade_num = grade_map.get(dr_grade, 0)
                severity_counts[grade_num] = severity_counts.get(grade_num, 0) + 1

    # Predictions over time (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    predictions_over_time = await db.execute(
        select(Prediction.created_at).where(Prediction.created_at >= thirty_days_ago)
    )
    timeline_map: dict[str, int] = {}
    for (created_at,) in predictions_over_time.all():
        date_key = created_at.date().isoformat()
        timeline_map[date_key] = timeline_map.get(date_key, 0) + 1

    # Fill in missing dates
    date_range = []
    current = thirty_days_ago.date()
    end = datetime.utcnow().date()
    while current <= end:
        date_range.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    filled_timeline = [
        {"date": date, "predictions": timeline_map.get(date, 0)} for date in date_range
    ]

    # Patients by age groups - simplified
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
    else:
        age_distribution = {}

    # Gender distribution
    gender_result = await db.execute(
        select(Patient.gender, func.count(Patient.id)).group_by(Patient.gender)
    )
    gender_distribution = {row[0].value: row[1] for row in gender_result.all()}

    # Recent activity (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_predictions = await db.scalar(
        select(func.count(Prediction.id)).where(Prediction.created_at >= seven_days_ago)
    )
    recent_reports = await db.scalar(
        select(func.count(Report.id)).where(Report.created_at >= seven_days_ago)
    )
    recent_patients = await db.scalar(
        select(func.count(Patient.id)).where(Patient.created_at >= seven_days_ago)
    )

    # Average confidence score for successful predictions
    avg_confidence = await db.scalar(
        select(func.avg(Prediction.confidence_score)).where(
            Prediction.status == PredictionStatus.SUCCESS,
            Prediction.confidence_score.isnot(None),
        )
    )

    return {
        "totals": {
            "patients": total_patients or 0,
            "predictions": total_predictions or 0,
            "reports": total_reports or 0,
            "scans": total_scans or 0,
        },
        "prediction_status": prediction_status_counts,
        "report_status": report_status_counts,
        "severity_distribution": severity_counts,
        "predictions_timeline": filled_timeline,
        "age_distribution": age_distribution,
        "gender_distribution": gender_distribution,
        "recent_activity": {
            "new_patients": recent_patients or 0,
            "new_predictions": recent_predictions or 0,
            "new_reports": recent_reports or 0,
        },
        "avg_confidence": round(float(avg_confidence), 2) if avg_confidence else None,
    }
