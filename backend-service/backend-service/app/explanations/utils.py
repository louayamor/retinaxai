from __future__ import annotations

from app.models.severity_report import RiskLevel

RISK_LEVEL_ALIASES: dict[str, str] = {
    "very_high": "severe",
}


def normalize_risk_level(risk_level: str | None) -> RiskLevel:
    if not risk_level:
        return RiskLevel.MODERATE
    normalized = RISK_LEVEL_ALIASES.get(risk_level.lower(), risk_level.lower())
    try:
        return RiskLevel(normalized)
    except ValueError:
        return RiskLevel.MODERATE
