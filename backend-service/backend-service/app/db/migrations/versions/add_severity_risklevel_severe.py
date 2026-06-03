"""Align severity risk level enum with severe

Revision ID: add_severity_risklevel_severe
Revises: add_prediction_explanations
Create Date: 2026-04-22

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_severity_risklevel_severe"
down_revision: str | None = "add_prediction_explanations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_enum(
    table_name: str,
    column_name: str,
    type_name: str,
    new_values: list[str],
    mapping_sql: str,
) -> None:
    tmp_type = f"{type_name}_old"

    op.execute(f"ALTER TYPE {type_name} RENAME TO {tmp_type}")
    op.execute(f"CREATE TYPE {type_name} AS ENUM ({', '.join(repr(v) for v in new_values)})")
    op.execute(
        f"""
        ALTER TABLE {table_name}
        ALTER COLUMN {column_name}
        TYPE {type_name}
        USING ({mapping_sql})::{type_name}
        """
    )
    op.execute(f"DROP TYPE {tmp_type}")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "severity_reports" in inspector.get_table_names():
        allowed_values = {"low", "moderate", "high", "very_high", "severe"}
        existing_values = {
            value
            for value in bind.execute(sa.text("SELECT DISTINCT risk_level::text FROM severity_reports WHERE risk_level IS NOT NULL")).scalars()
            if value is not None
        }
        invalid_values = existing_values - allowed_values
        if invalid_values:
            raise RuntimeError(
                f"Unexpected risk_level values found before enum migration: {sorted(invalid_values)}"
            )

        _replace_enum(
            table_name="severity_reports",
            column_name="risk_level",
            type_name="risklevel",
            new_values=["low", "moderate", "high", "severe"],
            mapping_sql=(
                "CASE risk_level::text "
                "WHEN 'low' THEN 'low' "
                "WHEN 'moderate' THEN 'moderate' "
                "WHEN 'high' THEN 'high' "
                "WHEN 'very_high' THEN 'severe' "
                "WHEN 'severe' THEN 'severe' "
                "ELSE 'moderate' END"
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "severity_reports" not in inspector.get_table_names():
        return

    _replace_enum(
        table_name="severity_reports",
        column_name="risk_level",
        type_name="risklevel",
        new_values=["low", "moderate", "high", "very_high"],
        mapping_sql=(
            "CASE risk_level::text "
            "WHEN 'low' THEN 'low' "
            "WHEN 'moderate' THEN 'moderate' "
            "WHEN 'high' THEN 'high' "
            "WHEN 'severe' THEN 'very_high' "
            "WHEN 'very_high' THEN 'very_high' "
            "ELSE 'moderate' END"
        ),
    )
