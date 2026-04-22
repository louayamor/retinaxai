"""align enum values with current models

Revision ID: 0f4d6c8a9b21
Revises: 687b907da35d
Create Date: 2026-04-17 02:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0f4d6c8a9b21"
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

    if "patients" in inspector.get_table_names():
        _replace_enum(
            table_name="patients",
            column_name="gender",
            type_name="gender",
            new_values=["M", "F"],
            mapping_sql=(
                "CASE gender::text "
                "WHEN 'MALE' THEN 'M' "
                "WHEN 'FEMALE' THEN 'F' "
                "WHEN 'M' THEN 'M' "
                "WHEN 'F' THEN 'F' "
                "ELSE gender::text END"
            ),
        )

    if "predictions" in inspector.get_table_names():
        _replace_enum(
            table_name="predictions",
            column_name="status",
            type_name="predictionstatus",
            new_values=["pending", "success", "partial", "failed"],
            mapping_sql=(
                "CASE status::text "
                "WHEN 'PENDING' THEN 'pending' "
                "WHEN 'SUCCESS' THEN 'success' "
                "WHEN 'FAILED' THEN 'failed' "
                "WHEN 'PARTIAL' THEN 'partial' "
                "WHEN 'pending' THEN 'pending' "
                "WHEN 'success' THEN 'success' "
                "WHEN 'failed' THEN 'failed' "
                "WHEN 'partial' THEN 'partial' "
                "ELSE status::text END"
            ),
        )

    if "reports" in inspector.get_table_names():
        _replace_enum(
            table_name="reports",
            column_name="status",
            type_name="reportstatus",
            new_values=["generating", "completed", "failed"],
            mapping_sql=(
                "CASE status::text "
                "WHEN 'GENERATING' THEN 'generating' "
                "WHEN 'COMPLETED' THEN 'completed' "
                "WHEN 'FAILED' THEN 'failed' "
                "WHEN 'generating' THEN 'generating' "
                "WHEN 'completed' THEN 'completed' "
                "WHEN 'failed' THEN 'failed' "
                "ELSE status::text END"
            ),
        )


def downgrade() -> None:
    _replace_enum(
        table_name="reports",
        column_name="status",
        type_name="reportstatus",
        new_values=["GENERATING", "COMPLETED", "FAILED"],
        mapping_sql=(
            "CASE status::text "
            "WHEN 'generating' THEN 'GENERATING' "
            "WHEN 'completed' THEN 'COMPLETED' "
            "WHEN 'failed' THEN 'FAILED' "
            "ELSE status::text END"
        ),
    )

    _replace_enum(
        table_name="predictions",
        column_name="status",
        type_name="predictionstatus",
        new_values=["PENDING", "SUCCESS", "FAILED"],
        mapping_sql=(
            "CASE status::text "
            "WHEN 'pending' THEN 'PENDING' "
            "WHEN 'success' THEN 'SUCCESS' "
            "WHEN 'partial' THEN 'FAILED' "
            "WHEN 'failed' THEN 'FAILED' "
            "ELSE status::text END"
        ),
    )

    _replace_enum(
        table_name="patients",
        column_name="gender",
        type_name="gender",
        new_values=["MALE", "FEMALE"],
        mapping_sql=(
            "CASE gender::text "
            "WHEN 'M' THEN 'MALE' "
            "WHEN 'F' THEN 'FEMALE' "
            "ELSE gender::text END"
        ),
    )
