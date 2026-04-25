"""Add biomarker error code fields

Revision ID: add_biomarker_error_code
Revises: add_prediction_biomarker_columns
Create Date: 2026-04-22

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_biomarker_error_code"
down_revision: Union[str, None] = "add_prediction_biomarker_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}
    biomarker_columns = {column["name"] for column in inspector.get_columns("vascular_biomarkers")}

    if "biomarker_error_code" not in prediction_columns:
        op.add_column(
            "predictions",
            sa.Column("biomarker_error_code", sa.String(length=100), nullable=True),
        )

    if "error_code" not in biomarker_columns:
        op.add_column(
            "vascular_biomarkers",
            sa.Column("error_code", sa.String(length=100), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}
    biomarker_columns = {column["name"] for column in inspector.get_columns("vascular_biomarkers")}

    if "biomarker_error_code" in prediction_columns:
        op.drop_column("predictions", "biomarker_error_code")
    if "error_code" in biomarker_columns:
        op.drop_column("vascular_biomarkers", "error_code")
