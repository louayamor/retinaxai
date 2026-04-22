"""Repair missing biomarker columns on predictions

Revision ID: add_prediction_biomarker_columns
Revises: 96e2434a5f94
Create Date: 2026-04-22

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "add_prediction_biomarker_columns"
down_revision: Union[str, None] = "96e2434a5f94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {column["name"] for column in inspector.get_columns("predictions")}

    biomarker_status = postgresql.ENUM(
        "PENDING",
        "COMPLETED",
        "FAILED",
        name="biomarkerstatus",
        create_type=False,
    )
    biomarker_status.create(bind, checkfirst=True)

    if "biomarker_status" not in existing_columns:
        op.add_column(
            "predictions",
            sa.Column("biomarker_status", biomarker_status, nullable=True),
        )

    if "biomarker_error_message" not in existing_columns:
        op.add_column(
            "predictions",
            sa.Column("biomarker_error_message", sa.String(length=500), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("predictions", "biomarker_error_message")
    op.drop_column("predictions", "biomarker_status")
