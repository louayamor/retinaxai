"""Add vascular biomarkers table

Revision ID: add_vascular_biomarkers
Revises: add_prediction_explanations
Create Date: 2026-04-21

"""

from collections.abc import Sequence
from typing import Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "add_vascular_biomarkers"
down_revision: Union[str, None] = "add_prediction_explanations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    biomarker_status = postgresql.ENUM(
        "PENDING",
        "COMPLETED",
        "FAILED",
        name="biomarkerstatus",
        create_type=False,
    )
    biomarker_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "vascular_biomarkers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("eye_side", sa.String(length=10), nullable=True),
        sa.Column("service_name", sa.String(length=100), nullable=False),
        sa.Column("service_version", sa.String(length=50), nullable=False),
        sa.Column("contract_version", sa.String(length=20), nullable=False),
        sa.Column("status", biomarker_status, nullable=False, server_default="PENDING"),
        sa.Column("biomarkers", postgresql.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_vascular_biomarkers_patient_id", "vascular_biomarkers", ["patient_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_vascular_biomarkers_patient_id", table_name="vascular_biomarkers")
    op.drop_table("vascular_biomarkers")
    op.execute("DROP TYPE IF EXISTS biomarkerstatus")
