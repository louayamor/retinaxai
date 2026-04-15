"""Add prediction_explanations table

Revision ID: add_prediction_explanations
Revises:
Create Date: 2026-04-15

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "add_prediction_explanations"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create explanation status enum
    explanation_status = postgresql.ENUM(
        "pending",
        "processing",
        "completed",
        "failed",
        name="explanationstatus",
        create_type=False,
    )
    explanation_status.create(op.get_bind(), checkfirst=True)

    # Create prediction_explanations table
    op.create_table(
        "prediction_explanations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.String(length=500), nullable=True),
        sa.Column("model_used", sa.String(length=100), nullable=False),
        sa.Column("status", explanation_status, nullable=False, default="pending"),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("shap_values", postgresql.JSON(), nullable=True),
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

    # Create gradcam_explanations table
    gradcam_status = postgresql.ENUM(
        "pending",
        "processing",
        "completed",
        "failed",
        name="gradcamstatus",
        create_type=False,
    )
    gradcam_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "gradcam_explanations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("left_eye_explanation", sa.Text(), nullable=True),
        sa.Column("right_eye_explanation", sa.Text(), nullable=True),
        sa.Column("highlighted_regions", postgresql.JSON(), nullable=True),
        sa.Column("model_used", sa.String(length=100), nullable=True),
        sa.Column("status", gradcam_status, nullable=True),
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

    # Create severity_reports table
    risk_level = postgresql.ENUM(
        "low", "moderate", "high", "very_high", name="risklevel", create_type=False
    )
    risk_level.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "severity_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("risk_level", risk_level, nullable=True),
        sa.Column("recommendations", postgresql.JSON(), nullable=True),
        sa.Column("model_used", sa.String(length=100), nullable=True),
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


def downgrade() -> None:
    op.drop_table("severity_reports")
    op.drop_table("gradcam_explanations")
    op.drop_table("prediction_explanations")

    op.execute("DROP TYPE IF EXISTS risklevel")
    op.execute("DROP TYPE IF EXISTS gradcamstatus")
    op.execute("DROP TYPE IF EXISTS explanationstatus")
