"""add chat sessions and messages tables

Revision ID: add_chat
Revises: add_notifications
Create Date: 2026-05-13 18:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_chat"
down_revision: str | tuple[str, ...] | None = (
    "0f4d6c8a9b21",
    "add_biomarker_error_code",
    "add_ocr_patient_id",
    "add_oct_fields_to_reports",
    "add_severity_risklevel_severe",
    "add_vascular_biomarkers",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("chat_sessions"):
        op.create_table(
            "chat_sessions",
            sa.Column(
                "id",
                sa.dialects.postgresql.UUID(as_uuid=True),
                primary_key=True,
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "user_id",
                sa.dialects.postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "title", sa.String(255), nullable=False, server_default="New Chat"
            ),
        )

    if not inspector.has_table("chat_messages"):
        op.create_table(
            "chat_messages",
            sa.Column(
                "id",
                sa.dialects.postgresql.UUID(as_uuid=True),
                primary_key=True,
                nullable=False,
            ),
            sa.Column(
                "session_id",
                sa.dialects.postgresql.UUID(as_uuid=True),
                sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "role",
                sa.Enum("user", "assistant", name="chatrole"),
                nullable=False,
            ),
            sa.Column("content", sa.Text, nullable=False),
            sa.Column("sources", sa.dialects.postgresql.JSONB, nullable=True),
            sa.Column("chart", sa.dialects.postgresql.JSONB, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.execute("DROP TYPE IF EXISTS chatrole")
    op.drop_table("chat_sessions")
