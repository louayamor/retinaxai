"""add notifications table

Revision ID: add_notifications
Revises: add_auth_sessions
Create Date: 2026-04-15 10:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_notifications"
down_revision: str | None = "add_access_token_jti"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("notifications"):
        op.create_table(
            "notifications",
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
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("type", sa.String(length=50), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("message", sa.String(length=1000), nullable=False),
            sa.Column("read", sa.Boolean(), nullable=False, server_default=sa.false()),
        )

    existing_indexes = {index['name'] for index in inspector.get_indexes('notifications')} if inspector.has_table('notifications') else set()
    if "ix_notifications_user_id" not in existing_indexes:
        op.create_index("ix_notifications_user_id", "notifications", ["user_id"])
    if "ix_notifications_type" not in existing_indexes:
        op.create_index("ix_notifications_type", "notifications", ["type"])
    if "ix_notifications_read" not in existing_indexes:
        op.create_index("ix_notifications_read", "notifications", ["read"])


def downgrade() -> None:
    op.drop_index("ix_notifications_read", table_name="notifications")
    op.drop_index("ix_notifications_type", table_name="notifications")
    op.drop_index("ix_notifications_user_id", table_name="notifications")
    op.drop_table("notifications")
