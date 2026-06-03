from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "55a592cbc1ab"
down_revision: str | None = "add_chat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "auth_sessions",
        sa.Column("refresh_token_jti", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "auth_sessions",
        sa.Column(
            "token_family",
            sa.UUID(),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
    )
    op.alter_column("auth_sessions", "token_family", server_default=None)
    op.create_index(
        op.f("ix_auth_sessions_refresh_token_jti"),
        "auth_sessions",
        ["refresh_token_jti"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_auth_sessions_refresh_token_jti"), table_name="auth_sessions")
    op.drop_column("auth_sessions", "token_family")
    op.drop_column("auth_sessions", "refresh_token_jti")
