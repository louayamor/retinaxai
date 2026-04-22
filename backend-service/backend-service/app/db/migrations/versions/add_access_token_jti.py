"""Add access_token_jti to auth_sessions

Revision ID: add_access_token_jti
Revises: 687b907da35d
Create Date: 2026-04-13 18:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "add_access_token_jti"
down_revision: str | None = "687b907da35d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {column['name'] for column in inspector.get_columns('auth_sessions')}
    existing_indexes = {index['name'] for index in inspector.get_indexes('auth_sessions')} if inspector.has_table('auth_sessions') else set()

    if "access_token_jti" not in existing_columns:
        op.add_column(
            "auth_sessions",
            sa.Column("access_token_jti", sa.String(length=36), nullable=True),
        )
    if "ix_auth_sessions_access_token_jti" not in existing_indexes:
        op.create_index(
            "ix_auth_sessions_access_token_jti",
            "auth_sessions",
            ["access_token_jti"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index("ix_auth_sessions_access_token_jti", table_name="auth_sessions")
    op.drop_column("auth_sessions", "access_token_jti")
