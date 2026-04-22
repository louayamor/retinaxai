"""add auth sessions

Revision ID: add_auth_sessions
Revises: 96e2434a5f94
Create Date: 2026-04-04 10:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = 'add_auth_sessions'
down_revision: str | None = '96e2434a5f94'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table('auth_sessions'):
        op.create_table(
            'auth_sessions',
            sa.Column('id', sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('user_id', sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('refresh_token', sa.String(length=512), nullable=False, unique=True),
            sa.Column('revoked', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        )

    existing_indexes = {index['name'] for index in inspector.get_indexes('auth_sessions')} if inspector.has_table('auth_sessions') else set()
    if 'ix_auth_sessions_user_id' not in existing_indexes:
        op.create_index('ix_auth_sessions_user_id', 'auth_sessions', ['user_id'], unique=False)
    if 'ix_auth_sessions_refresh_token' not in existing_indexes:
        op.create_index('ix_auth_sessions_refresh_token', 'auth_sessions', ['refresh_token'], unique=True)
    if 'ix_auth_sessions_expires_at' not in existing_indexes:
        op.create_index('ix_auth_sessions_expires_at', 'auth_sessions', ['expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_auth_sessions_expires_at', table_name='auth_sessions')
    op.drop_index('ix_auth_sessions_refresh_token', table_name='auth_sessions')
    op.drop_index('ix_auth_sessions_user_id', table_name='auth_sessions')
    op.drop_table('auth_sessions')
