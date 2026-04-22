"""Add modality to mri_scans

Revision ID: add_mri_scans_modality
Revises: add_oct_reports
Create Date: 2026-04-10 00:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = 'add_mri_scans_modality'
down_revision: str | None = 'add_oct_reports'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {column['name'] for column in inspector.get_columns('mri_scans')}

    if 'modality' not in existing_columns:
        op.add_column(
            'mri_scans',
            sa.Column('modality', sa.String(length=50), nullable=False, server_default='fundus')
        )


def downgrade() -> None:
    op.drop_column('mri_scans', 'modality')
