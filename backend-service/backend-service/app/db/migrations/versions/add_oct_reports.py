"""add oct_reports table

Revision ID: add_oct_reports
Revises: add_auth_sessions
Create Date: 2026-04-04 19:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = 'add_oct_reports'
down_revision: str | None = 'add_auth_sessions'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table('oct_reports'):
        op.create_table(
            'oct_reports',
            sa.Column('id', sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('patient_id', sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
            sa.Column('eye', sa.String(length=2), nullable=False),
            sa.Column('source_file', sa.String(length=255), nullable=False),
            sa.Column('dr_grade', sa.String(length=50), nullable=True),
            sa.Column('edema', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('erm_status', sa.String(length=50), nullable=True),
            sa.Column('image_quality', sa.Float(), nullable=True),
            sa.Column('thickness_center_fovea', sa.Float(), nullable=True),
            sa.Column('thickness_average_thickness', sa.Float(), nullable=True),
            sa.Column('thickness_total_volume_mm3', sa.Float(), nullable=True),
            sa.Column('thickness_inner_superior', sa.Float(), nullable=True),
            sa.Column('thickness_inner_nasal', sa.Float(), nullable=True),
            sa.Column('thickness_inner_inferior', sa.Float(), nullable=True),
            sa.Column('thickness_inner_temporal', sa.Float(), nullable=True),
            sa.Column('thickness_outer_superior', sa.Float(), nullable=True),
            sa.Column('thickness_outer_nasal', sa.Float(), nullable=True),
            sa.Column('thickness_outer_inferior', sa.Float(), nullable=True),
            sa.Column('thickness_outer_temporal', sa.Float(), nullable=True),
        )

    existing_indexes = {index['name'] for index in inspector.get_indexes('oct_reports')} if inspector.has_table('oct_reports') else set()
    if 'ix_oct_reports_patient_id' not in existing_indexes:
        op.create_index('ix_oct_reports_patient_id', 'oct_reports', ['patient_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_oct_reports_patient_id', table_name='oct_reports')
    op.drop_table('oct_reports')
