"""add_oct_fields_to_reports

Revision ID: add_oct_fields_to_reports
Revises: 687b907da35d
Create Date: 2026-04-18 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'add_oct_fields_to_reports'
down_revision: str | None = '687b907da35d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {column['name'] for column in inspector.get_columns('reports')}

    if 'reporttype' not in {enum['name'] for enum in inspector.get_enums()}:
        op.execute("CREATE TYPE reporttype AS ENUM ('LLM', 'OCT')")

    if 'report_type' not in existing_columns:
        op.add_column('reports', sa.Column('report_type', sa.Enum('LLM', 'OCT', name='reporttype'), server_default='LLM', nullable=False))
    if 'prediction_id' in existing_columns:
        op.alter_column('reports', 'prediction_id', existing_type=sa.dialects.postgresql.UUID(as_uuid=True), nullable=True)

    for column_name, column_type in [
        ('eye', sa.String(length=2)),
        ('source_file', sa.String(length=255)),
        ('dr_grade', sa.String(length=50)),
        ('edema', sa.Boolean()),
        ('erm_status', sa.String(length=50)),
        ('image_quality', sa.Float()),
        ('thickness_center_fovea', sa.Float()),
        ('thickness_average_thickness', sa.Float()),
        ('thickness_total_volume_mm3', sa.Float()),
        ('thickness_inner_superior', sa.Float()),
        ('thickness_inner_nasal', sa.Float()),
        ('thickness_inner_inferior', sa.Float()),
        ('thickness_inner_temporal', sa.Float()),
        ('thickness_outer_superior', sa.Float()),
        ('thickness_outer_nasal', sa.Float()),
        ('thickness_outer_inferior', sa.Float()),
        ('thickness_outer_temporal', sa.Float()),
    ]:
        if column_name not in existing_columns:
            op.add_column('reports', sa.Column(column_name, column_type, nullable=True))


def downgrade() -> None:
    op.drop_column('reports', 'thickness_outer_temporal')
    op.drop_column('reports', 'thickness_outer_inferior')
    op.drop_column('reports', 'thickness_outer_nasal')
    op.drop_column('reports', 'thickness_outer_superior')
    op.drop_column('reports', 'thickness_inner_temporal')
    op.drop_column('reports', 'thickness_inner_inferior')
    op.drop_column('reports', 'thickness_inner_nasal')
    op.drop_column('reports', 'thickness_inner_superior')
    op.drop_column('reports', 'thickness_total_volume_mm3')
    op.drop_column('reports', 'thickness_average_thickness')
    op.drop_column('reports', 'thickness_center_fovea')
    op.drop_column('reports', 'image_quality')
    op.drop_column('reports', 'erm_status')
    op.drop_column('reports', 'edema')
    op.drop_column('reports', 'dr_grade')
    op.drop_column('reports', 'source_file')
    op.drop_column('reports', 'eye')
    op.alter_column('reports', 'prediction_id', existing_type=sa.dialects.postgresql.UUID(as_uuid=True), nullable=False)
    op.drop_column('reports', 'report_type')
    op.execute('DROP TYPE IF EXISTS reporttype')
