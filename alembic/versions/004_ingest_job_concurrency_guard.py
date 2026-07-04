"""Add partial unique index preventing concurrent ingest jobs per restaurant

Revision ID: 004
Revises: 003
Create Date: 2026-07-03
"""

import sqlalchemy as sa
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_ingest_job_one_active_per_restaurant",
        "ingest_job",
        ["restaurant_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    op.drop_index("ix_ingest_job_one_active_per_restaurant", table_name="ingest_job")
