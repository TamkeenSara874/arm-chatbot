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
    # Existing dev/prod data can already have multiple stale pending/processing
    # rows per restaurant (e.g. a job whose container was killed before it
    # reached a terminal status) -- the unique index below would reject those
    # as duplicates and fail the migration. Close out every such row except
    # the most recent one per restaurant before adding the constraint.
    op.execute(
        """
        UPDATE ingest_job
        SET status = 'failed',
            error_message = 'Superseded by a newer ingest job for this restaurant '
                            '(closed automatically while adding the one-active-job '
                            'per-restaurant constraint).'
        WHERE status IN ('pending', 'processing')
          AND id NOT IN (
              SELECT DISTINCT ON (restaurant_id) id
              FROM ingest_job
              WHERE status IN ('pending', 'processing')
              ORDER BY restaurant_id, created_at DESC
          )
        """
    )
    op.create_index(
        "ix_ingest_job_one_active_per_restaurant",
        "ingest_job",
        ["restaurant_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    op.drop_index("ix_ingest_job_one_active_per_restaurant", table_name="ingest_job")
