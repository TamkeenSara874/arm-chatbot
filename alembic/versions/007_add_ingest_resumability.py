"""Add pipeline_version to review_chunk_meta, skipped_already_processed to ingest_job

Revision ID: 007
Revises: 006
Create Date: 2026-07-06
"""

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, no backfill: existing rows get NULL, which a resumability skip
    # check (WHERE pipeline_version = current) will never match -- a pipeline
    # version bump correctly forces reprocessing of pre-existing rows instead
    # of silently skipping them as "already done".
    op.add_column(
        "review_chunk_meta",
        sa.Column("pipeline_version", sa.String(20), nullable=True),
    )
    op.add_column(
        "ingest_job",
        sa.Column("skipped_already_processed", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingest_job", "skipped_already_processed")
    op.drop_column("review_chunk_meta", "pipeline_version")
