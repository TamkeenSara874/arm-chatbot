"""Add ingest_manifest table

Revision ID: 003
Revises: 002
Create Date: 2026-07-02
"""

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingest_manifest",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("collection_name", sa.String(128), nullable=False, unique=True),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("pipeline_version", sa.String(32), nullable=False),
        sa.Column("review_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("ingest_manifest")
