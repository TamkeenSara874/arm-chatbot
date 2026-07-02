"""Add date_inferred to review_chunk_meta

Revision ID: 002
Revises: 001
Create Date: 2026-07-02
"""

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "review_chunk_meta",
        sa.Column(
            "date_inferred",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )


def downgrade() -> None:
    op.drop_column("review_chunk_meta", "date_inferred")
