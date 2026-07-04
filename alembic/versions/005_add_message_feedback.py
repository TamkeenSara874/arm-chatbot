"""Add feedback column to chat_message

Revision ID: 005
Revises: 004
Create Date: 2026-07-04
"""

import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_message",
        sa.Column("feedback", sa.String(10), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_message", "feedback")
