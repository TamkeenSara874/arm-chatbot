"""Add summary_message_count to chat_session

Revision ID: 009
Revises: 008
Create Date: 2026-07-21
"""

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, no backfill. NULL means "never summarized", which is exactly
    # right for the two kinds of row that already exist:
    #   - summary IS NULL     -> genuinely never summarized.
    #   - summary IS NOT NULL -> written by the old one-shot path, which never
    #     recorded how much it covered. Treating it as NULL makes the next
    #     refresh re-summarize from the start of the conversation once, after
    #     which the count is accurate. That is a single extra summary call per
    #     pre-existing summarized session, and it is preferable to guessing a
    #     coverage number we cannot actually know.
    op.add_column(
        "chat_session",
        sa.Column("summary_message_count", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_session", "summary_message_count")
