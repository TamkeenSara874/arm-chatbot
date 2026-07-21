"""Add chat_correction_vote (per-session consensus dedup + audit trail), is_rejected flag

Revision ID: 008
Revises: 007
Create Date: 2026-07-20

Closes a real gap: correction_count previously incremented on any matching
submission with no check on who submitted it, so a single session hitting
POST /chat/correct 3 times reached "confirmed consensus" (which overrides
real review evidence in the generation prompt) with zero genuine
corroboration. One row per (correction_id, session_id) makes the count
reflect distinct sessions, enforced by a unique constraint rather than
trusted client input.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_correction_vote",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "correction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_correction.id"),
            nullable=False,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Best-effort audit field, not an access-control decision -- IPs are
        # unreliable behind proxies/NAT, so nothing here relies on it beyond
        # giving a human something to look at after the fact.
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_correction_vote_correction_id", "chat_correction_vote", ["correction_id"]
    )
    op.create_index(
        "ix_correction_vote_session_id", "chat_correction_vote", ["session_id"]
    )
    op.create_unique_constraint(
        "uq_correction_vote_correction_session",
        "chat_correction_vote",
        ["correction_id", "session_id"],
    )
    # Soft-reject: an admin-rejected correction keeps its row (and vote
    # history) for audit purposes -- only its Qdrant point is deleted, which
    # is what actually stops it from ever being surfaced/applied again.
    op.add_column(
        "chat_correction",
        sa.Column("is_rejected", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("chat_correction", "is_rejected")
    op.drop_table("chat_correction_vote")
