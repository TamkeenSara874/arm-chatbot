"""Create all tables

Revision ID: 001
Revises:
Create Date: 2026-07-01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # chat_session
    op.create_table(
        "chat_session",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("restaurant_id", sa.Integer(), nullable=False),
        sa.Column("user_identifier", sa.String(255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("dietary_flags", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chat_session_restaurant_id", "chat_session", ["restaurant_id"])
    op.create_index("ix_chat_session_last_activity", "chat_session", ["last_activity_at"])

    # chat_message
    op.create_table(
        "chat_message",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True),
                  sa.ForeignKey("chat_session.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("retrieved_chunk_ids", ARRAY(sa.String()), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chat_message_session_id", "chat_message", ["session_id"])
    op.create_index("ix_chat_message_created_at", "chat_message", ["created_at"])

    # chat_correction
    op.create_table(
        "chat_correction",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("qdrant_point_id", sa.String(255), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("restaurant_id", sa.Integer(), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("original_response", sa.Text(), nullable=False),
        sa.Column("corrected_response", sa.Text(), nullable=False),
        sa.Column("correction_count", sa.Integer(), server_default="1"),
        sa.Column("is_consensus", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chat_correction_restaurant_id", "chat_correction", ["restaurant_id"])

    # review_chunk_meta
    op.create_table(
        "review_chunk_meta",
        sa.Column("chunk_id", sa.String(255), primary_key=True),
        sa.Column("restaurant_id", sa.Integer(), nullable=False),
        sa.Column("review_id", sa.String(255), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=True),
        sa.Column("full_review", sa.Text(), nullable=True),
        sa.Column("has_content", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("sentiment_label", sa.String(50), nullable=True),
        sa.Column("sentiment_rating_agree", sa.Boolean(), nullable=True),
        sa.Column("review_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("username", sa.String(255), nullable=True),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("has_injection_attempt", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Composite indexes are critical for count_query performance
    op.create_index("ix_chunk_restaurant_chunk_index", "review_chunk_meta",
                    ["restaurant_id", "chunk_index"])
    op.create_index("ix_chunk_restaurant_sentiment", "review_chunk_meta",
                    ["restaurant_id", "sentiment_label"])
    op.create_index("ix_chunk_restaurant_date", "review_chunk_meta",
                    ["restaurant_id", "review_date"])
    op.create_index("ix_chunk_restaurant_rating", "review_chunk_meta",
                    ["restaurant_id", "rating"])
    op.create_index("ix_chunk_restaurant_content", "review_chunk_meta",
                    ["restaurant_id", "has_content"])

    # ingest_job
    op.create_table(
        "ingest_job",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("restaurant_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("progress_pct", sa.Integer(), server_default="0"),
        sa.Column("total_reviews", sa.Integer(), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column("skipped_empty", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_ingest_job_restaurant_id", "ingest_job", ["restaurant_id"])
    op.create_index("ix_ingest_job_status", "ingest_job", ["status"])


def downgrade() -> None:
    op.drop_table("ingest_job")
    op.drop_index("ix_chunk_restaurant_content", "review_chunk_meta")
    op.drop_index("ix_chunk_restaurant_rating", "review_chunk_meta")
    op.drop_index("ix_chunk_restaurant_date", "review_chunk_meta")
    op.drop_index("ix_chunk_restaurant_sentiment", "review_chunk_meta")
    op.drop_index("ix_chunk_restaurant_chunk_index", "review_chunk_meta")
    op.drop_table("review_chunk_meta")
    op.drop_table("chat_correction")
    op.drop_table("chat_message")
    op.drop_table("chat_session")
