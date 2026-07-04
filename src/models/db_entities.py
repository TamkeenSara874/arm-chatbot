import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_session"
    __table_args__ = (
        Index("ix_chat_session_restaurant_id", "restaurant_id"),
        Index("ix_chat_session_last_activity", "last_activity_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    restaurant_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    dietary_flags: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"
    __table_args__ = (
        Index("ix_chat_message_session_id", "session_id"),
        Index("ix_chat_message_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    feedback: Mapped[str | None] = mapped_column(String(10), nullable=True)  # "up" | "down"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")


class ChatCorrection(Base):
    __tablename__ = "chat_correction"
    __table_args__ = (Index("ix_chat_correction_restaurant_id", "restaurant_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    qdrant_point_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    restaurant_id: Mapped[int] = mapped_column(Integer, nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    original_response: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_response: Mapped[str] = mapped_column(Text, nullable=False)
    correction_count: Mapped[int] = mapped_column(Integer, default=1)
    is_consensus: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewChunkMeta(Base):
    __tablename__ = "review_chunk_meta"
    __table_args__ = (
        # Composite indexes for count_query fast path
        Index("ix_chunk_restaurant_chunk_index", "restaurant_id", "chunk_index"),
        Index("ix_chunk_restaurant_sentiment", "restaurant_id", "sentiment_label"),
        Index("ix_chunk_restaurant_date", "restaurant_id", "review_date"),
        Index("ix_chunk_restaurant_rating", "restaurant_id", "rating"),
        Index("ix_chunk_restaurant_content", "restaurant_id", "has_content"),
    )

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    restaurant_id: Mapped[int] = mapped_column(Integer, nullable=False)
    review_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_review: Mapped[str | None] = mapped_column(Text, nullable=True)
    # True = has text content and a Qdrant embedding; False = rating/metadata only
    has_content: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    sentiment_rating_agree: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    review_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    has_injection_attempt: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    # True when createdAt was unparseable and datetime.now() was used as fallback
    date_inferred: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestJob(Base):
    __tablename__ = "ingest_job"
    __table_args__ = (
        Index("ix_ingest_job_restaurant_id", "restaurant_id"),
        Index("ix_ingest_job_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    restaurant_id: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )  # pending | processing | complete | failed
    progress_pct: Mapped[int] = mapped_column(Integer, server_default="0")
    total_reviews: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    skipped_empty: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestManifest(Base):
    """Tracks the last successfully seeded state of each Qdrant collection.

    The seed script compares both file_hash and pipeline_version against the
    stored values. A match on both means the collection is current; any
    mismatch triggers a full re-ingest. Bump PIPELINE_VERSION in
    ingest_worker.py whenever the embedding model, chunking strategy, or
    entity extraction prompt changes.
    """

    __tablename__ = "ingest_manifest"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    review_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
