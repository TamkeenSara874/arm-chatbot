from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    snippet: str
    username: str | None = None
    rating: float | None = None
    source: str | None = None
    sentiment: str | None = None
    sentiment_conflict: bool = False
    relevance: float = 0.0


class SubAnswer(BaseModel):
    sub_query: str
    answer: str


class ChatResponseSchema(BaseModel):
    """Structured output schema enforced at the API level via OpenAI beta.parse."""

    answer: str
    sub_answers: list[SubAnswer] = []
    evidence: list[EvidenceItem]
    confidence: float = Field(..., ge=0.0, le=1.0)
    caveats: str | None = None
    entity_counts: dict[str, int] = {}
    source_breakdown: dict[str, int] = {}


class DateFilter(BaseModel):
    from_date: str | None = Field(None, alias="from")
    to_date: str | None = Field(None, alias="to")

    model_config = {"populate_by_name": True}


class RatingFilter(BaseModel):
    min: float | None = None
    max: float | None = None


class DecomposedQuery(BaseModel):
    intent: str
    aspect_filter: str | None = None
    sentiment_filter: str | None = None
    entities: list[str] = []
    needs_aggregation: bool = False
    complexity: Literal["simple", "complex"] = "simple"
    sub_queries: list[str] = []
    rephrased_query: str = ""
    source_filter: str | None = None
    date_filter: DateFilter | None = None
    rating_filter: RatingFilter | None = None


# Request / response schemas for the API


class SessionCreateRequest(BaseModel):
    restaurant_id: int
    user_identifier: str | None = None


class SessionResponse(BaseModel):
    session_id: uuid.UUID
    restaurant_id: int


class MessageResponse(BaseModel):
    message_id: uuid.UUID
    role: str
    content: str
    confidence: float | None = None
    created_at: str


class ChatQueryRequest(BaseModel):
    session_id: uuid.UUID
    restaurant_id: int
    message: str = Field(..., min_length=1, max_length=2000)


class ChatQueryResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    response: ChatResponseSchema
    cached: bool = False
    complexity: str
    model_used: str


class CorrectionRequest(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    corrected_response: str = Field(..., min_length=1, max_length=4000)


class CorrectionResponse(BaseModel):
    correction_id: uuid.UUID
    is_consensus: bool


class IngestJobResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    progress_pct: int
    total_reviews: int | None = None
    total_chunks: int | None = None
    skipped_empty: int | None = None
    error_message: str | None = None


class RestaurantListResponse(BaseModel):
    restaurant_ids: list[int]


class ReportRequest(BaseModel):
    session_id: uuid.UUID
    restaurant_id: int
    message: str = Field(..., min_length=1, max_length=2000)
    date_from: date | None = None
    date_to: date | None = None


class InsightsReport(BaseModel):
    """Structured output from the export_insights_report tool call."""

    restaurant_id: int
    generated_at: datetime
    date_from: date | None
    date_to: date | None
    total_reviews: int
    avg_rating: float | None
    rating_distribution: dict[str, int]
    sentiment_breakdown: dict[str, int]
    source_breakdown: dict[str, int]
    top_praised: list[tuple[str, int]]
    top_complained: list[tuple[str, int]]
    summary: str
    markdown: str


class ReportResponse(BaseModel):
    restaurant_id: int
    report: InsightsReport
    model_used: str
