from prometheus_client import Counter, Gauge, Histogram

llm_request_total = Counter(
    "llm_request_total",
    "Total LLM API requests by provider, model and intent",
    ["provider", "model", "intent"],
)

llm_request_latency = Histogram(
    "llm_request_latency_seconds",
    "End-to-end LLM request latency in seconds",
    ["provider", "model"],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

retrieval_latency = Histogram(
    "retrieval_latency_seconds",
    "Hybrid retrieval latency (ANN + BM25 + RRF) in seconds",
    buckets=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)

cache_hit_total = Counter(
    "cache_hit_total",
    "Redis query cache outcomes",
    ["result"],  # "hit" | "miss"
)

active_sessions_gauge = Gauge(
    "active_sessions_gauge",
    "Number of active chat sessions (created in last 24h)",
)

guardrail_triggered_total = Counter(
    "guardrail_triggered_total",
    "Number of queries blocked by the guardrail",
    ["type"],  # "out_of_scope" | "ui_question" | ... | "crisis_language"
)

injection_attempts_detected_total = Counter(
    "injection_attempts_detected_total",
    "Injection pattern matches found at ingestion time",
    ["restaurant_id"],
)

ingest_reviews_total = Counter(
    "ingest_reviews_total",
    "Reviews processed during ingestion",
    ["status"],  # "ingested" | "skipped_empty" | "skipped_invalid"
)

count_query_total = Counter(
    "count_query_total",
    "count_query fast-path executions (no LLM cost)",
)

output_validation_failed_total = Counter(
    "output_validation_failed_total",
    "LLM outputs rejected by validate_llm_output()",
)

report_generated_total = Counter(
    "report_generated_total",
    "Insights reports generated via export_insights_report tool call",
    ["restaurant_id"],
)

pipeline_stage_latency = Histogram(
    "pipeline_stage_latency_seconds",
    "Latency per pipeline stage: decomp, retrieval, rerank, ranking, generation",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)

rerank_latency = Histogram(
    "rerank_latency_seconds",
    "Cross-encoder reranking latency for a single candidate batch",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
)

request_cost_usd = Histogram(
    "request_cost_usd",
    "Estimated USD cost per chat request (LLM tokens only)",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
)
