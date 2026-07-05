export interface EvidenceItem {
  snippet: string;
  username?: string | null;
  rating?: number | null;
  source?: string | null;
  sentiment?: string | null;
  sentiment_conflict: boolean;
  date_inferred: boolean;
  relevance: number;
}

export interface SubAnswer {
  sub_query: string;
  answer: string;
}

export interface ChatResponseSchema {
  answer: string;
  sub_answers: SubAnswer[];
  evidence: EvidenceItem[];
  confidence: number;
  caveats?: string | null;
  entity_counts: Record<string, number>;
  source_breakdown: Record<string, number>;
}

export interface SessionResponse {
  session_id: string;
  restaurant_id: number;
  created_at: string;
}

export interface ChatQueryRequest {
  session_id: string;
  restaurant_id: number;
  message: string;
}

export interface ChatQueryResponse {
  session_id: string;
  message_id: string;
  response: ChatResponseSchema;
  cached: boolean;
  complexity: string;
  model_used: string;
  latency_ms?: number;
  cost_usd?: number;
}

export interface ChatMessage {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  confidence?: number | null;
  created_at: string;
}

export interface SessionHistoryResponse {
  messages: ChatMessage[];
  total: number;
  page: number;
}

export interface CorrectionRequest {
  session_id: string;
  message_id: string;
  corrected_response: string;
}

export interface IngestJobResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  progress_pct: number;
  total_reviews?: number | null;
  total_chunks?: number | null;
  skipped_empty?: number | null;
  error_message?: string | null;
}

export interface ReportRequest {
  session_id: string;
  restaurant_id: number;
  message: string;
}

// Backend serializes list[tuple[str, int]] as a 2-element JSON array per entry.
export type EntityCount = [entity: string, count: number];

export interface InsightsReport {
  restaurant_id: number;
  generated_at: string;
  date_from?: string | null;
  date_to?: string | null;
  total_reviews: number;
  avg_rating: number | null;
  rating_distribution: Record<string, number>;
  sentiment_breakdown: Record<string, number>;
  source_breakdown: Record<string, number>;
  top_praised: EntityCount[];
  top_complained: EntityCount[];
  summary: string;
  markdown: string;
}

export interface ReportResponse {
  report: InsightsReport;
}
