import type {
  ChatMessage,
  CorrectionRequest,
  IngestJobResponse,
  InsightsReport,
  ReportRequest,
  RestaurantListResponse,
  SessionResponse,
} from '../types/api';

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '';
const API_KEY = (import.meta.env.VITE_API_KEY as string | undefined) ?? 'change-me-local-dev-key';

const JWT_KEY = 'arm-chatbot-jwt';

export function getStoredJwt(): string | null {
  return localStorage.getItem(JWT_KEY);
}

function storeJwt(token: string): void {
  localStorage.setItem(JWT_KEY, token);
}

export function clearJwt(): void {
  localStorage.removeItem(JWT_KEY);
}

function authHeaders(useJwt = false): Record<string, string> {
  const token = useJwt ? (getStoredJwt() ?? API_KEY) : API_KEY;
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`,
  };
}

async function request<T>(path: string, init: RequestInit = {}, useJwt = false): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(useJwt), ...(init.headers as Record<string, string> | undefined) },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const getApiKey = (): string => getStoredJwt() ?? API_KEY;

export const api = {
  getRestaurants: () => request<RestaurantListResponse>('/api/v1/restaurants'),

  login: async (restaurantId: number): Promise<void> => {
    const res = await request<{ access_token: string }>(
      '/api/v1/auth/token',
      { method: 'POST', body: JSON.stringify({ restaurant_id: restaurantId }) },
    );
    storeJwt(res.access_token);
  },

  createSession: (restaurantId: number) =>
    request<SessionResponse>(
      '/api/v1/chat/sessions',
      { method: 'POST', body: JSON.stringify({ restaurant_id: restaurantId }) },
      true,
    ),

  getHistory: (sessionId: string, page = 1) =>
    request<ChatMessage[]>(
      `/api/v1/chat/sessions/${sessionId}/history?page=${page}`,
      {},
      true,
    ),

  submitCorrection: (body: CorrectionRequest) =>
    request<{ ok: boolean }>(
      '/api/v1/chat/correct',
      { method: 'POST', body: JSON.stringify(body) },
      true,
    ),

  generateReport: (body: ReportRequest) =>
    request<{ report: InsightsReport }>(
      '/api/v1/chat/report',
      { method: 'POST', body: JSON.stringify(body) },
      true,
    ),

  getIngestStatus: (jobId: string) =>
    request<IngestJobResponse>(`/api/v1/ingest/${jobId}/status`, {}, true),
};
