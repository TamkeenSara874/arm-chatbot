import type {
  AnomalyAlertResponse,
  ChatMessage,
  CorrectionRequest,
  IngestJobResponse,
  InsightsReport,
  ReportRequest,
  SessionResponse,
  VoiceTranscribeResponse,
} from '../types/api';
import { useChatStore } from '../store/chatStore';

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '';
const API_KEY = (import.meta.env.VITE_API_KEY as string | undefined) ?? 'change-me-local-dev-key';

const JWT_KEY = 'arm-chatbot-jwt';

export class AuthExpiredError extends Error {
  constructor(message = 'Your session has expired. Please select your restaurant again.') {
    super(message);
    this.name = 'AuthExpiredError';
  }
}

export function getStoredJwt(): string | null {
  return localStorage.getItem(JWT_KEY);
}

function storeJwt(token: string): void {
  localStorage.setItem(JWT_KEY, token);
}

export function clearJwt(): void {
  localStorage.removeItem(JWT_KEY);
}

// Clearing the JWT and dropping restaurantId forces App's LoginPage back into
// view -- the same reset handleLogout() already does -- so an expired/missing
// token always routes the user back through a real login instead of silently
// continuing under the wrong credentials.
export function resetToLogin(): void {
  clearJwt();
  useChatStore.getState().setRestaurantId(null);
}

function authHeaders(useJwt = false): Record<string, string> {
  if (useJwt) {
    const token = getStoredJwt();
    if (!token) {
      resetToLogin();
      throw new AuthExpiredError();
    }
    return { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` };
  }
  return { 'Content-Type': 'application/json', Authorization: `Bearer ${API_KEY}` };
}

async function request<T>(path: string, init: RequestInit = {}, useJwt = false): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(useJwt), ...(init.headers as Record<string, string> | undefined) },
  });
  if (res.status === 401 && useJwt) {
    resetToLogin();
    throw new AuthExpiredError();
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** Returns the restaurant JWT or throws AuthExpiredError -- never falls back
 * to the shared static API key, which the JWT-only chat endpoints reject
 * anyway (a silent fallback here previously meant a missing/expired token
 * failed with a generic 401 instead of a clear "log in again" signal). */
export function getRequiredJwt(): string {
  const token = getStoredJwt();
  if (!token) {
    resetToLogin();
    throw new AuthExpiredError();
  }
  return token;
}

export const api = {
  login: async (restaurantId: number, restaurantKey: string): Promise<void> => {
    const res = await request<{ access_token: string }>(
      '/api/v1/auth/token',
      {
        method: 'POST',
        body: JSON.stringify({ restaurant_id: restaurantId, restaurant_key: restaurantKey }),
      },
    );
    storeJwt(res.access_token);
  },

  createSession: (restaurantId: number) =>
    request<SessionResponse>(
      '/api/v1/chat/sessions',
      { method: 'POST', body: JSON.stringify({ restaurant_id: restaurantId }) },
      true,
    ),

  // The backend route takes limit/offset, not page -- a `page` param was
  // silently ignored, so every reload always returned the same oldest-20
  // messages regardless of how long the session was. Requesting a limit
  // large enough to cover realistic sessions returns the full history in
  // one round trip without needing offset math or a backend contract change.
  getHistory: (sessionId: string, limit = 200) =>
    request<ChatMessage[]>(
      `/api/v1/chat/sessions/${sessionId}/history?limit=${limit}`,
      {},
      true,
    ),

  submitCorrection: (body: CorrectionRequest) =>
    request<{ ok: boolean }>(
      '/api/v1/chat/correct',
      { method: 'POST', body: JSON.stringify(body) },
      true,
    ),

  submitFeedback: (messageId: string) =>
    request<{ ok: boolean }>(
      `/api/v1/chat/${messageId}/feedback`,
      { method: 'POST', body: JSON.stringify({ message_id: messageId }) },
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

  getAlerts: () => request<AnomalyAlertResponse>('/api/v1/chat/alerts', {}, true),

  // Bypasses request()'s JSON Content-Type -- the browser must set its own
  // multipart/form-data boundary, which it only does when Content-Type is
  // left unset on a FormData body.
  transcribeVoice: async (audioBlob: Blob): Promise<VoiceTranscribeResponse> => {
    const token = getRequiredJwt();
    const body = new FormData();
    body.append('file', audioBlob, 'clip.webm');
    const res = await fetch(`${BASE}/api/v1/voice/transcribe`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body,
    });
    if (res.status === 401) {
      resetToLogin();
      throw new AuthExpiredError();
    }
    if (!res.ok) {
      const errBody = await res.text().catch(() => '');
      throw new Error(`${res.status}: ${errBody}`);
    }
    return res.json() as Promise<VoiceTranscribeResponse>;
  },
};
