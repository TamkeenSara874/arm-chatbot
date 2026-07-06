import { create } from 'zustand';
import type { ChatQueryResponse } from '../types/api';

export interface LocalMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  response?: ChatQueryResponse;
  isStreaming?: boolean;
  error?: string;
}

const STORAGE_KEY = 'arm-chatbot-session';

function loadPersistedSession(): { restaurantId: number | null; sessionId: string | null } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { restaurantId: null, sessionId: null };
    return JSON.parse(raw) as { restaurantId: number | null; sessionId: string | null };
  } catch {
    return { restaurantId: null, sessionId: null };
  }
}

function saveSession(restaurantId: number | null, sessionId: string | null): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ restaurantId, sessionId }));
  } catch { /* ignore quota errors */ }
}

// The backend recomputes this caveat fresh whenever most retrieved evidence
// predates DATA_STALENESS_DAYS -- true of nearly every query against a fixed,
// aging dataset. A per-session suppression (show once per conversation) still
// re-shows it every time a new conversation starts, which is exactly what
// prompted this: persist "seen" per restaurant in localStorage instead, so it
// surfaces once total until the dataset genuinely changes or storage is cleared.
const STALENESS_CAVEAT_PREFIX = 'Note: the most relevant reviews found are from over a year ago';
const STALENESS_SEEN_KEY_PREFIX = 'arm-chatbot-staleness-seen-';

function hasSeenStalenessCaveat(restaurantId: number | null): boolean {
  if (restaurantId == null) return false;
  try {
    return localStorage.getItem(`${STALENESS_SEEN_KEY_PREFIX}${restaurantId}`) === '1';
  } catch {
    return false;
  }
}

function markStalenessCaveatSeen(restaurantId: number | null): void {
  if (restaurantId == null) return;
  try {
    localStorage.setItem(`${STALENESS_SEEN_KEY_PREFIX}${restaurantId}`, '1');
  } catch { /* ignore quota errors */ }
}

// Strips the staleness caveat from a response if it's already been shown
// once for this restaurant; otherwise marks it seen and lets it through this
// one time. Other caveat types (e.g. a validation-failure warning) are left
// untouched -- only this specific, known-noisy caveat is ever suppressed.
function suppressRepeatStalenessCaveat(
  response: ChatQueryResponse,
  restaurantId: number | null
): ChatQueryResponse {
  const caveat = response.response.caveats;
  if (!caveat?.startsWith(STALENESS_CAVEAT_PREFIX)) return response;
  if (hasSeenStalenessCaveat(restaurantId)) {
    return { ...response, response: { ...response.response, caveats: null } };
  }
  markStalenessCaveatSeen(restaurantId);
  return response;
}

interface ChatStore {
  restaurantId: number | null;
  sessionId: string | null;
  messages: LocalMessage[];
  isStreaming: boolean;
  selectedMessageId: string | null;

  setRestaurantId: (id: number | null) => void;
  setSessionId: (id: string | null) => void;
  addUserMessage: (id: string, content: string) => void;
  startStreaming: (id: string) => void;
  appendToken: (id: string, token: string) => void;
  finalizeMessage: (id: string, response: ChatQueryResponse) => void;
  cancelStreaming: (id: string) => void;
  setMessageError: (id: string, error: string) => void;
  removeMessage: (id: string) => void;
  loadHistory: (history: LocalMessage[]) => void;
  setSelectedMessageId: (id: string | null) => void;
  newConversation: () => void;
}

const { restaurantId: _r, sessionId: _s } = loadPersistedSession();

export const useChatStore = create<ChatStore>((set, get) => ({
  restaurantId: _r,
  sessionId: _s,
  messages: [],
  isStreaming: false,
  selectedMessageId: null,

  setRestaurantId: (id) => {
    saveSession(id, null);
    set({ restaurantId: id, sessionId: null, messages: [], selectedMessageId: null });
  },

  setSessionId: (id) => {
    saveSession(get().restaurantId, id);
    set({ sessionId: id });
  },

  addUserMessage: (id, content) =>
    set((s) => ({ messages: [...s.messages, { id, role: 'user', content }] })),

  startStreaming: (id) =>
    set((s) => ({
      isStreaming: true,
      messages: [...s.messages, { id, role: 'assistant', content: '', isStreaming: true }],
    })),

  // Matches by the specific message id (not the isStreaming flag) so tokens
  // from a stale/orphaned stream -- e.g. one left running after a restaurant
  // switch -- can never land on a different message than the one they belong to.
  appendToken: (id, token) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      ),
    })),

  finalizeMessage: (id, response) =>
    set((s) => {
      const deduped = suppressRepeatStalenessCaveat(response, s.restaurantId);
      return {
        isStreaming: false,
        messages: s.messages.map((m) =>
          m.id === id
            ? { ...m, content: deduped.response.answer, response: deduped, isStreaming: false }
            : m
        ),
      };
    }),

  cancelStreaming: (id) =>
    set((s) => {
      const target = s.messages.find((m) => m.id === id);
      if (!target) return { isStreaming: false };
      const hasContent = Boolean(target.content?.trim());
      return {
        isStreaming: false,
        messages: hasContent
          ? s.messages.map((m) => (m.id === id ? { ...m, isStreaming: false } : m))
          : s.messages.filter((m) => m.id !== id),
      };
    }),

  setMessageError: (id, error) =>
    set((s) => ({
      isStreaming: false,
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, error, isStreaming: false, content: m.content || '' } : m
      ),
    })),

  removeMessage: (id) =>
    set((s) => ({ messages: s.messages.filter((m) => m.id !== id) })),

  loadHistory: (history) =>
    set((s) => ({
      messages: history.map((m) =>
        m.response
          ? { ...m, response: suppressRepeatStalenessCaveat(m.response, s.restaurantId) }
          : m
      ),
    })),

  setSelectedMessageId: (id) => set({ selectedMessageId: id }),

  newConversation: () => {
    saveSession(get().restaurantId, null);
    set({ sessionId: null, messages: [], isStreaming: false, selectedMessageId: null });
  },
}));
