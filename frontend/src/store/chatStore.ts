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
    set((s) => ({
      isStreaming: false,
      messages: s.messages.map((m) =>
        m.id === id
          ? { ...m, content: response.response.answer, response, isStreaming: false }
          : m
      ),
    })),

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

  loadHistory: (history) => set({ messages: history }),

  setSelectedMessageId: (id) => set({ selectedMessageId: id }),

  newConversation: () => {
    saveSession(get().restaurantId, null);
    set({ sessionId: null, messages: [], isStreaming: false, selectedMessageId: null });
  },
}));
