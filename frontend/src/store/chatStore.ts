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
  streamingToken: string;
  isStreaming: boolean;
  selectedMessageId: string | null;

  setRestaurantId: (id: number | null) => void;
  setSessionId: (id: string | null) => void;
  addUserMessage: (id: string, content: string) => void;
  startStreaming: (id: string) => void;
  appendToken: (token: string) => void;
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
  streamingToken: '',
  isStreaming: false,
  selectedMessageId: null,

  setRestaurantId: (id) => {
    saveSession(id, null);
    set({ restaurantId: id, sessionId: null, messages: [], streamingToken: '', selectedMessageId: null });
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
      streamingToken: '',
      messages: [...s.messages, { id, role: 'assistant', content: '', isStreaming: true }],
    })),

  appendToken: (token) =>
    set((s) => {
      const next = s.streamingToken + token;
      return {
        streamingToken: next,
        messages: s.messages.map((m) =>
          m.isStreaming ? { ...m, content: next } : m
        ),
      };
    }),

  finalizeMessage: (id, response) =>
    set((s) => ({
      isStreaming: false,
      streamingToken: '',
      messages: s.messages.map((m) =>
        m.id === id
          ? { ...m, content: response.response.answer, response, isStreaming: false }
          : m
      ),
    })),

  cancelStreaming: (id) =>
    set((s) => {
      const target = s.messages.find((m) => m.id === id);
      if (!target) return { isStreaming: false, streamingToken: '' };
      const hasContent = Boolean(target.content?.trim());
      return {
        isStreaming: false,
        streamingToken: '',
        messages: hasContent
          ? s.messages.map((m) => (m.id === id ? { ...m, isStreaming: false } : m))
          : s.messages.filter((m) => m.id !== id),
      };
    }),

  setMessageError: (id, error) =>
    set((s) => ({
      isStreaming: false,
      streamingToken: '',
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
    set({ sessionId: null, messages: [], streamingToken: '', isStreaming: false, selectedMessageId: null });
  },
}));
