import { useCallback, useRef } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useChatStore } from '../store/chatStore';
import { getApiKey } from '../services/api';
import type { ChatQueryRequest, ChatQueryResponse } from '../types/api';

const BASE = import.meta.env.VITE_API_URL ?? '';

function newId(): string {
  return crypto.randomUUID();
}

interface SseMessage {
  event: string;
  data: string;
  id?: string;
  retry?: number;
}

export function useSSE() {
  const abortRef = useRef<AbortController | null>(null);
  const assistantMsgIdRef = useRef<string | null>(null);

  // Shared by send() and regenerate() -- streams a query into the given
  // (already-created) assistant message id. Callers are responsible for
  // creating/removing message rows before calling this.
  const runStream = useCallback(async (assistantMsgId: string, request: ChatQueryRequest) => {
    if (abortRef.current) {
      abortRef.current.abort();
    }

    // Always read the latest actions from the store at call time — avoids stale closure
    // issues when send()/regenerate() is held across re-renders.
    const s = () => useChatStore.getState();

    const controller = new AbortController();
    abortRef.current = controller;
    assistantMsgIdRef.current = assistantMsgId;

    const handleOpen = async (res: Response): Promise<void> => {
      if (!res.ok) {
        const text = await res.text().catch(() => String(res.status));
        if (res.status === 503) {
          throw new Error('Service temporarily unavailable. Please try again in a moment.');
        }
        throw new Error(`Request failed (${res.status}): ${text}`);
      }
    };

    const handleMessage = (ev: SseMessage): void => {
      if (ev.event === 'token') {
        s().appendToken(ev.data);
      } else if (ev.event === 'done') {
        const response = JSON.parse(ev.data) as ChatQueryResponse;
        s().finalizeMessage(assistantMsgId, response);
      } else if (ev.event === 'error') {
        const errData = JSON.parse(ev.data) as { message?: string };
        s().setMessageError(
          assistantMsgId,
          errData.message ?? 'An error occurred. Please try again.'
        );
      }
    };

    const handleError = (err: unknown): void => {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      s().setMessageError(
        assistantMsgId,
        err instanceof Error ? err.message : 'Response interrupted. Please try again.'
      );
      throw err;
    };

    try {
      await fetchEventSource(`${BASE}/api/v1/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${getApiKey()}`,
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
        openWhenHidden: true,
        onopen: handleOpen,
        onmessage: handleMessage,
        onerror: handleError,
      });
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        const msg = err instanceof Error ? err.message : 'An unexpected error occurred.';
        s().setMessageError(assistantMsgId, msg);
      }
    }
  }, []);

  const send = useCallback(
    async (request: ChatQueryRequest) => {
      const s = () => useChatStore.getState();
      const userMsgId = newId();
      const assistantMsgId = newId();
      s().addUserMessage(userMsgId, request.message);
      s().startStreaming(assistantMsgId);
      await runStream(assistantMsgId, request);
    },
    [runStream]
  );

  // Re-asks the same question that produced `oldAssistantMsgId`, replacing
  // that response with a new one. Does not add a new user bubble -- the
  // original question is still shown above the regenerated answer.
  const regenerate = useCallback(
    async (oldAssistantMsgId: string, request: ChatQueryRequest) => {
      const s = () => useChatStore.getState();
      s().removeMessage(oldAssistantMsgId);
      const assistantMsgId = newId();
      s().startStreaming(assistantMsgId);
      await runStream(assistantMsgId, request);
    },
    [runStream]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    const id = assistantMsgIdRef.current;
    if (id) {
      useChatStore.getState().cancelStreaming(id);
      assistantMsgIdRef.current = null;
    }
  }, []);

  return { send, regenerate, cancel };
}
