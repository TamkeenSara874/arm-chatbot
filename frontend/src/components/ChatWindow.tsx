import { Mic, MessageSquare, Send, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useCreateSession } from '../hooks/useChat';
import { useSSE } from '../hooks/useSSE';
import { useVoiceInput } from '../hooks/useVoiceInput';
import { useChatStore } from '../store/chatStore';
import { MessageBubble } from './MessageBubble';

function EmptyState({
  restaurantId,
  onSelect,
}: {
  restaurantId: number;
  onSelect: (q: string) => void;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-aio-50">
        <MessageSquare size={24} className="text-aio-500" />
      </div>
      <div>
        <p className="text-sm font-semibold text-gray-700">Restaurant #{restaurantId}</p>
        <p className="mt-1 max-w-xs text-xs text-gray-400">
          Ask anything about your customer reviews. For example:
        </p>
      </div>
      <div className="flex flex-col gap-1.5 text-left">
        {[
          'What do customers say about our food?',
          'What are the most common complaints?',
          'How many positive reviews do we have?',
          'What should we improve based on feedback?',
        ].map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="rounded-lg border border-gray-100 bg-white px-3 py-2 text-left text-xs text-gray-500 shadow-sm transition hover:border-aio-200 hover:bg-aio-50 hover:text-aio-600"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function NoSessionState() {
  const { restaurantId } = useChatStore();
  const { mutate: createSession, isPending, isError } = useCreateSession();

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 py-16">
      {isError && (
        <p className="text-xs text-red-500">Could not start session. Please refresh.</p>
      )}
      <p className="text-sm text-gray-400">
        {isPending ? 'Starting session...' : 'Ready to start.'}
      </p>
      {!isPending && restaurantId != null && (
        <button
          onClick={() => createSession(restaurantId)}
          className="rounded-lg bg-aio-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-aio-600"
        >
          Start Conversation
        </button>
      )}
    </div>
  );
}

export function ChatWindow() {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { send, cancel } = useSSE();
  const { restaurantId, sessionId, messages, isStreaming } = useChatStore();
  // Fills the input with the transcription rather than auto-sending -- a
  // misheard/garbled clip is better shown for review than submitted blind.
  const voice = useVoiceInput((text) => setInput((prev) => (prev ? `${prev} ${text}` : text)));

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cancels any in-flight stream when the restaurant changes (cleanup runs
  // before the effect re-fires) and on unmount. Without this, switching
  // restaurants mid-stream left the old fetchEventSource request running in
  // the background with no client-side way to stop it.
  useEffect(() => {
    return () => cancel();
  }, [restaurantId, cancel]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isStreaming || !sessionId || !restaurantId) return;
    setInput('');
    void send({ session_id: sessionId, restaurant_id: restaurantId, message: text });
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-4 space-y-4">
        {!sessionId ? (
          <NoSessionState />
        ) : messages.length === 0 ? (
          <EmptyState
            restaurantId={restaurantId!}
            onSelect={(q) => {
              void send({ session_id: sessionId!, restaurant_id: restaurantId!, message: q });
            }}
          />
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      <div className="shrink-0 border-t border-gray-100 bg-white px-4 py-3">
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              !sessionId
                ? 'Select a restaurant to begin...'
                : 'Ask about your reviews...'
            }
            disabled={!sessionId || isStreaming}
            rows={1}
            maxLength={2000}
            className="flex-1 resize-none rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-800 placeholder-gray-400 outline-none transition focus:border-aio-400 focus:bg-white focus:ring-2 focus:ring-aio-100 disabled:opacity-50"
            style={{ maxHeight: '120px', overflowY: 'auto' }}
          />
          {voice.isSupported && (
            <button
              type="button"
              onClick={voice.isRecording ? voice.stopRecording : () => void voice.startRecording()}
              disabled={!sessionId || isStreaming || voice.isTranscribing}
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl shadow-sm transition disabled:opacity-40 ${
                voice.isRecording
                  ? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
              }`}
              title={voice.isRecording ? 'Stop recording' : 'Dictate your question'}
            >
              {voice.isRecording ? <Square size={14} /> : <Mic size={16} />}
            </button>
          )}
          {isStreaming ? (
            <button
              type="button"
              onClick={cancel}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gray-200 text-gray-500 transition hover:bg-gray-300"
              title="Stop"
            >
              <span className="h-3 w-3 rounded-sm bg-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() || !sessionId}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-aio-500 text-white shadow-sm transition hover:bg-aio-600 disabled:opacity-40"
              title="Send"
            >
              <Send size={16} />
            </button>
          )}
        </form>
        <div className="mt-1.5 flex items-center justify-between text-xs">
          <p className="text-red-500">{voice.error ?? (voice.isTranscribing ? 'Transcribing...' : '')}</p>
          <p className="text-gray-400">{input.length}/2000</p>
        </div>
      </div>
    </div>
  );
}
