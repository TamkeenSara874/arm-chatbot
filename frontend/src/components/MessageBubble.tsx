import { BookOpen, Clock, Database, DollarSign, RotateCcw } from 'lucide-react';
import { useState } from 'react';
import { useSSE } from '../hooks/useSSE';
import { useChatStore, type LocalMessage } from '../store/chatStore';
import { CorrectionModal } from './CorrectionModal';
import { FeedbackButtons } from './FeedbackButtons';

const SOURCE_COLORS: Record<string, string> = {
  Google: 'bg-blue-50 text-blue-700',
  Yelp: 'bg-red-50 text-red-600',
  TripAdvisor: 'bg-green-50 text-green-700',
  OpenTable: 'bg-orange-50 text-orange-700',
};

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1 px-0.5">
      <span className="h-2 w-2 rounded-full bg-gray-300 animate-dot-1" />
      <span className="h-2 w-2 rounded-full bg-gray-300 animate-dot-2" />
      <span className="h-2 w-2 rounded-full bg-gray-300 animate-dot-3" />
    </div>
  );
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? 'bg-green-50 text-green-600' :
    pct >= 50 ? 'bg-amber-50 text-amber-600' :
                'bg-red-50 text-red-500';
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>
      {pct}% confident
    </span>
  );
}

function ModelBadge({ model }: { model: string }) {
  if (!model || model === 'none' || model === 'guardrail') return null;
  if (model === 'direct_query') {
    return (
      <span className="flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
        <Database size={10} />
        Direct query
      </span>
    );
  }
  return (
    <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
      {model}
    </span>
  );
}

function QueryMetrics({ latencyMs, costUsd }: { latencyMs?: number; costUsd?: number }) {
  if (!latencyMs && !costUsd) return null;
  return (
    <div className="flex items-center gap-3 px-1">
      {latencyMs != null && latencyMs > 0 && (
        <span className="flex items-center gap-1 text-xs text-gray-400">
          <Clock size={10} />
          {latencyMs < 1000 ? `${latencyMs}ms` : `${(latencyMs / 1000).toFixed(1)}s`}
        </span>
      )}
      {costUsd != null && costUsd > 0 && (
        <span className="flex items-center gap-1 text-xs text-gray-400">
          <DollarSign size={10} />
          ~${costUsd < 0.001 ? costUsd.toFixed(5) : costUsd.toFixed(4)}
        </span>
      )}
    </div>
  );
}

interface MessageBubbleProps {
  message: LocalMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const [correcting, setCorrecting] = useState(false);
  const { selectedMessageId, setSelectedMessageId, messages, sessionId, restaurantId, isStreaming } =
    useChatStore();
  const { regenerate } = useSSE();
  const isSelected = selectedMessageId === message.id;

  const messageIndex = messages.findIndex((m) => m.id === message.id);
  const precedingUserMessage =
    messageIndex >= 0
      ? [...messages.slice(0, messageIndex)].reverse().find((m) => m.role === 'user')
      : undefined;

  function handleRegenerate() {
    if (!precedingUserMessage || !sessionId || !restaurantId || isStreaming) return;
    void regenerate(message.id, {
      session_id: sessionId,
      restaurant_id: restaurantId,
      message: precedingUserMessage.content,
    });
  }

  if (message.role === 'user') {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[72%] rounded-2xl rounded-tr-sm bg-aio-500 px-4 py-3 shadow-sm">
          <p className="text-sm text-white leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  const response = message.response;
  const evidenceCount = response?.response.evidence.length ?? 0;
  const sourceBreakdown = response?.response.source_breakdown ?? {};
  const hasSourceBreakdown = Object.keys(sourceBreakdown).length > 0;

  return (
    <div className="flex justify-start animate-fade-in">
      <div className="max-w-[80%] space-y-1.5">
        <div
          className={`rounded-2xl rounded-tl-sm border bg-white px-4 py-3 shadow-sm transition ${
            isSelected ? 'border-aio-200 ring-2 ring-aio-100' : 'border-gray-100'
          }`}
        >
          {message.isStreaming && !message.content ? (
            <TypingIndicator />
          ) : message.error ? (
            <div className="space-y-1">
              <p className="text-sm text-red-500">{message.error}</p>
              {message.content && (
                <p className="text-sm text-gray-700 leading-relaxed">{message.content}</p>
              )}
            </div>
          ) : (
            <p
              className={`text-sm text-gray-800 leading-relaxed whitespace-pre-wrap ${
                message.isStreaming ? 'streaming-cursor' : ''
              }`}
            >
              {message.content}
            </p>
          )}

          {response?.response.caveats && (
            <p className="mt-2 text-xs text-amber-600 border-t border-amber-100 pt-2">
              {response.response.caveats}
            </p>
          )}
        </div>

        {response && (
          <div className="flex flex-col gap-1.5 px-1">
            <div className="flex flex-wrap items-center gap-2">
              <ConfidenceBadge value={response.response.confidence} />

              {response.cached && (
                <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-500">
                  cached
                </span>
              )}

              <ModelBadge model={response.model_used} />

              {evidenceCount > 0 && (
                <button
                  onClick={() => setSelectedMessageId(isSelected ? null : message.id)}
                  className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium transition ${
                    isSelected
                      ? 'bg-aio-100 text-aio-600'
                      : 'bg-gray-100 text-gray-500 hover:bg-aio-50 hover:text-aio-500'
                  }`}
                >
                  <BookOpen size={11} />
                  {evidenceCount} review{evidenceCount !== 1 ? 's' : ''}
                </button>
              )}

              <FeedbackButtons message={message} onCorrect={() => setCorrecting(true)} />

              {precedingUserMessage && (
                <button
                  onClick={handleRegenerate}
                  disabled={isStreaming}
                  title="Regenerate response"
                  className="flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-500 transition hover:bg-aio-50 hover:text-aio-500 disabled:opacity-40"
                >
                  <RotateCcw size={11} />
                  Regenerate
                </button>
              )}
            </div>

            {hasSourceBreakdown && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-gray-400">Sources:</span>
                {Object.entries(sourceBreakdown)
                  .sort(([, a], [, b]) => b - a)
                  .map(([source, count]) => (
                    <span
                      key={source}
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        SOURCE_COLORS[source] ?? 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {source} {count}
                    </span>
                  ))}
              </div>
            )}

            <QueryMetrics latencyMs={response?.latency_ms} costUsd={response?.cost_usd} />
          </div>
        )}
      </div>

      {correcting && (
        <CorrectionModal message={message} onClose={() => setCorrecting(false)} />
      )}
    </div>
  );
}
