import { AlertTriangle, Calendar, Star, X } from 'lucide-react';
import { useChatStore } from '../store/chatStore';
import type { EvidenceItem } from '../types/api';

const SENTIMENT_STYLES: Record<string, string> = {
  Positive: 'bg-green-50 text-green-700 ring-1 ring-green-200',
  Negative: 'bg-red-50 text-red-700 ring-1 ring-red-200',
  Mixed: 'bg-amber-50 text-amber-700 ring-1 ring-amber-200',
  Neutral: 'bg-gray-100 text-gray-600 ring-1 ring-gray-200',
};

const SOURCE_STYLES: Record<string, string> = {
  Google: 'bg-blue-50 text-blue-700',
  Yelp: 'bg-red-50 text-red-600',
  TripAdvisor: 'bg-green-50 text-green-700',
  'AIO Online': 'bg-aio-50 text-aio-600',
};

function Stars({ rating }: { rating: number }) {
  return (
    <span className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <Star
          key={n}
          size={11}
          className={n <= Math.round(rating) ? 'fill-amber-400 text-amber-400' : 'text-gray-200'}
        />
      ))}
      <span className="ml-1 text-xs font-medium text-gray-500">{rating.toFixed(1)}</span>
    </span>
  );
}

function EvidenceCard({
  item,
  index,
  matchPercent,
}: {
  item: EvidenceItem;
  index: number;
  matchPercent: number;
}) {
  const sentimentStyle = item.sentiment ? SENTIMENT_STYLES[item.sentiment] ?? SENTIMENT_STYLES.Neutral : SENTIMENT_STYLES.Neutral;
  const sourceStyle = item.source ? SOURCE_STYLES[item.source] ?? 'bg-gray-100 text-gray-600' : '';

  return (
    <div className="rounded-lg border border-gray-100 bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-start justify-between gap-2">
        <span className="text-xs font-semibold text-gray-400">#{index + 1}</span>
        <span className="text-xs font-medium text-aio-500">{matchPercent}% match</span>
      </div>

      <p className="mb-3 text-sm leading-relaxed text-gray-700">{item.snippet}</p>

      <div className="flex flex-wrap items-center gap-1.5">
        {item.rating != null && <Stars rating={item.rating} />}

        {item.sentiment && (
          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${sentimentStyle}`}>
            {item.sentiment}
          </span>
        )}

        {item.source && (
          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${sourceStyle}`}>
            {item.source}
          </span>
        )}

        {item.username && (
          <span className="text-xs text-gray-400">{item.username}</span>
        )}

        {item.date_inferred && (
          <span className="flex items-center gap-0.5 text-xs text-gray-400">
            <Calendar size={10} />
            date estimated
          </span>
        )}
      </div>

      {item.sentiment_conflict && (
        <div className="mt-2 flex items-start gap-1.5 rounded-md bg-amber-50 px-2 py-1.5">
          <AlertTriangle size={12} className="mt-0.5 shrink-0 text-amber-500" />
          <p className="text-xs text-amber-700">
            Star rating disagrees with review text. Text sentiment is used.
          </p>
        </div>
      )}
    </div>
  );
}

// Cross-encoder relevance scores are calibrated to be confidently near-0 or
// near-1 (trained as a direct question/passage classifier), not a smooth
// gradient -- a correctly-ranked-but-compressed evidence set (e.g. every raw
// score under 1%) would otherwise show a wall of "0% match" badges even
// though the ordering itself is meaningful. Min-max normalizing within this
// response's own evidence set surfaces that relative ordering: the best
// piece of evidence shown reads as ~100%, the weakest as ~0%, everything
// else scaled proportionally between.
function matchPercentages(evidence: EvidenceItem[]): number[] {
  if (evidence.length === 0) return [];
  const relevances = evidence.map((e) => e.relevance);
  const min = Math.min(...relevances);
  const max = Math.max(...relevances);
  const spread = max - min;
  if (spread < 1e-9) {
    // All equally (ir)relevant -- nothing to distinguish, show them as tied.
    return relevances.map(() => 100);
  }
  return relevances.map((r) => Math.round(((r - min) / spread) * 100));
}

export function EvidencePanel() {
  const { messages, selectedMessageId, setSelectedMessageId } = useChatStore();

  const selected = messages.find((m) => m.id === selectedMessageId);
  const evidence: EvidenceItem[] = selected?.response?.response.evidence ?? [];
  const percentages = matchPercentages(evidence);

  // The backend orders evidence by a composite score (relevance + recency +
  // rating) -- generation reasons over that order for good reason, since a
  // fresher or better-rated review can be more useful evidence than a
  // slightly-higher-relevance old one. But this panel's badge shows relevance
  // alone, so the two can disagree (a 47% card sitting above a 100% one) --
  // confusing for a human scanning cards top to bottom expecting the number
  // to match the order. Re-sort purely for display; the array powering
  // generation elsewhere is untouched.
  const sorted = evidence
    .map((item, originalIndex) => ({ item, percent: percentages[originalIndex] }))
    .sort((a, b) => b.percent - a.percent);

  return (
    <aside className="animate-slide-in fixed right-0 top-[64px] bottom-0 z-30 flex w-80 flex-col border-l border-gray-100 bg-gray-50">
      <div className="flex items-center justify-between border-b border-gray-100 bg-white px-4 py-3">
        <h2 className="text-sm font-semibold text-gray-800">
          Evidence
          <span className="ml-1.5 rounded-full bg-aio-100 px-2 py-0.5 text-xs font-medium text-aio-600">
            {evidence.length}
          </span>
        </h2>
        <button
          onClick={() => setSelectedMessageId(null)}
          className="rounded-md p-1 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
        >
          <X size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-2">
        {evidence.length === 0 ? (
          <p className="py-6 text-center text-sm text-gray-400">No evidence for this response.</p>
        ) : (
          sorted.map(({ item, percent }, i) => (
            <EvidenceCard key={i} item={item} index={i} matchPercent={percent} />
          ))
        )}
      </div>
    </aside>
  );
}
