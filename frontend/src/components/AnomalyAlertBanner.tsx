import { AlertTriangle, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useAnomalyAlert } from '../hooks/useChat';
import { useChatStore } from '../store/chatStore';

// Dismissal is intentionally in-memory only (not persisted to localStorage
// like the staleness caveat) -- this is a time-sensitive signal meant to be
// re-noticed on a fresh visit, not a repetitive disclaimer to suppress
// permanently. Resets whenever the restaurant changes.
export function AnomalyAlertBanner() {
  const restaurantId = useChatStore((s) => s.restaurantId);
  const { data } = useAnomalyAlert(restaurantId);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    setDismissed(false);
  }, [restaurantId]);

  if (!data?.detected || !data.message || dismissed) return null;

  return (
    <div className="flex shrink-0 items-start gap-2 border-b border-amber-200 bg-amber-50 px-5 py-2.5">
      <AlertTriangle size={15} className="mt-0.5 shrink-0 text-amber-500" />
      <p className="flex-1 text-xs font-medium text-amber-800">{data.message}</p>
      <button
        onClick={() => setDismissed(true)}
        title="Dismiss"
        className="shrink-0 rounded p-0.5 text-amber-500 transition hover:bg-amber-100 hover:text-amber-700"
      >
        <X size={14} />
      </button>
    </div>
  );
}
