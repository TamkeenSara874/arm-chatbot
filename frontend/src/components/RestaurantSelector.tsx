import { ChevronDown, Store } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useCreateSession, useRestaurants } from '../hooks/useChat';
import { api } from '../services/api';
import { useChatStore } from '../store/chatStore';

export function RestaurantSelector() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { data, isLoading, isError, refetch } = useRestaurants();
  const { mutate: createSession, isPending } = useCreateSession();
  const { restaurantId, setRestaurantId, newConversation } = useChatStore();

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  async function select(id: number) {
    setOpen(false);
    setRestaurantId(id);
    newConversation();
    await api.login(id);
    createSession(id);
  }

  const label = restaurantId != null ? `Restaurant #${restaurantId}` : 'Select restaurant';

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={isPending}
        className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:border-aio-400 hover:text-aio-500 disabled:opacity-50"
      >
        <Store size={15} className={isLoading ? 'text-gray-400 animate-pulse' : 'text-aio-500'} />
        <span>{label}</span>
        <ChevronDown size={14} className={`text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 min-w-[200px] rounded-lg border border-gray-100 bg-white py-1 shadow-lg animate-fade-in">
          {isLoading && (
            <p className="px-3 py-2 text-xs text-gray-400">Connecting to server...</p>
          )}
          {isError && (
            <div className="px-3 py-2 space-y-1.5">
              <p className="text-xs text-red-500">Server unavailable. Is the backend running?</p>
              <button
                onClick={() => { void refetch(); }}
                className="text-xs font-medium text-aio-500 hover:underline"
              >
                Retry
              </button>
            </div>
          )}
          {!isLoading && !isError && data?.restaurant_ids.length === 0 && (
            <p className="px-3 py-2 text-xs text-gray-400">No restaurants found. Ingest reviews first.</p>
          )}
          {data?.restaurant_ids.map((id) => (
            <button
              key={id}
              onClick={() => select(id)}
              className={`flex w-full items-center gap-2 px-3 py-2 text-sm transition hover:bg-aio-50 ${
                restaurantId === id ? 'bg-aio-50 font-medium text-aio-500' : 'text-gray-700'
              }`}
            >
              <Store size={13} />
              Restaurant #{id}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
