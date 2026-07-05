import { MessageSquare } from 'lucide-react';
import { useState } from 'react';
import { useCreateSession } from '../hooks/useChat';
import { api } from '../services/api';
import { useChatStore } from '../store/chatStore';

const EXAMPLE_QUESTIONS = [
  'What do customers praise most?',
  'What are the top complaints?',
  'How many positive reviews do I have?',
  'What should I improve based on feedback?',
];

export function LoginPage() {
  const [restaurantIdInput, setRestaurantIdInput] = useState('');
  const [restaurantKey, setRestaurantKey] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isPending, setIsPending] = useState(false);

  const { mutate: createSession } = useCreateSession();
  const { setRestaurantId, newConversation } = useChatStore();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = Number(restaurantIdInput);
    if (!Number.isInteger(id) || id < 1) {
      setError('Enter a valid restaurant ID');
      return;
    }
    setError(null);
    setIsPending(true);
    try {
      await api.login(id, restaurantKey);
      setRestaurantId(id);
      newConversation();
      createSession(id);
    } catch {
      setError('Invalid restaurant ID or access key');
    } finally {
      setIsPending(false);
    }
  }

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-10 bg-gray-50 px-6 py-16 text-center">
      <div className="space-y-3">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-aio-400 to-aio-600 shadow-lg">
          <MessageSquare size={26} className="text-white" />
        </div>
        <h1 className="text-2xl font-bold text-gray-900">ARM Review Chatbot</h1>
        <p className="max-w-md text-sm text-gray-500 leading-relaxed">
          Ask plain-English questions about your customer reviews and get instant,
          evidence-backed answers from real feedback.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="w-full max-w-sm space-y-3 rounded-xl border border-gray-100 bg-white p-6 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">
          Log in to your restaurant
        </p>
        <input
          type="number"
          min={1}
          placeholder="Restaurant ID"
          value={restaurantIdInput}
          onChange={(e) => setRestaurantIdInput(e.target.value)}
          className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm shadow-sm focus:border-aio-400 focus:outline-none"
        />
        <input
          type="password"
          placeholder="Access key"
          value={restaurantKey}
          onChange={(e) => setRestaurantKey(e.target.value)}
          className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm shadow-sm focus:border-aio-400 focus:outline-none"
        />
        <button
          type="submit"
          disabled={isPending}
          className="w-full rounded-lg bg-aio-500 px-3 py-2 text-sm font-medium text-white transition hover:bg-aio-600 disabled:opacity-50"
        >
          {isPending ? 'Logging in...' : 'Log in'}
        </button>
        {error && <p className="text-xs text-red-500">{error}</p>}
      </form>

      <div className="w-full max-w-lg space-y-2">
        <p className="text-xs text-gray-400">Try asking</p>
        <div className="flex flex-wrap justify-center gap-2">
          {EXAMPLE_QUESTIONS.map((q) => (
            <span
              key={q}
              className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-500 shadow-sm"
            >
              {q}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
