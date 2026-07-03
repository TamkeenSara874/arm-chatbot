import { FileBarChart, LogOut, MessageSquare, Plus } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ChatWindow } from './components/ChatWindow';
import { EvidencePanel } from './components/EvidencePanel';
import { ReportView } from './components/ReportView';
import { RestaurantSelector } from './components/RestaurantSelector';
import { useSessionHistory } from './hooks/useChat';
import { clearJwt } from './services/api';
import { useChatStore } from './store/chatStore';

function useHistoryRestore() {
  const sessionId = useChatStore((s) => s.sessionId);
  const hasMessages = useChatStore((s) => s.messages.length > 0);
  const loadHistory = useChatStore((s) => s.loadHistory);
  const setSessionId = useChatStore((s) => s.setSessionId);

  // Only fetch when we have a stored session but no in-memory messages (i.e. after reload)
  const { data, error } = useSessionHistory(!hasMessages ? sessionId : null);

  useEffect(() => {
    if (!data?.length) return;
    loadHistory(
      data.map((m) => ({
        id: m.message_id,
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }))
    );
  }, [data, loadHistory]);

  useEffect(() => {
    if (error && sessionId) {
      // Stored session no longer exists on the server — clear it
      setSessionId(null);
    }
  }, [error, sessionId, setSessionId]);
}

const EXAMPLE_QUESTIONS = [
  'What do customers praise most?',
  'What are the top complaints?',
  'How many positive reviews do I have?',
  'What should I improve based on feedback?',
];

function AioLogo() {
  return (
    <div className="flex items-center gap-2">
      <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-aio-500 shadow-sm">
        <MessageSquare size={14} className="text-white" />
      </div>
      <span className="text-base font-bold tracking-tight text-gray-800">
        ARM <span className="text-aio-500">Review Chatbot</span>
      </span>
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-6 py-16 text-center">
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

      <div className="w-full max-w-sm space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">
          Select a restaurant to begin
        </p>
        <RestaurantSelector />
      </div>

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

export default function App() {
  const [showReport, setShowReport] = useState(false);
  const { restaurantId, selectedMessageId, newConversation, setRestaurantId } = useChatStore();
  const showEvidencePanel = selectedMessageId !== null;

  useHistoryRestore();

  function handleLogout() {
    clearJwt();
    setRestaurantId(null);
  }

  return (
    <div className="flex h-screen flex-col bg-gray-50">
      <header className="flex shrink-0 items-center justify-between border-b border-gray-100 bg-white px-5 py-3 shadow-sm z-20">
        <div className="flex items-center gap-3">
          <AioLogo />
          <span className="hidden sm:block h-4 w-px bg-gray-200" />
          <nav className="hidden sm:flex items-center gap-1 text-xs text-gray-400">
            <span>AIO Platform</span>
            <span>/</span>
            <span className="font-medium text-aio-500">Review Chatbot</span>
          </nav>
        </div>

        <div className="flex items-center gap-2">
          <RestaurantSelector />

          {restaurantId != null && (
            <>
              <button
                onClick={newConversation}
                title="New conversation"
                className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-2 text-xs font-medium text-gray-600 transition hover:border-aio-300 hover:text-aio-500"
              >
                <Plus size={13} />
                New Chat
              </button>

              <button
                onClick={() => setShowReport(true)}
                className="flex items-center gap-1.5 rounded-lg bg-aio-500 px-3 py-2 text-xs font-medium text-white transition hover:bg-aio-600"
              >
                <FileBarChart size={13} />
                Report
              </button>

              <button
                onClick={handleLogout}
                title="Log out and switch restaurant"
                className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-2 text-xs font-medium text-gray-600 transition hover:border-red-300 hover:text-red-500"
              >
                <LogOut size={13} />
                Log Out
              </button>
            </>
          )}
        </div>
      </header>

      <main className="relative flex flex-1 overflow-hidden">
        <div
          className={`flex flex-1 flex-col overflow-hidden transition-all duration-200 ${
            showEvidencePanel ? 'mr-80' : ''
          }`}
        >
          {restaurantId == null ? <WelcomeScreen /> : <ChatWindow />}
        </div>

        {showEvidencePanel && <EvidencePanel />}
      </main>

      {showReport && <ReportView onClose={() => setShowReport(false)} />}
    </div>
  );
}
