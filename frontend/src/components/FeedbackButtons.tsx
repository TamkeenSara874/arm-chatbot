import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { useState } from 'react';
import { api } from '../services/api';
import type { LocalMessage } from '../store/chatStore';

interface FeedbackButtonsProps {
  message: LocalMessage;
  onCorrect: () => void;
}

export function FeedbackButtons({ message, onCorrect }: FeedbackButtonsProps) {
  const [thumbsUp, setThumbsUp] = useState(false);

  if (!message.response) return null;

  function handleThumbsUp() {
    // response.message_id is the *user* message id (same convention
    // CorrectionModal uses) -- message.id is only a client-generated id for
    // the assistant bubble and was never persisted as a real row id.
    const messageId = message.response?.message_id ?? message.id;
    setThumbsUp((v) => {
      const next = !v;
      if (next) {
        // Fire-and-forget: local state is the source of truth for the UI,
        // this just persists the signal server-side. There's no un-feedback
        // endpoint, so only submit when turning it on.
        void api.submitFeedback(messageId).catch(() => {});
      }
      return next;
    });
  }

  return (
    <div className="mt-2 flex items-center gap-1">
      <button
        onClick={handleThumbsUp}
        title="This was helpful"
        className={`rounded-md p-1.5 text-xs transition ${
          thumbsUp
            ? 'bg-green-50 text-green-600'
            : 'text-gray-300 hover:bg-gray-100 hover:text-gray-500'
        }`}
      >
        <ThumbsUp size={13} />
      </button>
      <button
        onClick={onCorrect}
        title="This was wrong — submit a correction"
        className="rounded-md p-1.5 text-gray-300 transition hover:bg-red-50 hover:text-red-400"
      >
        <ThumbsDown size={13} />
      </button>
    </div>
  );
}
