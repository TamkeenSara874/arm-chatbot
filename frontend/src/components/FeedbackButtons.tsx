import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { useState } from 'react';
import type { LocalMessage } from '../store/chatStore';

interface FeedbackButtonsProps {
  message: LocalMessage;
  onCorrect: () => void;
}

export function FeedbackButtons({ message, onCorrect }: FeedbackButtonsProps) {
  const [thumbsUp, setThumbsUp] = useState(false);

  if (!message.response) return null;

  return (
    <div className="mt-2 flex items-center gap-1">
      <button
        onClick={() => setThumbsUp((v) => !v)}
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
