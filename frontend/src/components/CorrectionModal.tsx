import { X } from 'lucide-react';
import { useState } from 'react';
import { useCorrect } from '../hooks/useChat';
import { useChatStore } from '../store/chatStore';
import type { LocalMessage } from '../store/chatStore';

interface CorrectionModalProps {
  message: LocalMessage;
  onClose: () => void;
}

export function CorrectionModal({ message, onClose }: CorrectionModalProps) {
  const [correction, setCorrection] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const { sessionId } = useChatStore();
  const { mutate: submitCorrection, isPending } = useCorrect();

  const originalResponse = message.response?.response.answer ?? message.content;
  const messageId = message.response?.message_id ?? message.id;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!correction.trim() || !sessionId) return;

    submitCorrection(
      {
        session_id: sessionId,
        message_id: messageId,
        corrected_response: correction.trim(),
      },
      {
        onSuccess: () => setSubmitted(true),
      }
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
      <div className="mx-4 w-full max-w-lg rounded-xl bg-white shadow-xl animate-fade-in">
        <div className="flex items-center justify-between border-b border-gray-100 px-5 py-4">
          <h2 className="text-sm font-semibold text-gray-800">Submit Correction</h2>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
          >
            <X size={16} />
          </button>
        </div>

        {submitted ? (
          <div className="p-5 text-center">
            <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-green-50">
              <span className="text-green-600 text-lg">✓</span>
            </div>
            <p className="text-sm font-medium text-gray-800">Thank you for the correction.</p>
            <p className="mt-1 text-xs text-gray-400">
              Your feedback helps improve future responses.
            </p>
            <button
              onClick={onClose}
              className="mt-4 rounded-lg bg-aio-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-aio-600"
            >
              Close
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="p-5 space-y-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">
                Original response
              </label>
              <div className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2.5 text-sm text-gray-600 leading-relaxed max-h-32 overflow-y-auto scrollbar-thin">
                {originalResponse}
              </div>
            </div>

            <div>
              <label
                htmlFor="correction"
                className="mb-1.5 block text-xs font-medium text-gray-700"
              >
                What should the correct answer be?
              </label>
              <textarea
                id="correction"
                value={correction}
                onChange={(e) => setCorrection(e.target.value)}
                rows={4}
                maxLength={4000}
                placeholder="Enter the correct response based on your restaurant's reviews..."
                className="w-full resize-none rounded-lg border border-gray-200 px-3 py-2.5 text-sm text-gray-800 placeholder-gray-400 outline-none transition focus:border-aio-400 focus:ring-2 focus:ring-aio-100"
              />
              <p className="mt-1 text-right text-xs text-gray-400">{correction.length}/4000</p>
            </div>

            <div className="flex gap-2 justify-end">
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-600 transition hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!correction.trim() || isPending}
                className="rounded-lg bg-aio-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-aio-600 disabled:opacity-50"
              >
                {isPending ? 'Submitting...' : 'Submit'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
