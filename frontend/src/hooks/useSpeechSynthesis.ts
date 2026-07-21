import { useCallback, useEffect, useState } from 'react';

interface UseSpeechSynthesisResult {
  isSpeaking: boolean;
  isSupported: boolean;
  speak: (text: string) => void;
  stop: () => void;
}

/** Reads text aloud via the browser's built-in SpeechSynthesis -- free,
 * zero setup, no backend call. Quality/voice varies by OS, which is the
 * accepted tradeoff for a $0 V1 (see the voice-mode research this followed
 * from); a nicer server-side voice can replace just this hook later without
 * touching the dictation half. */
export function useSpeechSynthesis(): UseSpeechSynthesisResult {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const isSupported = typeof window !== 'undefined' && 'speechSynthesis' in window;

  // A message re-rendering (e.g. a streaming token arriving) must not orphan
  // an utterance that's still reading the previous text out loud.
  useEffect(() => {
    return () => {
      if (isSupported) window.speechSynthesis.cancel();
    };
  }, [isSupported]);

  const stop = useCallback(() => {
    if (!isSupported) return;
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, [isSupported]);

  const speak = useCallback(
    (text: string) => {
      if (!isSupported || !text.trim()) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      setIsSpeaking(true);
      window.speechSynthesis.speak(utterance);
    },
    [isSupported],
  );

  return { isSpeaking, isSupported, speak, stop };
}
