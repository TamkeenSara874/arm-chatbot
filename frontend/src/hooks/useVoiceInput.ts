import { useCallback, useRef, useState } from 'react';
import { api } from '../services/api';

interface UseVoiceInputResult {
  isRecording: boolean;
  isTranscribing: boolean;
  error: string | null;
  // Undefined when the browser has no mic/MediaRecorder support at all --
  // callers use this to hide the mic button entirely rather than show one
  // that always fails.
  isSupported: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
}

function getSupportedMimeType(): string | undefined {
  // Safari/iOS don't support audio/webm at all -- mp4 is its native
  // MediaRecorder output. Trying webm first matches what Chrome/Firefox
  // (the majority of desktop restaurant-owner traffic) produce natively.
  const candidates = ['audio/webm', 'audio/mp4', 'audio/ogg'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

/** Records a short voice clip and transcribes it via the backend's
 * Groq-Whisper-backed /voice/transcribe endpoint. onTranscribed receives the
 * resulting text -- the caller decides what to do with it (e.g. fill the
 * chat input) rather than this hook assuming auto-send, since a
 * misheard/garbled transcription is better shown to the user for review
 * than submitted blind. */
export function useVoiceInput(onTranscribed: (text: string) => void): UseVoiceInputResult {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const isSupported =
    typeof MediaRecorder !== 'undefined' &&
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia;

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = getSupportedMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        const blob = new Blob(chunksRef.current, { type: mimeType ?? 'audio/webm' });
        chunksRef.current = [];
        setIsTranscribing(true);
        api
          .transcribeVoice(blob)
          .then((res) => onTranscribed(res.text))
          .catch(() => setError('Could not transcribe audio. Please try again or type instead.'))
          .finally(() => setIsTranscribing(false));
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch {
      setError('Microphone access was denied or is unavailable.');
    }
  }, [onTranscribed]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  }, []);

  return { isRecording, isTranscribing, error, isSupported, startRecording, stopRecording };
}
