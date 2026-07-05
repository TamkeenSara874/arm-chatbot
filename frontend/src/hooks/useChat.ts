import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../services/api';
import { useChatStore } from '../store/chatStore';
import type { ChatMessage, CorrectionRequest, ReportRequest } from '../types/api';

export function useCreateSession() {
  const setSessionId = useChatStore((s) => s.setSessionId);
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (restaurantId: number) => api.createSession(restaurantId),
    onSuccess: (data) => {
      setSessionId(data.session_id);
      void queryClient.invalidateQueries({ queryKey: ['restaurants'] });
    },
  });
}

export function useSessionHistory(sessionId: string | null) {
  return useQuery<ChatMessage[]>({
    queryKey: ['history', sessionId],
    queryFn: () => api.getHistory(sessionId!),
    enabled: !!sessionId,
    staleTime: Infinity,
  });
}

export function useCorrect() {
  return useMutation({
    mutationFn: (body: CorrectionRequest) => api.submitCorrection(body),
  });
}

export function useReport() {
  return useMutation({
    mutationFn: (body: ReportRequest) => api.generateReport(body),
  });
}
