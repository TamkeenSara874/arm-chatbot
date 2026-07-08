import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../services/api';
import { useChatStore } from '../store/chatStore';
import type { AnomalyAlertResponse, ChatMessage, CorrectionRequest, ReportRequest } from '../types/api';

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

// The backend caches its own result for 12h (src/core/anomaly.py), so
// polling faster than that never gets fresher data -- 30 minutes just means
// a tab left open across that 12h boundary picks up a newly-detected alert
// reasonably promptly, without hammering the endpoint for no reason.
export function useAnomalyAlert(restaurantId: number | null) {
  return useQuery<AnomalyAlertResponse>({
    queryKey: ['alerts', restaurantId],
    queryFn: () => api.getAlerts(),
    enabled: restaurantId != null,
    refetchInterval: 30 * 60 * 1000,
    staleTime: 5 * 60 * 1000,
  });
}
