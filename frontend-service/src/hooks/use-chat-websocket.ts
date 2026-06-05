'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { AnalyticsChartSpec, ChatMessage } from '@/lib/api';

const WS_BASE = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL.replace(/^http/, 'ws')
  : 'ws://localhost:8000';

export interface ThinkingState {
  stage: string;
  message: string;
}

export type ChatWsStatus = 'idle' | 'connecting' | 'sending' | 'error';

interface FinalPayload {
  type: 'final';
  summary: string;
  chart: AnalyticsChartSpec | null;
  sources: Array<{ artifact_id: string; snippet: string }>;
  error: string | null;
}

type WsMessage =
  | FinalPayload
  | { type: 'thinking'; stage: string; message: string }
  | { type: 'error'; message: string };

export function useChatWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const resolveRef = useRef<((value: FinalPayload) => void) | null>(null);
  const rejectRef = useRef<((reason: unknown) => void) | null>(null);

  const [status, setStatus] = useState<ChatWsStatus>('idle');
  const [thinking, setThinking] = useState<ThinkingState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('idle');
    setThinking(null);
    setError(null);
    resolveRef.current = null;
    rejectRef.current = null;
  }, []);

  const send = useCallback(
    (question: string, messages: ChatMessage[], top_k = 5): Promise<FinalPayload> => {
      disconnect();

      return new Promise((resolve, reject) => {
        resolveRef.current = resolve;
        rejectRef.current = reject;

        setThinking(null);
        setError(null);
        setStatus('connecting');

        const ws = new WebSocket(`${WS_BASE}/ws/chat`);
        wsRef.current = ws;

        ws.onopen = () => {
          setStatus('sending');
          ws.send(
            JSON.stringify({
              type: 'chat',
              question,
              messages: messages.map((m) => ({ role: m.role, content: m.content })),
              top_k,
            }),
          );
        };

        ws.onmessage = (event) => {
          try {
            const data: WsMessage = JSON.parse(event.data);

            if (data.type === 'thinking') {
              setThinking({ stage: data.stage, message: data.message });
              return;
            }

            if (data.type === 'final') {
              setStatus('idle');
              setThinking(null);
              ws.close();
              resolveRef.current = null;
              rejectRef.current = null;
              resolve(data);
              return;
            }

            if (data.type === 'error') {
              setStatus('error');
              setThinking(null);
              setError(data.message);
              ws.close();
              rejectRef.current = null;
              reject(new Error(data.message));
              return;
            }
          } catch {
            console.warn('[ChatWS] Invalid message:', event.data);
          }
        };

        ws.onerror = () => {
          setStatus('error');
          setError('WebSocket connection failed');
          reject(new Error('WebSocket connection failed'));
        };
      });
    },
    [disconnect],
  );

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return { status, thinking, error, send, disconnect };
}
