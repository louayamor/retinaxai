'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { toast } from 'sonner';

export interface PatientWebSocketOptions {
  patientId: string;
  onPredictionComplete?: (data: PredictionEventData) => void;
  onXAIReady?: (data: XAIEventData) => void;
  onSeverityReady?: (data: SeverityEventData) => void;
  onGradCAMReady?: (data: GradCAMEventData) => void;
}

export interface PredictionEventData {
  prediction_id: string;
  patient_id: string;
  status: string;
  dr_grade: number;
  confidence: number;
  overall_severity: string;
  triggers_xai: boolean;
  timestamp: string;
  error?: string;
}

export interface XAIEventData {
  prediction_id: string;
  patient_id?: string;
  status: string;
  progress: number;
  message: string;
  explanation_id?: string;
  content?: string;
  summary?: string;
  details?: Record<string, unknown>;
  timestamp: string;
  error?: string;
}

export interface SeverityEventData extends XAIEventData {
  details: {
    risk_level: string;
    recommendations: string[];
    content?: string;
    summary?: string;
  };
}

export interface GradCAMEventData extends XAIEventData {
  details: {
    left_eye?: string;
    right_eye?: string;
    highlighted_regions?: {
      left_eye: string[];
      right_eye: string[];
    };
  };
}

interface WebSocketMessage {
  event: string;
  data: Record<string, unknown>;
}

export function usePatientWebSocket({
  patientId,
  onPredictionComplete,
  onXAIReady,
  onSeverityReady,
  onGradCAMReady,
}: PatientWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WebSocketMessage | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const patientIdRef = useRef(patientId);

  patientIdRef.current = patientId;

  const handleMessage = useCallback((event: string, data: Record<string, unknown>) => {
    const predictionData = data as unknown as PredictionEventData;
    const xaiData = data as unknown as XAIEventData;

    if (event === 'prediction.completed') {
      if (onPredictionComplete && predictionData.prediction_id) {
        onPredictionComplete(predictionData);
      }
      toast.success('Prediction Complete', {
        description: `DR Grade: ${predictionData.dr_grade}, Severity: ${predictionData.overall_severity}`,
      });
    } else if (event === 'prediction.failed') {
      toast.error('Prediction Failed', {
        description: predictionData.error || 'An unknown error occurred',
      });
    } else if (event === 'xai.explanation_ready') {
      if (onXAIReady && xaiData) {
        onXAIReady(xaiData);
      }
      toast.success('Explanation Ready', {
        description: xaiData.message || 'AI explanation has been generated',
      });
    } else if (event === 'xai.gradcam_ready') {
      if (onGradCAMReady && xaiData) {
        onGradCAMReady(xaiData as unknown as GradCAMEventData);
      }
      toast.info('GradCAM Analysis Ready', {
        description: 'View the detailed heatmap analysis',
      });
    } else if (event === 'xai.severity_ready') {
      if (onSeverityReady && xaiData) {
        onSeverityReady(xaiData as unknown as SeverityEventData);
      }
      toast.success('Risk Assessment Complete', {
        description: xaiData.message || 'Risk level has been determined',
      });
    } else if (event.startsWith('xai.') && xaiData.status === 'completed') {
      toast.success('XAI Processing Complete', {
        description: xaiData.message,
      });
    } else if (event.startsWith('xai.') && xaiData.status === 'failed') {
      toast.error('XAI Processing Failed', {
        description: xaiData.message || 'An error occurred during processing',
      });
    }

    setLastEvent({ event, data });
  }, [onPredictionComplete, onXAIReady, onSeverityReady, onGradCAMReady]);

  const connect = useCallback(() => {
    const WS_URL = (process.env.NEXT_PUBLIC_API_URL?.replace('http', 'ws') || 'ws://localhost:8000') + '/ws';
    
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    wsRef.current = new WebSocket(WS_URL);

    wsRef.current.onopen = () => {
      setConnected(true);
      console.log('[PatientWS] Connected for patient:', patientIdRef.current);

      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: `prediction:${patientIdRef.current}` }
      }));

      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'notifications' }
      }));
    };

    wsRef.current.onclose = () => {
      setConnected(false);
      console.log('[PatientWS] Disconnected, reconnecting in 3s...');
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    wsRef.current.onerror = (error) => {
      console.error('[PatientWS] Connection error:', error);
    };

    wsRef.current.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        console.log('[PatientWS] Message:', message.event, message.data);
        handleMessage(message.event, message.data as Record<string, unknown>);
      } catch {
        console.warn('[PatientWS] Invalid JSON:', event.data);
      }
    };
  }, [handleMessage]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const send = useCallback((event: string, data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event, data }));
    }
  }, []);

  return {
    connected,
    lastEvent,
    send,
    reconnect: connect,
  };
}
