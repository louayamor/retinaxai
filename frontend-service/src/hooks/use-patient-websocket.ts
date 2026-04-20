'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { toast } from 'sonner';

export interface PatientWebSocketOptions {
  patientId: string;
  onPredictionComplete?: (data: PredictionEventData) => void;
  onXAIReady?: (data: XAIEventData) => void;
  onSeverityReady?: (data: SeverityEventData) => void;
  onGradCAMReady?: (data: GradCAMEventData) => void;
  onLogMessage?: (data: LogMessageData) => void;
}

export interface LogMessageData {
  prediction_id: string;
  patient_id: string;
  step: string;
  status: "info" | "success" | "warning" | "error";
  message: string;
  timestamp: string;
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
  onLogMessage,
}: PatientWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WebSocketMessage | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const patientIdRef = useRef(patientId);
  const isConnectingRef = useRef(false);

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
    } else if (event === 'prediction.log') {
      const logData = data as unknown as LogMessageData;
      if (onLogMessage && logData) {
        onLogMessage(logData);
      }
    } else if (event === 'training_stage') {
      const logData = data as unknown as LogMessageData;
      if (onLogMessage && logData) {
        onLogMessage({
          prediction_id: String(logData.prediction_id || ''),
          patient_id: String(logData.patient_id || patientIdRef.current),
          step: String(logData.step || 'training'),
          status: (logData.status as LogMessageData['status']) || 'info',
          message: String(logData.message || ''),
          timestamp: String((logData.timestamp as string) || new Date().toISOString()),
        });
      }
    }

    setLastEvent({ event, data });
  }, [onPredictionComplete, onXAIReady, onSeverityReady, onGradCAMReady, onLogMessage]);

  const connect = useCallback(() => {
    const WS_URL = (process.env.NEXT_PUBLIC_API_URL?.replace('http', 'ws') || 'ws://localhost:8000') + '/ws';
    
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING || isConnectingRef.current) {
      return;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    isConnectingRef.current = true;
    wsRef.current = new WebSocket(WS_URL);

    wsRef.current.onopen = () => {
      isConnectingRef.current = false;
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

      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'training:imaging' }
      }));

      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'training:clinical' }
      }));
    };

    wsRef.current.onclose = () => {
      isConnectingRef.current = false;
      if (wsRef.current) {
        setConnected(false);
        wsRef.current = null;
      }
    };
    
    wsRef.current.onerror = () => {
      isConnectingRef.current = false;
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
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      isConnectingRef.current = false;
    };
  }, []);

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
