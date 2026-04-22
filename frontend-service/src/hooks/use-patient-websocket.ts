'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { toast } from 'sonner';

export interface PatientWebSocketOptions {
  patientId: string;
  onPredictionComplete?: (data: PredictionEventData) => void;
  onPredictionFailed?: (data: PredictionEventData) => void;
  onBiomarkerUpdate?: (data: BiomarkerEventData) => void;
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

export interface BiomarkerEventData {
  prediction_id: string;
  patient_id: string;
  eye_side: 'left' | 'right' | string;
  status: string;
  progress: number;
  message: string;
  biomarkers?: Record<string, unknown>;
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
  onPredictionFailed,
  onBiomarkerUpdate,
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
  const pendingMessagesRef = useRef<Array<{ event: string; data: Record<string, unknown> }>>([]);

  patientIdRef.current = patientId;

  const handleMessage = useCallback((event: string, data: Record<string, unknown>) => {
    const predictionData = data as unknown as PredictionEventData;
    const xaiData = data as unknown as XAIEventData;
    const biomarkerData = data as unknown as BiomarkerEventData;

    console.log('[PatientWS] Incoming event:', event, data);

    if (event === 'prediction.completed') {
      console.log('[PatientWS] prediction.completed received', predictionData);
      if (onPredictionComplete && predictionData.prediction_id) {
        onPredictionComplete(predictionData);
      }
      toast.success('Prediction Complete', {
        description: `DR Grade: ${predictionData.dr_grade}, Severity: ${predictionData.overall_severity}`,
      });
    } else if (event === 'prediction.failed') {
      console.log('[PatientWS] prediction.failed received', predictionData);
      if (onPredictionFailed && predictionData.prediction_id) {
        onPredictionFailed(predictionData);
      }
      toast.error('Prediction Failed', {
        description: predictionData.error || 'An unknown error occurred',
      });
    } else if (event === 'xai.explanation_ready') {
      console.log('[PatientWS] xai.explanation_ready received', xaiData);
      if (onXAIReady && xaiData) {
        onXAIReady(xaiData);
      }
      toast.success('Explanation Ready', {
        description: xaiData.message || 'AI explanation has been generated',
      });
    } else if (event === 'xai.gradcam_ready') {
      console.log('[PatientWS] xai.gradcam_ready received', xaiData);
      if (onGradCAMReady && xaiData) {
        onGradCAMReady(xaiData as unknown as GradCAMEventData);
      }
      toast.info('GradCAM Analysis Ready', {
        description: 'View the detailed heatmap analysis',
      });
    } else if (event === 'xai.severity_ready') {
      console.log('[PatientWS] xai.severity_ready received', xaiData);
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
      console.log('[PatientWS] prediction.log received', data);
      const logData = data as unknown as LogMessageData;
      if (onLogMessage && logData) {
        onLogMessage(logData);
      }
    } else if (event.startsWith('biomarker.')) {
      console.log('[PatientWS] biomarker event received', biomarkerData);
      if (onBiomarkerUpdate && biomarkerData) {
        onBiomarkerUpdate(biomarkerData);
      }
      if (biomarkerData.status === 'completed') {
        toast.success(`${String(biomarkerData.eye_side).toUpperCase()} Biomarkers Ready`, {
          description: biomarkerData.message,
        });
      } else if (biomarkerData.status === 'failed') {
        toast.error('Biomarker Extraction Failed', {
          description: biomarkerData.message || biomarkerData.error || 'An error occurred',
        });
      }
    } else if (event === 'training_stage') {
      console.log('[PatientWS] training_stage received', data);
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
    } else if (event === 'subscribed' || event === 'unsubscribed') {
      console.log(`[PatientWS] ${event} acknowledgement`, data);
    } else if (event === 'connected') {
      console.log('[PatientWS] transport connected', data);
    } else if (event === 'error') {
      console.log('[PatientWS] transport error event', data);
    } else {
      console.log('[PatientWS] unhandled event received', event, data);
    }

    setLastEvent({ event, data });
  }, [onPredictionComplete, onPredictionFailed, onBiomarkerUpdate, onXAIReady, onSeverityReady, onGradCAMReady, onLogMessage]);

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
    console.log('[PatientWS] connecting', { patientId: patientIdRef.current, url: WS_URL });
    wsRef.current = new WebSocket(WS_URL);

    wsRef.current.onopen = () => {
      isConnectingRef.current = false;
      setConnected(true);
      console.log('[PatientWS] Connected for patient:', patientIdRef.current);

      const predictionRoomMessage = JSON.stringify({
        event: 'subscribe',
        data: { room: `prediction:${patientIdRef.current}` }
      });
      console.log('[PatientWS] sending subscribe', predictionRoomMessage);
      wsRef.current?.send(predictionRoomMessage);

      while (pendingMessagesRef.current.length > 0) {
        const next = pendingMessagesRef.current.shift();
        if (!next) continue;
        const payload = JSON.stringify(next);
        console.log('[PatientWS] flushing queued message', next);
        wsRef.current?.send(payload);
      }

      const notificationsMessage = JSON.stringify({
        event: 'subscribe',
        data: { room: 'notifications' }
      });
      console.log('[PatientWS] sending subscribe', notificationsMessage);
      wsRef.current?.send(notificationsMessage);

      const trainingImagingMessage = JSON.stringify({
        event: 'subscribe',
        data: { room: 'training:imaging' }
      });
      console.log('[PatientWS] sending subscribe', trainingImagingMessage);
      wsRef.current?.send(trainingImagingMessage);

      const trainingClinicalMessage = JSON.stringify({
        event: 'subscribe',
        data: { room: 'training:clinical' }
      });
      console.log('[PatientWS] sending subscribe', trainingClinicalMessage);
      wsRef.current?.send(trainingClinicalMessage);
    };

    wsRef.current.onclose = () => {
      isConnectingRef.current = false;
      console.log('[PatientWS] connection closed', { patientId: patientIdRef.current });
      pendingMessagesRef.current = [];
      if (wsRef.current) {
        setConnected(false);
        wsRef.current = null;
      }
    };
    
    wsRef.current.onerror = () => {
      isConnectingRef.current = false;
      console.log('[PatientWS] connection error flag set', { patientId: patientIdRef.current });
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
        console.log('[PatientWS] raw non-JSON message', event.data);
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
      const payload = JSON.stringify({ event, data });
      console.log('[PatientWS] sending message', { event, data });
      wsRef.current.send(payload);
    } else {
      console.log('[PatientWS] queueing message until socket opens', { event, data, readyState: wsRef.current?.readyState });
      pendingMessagesRef.current.push({ event, data });
    }
  }, []);

  return {
    connected,
    lastEvent,
    send,
    reconnect: connect,
  };
}
