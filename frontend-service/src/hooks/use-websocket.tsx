'use client';

import { createContext, useContext, ReactNode, useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = (process.env.NEXT_PUBLIC_API_URL?.replace('http', 'ws') || 'ws://localhost:8000') + '/ws';

interface WebSocketContextValue {
  connected: boolean;
  subscribe: (event: string, callback: (data: unknown) => void) => () => void;
  emit: (event: string, data: unknown) => void;
}

const SocketContext = createContext<WebSocketContextValue | null>(null);

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const listenersRef = useRef<Map<string, Set<(data: unknown) => void>>>(new Map());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 15;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    if (reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
      console.log('[WS] Max reconnect attempts reached, giving up');
      return;
    }
    
    wsRef.current = new WebSocket(WS_URL);
    
    wsRef.current.onopen = () => {
      setConnected(true);
      reconnectAttemptRef.current = 0;
      console.log('[WS] Connected');
      // Subscribe to training events
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'training:imaging' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'training:clinical' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'llmops' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'notifications' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'xai:prediction' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'xai:gradcam' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'xai:severity' }
      }));
      wsRef.current?.send(JSON.stringify({
        event: 'subscribe',
        data: { room: 'mlops.monitor' }
      }));
    };
    
    wsRef.current.onclose = () => {
      setConnected(false);
      reconnectAttemptRef.current += 1;
      const backoff = Math.min(2 ** reconnectAttemptRef.current, 30) * 1000;
      console.log(`[WS] Disconnected, reconnecting in ${backoff / 1000}s (attempt ${reconnectAttemptRef.current}/${MAX_RECONNECT_ATTEMPTS})...`);
      reconnectTimeoutRef.current = setTimeout(connect, backoff);
    };
    
    wsRef.current.onerror = () => {
      console.log('[WS] Connection error');
      wsRef.current?.close();
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('[WS] Message received:', message);
        const data = message.data || message;
        
        const callbacks = listenersRef.current.get(message.event);
        if (callbacks) {
          callbacks.forEach(cb => cb(data));
        }
        const allCallbacks = listenersRef.current.get('*');
        if (allCallbacks) {
          allCallbacks.forEach(cb => cb(message));
        }
      } catch {
        console.warn('[WS] Invalid JSON:', event.data);
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  const subscribe = useCallback((event: string, callback: (data: unknown) => void) => {
    if (!listenersRef.current.has(event)) {
      listenersRef.current.set(event, new Set());
    }
    listenersRef.current.get(event)!.add(callback);
    
    return () => {
      listenersRef.current.get(event)?.delete(callback);
    };
  }, []);

  const emit = useCallback((event: string, data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event, data }));
    }
  }, []);

  return (
    <SocketContext.Provider value={{ connected, subscribe, emit }}>
      {children}
    </SocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
}
