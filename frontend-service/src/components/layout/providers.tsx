'use client';

import React from 'react';
import { WebSocketProvider } from '@/hooks/use-websocket';
import { AuthProvider } from '@/providers/auth-context';

export default function Providers({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthProvider>
      <WebSocketProvider>
        {children}
      </WebSocketProvider>
    </AuthProvider>
  );
}