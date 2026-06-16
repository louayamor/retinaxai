'use client';

import { useWebSocket } from '@/hooks/use-websocket';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

export function ConnectionStatus() {
  const { connected } = useWebSocket();

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            role='status'
            aria-live='polite'
            aria-label={connected ? 'Real-time updates connected' : 'Real-time updates offline'}
            className={cn(
            'flex items-center gap-2 cursor-default px-3 py-1.5 rounded-full border text-xs font-medium transition-colors',
            connected 
              ? 'border-green-500/50 bg-green-500/10 text-green-600' 
              : 'border-orange-500/50 bg-orange-500/10 text-orange-600'
          )}>
            {connected ? (
              <>
                <Wifi className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Offline</span>
              </>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{connected ? 'Real-time updates connected' : 'Real-time updates offline'}</p>
          <p className="text-xs text-muted-foreground mt-1">
            {connected ? 'Events will appear instantly' : 'Page will refresh on reconnect'}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}