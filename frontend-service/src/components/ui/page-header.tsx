'use client';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { RefreshCw } from 'lucide-react';
import type { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  metadata?: ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  actions,
  metadata,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn(
      'flex flex-col md:flex-row md:items-start justify-between gap-3',
      className
    )}>
      <div className='min-w-0 flex-1'>
        <h1 className='text-xl font-bold tracking-tight'>{title}</h1>
        {description && (
          <p className='text-sm text-muted-foreground mt-0.5'>{description}</p>
        )}
        {metadata && (
          <div className='flex items-center gap-3 mt-1.5 text-xs text-muted-foreground'>
            {metadata}
          </div>
        )}
      </div>
      {actions && (
        <div className='flex items-center gap-2 shrink-0'>
          {actions}
        </div>
      )}
    </div>
  );
}

interface RefreshButtonProps {
  onClick: () => void;
  loading?: boolean;
  label?: string;
}

export function RefreshButton({
  onClick,
  loading,
  label = 'Refresh',
}: RefreshButtonProps) {
  return (
    <Button
      variant='outline'
      size='sm'
      onClick={onClick}
      disabled={loading}
    >
      <RefreshCw className={cn('h-4 w-4 mr-2', loading && 'animate-spin')} />
      {label}
    </Button>
  );
}
