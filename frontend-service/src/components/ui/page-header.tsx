'use client';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { RefreshCw } from 'lucide-react';
import type { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn(
      'flex flex-col md:flex-row md:items-center justify-between gap-4',
      className
    )}>
      <div>
        <h1 className='text-2xl font-bold tracking-tight'>{title}</h1>
        {description && (
          <p className='text-sm text-muted-foreground mt-1'>{description}</p>
        )}
      </div>
      {actions && (
        <div className='flex items-center gap-2'>
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
