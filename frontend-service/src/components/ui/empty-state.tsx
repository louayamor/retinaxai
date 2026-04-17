'use client';

import type { ComponentType } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

interface EmptyStateProps {
  icon?: ComponentType<{ className?: string }>;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div className={cn(
      'flex flex-col items-center justify-center py-12 px-4 text-center',
      className
    )}>
      {Icon && (
        <div className='relative mb-4'>
          <Icon className='h-16 w-16 text-muted-foreground/30' />
        </div>
      )}
      <h3 className='text-lg font-semibold text-foreground mb-1'>{title}</h3>
      {description && (
        <p className='text-sm text-muted-foreground max-w-sm mb-4'>
          {description}
        </p>
      )}
      {action && (
        <Button onClick={action.onClick} variant='outline' size='sm'>
          {action.label}
        </Button>
      )}
    </div>
  );
}

interface LoadingStateProps {
  message?: string;
  className?: string;
}

export function LoadingState({
  message = 'Loading...',
  className,
}: LoadingStateProps) {
  return (
    <div className={cn(
      'flex items-center justify-center py-12',
      className
    )}>
      <div className='text-center'>
        <div className='animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mx-auto mb-3' />
        <p className='text-sm text-muted-foreground'>{message}</p>
      </div>
    </div>
  );
}
