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
      'flex flex-col items-center justify-center py-12 px-4 text-center relative',
      className
    )}>
      {/* Brand-tinted background accent */}
      <div className='absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 rounded-full bg-[var(--brand-teal)]/5 blur-3xl pointer-events-none' />

      {Icon && (
        <div className='relative mb-4 flex items-center justify-center'>
          <div className='absolute inset-0 rounded-full bg-[var(--brand-teal)]/10 blur-sm scale-150' />
          <Icon className='h-12 w-12 text-muted-foreground/40 relative' />
        </div>
      )}
      <h3 className='text-base font-semibold text-foreground mb-1'>{title}</h3>
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
        <div className='animate-spin rounded-full h-7 w-7 border-2 border-[var(--brand-teal)] border-t-transparent mx-auto mb-3' />
        <p className='text-sm text-muted-foreground'>{message}</p>
      </div>
    </div>
  );
}
