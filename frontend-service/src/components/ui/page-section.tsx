'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

interface PageSectionProps {
  title: string;
  description?: string;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
  padding?: 'compact' | 'default' | 'comfortable' | 'none';
  headerAction?: ReactNode;
}

const PADDING_CLASSES = {
  compact: 'p-3 md:p-4',
  default: 'p-4 md:p-6',
  comfortable: 'p-6 md:p-8',
  none: '',
};

export function PageSection({
  title,
  description,
  icon,
  children,
  className,
  contentClassName,
  padding = 'default',
  headerAction,
}: PageSectionProps) {
  return (
    <Card className={cn('shadow-sm', className)}>
      <CardHeader className={cn(
        'px-4 md:px-6',
        headerAction && 'flex flex-row items-center justify-between'
      )}>
        <div className='space-y-1'>
          {title && (
            <CardTitle className='flex items-center gap-2 text-lg font-semibold'>
              {icon}
              {title}
            </CardTitle>
          )}
          {description && (
            <CardDescription>{description}</CardDescription>
          )}
        </div>
        {headerAction && <div>{headerAction}</div>}
      </CardHeader>
      <CardContent className={cn(PADDING_CLASSES[padding], contentClassName)}>
        {children}
      </CardContent>
    </Card>
  );
}

interface PageGridProps {
  children: ReactNode;
  columns?: 1 | 2 | 3 | 4;
  className?: string;
  gap?: 'sm' | 'md' | 'lg';
}

const GRID_GAP = {
  sm: 'gap-3',
  md: 'gap-4',
  lg: 'gap-6',
} as const;

export function PageGrid({
  children,
  columns = 2,
  className,
  gap = 'md',
}: PageGridProps) {
  const colClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={cn('grid', colClasses[columns], GRID_GAP[gap], className)}>
      {children}
    </div>
  );
}
