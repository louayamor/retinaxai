'use client';

import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

type GridColsValue = 1 | 2 | 3 | 4 | 5 | 6;

interface StatsRowProps {
  children: ReactNode;
  columns?: GridColsValue;
  className?: string;
  gap?: 'sm' | 'md' | 'lg';
}

const GAP_CLASSES = {
  sm: 'gap-3',
  md: 'gap-4',
  lg: 'gap-6',
} as const;

const COL_CLASSES: Record<GridColsValue, string> = {
  1: 'grid-cols-1',
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-2 lg:grid-cols-4',
  5: 'grid-cols-2 lg:grid-cols-5',
  6: 'grid-cols-2 lg:grid-cols-6',
};

export function StatsRow({
  children,
  columns = 4,
  className,
  gap = 'md',
}: StatsRowProps) {
  return (
    <div
      className={cn(
        'grid',
        COL_CLASSES[columns],
        GAP_CLASSES[gap],
        className
      )}
    >
      {children}
    </div>
  );
}
