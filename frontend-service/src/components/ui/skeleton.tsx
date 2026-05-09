import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

function Skeleton({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot='skeleton'
      className={cn('bg-accent animate-pulse rounded-md', className)}
      {...props}
    />
  );
}

/** Skeleton for a single table row */
function SkeletonTableRow({ columns = 5 }: { columns?: number }) {
  return (
    <div className='flex items-center gap-4 px-4 py-3'>
      {Array.from({ length: columns }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn('h-4', i === 0 ? 'w-1/4' : 'flex-1')}
        />
      ))}
    </div>
  );
}

/** Skeleton for a stat/metric card */
function SkeletonStatCard() {
  return (
    <div className='rounded-lg border bg-card p-4 space-y-2'>
      <Skeleton className='h-3 w-20' />
      <Skeleton className='h-7 w-16' />
      <Skeleton className='h-2 w-24' />
    </div>
  );
}

/** Skeleton for chart area */
function SkeletonChart({ height = 200 }: { height?: number }) {
  return (
    <div className='rounded-lg border bg-card p-4'>
      <Skeleton className='h-3 w-24 mb-3' />
      <Skeleton className={cn('w-full rounded-md')} style={{ height }} />
    </div>
  );
}

/** Skeleton for a card with header + content */
function SkeletonCard() {
  return (
    <div className='rounded-lg border bg-card p-4 space-y-3'>
      <div className='flex items-center justify-between'>
        <Skeleton className='h-4 w-1/3' />
        <Skeleton className='h-4 w-4 rounded-full' />
      </div>
      <Skeleton className='h-3 w-3/4' />
    </div>
  );
}

export {
  Skeleton,
  SkeletonTableRow,
  SkeletonStatCard,
  SkeletonChart,
  SkeletonCard,
};
