'use client';

import { cn } from '@/lib/utils';
import type { DRSeverity } from '@/types';

const GRADE_ORDER: DRSeverity[] = ['no_dr', 'mild', 'moderate', 'severe', 'proliferative'];

const GRADE_CONFIG: Record<DRSeverity, { label: string; color: string; position: number }> = {
  no_dr: { label: 'No DR', color: 'bg-emerald-500', position: 0 },
  mild: { label: 'Mild', color: 'bg-lime-500', position: 1 },
  moderate: { label: 'Moderate', color: 'bg-amber-500', position: 2 },
  severe: { label: 'Severe', color: 'bg-orange-500', position: 3 },
  proliferative: { label: 'Proliferative', color: 'bg-rose-500', position: 4 },
};

interface SeverityScaleProps {
  grade: DRSeverity;
  className?: string;
}

export function SeverityScale({ grade, className }: SeverityScaleProps) {
  const config = GRADE_CONFIG[grade] ?? GRADE_CONFIG.no_dr;

  return (
    <div className={cn('space-y-1', className)} role='figure' aria-label={`DR severity: ${config.label}`}>
      <div className='flex items-center justify-between text-xs text-muted-foreground'>
        <span>Normal</span>
        <span>Proliferative</span>
      </div>
      <div className='relative h-2 w-full overflow-hidden rounded-full bg-muted' aria-hidden='true'>
        <div
          className='absolute top-0 left-0 h-full rounded-full bg-gradient-to-r from-emerald-400 via-yellow-400 to-rose-500'
          style={{ width: '100%' }}
        />
        <div
          className='absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border-2 border-background shadow-md transition-all'
          style={{
            left: `${(config.position / (GRADE_ORDER.length - 1)) * 100}%`,
            backgroundColor: 'white',
          }}
        />
      </div>
      <div className='flex items-center justify-between gap-2'>
        <span className='text-sm font-semibold'>{config.label}</span>
        <span className='text-xs text-muted-foreground'>
          Grade {config.position} of {GRADE_ORDER.length - 1}
        </span>
      </div>
    </div>
  );
}
