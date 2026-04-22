'use client';

import { cn } from '@/lib/utils';

interface PredictionProgressProps {
  progress: number;
  stage: string;
  message?: string;
  status: 'idle' | 'uploading' | 'predicting' | 'biomarker' | 'xai' | 'reporting' | 'completed' | 'failed';
  className?: string;
}

const STATUS_COLORS: Record<PredictionProgressProps['status'], string> = {
  idle: 'bg-muted-foreground/40',
  uploading: 'bg-blue-500',
  predicting: 'bg-cyan-500',
  biomarker: 'bg-emerald-500',
  xai: 'bg-violet-500',
  reporting: 'bg-amber-500',
  completed: 'bg-emerald-500',
  failed: 'bg-rose-500',
};

const STAGE_LABELS: Record<string, string> = {
  upload: 'Uploading scans',
  prediction: 'Running prediction',
  biomarker_left: 'Extracting left-eye biomarkers',
  biomarker_right: 'Extracting right-eye biomarkers',
  biomarker: 'Extracting biomarkers',
  xai: 'Generating explanations',
  report: 'Generating report',
  completed: 'Completed',
  failed: 'Failed',
};

export function PredictionProgress({ progress, stage, message, status, className }: PredictionProgressProps) {
  const label = STAGE_LABELS[stage] || stage;

  return (
    <div className={cn('space-y-2 rounded-lg border bg-muted/30 p-4', className)}>
      <div className='flex items-center justify-between gap-3'>
        <div className='min-w-0'>
          <p className='text-sm font-medium'>{label}</p>
          {message && <p className='text-xs text-muted-foreground'>{message}</p>}
        </div>
        <span className='text-xs font-medium tabular-nums text-muted-foreground'>
          {Math.max(0, Math.min(100, Math.round(progress)))}%
        </span>
      </div>

      <div className='h-2 w-full overflow-hidden rounded-full bg-muted'>
        <div
          className={cn('h-full rounded-full transition-all duration-300', STATUS_COLORS[status])}
          style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
        />
      </div>
    </div>
  );
}
