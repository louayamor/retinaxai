'use client';

import { cn } from '@/lib/utils';

interface TrainingProgressProps {
  stage: string;
  progress: number;
  status: 'started' | 'running' | 'completed' | 'failed';
  message?: string;
  className?: string;
}

const STAGE_LABELS: Record<string, string> = {
  pipeline: 'Pipeline',
  data_ingestion: 'Data Ingestion',
  data_cleaning: 'Data Cleaning',
  data_transformation: 'Data Transformation',
  model_training: 'Model Training',
  model_evaluation: 'Model Evaluation',
};

const STATUS_COLORS = {
  started: 'bg-blue-500',
  running: 'bg-yellow-500 animate-pulse',
  completed: 'bg-green-500',
  failed: 'bg-red-500',
};

const STATUS_BADGE = {
  started: 'bg-blue-500/20 text-blue-400 border-blue-500/50',
  running: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
  completed: 'bg-green-500/20 text-green-400 border-green-500/50',
  failed: 'bg-red-500/20 text-red-400 border-red-500/50',
};

export function TrainingProgress({ stage, progress, status, message, className }: TrainingProgressProps) {
  const stageLabel = STAGE_LABELS[stage] || stage;
  const clamped = Math.max(0, Math.min(100, progress));
  const numeric = Math.round(clamped);

  return (
    <div className={cn('space-y-2', className)} role='group' aria-label='Training progress'>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{stageLabel}</span>
        <span className={cn(
          'text-xs px-2 py-0.5 rounded-full border',
          STATUS_BADGE[status]
        )}>
          {status}
        </span>
      </div>
      
      <div
        className="w-full h-2 bg-muted rounded-full overflow-hidden"
        role='progressbar'
        aria-valuenow={numeric}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${stageLabel}: ${numeric}% complete`}
      >
        <div
          className={cn('h-full transition-all duration-300', STATUS_COLORS[status])}
          style={{ width: `${clamped}%` }}
        />
      </div>

      {message && (
        <p className="text-xs text-muted-foreground">{message}</p>
      )}
    </div>
  );
}