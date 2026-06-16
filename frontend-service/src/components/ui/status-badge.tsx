'use client';

import { Badge } from '@/components/ui/badge';
import type { DRSeverity } from '@/types';

const GRADE_ORDER: DRSeverity[] = ['no_dr', 'mild', 'moderate', 'severe', 'proliferative'];

const DR_GRADE_CONFIG: Record<DRSeverity, { label: string; ordinal: string; variant: string }> = {
  no_dr: { label: 'No DR', ordinal: 'Grade 0 of 4: No Diabetic Retinopathy', variant: 'dr_no_dr' },
  mild: { label: 'Mild', ordinal: 'Grade 1 of 4: Mild Non-Proliferative DR', variant: 'dr_mild' },
  moderate: { label: 'Moderate', ordinal: 'Grade 2 of 4: Moderate Non-Proliferative DR', variant: 'dr_moderate' },
  severe: { label: 'Severe', ordinal: 'Grade 3 of 4: Severe Non-Proliferative DR', variant: 'dr_severe' },
  proliferative: { label: 'Proliferative', ordinal: 'Grade 4 of 4: Proliferative DR', variant: 'dr_proliferative' },
};

interface DRGradeBadgeProps {
  grade: DRSeverity | number | string;
  showLabel?: boolean;
  size?: 'sm' | 'default';
}

export function DRGradeBadge({ grade, showLabel = true, size = 'default' }: DRGradeBadgeProps) {
  const gradeStr = String(grade).toLowerCase();
  
  let config = DR_GRADE_CONFIG.no_dr;
  
  if (gradeStr === '0' || gradeStr === 'no_dr' || gradeStr === 'none') {
    config = DR_GRADE_CONFIG.no_dr;
  } else if (gradeStr === '1' || gradeStr === 'mild') {
    config = DR_GRADE_CONFIG.mild;
  } else if (gradeStr === '2' || gradeStr === 'moderate') {
    config = DR_GRADE_CONFIG.moderate;
  } else if (gradeStr === '3' || gradeStr === 'severe') {
    config = DR_GRADE_CONFIG.severe;
  } else if (gradeStr === '4' || gradeStr === 'proliferative') {
    config = DR_GRADE_CONFIG.proliferative;
  }

  return (
    <Badge
      variant={config.variant as 'dr_no_dr'}
      className={size === 'sm' ? 'text-xs px-1.5 py-0' : ''}
      role='status'
    >
      <span aria-hidden='true'>{showLabel ? config.label : gradeStr}</span>
      <span className='sr-only'>{config.ordinal}</span>
    </Badge>
  );
}

interface StatusBadgeProps {
  status: 'pending' | 'success' | 'failed' | 'running' | string;
  label?: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const statusConfig: Record<string, { label: string; variant: string }> = {
    pending: { label: 'Pending', variant: 'warning' },
    running: { label: 'Running', variant: 'info' },
    success: { label: 'Success', variant: 'success' },
    completed: { label: 'Completed', variant: 'success' },
    failed: { label: 'Failed', variant: 'destructive' },
    error: { label: 'Error', variant: 'destructive' },
  };

  const config = statusConfig[status] || { label: status, variant: 'secondary' };

  return (
    <Badge variant={config.variant as 'success'}>
      {label || config.label}
    </Badge>
  );
}
