'use client';

import { Badge } from '@/components/ui/badge';
import type { DRSeverity } from '@/types';

const DR_GRADE_CONFIG: Record<DRSeverity, { label: string; variant: string }> = {
  no_dr: { label: 'No DR', variant: 'dr_no_dr' },
  mild: { label: 'Mild', variant: 'dr_mild' },
  moderate: { label: 'Moderate', variant: 'dr_moderate' },
  severe: { label: 'Severe', variant: 'dr_severe' },
  proliferative: { label: 'Proliferative', variant: 'dr_proliferative' },
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
    >
      {showLabel ? config.label : gradeStr}
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
