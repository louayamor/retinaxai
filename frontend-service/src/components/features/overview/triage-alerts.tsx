'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, ArrowRight, Eye, Loader2 } from 'lucide-react';
import type { DRSeverity } from '@/types';

interface TriageAlert {
  patient_id: string;
  patient_name: string;
  grade: DRSeverity;
  confidence: number;
  prediction_id: string;
  created_at: string;
}

export function TriageAlerts() {
  const [alerts, setAlerts] = useState<TriageAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${BASE}/api/v1/predictions?page=1&size=20&sort_by=created_at&sort_order=desc`, {
      credentials: 'include',
    })
      .then((r) => r.json())
      .then((json) => {
        const items: TriageAlert[] = (json.items || [])
          .filter((p: { status: string; output_payload?: { dr_grade?: string } }) =>
            p.status === 'success' && ['severe', 'proliferative'].includes(p.output_payload?.dr_grade ?? '')
          )
          .slice(0, 5)
          .map((p: { id: string; patient_id: string; patient_name?: string; confidence_score: number | null; output_payload?: { dr_grade?: string }; created_at: string }) => ({
            patient_id: p.patient_id,
            patient_name: p.patient_name || 'Unknown',
            grade: (p.output_payload?.dr_grade ?? 'no_dr') as DRSeverity,
            confidence: p.confidence_score ?? 0,
            prediction_id: p.id,
            created_at: p.created_at,
          }));
        setAlerts(items);
      })
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle className='flex items-center gap-2 text-base'><AlertTriangle className='h-4 w-4 text-amber-500' /> Urgent Reviews</CardTitle></CardHeader>
        <CardContent className='flex items-center justify-center py-6'>
          <Loader2 className='h-5 w-5 animate-spin text-muted-foreground' />
        </CardContent>
      </Card>
    );
  }

  if (alerts.length === 0) return null;

  return (
    <Card className='border-amber-500/30 bg-amber-500/5'>
      <CardHeader className='pb-2'>
        <CardTitle className='flex items-center gap-2 text-base'>
          <AlertTriangle className='h-4 w-4 text-amber-500' />
          Urgent Reviews
          <Badge variant='destructive' className='ml-auto'>{alerts.length} pending</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className='space-y-2'>
          {alerts.map((a) => (
            <div key={a.prediction_id} className='flex items-center justify-between rounded-md border bg-background px-3 py-2 text-sm'>
              <div className='min-w-0'>
                <p className='font-medium truncate'>{a.patient_name}</p>
                <p className='text-xs text-muted-foreground'>
                  {a.grade === 'proliferative' ? 'Proliferative DR' : 'Severe NPDR'} · {Math.round(a.confidence * 100)}% confidence
                </p>
              </div>
              <Button
                variant='ghost'
                size='icon'
                className='h-8 w-8 shrink-0'
                onClick={() => router.push(`/dashboard/clinical/predictions/${a.prediction_id}/gradcam`)}
                aria-label={`Review ${a.patient_name}`}
              >
                <Eye className='h-4 w-4' />
              </Button>
            </div>
          ))}
          <Button
            variant='link'
            size='sm'
            className='w-full text-xs'
            onClick={() => router.push('/dashboard/clinical/predictions?tab=predictions')}
          >
            View all predictions <ArrowRight className='ml-1 h-3 w-3' />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
