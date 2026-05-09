'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader, RefreshButton } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Loader2, Play, RefreshCw, Square, CheckCircle2, WifiOff } from 'lucide-react';
import { useWebSocket } from '@/hooks/use-websocket';
import { TrainingProgress } from '@/components/training-progress';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';

interface Metrics {
  imaging?: {
    accuracy?: number;
    quadratic_weighted_kappa?: number;
    roc_auc_macro?: number;
    num_samples?: number;
  };
  clinical?: {
    accuracy?: number;
    quadratic_weighted_kappa?: number;
    roc_auc_macro?: number;
    num_samples?: number;
  };
}

interface JobStatus {
  job_id: string | null;
  pipeline: string | null;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export default function ModelsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<{
    stage: string;
    progress: number;
    status: 'started' | 'running' | 'completed' | 'failed';
    message?: string;
  } | null>(null);

  const { connected, subscribe } = useWebSocket();

  useEffect(() => {
    const unsubStage = subscribe('training_stage', (data: unknown) => {
      const eventData = data as { stage: string; status: string; progress: number; message?: string; error?: string };
      const { stage, status, progress, message, error } = eventData;
      setTrainingProgress({ stage, progress, status: status as 'started' | 'running' | 'completed' | 'failed', message });

      if (status === 'completed') {
        toast.success(message || 'Training completed successfully!');
        setTraining(null);
        void fetchMetrics();
        void fetchStatus();
      } else if (status === 'failed') {
        toast.error(error || message || 'Training failed');
        setTraining(null);
      } else if (status === 'started' || status === 'running') {
        toast(message || `Stage: ${stage}`, { icon: '🔄' });
      }
    });

    return () => {
      unsubStage();
    };
  }, [subscribe]);

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${MLOPS_BASE}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.warn('MLOps service not available:', err);
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${MLOPS_BASE}/status`);
      if (res.ok) {
        const data = await res.json();
        setJobStatus(data);
      }
    } catch (err) {
      console.warn('MLOps service not available:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    await Promise.all([fetchMetrics(), fetchStatus()]);
    setLoading(false);
  };

  const triggerTraining = async (pipeline: 'imaging' | 'clinical') => {
    setTraining(pipeline);
    try {
      const res = await fetch(`${MLOPS_BASE}/api/train/${pipeline}`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        console.log(`Training started: ${data.job_id}`);
        setTimeout(fetchStatus, 2000);
      }
    } catch (err) {
      console.error(`Failed to start ${pipeline} training:`, err);
    } finally {
      setTraining(null);
    }
  };

  const stopTraining = async () => {
    if (!jobStatus?.job_id) return;
    try {
      const res = await fetch(`${MLOPS_BASE}/api/train/${jobStatus.job_id}/stop`, {
        method: 'POST',
      });
      if (res.ok) {
        toast.info('Training stop requested');
        setTraining(null);
        void fetchStatus();
      }
    } catch (err) {
      console.error('Failed to stop training:', err);
    }
  };

  useEffect(() => {
    void fetchMetrics();
    void fetchStatus();
  }, []);

  const radarData = [
    {
      metric: 'Accuracy',
      imaging: metrics?.imaging?.accuracy || 0,
      clinical: metrics?.clinical?.accuracy || 0,
    },
    {
      metric: 'QWK',
      imaging: metrics?.imaging?.quadratic_weighted_kappa || 0,
      clinical: metrics?.clinical?.quadratic_weighted_kappa || 0,
    },
    {
      metric: 'AUC',
      imaging: metrics?.imaging?.roc_auc_macro || 0,
      clinical: metrics?.clinical?.roc_auc_macro || 0,
    },
  ];

  const isTraining = jobStatus?.status === 'running' || jobStatus?.status === 'pending';

  if (loading && !metrics) {
    return (
      <PageContainer>
        <div className='flex items-center justify-center h-[60vh]'>
          <Loader2 className='h-8 w-8 animate-spin text-muted-foreground' />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHeader
        title='AI Models'
        description='Model performance, training and evaluation metrics of EfficientNet-B3 + XGBoost'
      />

      <div className='rounded-lg border bg-card p-4'>
        <div className='flex items-center justify-between mb-4'>
          <div className='flex items-center gap-2'>
            <Play className='h-5 w-5' />
            <h3 className='font-semibold'>Training Pipeline</h3>
          </div>
          <RefreshButton onClick={fetchData} loading={loading} />
        </div>
        <div className='flex flex-wrap gap-4 items-center'>
          <Button
            onClick={() => void triggerTraining('imaging')}
            disabled={isTraining || training === 'imaging'}
            className='bg-[var(--brand-teal)] hover:bg-[#1a9a9a]'
          >
            {training === 'imaging' ? (
              <Loader2 className='mr-2 h-4 w-4 animate-spin' />
            ) : (
              <Play className='mr-2 h-4 w-4' />
            )}
            Train Imaging
          </Button>
          <Button
            onClick={() => void triggerTraining('clinical')}
            disabled={isTraining || training === 'clinical'}
            variant='outline'
          >
            {training === 'clinical' ? (
              <Loader2 className='mr-2 h-4 w-4 animate-spin' />
            ) : (
              <Play className='mr-2 h-4 w-4' />
            )}
            Train Clinical
          </Button>
          {isTraining && (
            <Button onClick={stopTraining} variant='destructive' size='sm'>
              <Square className='mr-2 h-4 w-4' />
              Stop
            </Button>
          )}
          {isTraining && (
            <Badge variant='secondary' className='ml-auto'>
              <Loader2 className='mr-1 h-3 w-3 animate-spin' />
              {jobStatus?.pipeline} — {jobStatus?.status}
            </Badge>
          )}
          {!connected && (
            <Badge variant='outline' className='ml-2 text-orange-500'>
              <WifiOff className='mr-1 h-3 w-3' />
              Offline
            </Badge>
          )}
        </div>
        {trainingProgress && (
          <div className='mt-4 p-4 bg-muted/50 rounded-lg'>
            <TrainingProgress
              stage={trainingProgress.stage}
              progress={trainingProgress.progress}
              status={trainingProgress.status}
              message={trainingProgress.message}
            />
          </div>
        )}
        {jobStatus?.error && (
          <p className='text-sm text-destructive mt-4'>{jobStatus.error}</p>
        )}
      </div>

      <div className='grid grid-cols-1 md:grid-cols-2 gap-5'>
        <Card>
          <CardHeader>
            <CardTitle className='flex items-center gap-2'>
              Imaging Model
              <Badge variant='secondary'>EfficientNet-B3</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className='text-muted-foreground'>Loading...</p>
            ) : metrics?.imaging ? (
              <div className='grid grid-cols-3 gap-4'>
                <div>
                  <p className='text-sm text-muted-foreground'>Accuracy</p>
                  <p className='text-2xl font-bold'>
                    {((metrics.imaging.accuracy ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>QWK</p>
                  <p className='text-2xl font-bold'>
                    {(metrics.imaging.quadratic_weighted_kappa ?? 0).toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>AUC</p>
                  <p className='text-2xl font-bold'>
                    {(metrics.imaging.roc_auc_macro ?? 0).toFixed(3)}
                  </p>
                </div>
              </div>
            ) : (
              <p className='text-muted-foreground'>No metrics available. Train the model first.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className='flex items-center gap-2'>
              Clinical Model
              <Badge variant='secondary'>XGBoost</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className='text-muted-foreground'>Loading...</p>
            ) : metrics?.clinical ? (
              <div className='grid grid-cols-3 gap-4'>
                <div>
                  <p className='text-sm text-muted-foreground'>Accuracy</p>
                  <p className='text-2xl font-bold'>
                    {((metrics.clinical.accuracy ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>QWK</p>
                  <p className='text-2xl font-bold'>
                    {(metrics.clinical.quadratic_weighted_kappa ?? 0).toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>AUC</p>
                  <p className='text-2xl font-bold'>
                    {(metrics.clinical.roc_auc_macro ?? 0).toFixed(3)}
                  </p>
                </div>
              </div>
            ) : (
              <p className='text-muted-foreground'>No metrics available. Train the model first.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className='rounded-lg border bg-card p-4'>
        <h3 className='font-semibold mb-4'>Model Comparison</h3>
        <div className='h-[300px]'>
          <ResponsiveContainer width='100%' height='100%'>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey='metric' />
              <PolarRadiusAxis domain={[0, 1]} />
              <Radar
                name='Imaging'
                dataKey='imaging'
                stroke='var(--brand-teal)'
                fill='var(--brand-teal)'
                fillOpacity={0.4}
              />
              <Radar
                name='Clinical'
                dataKey='clinical'
                stroke='var(--brand-gold)'
                fill='var(--brand-gold)'
                fillOpacity={0.4}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className='grid grid-cols-1 md:grid-cols-2 gap-5'>
        <Card>
          <CardHeader>
            <CardTitle>MLOps Service</CardTitle>
          </CardHeader>
          <CardContent>
            <div className='flex items-center gap-3'>
              <div className='h-3 w-3 rounded-full bg-green-500 animate-pulse' />
              <span className='text-sm'>Operational</span>
            </div>
            <Separator className='my-4' />
            <div className='grid grid-cols-2 gap-4 text-sm'>
              <div>
                <p className='text-muted-foreground'>Job Status</p>
                <Badge variant={isTraining ? 'default' : 'secondary'}>
                  {jobStatus?.status || 'idle'}
                </Badge>
              </div>
              <div>
                <p className='text-muted-foreground'>Pipeline</p>
                <p className='font-medium'>{jobStatus?.pipeline || '—'}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className='space-y-4'>
              <div className='flex justify-between'>
                <span className='text-sm text-muted-foreground'>Imaging Samples</span>
                <span className='font-medium'>{metrics?.imaging?.num_samples ?? '—'}</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-muted-foreground'>Clinical Samples</span>
                <span className='font-medium'>{metrics?.clinical?.num_samples ?? '—'}</span>
              </div>
              <div className='flex justify-between'>
                <span className='text-sm text-muted-foreground'>Last Training</span>
                <span className='font-medium'>
                  {jobStatus?.completed_at
                    ? new Date(jobStatus.completed_at).toLocaleDateString()
                    : 'Never'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  );
}
