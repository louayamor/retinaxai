'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHero } from '@/components/ui/page-hero';
import { PageSection, PageGrid } from '@/components/ui/page-section';
import { StatsRow } from '@/components/ui/stats-row';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { StatsCard } from '@/components/ui/stats-card';
import {
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Database,
  TrendingUp,
} from 'lucide-react';
import { toast } from 'sonner';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';

const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';

interface Job {
  job_id: string;
  pipeline: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  metrics?: { accuracy?: number };
  error?: string;
}

interface DriftStatus {
  pipeline: string;
  status: string;
  psi_threshold: number;
  overall_psi?: number;
  drift_detected?: boolean;
  last_checked: string;
  features_drifted?: string[];
}

interface Feature {
  key: string;
  value: string;
  created_at: string;
}

interface Metrics {
  imaging?: { accuracy?: number; quadratic_weighted_kappa?: number; roc_auc_macro?: number; precision_macro?: number; recall_macro?: number; num_samples?: number };
  clinical?: { accuracy?: number; quadratic_weighted_kappa?: number; roc_auc_macro?: number; precision_macro?: number; recall_macro?: number; num_samples?: number };
}

const PIPELINES = ['imaging', 'clinical'];

export default function MLOpsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [driftStatus, setDriftStatus] = useState<Record<string, DriftStatus>>({});
  const [driftHistory, setDriftHistory] = useState<Array<{ pipeline: string; psi: number; status: string; timestamp: string }>>([]);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [jobsRes, driftRes, featuresRes, metricsRes] = await Promise.all([
        fetch(`${MLOPS_BASE}/api/train/jobs`).then(r => r.json()).catch(() => ({ jobs: [], total: 0 })),
        fetch(`${MLOPS_BASE}/drift/history`).then(r => r.json()).catch(() => ({ history: [] })),
        fetch(`${MLOPS_BASE}/features/list`).then(r => r.json()).catch(() => ({ features: [], total: 0 })),
        fetch(`${MLOPS_BASE}/metrics`).then(r => r.json()).catch(() => ({})),
      ]);

      setJobs(jobsRes.jobs || []);
      setDriftHistory(driftRes.history || []);
      setFeatures(featuresRes.features || []);
      setMetrics(metricsRes);

      const driftStatusMap: Record<string, DriftStatus> = {};
      for (const pipeline of PIPELINES) {
        try {
          const res = await fetch(`${MLOPS_BASE}/drift/status/${pipeline}`);
          driftStatusMap[pipeline] = await res.json();
        } catch {
          driftStatusMap[pipeline] = { pipeline, status: 'stable', psi_threshold: 0.3, last_checked: new Date().toISOString() };
        }
      }
      setDriftStatus(driftStatusMap);
    } catch (error) {
      console.error('Failed to fetch MLOps data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleRetrain = async (pipeline: string) => {
    setRetraining(pipeline);
    try {
      const res = await fetch(`${MLOPS_BASE}/automation/drift-retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pipeline }),
      });
      const data = await res.json();
      if (data.job_id) {
        toast.success(`Retraining triggered: ${data.job_id}`);
      } else {
        toast.warning(`No retraining needed (PSI: ${data.psi?.toFixed(3) || 'N/A'})`);
      }
      void fetchData();
    } catch {
      toast.error('Failed to trigger retraining');
    } finally {
      setRetraining(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      case 'stable': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'drifted':
      case 'drift_detected': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getDriftChartData = () => {
    const grouped: Record<string, { timestamp: string; imaging?: number; clinical?: number }> = {};
    driftHistory.forEach(h => {
      const date = new Date(h.timestamp).toLocaleDateString();
      if (!grouped[date]) grouped[date] = { timestamp: date };
      if (h.pipeline === 'imaging') grouped[date].imaging = h.psi;
      if (h.pipeline === 'clinical') grouped[date].clinical = h.psi;
    });
    return Object.values(grouped).slice(-14);
  };

  const getMetricsRadarData = () => {
    if (!metrics) return [];
    const im = metrics.imaging || {};
    const cl = metrics.clinical || {};
    return [
      { metric: 'Acc', imaging: im.accuracy ? im.accuracy * 100 : 0, clinical: cl.accuracy ? cl.accuracy * 100 : 0 },
      { metric: 'Kappa', imaging: im.quadratic_weighted_kappa ? im.quadratic_weighted_kappa * 100 : 0, clinical: cl.quadratic_weighted_kappa ? cl.quadratic_weighted_kappa * 100 : 0 },
      { metric: 'AUC', imaging: im.roc_auc_macro ? im.roc_auc_macro * 100 : 0, clinical: cl.roc_auc_macro ? cl.roc_auc_macro * 100 : 0 },
      { metric: 'Prec', imaging: im.precision_macro ? im.precision_macro * 100 : undefined, clinical: cl.precision_macro ? cl.precision_macro * 100 : undefined },
      { metric: 'Rec', imaging: im.recall_macro ? im.recall_macro * 100 : undefined, clinical: cl.recall_macro ? cl.recall_macro * 100 : undefined },
    ];
  };

  if (loading && !metrics) {
    return (
      <PageContainer>
        <div className='flex items-center justify-center h-[60vh]'>
          <div className='animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full' />
        </div>
      </PageContainer>
    );
  }

  const activeJobs = jobs.filter(j => j.status === 'running').length;
  const overallDrift = Object.values(driftStatus).some(d => d.status === 'drift_detected');

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHero
        title='MLOps Monitor'
        description='Training pipelines, drift detection, and model performance'
      />

      <StatsRow columns={4}>
        <StatsCard
          title='Active Jobs'
          value={`${activeJobs}/${jobs.length}`}
          icon={Activity}
          subtitle='Training jobs'
        />
        <StatsCard
          title='Feature Store'
          value={features.length}
          icon={Database}
          subtitle='Stored features'
        />
        <StatsCard
          title='Drift Status'
          value={overallDrift ? 'Drifted' : 'Stable'}
          icon={overallDrift ? AlertTriangle : CheckCircle2}
          color={overallDrift ? '#ef4444' : '#22c55e'}
        />
        <StatsCard
          title='Imaging Accuracy'
          value={metrics?.imaging?.accuracy != null ? `${(metrics.imaging.accuracy * 100).toFixed(1)}%` : 'N/A'}
          icon={Brain}
          color='#3b82f6'
        />
      </StatsRow>

      <PageGrid columns={2}>
        {metrics?.imaging && (
          <Card>
            <CardHeader>
              <CardTitle className='flex items-center gap-2'>
                <Brain className='h-5 w-5 text-blue-600' />
                EfficientNet-B3 (Imaging)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className='grid grid-cols-3 gap-4'>
                <div>
                  <p className='text-sm text-muted-foreground'>Accuracy</p>
                  <p className='text-xl font-bold'>
                    {metrics.imaging.accuracy != null ? `${(metrics.imaging.accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>Kappa</p>
                  <p className='text-xl font-bold'>{metrics.imaging.quadratic_weighted_kappa?.toFixed(2) || 'N/A'}</p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>AUC</p>
                  <p className='text-xl font-bold'>
                    {metrics.imaging.roc_auc_macro != null ? `${(metrics.imaging.roc_auc_macro * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        {metrics?.clinical && (
          <Card>
            <CardHeader>
              <CardTitle className='flex items-center gap-2'>
                <Brain className='h-5 w-5 text-purple-600' />
                XGBoost (Clinical)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className='grid grid-cols-3 gap-4'>
                <div>
                  <p className='text-sm text-muted-foreground'>Accuracy</p>
                  <p className='text-xl font-bold'>
                    {metrics.clinical.accuracy != null ? `${(metrics.clinical.accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>Kappa</p>
                  <p className='text-xl font-bold'>{metrics.clinical.quadratic_weighted_kappa?.toFixed(2) || 'N/A'}</p>
                </div>
                <div>
                  <p className='text-sm text-muted-foreground'>AUC</p>
                  <p className='text-xl font-bold'>
                    {metrics.clinical.roc_auc_macro != null ? `${(metrics.clinical.roc_auc_macro * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </PageGrid>

      {getMetricsRadarData().length > 0 && (
        <PageSection title='Model Performance Comparison'>
          <div className='h-[300px]'>
            <ResponsiveContainer width='100%' height='100%'>
              <RadarChart data={getMetricsRadarData()}>
                <PolarGrid />
                <PolarAngleAxis dataKey='metric' />
                <PolarRadiusAxis domain={[0, 100]} />
                <Radar name='Imaging' dataKey='imaging' stroke='#2563eb' fill='#2563eb' fillOpacity={0.2} />
                <Radar name='Clinical' dataKey='clinical' stroke='#9333ea' fill='#9333ea' fillOpacity={0.2} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </PageSection>
      )}

      <PageGrid columns={2}>
        <PageSection title='Training Jobs' icon={<Activity className='h-5 w-5' />}>
          {jobs.length === 0 ? (
            <p className='text-sm text-muted-foreground py-4 text-center'>No training jobs</p>
          ) : (
            <div className='space-y-2'>
              {jobs.slice(0, 8).map((job) => (
                <div key={job.job_id} className='flex items-center justify-between p-3 rounded-lg border'>
                  <div>
                    <span className='font-mono text-sm'>{job.job_id.slice(0, 8)}</span>
                    <span className='text-muted-foreground ml-2 capitalize text-sm'>{job.pipeline}</span>
                  </div>
                  <div className='flex items-center gap-2'>
                    <span className='text-muted-foreground text-sm'>
                      {job.started_at ? new Date(job.started_at).toLocaleDateString() : 'Pending'}
                    </span>
                    <Badge className={getStatusColor(job.status)}>{job.status}</Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </PageSection>

        <PageSection title='Drift Status' icon={<TrendingUp className='h-5 w-5' />}>
          <div className='space-y-3'>
            {PIPELINES.map((pipeline) => {
              const status = driftStatus[pipeline];
              return (
                <div key={pipeline} className='p-3 rounded-lg border'>
                  <div className='flex items-center justify-between mb-2'>
                    <span className='capitalize text-sm font-medium'>{pipeline}</span>
                    <Badge className={getStatusColor(status?.status || 'unknown')}>
                      {status?.status || 'Unknown'}
                    </Badge>
                  </div>
                  <div className='grid grid-cols-2 gap-2 text-sm'>
                    <div><span className='text-muted-foreground'>PSI:</span> <span className='font-medium'>{status?.overall_psi?.toFixed(3) || 'N/A'}</span></div>
                    <div><span className='text-muted-foreground'>Threshold:</span> <span className='font-medium'>{status?.psi_threshold?.toFixed(1) || '0.3'}</span></div>
                  </div>
                  {status?.features_drifted && status.features_drifted.length > 0 && (
                    <div className='mt-2 flex flex-wrap gap-1'>
                      {status.features_drifted.slice(0, 3).map((f) => (
                        <Badge key={f} variant='outline'>{f}</Badge>
                      ))}
                    </div>
                  )}
                  <Button
                    variant='outline'
                    size='sm'
                    className='w-full mt-2'
                    onClick={() => void handleRetrain(pipeline)}
                    disabled={retraining === pipeline}
                  >
                    {retraining === pipeline ? (
                      <RefreshCw className='h-4 w-4 animate-spin' />
                    ) : (
                      <>
                        <RefreshCw className='h-4 w-4 mr-2' />
                        Retrain
                      </>
                    )}
                  </Button>
                </div>
              );
            })}
          </div>
        </PageSection>
      </PageGrid>

      <PageGrid columns={2}>
        <PageSection title='Drift History (PSI)'>
          <div className='h-[200px]'>
            {getDriftChartData().length > 0 ? (
              <ResponsiveContainer width='100%' height='100%'>
                <LineChart data={getDriftChartData()}>
                  <CartesianGrid strokeDasharray='3 3' />
                  <XAxis dataKey='timestamp' />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line type='monotone' dataKey='imaging' stroke='#2563eb' strokeWidth={2} name='Imaging' dot={false} />
                  <Line type='monotone' dataKey='clinical' stroke='#9333ea' strokeWidth={2} name='Clinical' dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className='text-sm text-muted-foreground flex items-center justify-center h-full'>No drift history</p>
            )}
          </div>
        </PageSection>

        <PageSection title='Feature Store' icon={<Database className='h-5 w-5' />} headerAction={<Badge variant='secondary'>{features.length}</Badge>}>
          {features.length === 0 ? (
            <p className='text-sm text-muted-foreground py-4 text-center'>No features in store</p>
          ) : (
            <div className='space-y-2'>
              {features.slice(0, 10).map((feat, i) => (
                <div key={i} className='flex items-center justify-between p-2 rounded border'>
                  <span className='font-mono text-sm truncate max-w-[150px]'>{feat.key}</span>
                  <span className='text-muted-foreground text-sm'>
                    {feat.created_at ? new Date(feat.created_at).toLocaleDateString() : 'N/A'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </PageSection>
      </PageGrid>
    </PageContainer>
  );
}
