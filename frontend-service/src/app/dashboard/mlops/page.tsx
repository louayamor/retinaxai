'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
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
  Cpu,
  Gauge,
  FileText,
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

interface PrometheusMetrics {
  training_runs_total: number;
  active_training_jobs: number;
  best_val_accuracy_imaging: number | null;
  best_val_accuracy_clinical: number | null;
  drift_detected_imaging: number | null;
  drift_detected_clinical: number | null;
  evidently_dataset_shift_imaging: number | null;
  evidently_dataset_shift_clinical: number | null;
  evidently_features_drifted_imaging: number | null;
  evidently_features_drifted_clinical: number | null;
  inference_latency_p95: number | null;
  gpu_utilization: number | null;
}

const PIPELINES = ['imaging', 'clinical'];

export default function MLOpsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [driftStatus, setDriftStatus] = useState<Record<string, DriftStatus>>({});
  const [driftHistory, setDriftHistory] = useState<Array<{ pipeline: string; overall_psi: number; status: string; generated_at: string; drift_detected: boolean }>>([]);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [promMetrics, setPromMetrics] = useState<PrometheusMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [jobsRes, driftRes, featuresRes, metricsRes, promRes] = await Promise.all([
        fetch(`${MLOPS_BASE}/api/train/jobs`).then(r => r.json()).catch(() => ({ jobs: [], total: 0 })),
        fetch(`${MLOPS_BASE}/drift/history`).then(r => r.json()).catch(() => ({ history: [] })),
        fetch(`${MLOPS_BASE}/features/list`).then(r => r.json()).catch(() => ({ features: [], total: 0 })),
        fetch(`${MLOPS_BASE}/metrics`).then(r => r.json()).catch(() => ({})),
        fetch(`${MLOPS_BASE}/metrics/prometheus`).then(r => r.json()).catch(() => null),
      ]);

      setJobs(jobsRes.jobs || []);
      setDriftHistory(driftRes.history || []);
      setFeatures(featuresRes.features || []);
      setMetrics(metricsRes);
      setPromMetrics(promRes);

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
      const date = new Date(h.generated_at).toLocaleDateString();
      if (!grouped[date]) grouped[date] = { timestamp: date };
      if (h.pipeline === 'imaging') grouped[date].imaging = h.overall_psi;
      if (h.pipeline === 'clinical') grouped[date].clinical = h.overall_psi;
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
  const evidentlyShiftImaging = promMetrics?.evidently_dataset_shift_imaging ?? 0;
  const evidentlyShiftClinical = promMetrics?.evidently_dataset_shift_clinical ?? 0;
  const evidentlyFeaturesImaging = promMetrics?.evidently_features_drifted_imaging ?? 0;
  const evidentlyFeaturesClinical = promMetrics?.evidently_features_drifted_clinical ?? 0;
  const hasEvidentlyData = evidentlyShiftImaging > 0 || evidentlyShiftClinical > 0 || evidentlyFeaturesImaging > 0 || evidentlyFeaturesClinical > 0;

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHeader
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

      {promMetrics && (
        <div className='rounded-lg border bg-card p-4'>
          <h3 className='font-semibold mb-4'>Live Prometheus Metrics</h3>
          <div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
            <Card>
              <CardContent className='pt-4'>
                <div className='flex items-center gap-2'>
                  <Activity className='h-4 w-4 text-muted-foreground' />
                  <div>
                    <p className='text-sm text-muted-foreground'>Training Runs</p>
                    <p className='text-xl font-bold'>{promMetrics.training_runs_total?.toFixed(0) ?? '0'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className='pt-4'>
                <div className='flex items-center gap-2'>
                  <Gauge className='h-4 w-4 text-muted-foreground' />
                  <div>
                    <p className='text-sm text-muted-foreground'>P95 Latency</p>
                    <p className='text-xl font-bold'>{promMetrics.inference_latency_p95 != null ? `${(promMetrics.inference_latency_p95 * 1000).toFixed(0)}ms` : 'N/A'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className='pt-4'>
                <div className='flex items-center gap-2'>
                  <Cpu className='h-4 w-4 text-muted-foreground' />
                  <div>
                    <p className='text-sm text-muted-foreground'>GPU Util</p>
                    <p className='text-xl font-bold'>{promMetrics.gpu_utilization != null ? `${promMetrics.gpu_utilization.toFixed(0)}%` : 'N/A'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className='pt-4'>
                <div className='flex items-center gap-2'>
                  <TrendingUp className='h-4 w-4 text-muted-foreground' />
                  <div>
                    <p className='text-sm text-muted-foreground'>Drift Detected</p>
                    <p className='text-xl font-bold'>{promMetrics.drift_detected_imaging === 1 || promMetrics.drift_detected_clinical === 1 ? 'Yes' : 'No'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      <div className='grid grid-cols-1 md:grid-cols-2 gap-5'>
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
      </div>

      {getMetricsRadarData().length > 0 && (
        <div className='rounded-lg border bg-card p-4'>
          <h3 className='font-semibold mb-4'>Model Performance Comparison</h3>
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
        </div>
      )}

      <div className='grid grid-cols-1 md:grid-cols-2 gap-5'>
        <div className='rounded-lg border bg-card p-4'>
          <div className='flex items-center gap-2 mb-4'>
            <Activity className='h-5 w-5' />
            <h3 className='font-semibold'>Training Jobs</h3>
          </div>
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
        </div>

        <div className='rounded-lg border bg-card p-4'>
          <div className='flex items-center gap-2 mb-4'>
            <TrendingUp className='h-5 w-5' />
            <h3 className='font-semibold'>Drift Status</h3>
          </div>
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
        </div>
      </div>

      {hasEvidentlyData && (
        <div className='rounded-lg border bg-card p-4'>
          <div className='flex items-center gap-2 mb-4'>
            <FileText className='h-5 w-5' />
            <h3 className='font-semibold'>Evidently Drift Analysis</h3>
          </div>
          <div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
            <Card>
              <CardContent className='pt-4'>
                <p className='text-sm text-muted-foreground'>Imaging Shift Score</p>
                <p className='text-xl font-bold'>{evidentlyShiftImaging.toFixed(4)}</p>
                <p className='text-xs text-muted-foreground mt-1'>Threshold: 0.5</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className='pt-4'>
                <p className='text-sm text-muted-foreground'>Clinical Shift Score</p>
                <p className='text-xl font-bold'>{evidentlyShiftClinical.toFixed(4)}</p>
                <p className='text-xs text-muted-foreground mt-1'>Threshold: 0.5</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className='pt-4'>
                <p className='text-sm text-muted-foreground'>Imaging Features Drifted</p>
                <p className='text-xl font-bold'>{evidentlyFeaturesImaging}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className='pt-4'>
                <p className='text-sm text-muted-foreground'>Clinical Features Drifted</p>
                <p className='text-xl font-bold'>{evidentlyFeaturesClinical}</p>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      <div className='grid grid-cols-1 md:grid-cols-2 gap-5'>
        <div className='rounded-lg border bg-card p-4'>
          <h3 className='font-semibold mb-4'>Drift History (PSI)</h3>
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
        </div>

        <div className='rounded-lg border bg-card p-4'>
          <div className='flex items-center justify-between mb-4'>
            <div className='flex items-center gap-2'>
              <Database className='h-5 w-5' />
              <h3 className='font-semibold'>Feature Store</h3>
            </div>
            <Badge variant='secondary'>{features.length}</Badge>
          </div>
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
        </div>
      </div>
    </PageContainer>
  );
}
