'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle2,
  Clock,
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

const MetricRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex justify-between text-xs">
    <span className="text-muted-foreground">{label}</span>
    <span className="font-medium">{value}</span>
  </div>
);

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
      fetchData();
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
        <div className="flex items-center justify-center h-[60vh]">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
        </div>
      </PageContainer>
    );
  }

  const activeJobs = jobs.filter(j => j.status === 'running').length;
  const overallDrift = Object.values(driftStatus).some(d => d.status === 'drift_detected');

  return (
    <PageContainer>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">MLOps Monitor</h1>
            <p className="text-sm text-muted-foreground">Training pipelines, drift detection, and model performance</p>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Activity className="h-3 w-3" /> Active Jobs
            </div>
            <div className="text-xl font-bold">{activeJobs}<span className="text-sm font-normal text-muted-foreground">/{jobs.length}</span></div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Database className="h-3 w-3" /> Feature Store
            </div>
            <div className="text-xl font-bold">{features.length}</div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <TrendingUp className="h-3 w-3" /> Drift Status
            </div>
            <div className="flex items-center gap-1">
              {overallDrift ? (
                <><AlertTriangle className="h-4 w-4 text-red-500" /><span className="text-sm font-medium text-red-500">Drifted</span></>
              ) : (
                <><CheckCircle2 className="h-4 w-4 text-green-500" /><span className="text-sm font-medium text-green-500">Stable</span></>
              )}
            </div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Brain className="h-3 w-3" /> Imaging Acc
            </div>
            <div className="text-xl font-bold">
              {metrics?.imaging?.accuracy != null ? `${(metrics.imaging.accuracy * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </Card>
        </div>

        {/* Model Metrics - Compact Grid */}
        <div className="grid grid-cols-2 gap-3">
          {metrics?.imaging && (
            <Card className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="h-4 w-4 text-blue-600" />
                <span className="font-semibold text-sm">EfficientNet-B3 (Imaging)</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                <MetricRow label="Acc" value={metrics.imaging.accuracy != null ? `${(metrics.imaging.accuracy * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Kappa" value={metrics.imaging.quadratic_weighted_kappa?.toFixed(2) || 'N/A'} />
                <MetricRow label="AUC" value={metrics.imaging.roc_auc_macro != null ? `${(metrics.imaging.roc_auc_macro * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Prec" value={metrics.imaging.precision_macro != null ? `${(metrics.imaging.precision_macro * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Rec" value={metrics.imaging.recall_macro != null ? `${(metrics.imaging.recall_macro * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Samples" value={metrics.imaging.num_samples?.toString() || 'N/A'} />
              </div>
            </Card>
          )}
          {metrics?.clinical && (
            <Card className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="h-4 w-4 text-purple-600" />
                <span className="font-semibold text-sm">XGBoost (Clinical)</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                <MetricRow label="Acc" value={metrics.clinical.accuracy != null ? `${(metrics.clinical.accuracy * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Kappa" value={metrics.clinical.quadratic_weighted_kappa?.toFixed(2) || 'N/A'} />
                <MetricRow label="AUC" value={metrics.clinical.roc_auc_macro != null ? `${(metrics.clinical.roc_auc_macro * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Prec" value={metrics.clinical.precision_macro != null ? `${(metrics.clinical.precision_macro * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Rec" value={metrics.clinical.recall_macro != null ? `${(metrics.clinical.recall_macro * 100).toFixed(1)}%` : 'N/A'} />
                <MetricRow label="Samples" value={metrics.clinical.num_samples?.toString() || 'N/A'} />
              </div>
            </Card>
          )}
        </div>

        {/* Radar Chart */}
        {getMetricsRadarData().length > 0 && (
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm">Model Performance Comparison</CardTitle>
            </CardHeader>
            <div className="h-[180px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={getMetricsRadarData()}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 8 }} />
                  <Radar name="Imaging" dataKey="imaging" stroke="#2563eb" fill="#2563eb" fillOpacity={0.2} />
                  <Radar name="Clinical" dataKey="clinical" stroke="#9333ea" fill="#9333ea" fillOpacity={0.2} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}

        {/* Training Jobs & Drift Status - Side by Side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Training Jobs */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Training Jobs
                <Badge variant="secondary" className="ml-auto text-xs">{jobs.length}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {jobs.length === 0 ? (
                <p className="text-xs text-muted-foreground py-4 text-center">No training jobs</p>
              ) : (
                <div className="space-y-1 max-h-[200px] overflow-y-auto">
                  {jobs.slice(0, 8).map((job) => (
                    <div key={job.job_id} className="flex items-center justify-between p-2 rounded border text-xs">
                      <div>
                        <span className="font-mono text-[10px]">{job.job_id.slice(0, 8)}</span>
                        <span className="text-muted-foreground ml-1 capitalize">{job.pipeline}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground text-[10px]">
                          {job.started_at ? new Date(job.started_at).toLocaleDateString() : 'Pending'}
                        </span>
                        <Badge className={`${getStatusColor(job.status)} text-[10px] py-0`}>{job.status}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Drift Status */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Drift Status
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 space-y-2">
              {PIPELINES.map((pipeline) => {
                const status = driftStatus[pipeline];
                return (
                  <div key={pipeline} className="p-2 rounded border">
                    <div className="flex items-center justify-between mb-1">
                      <span className="capitalize text-xs font-medium">{pipeline}</span>
                      <Badge className={`${getStatusColor(status?.status || 'unknown')} text-[10px] py-0`}>
                        {status?.status || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      <div><span className="text-muted-foreground">PSI:</span> <span className="font-medium">{status?.overall_psi?.toFixed(3) || 'N/A'}</span></div>
                      <div><span className="text-muted-foreground">Threshold:</span> <span className="font-medium">{status?.psi_threshold?.toFixed(1) || '0.3'}</span></div>
                    </div>
                    {status?.features_drifted && status.features_drifted.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {status.features_drifted.slice(0, 3).map((f) => (
                          <Badge key={f} variant="outline" className="text-[10px] py-0">{f}</Badge>
                        ))}
                      </div>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full mt-1 h-6 text-xs"
                      onClick={() => handleRetrain(pipeline)}
                      disabled={retraining === pipeline}
                    >
                      {retraining === pipeline ? <div className="animate-spin h-3 w-3 border border-primary border-t-transparent rounded-full" /> : <RefreshCw className="h-3 w-3" />}
                      <span className="ml-1">Retrain</span>
                    </Button>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </div>

        {/* Drift History & Features - Side by Side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Drift History Chart */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm">Drift History (PSI)</CardTitle>
            </CardHeader>
            <div className="h-[140px]">
              {getDriftChartData().length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getDriftChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" tick={{ fontSize: 8 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 8 }} />
                    <Tooltip wrapperStyle={{ fontSize: 10 }} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line type="monotone" dataKey="imaging" stroke="#2563eb" strokeWidth={1.5} name="Imaging" dot={false} />
                    <Line type="monotone" dataKey="clinical" stroke="#9333ea" strokeWidth={1.5} name="Clinical" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-xs text-muted-foreground flex items-center justify-center h-full">No drift history</p>
              )}
            </div>
          </Card>

          {/* Feature Store */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Database className="h-4 w-4" />
                Feature Store
                <Badge variant="secondary" className="ml-auto text-xs">{features.length}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {features.length === 0 ? (
                <p className="text-xs text-muted-foreground py-4 text-center">No features in store</p>
              ) : (
                <div className="space-y-1 max-h-[140px] overflow-y-auto">
                  {features.slice(0, 10).map((feat, i) => (
                    <div key={i} className="flex items-center justify-between p-1.5 rounded border text-xs">
                      <span className="font-mono text-[10px] truncate max-w-[120px]">{feat.key}</span>
                      <span className="text-muted-foreground text-[10px] truncate max-w-[80px]">
                        {feat.created_at ? new Date(feat.created_at).toLocaleDateString() : 'N/A'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </PageContainer>
  );
}
