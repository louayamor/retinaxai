'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Activity, 
  Server, 
  Brain, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle2,
  Clock,
  Loader2,
  RefreshCw,
  Database,
  Gauge
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
  BarChart,
  Bar,
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
  metrics?: { accuracy?: number; loss?: number; val_accuracy?: number };
  error?: string;
}

interface DriftStatus {
  pipeline: string;
  status: 'stable' | 'warning' | 'drifted';
  psi_threshold: number;
  current_psi?: number;
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
  training?: { total_runs?: number; active_jobs?: number };
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
    setLoading(true);
    try {
      const [statusRes, driftRes, featuresRes, metricsRes] = await Promise.all([
        fetch(`${MLOPS_BASE}/api/status`).then(r => r.json()).catch(() => ({ job: null, jobs: [] })),
        fetch(`${MLOPS_BASE}/api/drift/history`).then(r => r.json()).catch(() => ({ history: [] })),
        fetch(`${MLOPS_BASE}/api/features/list`).then(r => r.json()).catch(() => ({ features: [], total: 0 })),
        fetch(`${MLOPS_BASE}/api/metrics`).then(r => r.json()).catch(() => ({})),
      ]);

      setJobs(statusRes.jobs || []);
      setDriftHistory(driftRes.history || []);
      setFeatures(featuresRes.features || []);
      setMetrics(metricsRes);

      const driftStatusMap: Record<string, DriftStatus> = {};
      for (const pipeline of PIPELINES) {
        try {
          const res = await fetch(`${MLOPS_BASE}/api/drift/status/${pipeline}`);
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
      const res = await fetch(`${MLOPS_BASE}/api/automation/drift-retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pipeline }),
      });
      const data = await res.json();
      toast.success(`Retraining triggered: ${data.job_id}`);
      fetchData();
    } catch (error) {
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
      case 'drifted': return 'bg-red-500';
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
      { metric: 'Accuracy', imaging: im.accuracy ? im.accuracy * 100 : 0, clinical: cl.accuracy ? cl.accuracy * 100 : 0 },
      { metric: 'Kappa', imaging: im.quadratic_weighted_kappa ? im.quadratic_weighted_kappa * 100 : 0, clinical: cl.quadratic_weighted_kappa ? cl.quadratic_weighted_kappa * 100 : 0 },
      { metric: 'AUC', imaging: im.roc_auc_macro ? im.roc_auc_macro * 100 : 0, clinical: cl.roc_auc_macro ? cl.roc_auc_macro * 100 : 0 },
      { metric: 'Precision', imaging: im.precision_macro ? im.precision_macro * 100 : 0, clinical: cl.precision_macro ? cl.precision_macro * 100 : 0 },
      { metric: 'Recall', imaging: im.recall_macro ? im.recall_macro * 100 : 0, clinical: cl.recall_macro ? cl.recall_macro * 100 : 0 },
    ];
  };

  if (loading && !metrics) {
    return (
      <PageContainer>
        <div className="flex items-center justify-center h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">MLOps Monitor</h1>
            <p className="text-muted-foreground">Training pipelines, drift detection, and model performance</p>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="drift">Drift Detection</TabsTrigger>
            <TabsTrigger value="features">Feature Store</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {/* Model Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Gauge className="h-5 w-5" />
                  Model Performance Metrics
                </CardTitle>
                <CardDescription>Current production model performance</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2">
                  {/* Imaging Model */}
                  <div className="space-y-3 p-4 rounded-lg border bg-gradient-to-r from-blue-50/50 to-cyan-50/30">
                    <h3 className="font-semibold flex items-center gap-2">
                      <Brain className="h-4 w-4 text-blue-600" />
                      EfficientNet-B3 (Imaging)
                    </h3>
                    {metrics?.imaging ? (
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div><span className="text-muted-foreground">Accuracy:</span> {(metrics.imaging.accuracy! * 100).toFixed(1)}%</div>
                        <div><span className="text-muted-foreground">Kappa:</span> {metrics.imaging.quadratic_weighted_kappa?.toFixed(2)}</div>
                        <div><span className="text-muted-foreground">AUC:</span> {(metrics.imaging.roc_auc_macro! * 100).toFixed(1)}%</div>
                        <div><span className="text-muted-foreground">Samples:</span> {metrics.imaging.num_samples || 'N/A'}</div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No metrics available</p>
                    )}
                  </div>
                  {/* Clinical Model */}
                  <div className="space-y-3 p-4 rounded-lg border bg-gradient-to-r from-purple-50/50 to-pink-50/30">
                    <h3 className="font-semibold flex items-center gap-2">
                      <Brain className="h-4 w-4 text-purple-600" />
                      XGBoost (Clinical)
                    </h3>
                    {metrics?.clinical ? (
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div><span className="text-muted-foreground">Accuracy:</span> {(metrics.clinical.accuracy! * 100).toFixed(1)}%</div>
                        <div><span className="text-muted-foreground">Kappa:</span> {metrics.clinical.quadratic_weighted_kappa?.toFixed(2)}</div>
                        <div><span className="text-muted-foreground">AUC:</span> {(metrics.clinical.roc_auc_macro! * 100).toFixed(1)}%</div>
                        <div><span className="text-muted-foreground">Samples:</span> {metrics.clinical.num_samples || 'N/A'}</div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No metrics available</p>
                    )}
                  </div>
                </div>
                
                {getMetricsRadarData().length > 0 && (
                  <div className="mt-6 h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={getMetricsRadarData()}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} />
                        <Radar name="Imaging" dataKey="imaging" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
                        <Radar name="Clinical" dataKey="clinical" stroke="#9333ea" fill="#9333ea" fillOpacity={0.3} />
                        <Legend />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Active Training Jobs</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{jobs.filter(j => j.status === 'running').length}</div>
                  <p className="text-xs text-muted-foreground">of {jobs.length} total jobs</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Feature Store</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{features.length}</div>
                  <p className="text-xs text-muted-foreground">cached features</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Drift Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {Object.values(driftStatus).some(d => d.status === 'drifted') ? (
                      <>
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                        <span className="text-red-500 font-medium">Drifted</span>
                      </>
                    ) : Object.values(driftStatus).some(d => d.status === 'warning') ? (
                      <>
                        <AlertTriangle className="h-5 w-5 text-yellow-500" />
                        <span className="text-yellow-500 font-medium">Warning</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                        <span className="text-green-500 font-medium">Stable</span>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Training Tab */}
          <TabsContent value="training" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Training Jobs
                </CardTitle>
                <CardDescription>Recent and active training pipelines</CardDescription>
              </CardHeader>
              <CardContent>
                {jobs.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No training jobs found</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {jobs.slice(0, 10).map((job) => (
                      <div key={job.job_id} className="flex items-center justify-between p-3 rounded-lg border">
                        <div className="space-y-1">
                          <div className="font-mono text-sm">{job.job_id.slice(0, 8)}...</div>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <span className="capitalize">{job.pipeline}</span>
                            <span>•</span>
                            <span>{job.started_at ? new Date(job.started_at).toLocaleString() : 'Pending'}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge className={getStatusColor(job.status)}>{job.status}</Badge>
                          {job.metrics?.accuracy && (
                            <p className="text-xs text-muted-foreground mt-1">Acc: {(job.metrics.accuracy * 100).toFixed(1)}%</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Drift Tab */}
          <TabsContent value="drift" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              {PIPELINES.map((pipeline) => {
                const status = driftStatus[pipeline];
                return (
                  <Card key={pipeline}>
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span className="capitalize">{pipeline} Pipeline</span>
                        <Badge className={getStatusColor(status?.status || 'unknown')}>
                          {status?.status || 'Unknown'}
                        </Badge>
                      </CardTitle>
                      <CardDescription>
                        Last checked: {status?.last_checked ? new Date(status.last_checked).toLocaleString() : 'Never'}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Current PSI</p>
                          <p className="text-2xl font-bold">{status?.current_psi?.toFixed(3) || 'N/A'}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Threshold</p>
                          <p className="text-2xl font-bold">{status?.psi_threshold || 0.3}</p>
                        </div>
                      </div>
                      {status?.features_drifted && status.features_drifted.length > 0 && (
                        <div>
                          <p className="text-sm font-medium mb-2">Drifted Features:</p>
                          <div className="flex flex-wrap gap-1">
                            {status.features_drifted.map((f) => (
                              <Badge key={f} variant="outline" className="text-xs">{f}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => handleRetrain(pipeline)}
                        disabled={retraining === pipeline}
                      >
                        {retraining === pipeline ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
                        Trigger Retraining
                      </Button>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            {/* Drift History Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Drift History (PSI over time)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={getDriftChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" tick={{ fontSize: 12 }} />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="imaging" stroke="#2563eb" strokeWidth={2} name="Imaging" />
                      <Line type="monotone" dataKey="clinical" stroke="#9333ea" strokeWidth={2} name="Clinical" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Features Tab */}
          <TabsContent value="features" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Feature Store
                </CardTitle>
                <CardDescription>Cached feature values for inference</CardDescription>
              </CardHeader>
              <CardContent>
                {features.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No features in store</p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[500px] overflow-y-auto">
                    {features.map((feat, i) => (
                      <div key={i} className="flex items-center justify-between p-2 rounded border text-sm">
                        <div>
                          <span className="font-mono text-xs">{feat.key}</span>
                          <span className="text-muted-foreground mx-2">:</span>
                          <span className="text-xs">{feat.value}</span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {new Date(feat.created_at).toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageContainer>
  );
}