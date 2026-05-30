'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import {
  getSystemHealth,
  getMLOpsStatus,
  getMLOpsDriftStatus,
  getLLMOpsRagStatus,
  startMLOpsTraining,
  triggerLLMOpsReindex,
  triggerMLOpsDriftRetrain,
  type SystemHealth,
  type MLOpsJob,
  type MLOpsDriftStatus,
  type LLMOpsRagStatus,
} from '@/lib/api';
import {
  Activity,
  AlertTriangle,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Loader2,
  Server,
  Wifi,
  Zap,
} from 'lucide-react';

const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';

interface SystemMetrics {
  cpu: { usage_percent: number };
  memory: { total_gb: number; used_gb: number; available_gb: number; usage_percent: number };
  disk: { total_gb: number; used_gb: number; free_gb: number; usage_percent: number };
  load: number;
  network: { rx_mbps: number; tx_mbps: number };
}

interface GPUInfo {
  name: string;
  utilization_pct: number;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_pct: number;
  temperature_c: number;
  power_w: number;
}

interface AlertItem {
  labels: Record<string, string>;
  annotations: Record<string, string>;
  starts_at: string;
  status: string;
  fingerprint?: string;
}

const healthBadge = (status: string | null | undefined) => {
  if (!status) return <Badge variant="outline" className="text-muted-foreground">checking</Badge>;
  if (status === 'healthy') return <Badge className="bg-green-500 hover:bg-green-600 text-white">healthy</Badge>;
  return <Badge variant="destructive">unreachable</Badge>;
};

const statusBadge = (status: string | undefined) => {
  switch (status) {
    case 'completed': return <Badge className="bg-green-500 hover:bg-green-600 text-white">completed</Badge>;
    case 'running': return <Badge className="bg-blue-500 hover:bg-blue-600 text-white">running</Badge>;
    case 'failed': return <Badge variant="destructive">failed</Badge>;
    case 'stable': return <Badge className="bg-green-500 hover:bg-green-600 text-white">stable</Badge>;
    case 'warning': return <Badge className="bg-yellow-500 hover:bg-yellow-600 text-white">warning</Badge>;
    case 'drifted': return <Badge variant="destructive">drifted</Badge>;
    default: return <Badge variant="outline">{status || 'unknown'}</Badge>;
  }
};

export default function EngineeringDashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [gpu, setGpu] = useState<{ gpus: GPUInfo[] } | null>(null);
  const [jobs, setJobs] = useState<MLOpsJob[]>([]);
  const [drift, setDrift] = useState<MLOpsDriftStatus | null>(null);
  const [rag, setRag] = useState<LLMOpsRagStatus | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [driftRetraining, setDriftRetraining] = useState(false);

  const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const fetchData = useCallback(async () => {
    const [healthData, metricsData, gpuData, statusData, driftData, ragData] = await Promise.all([
      getSystemHealth().catch(() => null),
      fetch(`${API}/api/v1/system/metrics`, { credentials: 'include' }).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`${API}/api/v1/system/gpu`, { credentials: 'include' }).then(r => r.ok ? r.json() : null).catch(() => null),
      getMLOpsStatus().catch(() => null),
      getMLOpsDriftStatus('imaging').catch(() => null),
      getLLMOpsRagStatus().catch(() => null),
    ]);

    setHealth(healthData);
    setMetrics(metricsData);
    setGpu(gpuData);
    if (statusData) setJobs(statusData.jobs ?? []);
    setDrift(driftData);
    setRag(ragData);

    try {
      const res = await fetch(`${MLOPS_BASE}/metrics/alerts`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts ?? []);
      }
    } catch {}

    setLoading(false);
  }, [API]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleTrain = async () => {
    setTraining(true);
    try {
      const data = await startMLOpsTraining('imaging');
      toast.success(`Training started: ${data.job_id}`);
      fetchData();
    } catch (err) {
      toast.error(`Failed to start training: ${String(err).slice(0, 120)}`);
    } finally {
      setTraining(false);
    }
  };

  const handleReindex = async () => {
    setReindexing(true);
    try {
      const data = await triggerLLMOpsReindex();
      toast.success(`Reindex triggered: ${data.job_id}`);
      fetchData();
    } catch (err) {
      toast.error(`Failed to reindex RAG: ${String(err).slice(0, 120)}`);
    } finally {
      setReindexing(false);
    }
  };

  const handleDriftRetrain = async () => {
    setDriftRetraining(true);
    try {
      const data = await triggerMLOpsDriftRetrain('imaging');
      toast.success(`Drift retrain: ${data.message}`);
      fetchData();
    } catch (err) {
      toast.error(`Failed to trigger drift retrain: ${String(err).slice(0, 120)}`);
    } finally {
      setDriftRetraining(false);
    }
  };

  const latestImagingJob = jobs
    .filter(j => j.pipeline === 'imaging')
    .sort((a, b) => new Date(b.started_at || 0).getTime() - new Date(a.started_at || 0).getTime())[0];

  return (
    <PageContainer>
      <div className="flex flex-1 flex-col gap-6 min-h-0">
        <div className="rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm">
          <div className="relative z-10 space-y-1">
            <h1 className="text-xl font-bold tracking-tight">Engineering Dashboard</h1>
            <p className="max-w-xl text-sm text-white/70">System monitoring, model performance, and infrastructure</p>
          </div>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <Zap className="h-4 w-4" /> Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-3">
            <Button onClick={handleTrain} disabled={training} size="sm">
              {training && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}
              {training ? 'Starting...' : 'Train Imaging'}
            </Button>
            <Button onClick={handleReindex} disabled={reindexing} variant="secondary" size="sm">
              {reindexing && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}
              {reindexing ? 'Reindexing...' : 'Reindex RAG'}
            </Button>
            <Button onClick={handleDriftRetrain} disabled={driftRetraining} variant="secondary" size="sm">
              {driftRetraining && <Loader2 className="mr-1 h-4 w-4 animate-spin" />}
              {driftRetraining ? 'Checking...' : 'Drift Retrain'}
            </Button>
          </CardContent>
        </Card>

        <div>
          <h2 className="text-sm font-semibold text-muted-foreground mb-3">Service Health</h2>
          <div className="grid gap-4 grid-cols-2 sm:grid-cols-3 md:grid-cols-5">
            {(['backend', 'mlops', 'llmops', 'redis', 'postgres'] as const).map(svc => (
              <Card key={svc}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-xs font-medium text-muted-foreground capitalize">{svc}</CardTitle>
                </CardHeader>
                <CardContent>{healthBadge(health?.[svc])}</CardContent>
              </Card>
            ))}
          </div>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-muted-foreground mb-3">System Resources</h2>
          <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
            <Card>
              <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                <CardTitle className="text-xs font-medium text-muted-foreground">CPU</CardTitle>
                <Cpu className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.cpu?.usage_percent?.toFixed(0) ?? '-'}%</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                <CardTitle className="text-xs font-medium text-muted-foreground">Memory</CardTitle>
                <HardDrive className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {metrics?.memory ? `${metrics.memory.used_gb.toFixed(1)} GB` : '-'}
                </div>
                <p className="text-xs text-muted-foreground">
                  of {metrics?.memory?.total_gb.toFixed(0) ?? '-'} GB ({metrics?.memory?.usage_percent.toFixed(0) ?? '-'}%)
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                <CardTitle className="text-xs font-medium text-muted-foreground">Disk</CardTitle>
                <Database className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics?.disk?.usage_percent?.toFixed(0) ?? '-'}%</div>
                <p className="text-xs text-muted-foreground">
                  {metrics?.disk?.used_gb.toFixed(0) ?? '-'} / {metrics?.disk?.total_gb.toFixed(0) ?? '-'} GB
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                <CardTitle className="text-xs font-medium text-muted-foreground">GPU</CardTitle>
                <Gauge className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                {gpu?.gpus?.length ? (
                  <>
                    <div className="text-2xl font-bold">{gpu.gpus[0].utilization_pct.toFixed(0)}%</div>
                    <p className="text-xs text-muted-foreground">
                      {gpu.gpus[0].temperature_c.toFixed(0)}&deg;C &middot; {gpu.gpus[0].power_w.toFixed(0)}W &middot; {gpu.gpus[0].name}
                    </p>
                  </>
                ) : (
                  <div className="text-2xl font-bold text-muted-foreground">N/A</div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" /> Pipeline Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Imaging Pipeline</span>
                {latestImagingJob ? statusBadge(latestImagingJob.status) : statusBadge(undefined)}
              </div>
              {latestImagingJob?.job_id && (
                <p className="text-xs text-muted-foreground">Job: {latestImagingJob.job_id.slice(0, 8)}&hellip;</p>
              )}
              <div className="flex items-center justify-between">
                <span className="text-sm">Data Drift</span>
                {drift ? statusBadge(drift.status) : statusBadge(undefined)}
              </div>
              {drift?.current_psi !== undefined && (
                <p className="text-xs text-muted-foreground">PSI: {drift.current_psi.toFixed(3)}</p>
              )}
              <div className="flex items-center justify-between">
                <span className="text-sm">RAG Pipeline</span>
                {rag ? (
                  <Badge variant="outline">{rag.total_documents} docs</Badge>
                ) : statusBadge(undefined)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" /> Recent Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              {alerts.length === 0 ? (
                <p className="text-sm text-muted-foreground">No active alerts</p>
              ) : (
                <div className="space-y-2 max-h-[200px] overflow-y-auto">
                  {alerts.slice(0, 10).map((a, i) => (
                    <div key={a.fingerprint ?? i} className="flex items-start gap-2 text-sm">
                      {a.status === 'firing' ? (
                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500" />
                      ) : (
                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-yellow-500" />
                      )}
                      <div className="min-w-0">
                        <p className="truncate font-medium">{a.labels?.alertname || 'Alert'}</p>
                        <p className="truncate text-xs text-muted-foreground">{a.annotations?.summary || a.annotations?.description || ''}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div>
          <h2 className="text-sm font-semibold text-muted-foreground mb-3">Quick Links</h2>
          <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
            <Link href="/dashboard/engineering/mlops">
              <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
                <CardContent className="flex flex-col items-center gap-2 py-4">
                  <Server className="h-6 w-6 text-cyan-500" />
                  <span className="text-sm font-medium">MLOps Monitor</span>
                </CardContent>
              </Card>
            </Link>
            <Link href="/dashboard/engineering/llmops">
              <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
                <CardContent className="flex flex-col items-center gap-2 py-4">
                  <Activity className="h-6 w-6 text-purple-500" />
                  <span className="text-sm font-medium">LLMOps Monitor</span>
                </CardContent>
              </Card>
            </Link>
            <Link href="/dashboard/engineering/models">
              <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
                <CardContent className="flex flex-col items-center gap-2 py-4">
                  <Cpu className="h-6 w-6 text-amber-500" />
                  <span className="text-sm font-medium">AI Models</span>
                </CardContent>
              </Card>
            </Link>
            <Link href="/dashboard/engineering/system">
              <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
                <CardContent className="flex flex-col items-center gap-2 py-4">
                  <Wifi className="h-6 w-6 text-green-500" />
                  <span className="text-sm font-medium">System Stats</span>
                </CardContent>
              </Card>
            </Link>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}
