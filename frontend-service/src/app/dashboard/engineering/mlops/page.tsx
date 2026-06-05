'use client';

import { useEffect, useState, useMemo, useCallback } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
import { StatsRow } from '@/components/ui/stats-row';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { StatsCard } from '@/components/ui/stats-card';
import {
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle2,
  Database,
  TrendingUp,
  Cpu,
  Gauge,
  FileText,
  HardDrive,
  Wifi,
  Waves,
  ExternalLink,
  BarChart3,
} from 'lucide-react';
import { toast } from 'sonner';
import { useWebSocket } from '@/hooks/use-websocket';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
import { getSystemMetricsRange, GRAFANA_KIOSK_URL, GRAFANA_PROXY_BASE, type PrometheusRangeResult } from '@/lib/api';

interface Metrics {
  imaging?: { accuracy?: number; quadratic_weighted_kappa?: number; roc_auc_macro?: number; precision_macro?: number; recall_macro?: number; num_samples?: number };
  clinical?: { accuracy?: number; quadratic_weighted_kappa?: number; roc_auc_macro?: number; precision_macro?: number; recall_macro?: number; num_samples?: number };
  training_summary?: {
    epoch_log?: Array<{ epoch?: number; val_qwk?: number }>;
    best_epoch?: number;
  };
}

interface PrometheusMetrics {
  training_runs_total: number;
  active_training_jobs: number;
  best_val_accuracy_imaging: number | null;
  drift_detected_imaging: number | null;
  evidently_dataset_shift_imaging: number | null;
  evidently_features_drifted_imaging: number | null;
  inference_latency_p95: number | null;
  gpu_utilization: number | null;
  gpu_memory_used_bytes?: number | null;
  gpu_memory_total_bytes?: number | null;
  cpu_utilization?: number | null;
  memory_total_bytes?: number | null;
  memory_available_bytes?: number | null;
  disk_total_bytes?: number | null;
  disk_free_bytes?: number | null;
  network_receive_bytes_per_second?: number | null;
  network_transmit_bytes_per_second?: number | null;
}

interface MonitorSnapshot {
  generated_at: string;
  metrics: { imaging?: Metrics['imaging'] | null; clinical?: Metrics['clinical'] | null };
  training_summary?: Metrics['training_summary'] | null;
  prometheus: PrometheusMetrics | null;
}

const MAX_HISTORY = 40;
const EMPTY_METRICS: Metrics = { imaging: undefined, clinical: undefined, training_summary: undefined };

const PROMQL = {
  cpu: '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
  memory: '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100',
  disk: '(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}) / node_filesystem_size_bytes{mountpoint="/"} * 100',
  netRx: 'rate(node_network_receive_bytes_total{device!="lo"}[5m]) * 8 / 1024 / 1024',
  netTx: 'rate(node_network_transmit_bytes_total{device!="lo"}[5m]) * 8 / 1024 / 1024',
  gpuUtil: 'nvidia_gpu_utilization',
  gpuMem: 'nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100',
};

const TIME_RANGES = ['6h', '24h', '7d'] as const;
type TimeRange = typeof TIME_RANGES[number];

const GRAFANA_VIEW_URL = `${GRAFANA_PROXY_BASE}/d/retinaxai-mlops-dashboard/retinaxai-mlops-dashboard?kiosk`;

const GRAFANA_PANEL_LINKS = [
  { id: 2, title: 'Training Runs' },
  { id: 12, title: 'Accuracy' },
  { id: 13, title: 'F1 Score' },
  { id: 8, title: 'Loss' },
  { id: 6, title: 'QWK Trend' },
  { id: 7, title: 'Drift PSI' },
  { id: 3, title: 'Latency' },
  { id: 19, title: 'GPU Util' },
  { id: 18, title: 'GPU Memory' },
] as const;

type RangeData = Record<string, PrometheusRangeResult>;

function formatTimestamp(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function toChartData(results: PrometheusRangeResult): Array<{ time: string; value: number }> {
  if (!results.length) return [];
  return results[0].values.map(([ts, v]) => ({
    time: formatTimestamp(ts),
    value: parseFloat(v),
  }));
}

export default function MLOpsPage() {
  const [snapshot, setSnapshot] = useState<MonitorSnapshot | null>(null);
  const [metrics, setMetrics] = useState<Metrics>(EMPTY_METRICS);
  const [promMetrics, setPromMetrics] = useState<PrometheusMetrics | null>(null);
  const [promHistory, setPromHistory] = useState<PrometheusMetrics[]>([]);
  const [driftHistory, setDriftHistory] = useState<
    Array<{ generated_at: string; overall_psi: number; status: string; current_samples: number; reference_samples: number }>
  >([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('ml');

  const [timeRange, setTimeRange] = useState<TimeRange>('6h');
  const [rangeData, setRangeData] = useState<RangeData>({});
  const [rangeLoading, setRangeLoading] = useState(false);

  const { subscribe } = useWebSocket();
  useEffect(() => {
    const unsubscribe = subscribe('mlops.monitor', (data) => {
      const incoming = data as MonitorSnapshot;
      setSnapshot(incoming);
      const nextMetrics: Metrics = {
        imaging: incoming.metrics?.imaging ?? undefined,
        clinical: incoming.metrics?.clinical ?? undefined,
        training_summary: incoming.training_summary ?? undefined,
      };
      setMetrics(nextMetrics);
      setPromMetrics(incoming.prometheus ?? null);
      const promData = incoming.prometheus;
      if (promData) {
        setPromHistory((prev) => [...prev.slice(-MAX_HISTORY + 1), promData]);
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, [subscribe]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (!snapshot) {
        toast.error('Waiting for MLOps live feed');
        setLoading(false);
      }
    }, 8000);
    return () => clearTimeout(timeout);
  }, [snapshot]);

  const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const MLOPS = `${API}/api/v1/system/mlops`;

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${MLOPS}/drift/history?limit=50`, { signal: controller.signal })
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.history) setDriftHistory(data.history);
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const fetchRangeData = useCallback(async (range: TimeRange) => {
    setRangeLoading(true);
    const queries: [string, string][] = [
      ['cpu', PROMQL.cpu],
      ['memory', PROMQL.memory],
      ['disk', PROMQL.disk],
      ['netRx', PROMQL.netRx],
      ['netTx', PROMQL.netTx],
    ];
    const results: RangeData = {};
    await Promise.allSettled(
      queries.map(async ([key, q]) => {
        results[key] = await getSystemMetricsRange(q, range, range === '7d' ? '30m' : '10m');
      })
    );
    setRangeData(results);
    setRangeLoading(false);
  }, []);

  useEffect(() => {
    fetchRangeData(timeRange);
  }, [timeRange, fetchRangeData]);

  const getMetricsRadarData = useMemo(() => {
    const im = metrics.imaging || {};
    return [
      { metric: 'Acc', imaging: im.accuracy ? im.accuracy * 100 : 0 },
      { metric: 'Kappa', imaging: im.quadratic_weighted_kappa ? im.quadratic_weighted_kappa * 100 : 0 },
      { metric: 'AUC', imaging: im.roc_auc_macro ? im.roc_auc_macro * 100 : 0 },
      { metric: 'Prec', imaging: im.precision_macro ? im.precision_macro * 100 : undefined },
      { metric: 'Rec', imaging: im.recall_macro ? im.recall_macro * 100 : undefined },
    ];
  }, [metrics]);

  const getQwkHistoryData = useMemo(() => {
    const epochLog = metrics.training_summary?.epoch_log || [];
    return epochLog
      .filter((ep) => typeof ep.epoch === 'number' && typeof ep.val_qwk === 'number')
      .map((ep) => ({
        epoch: ep.epoch as number,
        qwk: (ep.val_qwk as number) * 100,
      }));
  }, [metrics]);

  const promSparkData = useMemo(
    () =>
      promHistory.map((entry, index) => ({
        index,
        gpu: entry.gpu_utilization ?? 0,
        cpu: entry.cpu_utilization ?? 0,
        mem: entry.memory_total_bytes && entry.memory_available_bytes
          ? ((entry.memory_total_bytes - entry.memory_available_bytes) / entry.memory_total_bytes) * 100
          : 0,
      })),
    [promHistory]
  );

  const gpuMemoryPercent = promMetrics?.gpu_memory_total_bytes && promMetrics?.gpu_memory_used_bytes
    ? (promMetrics.gpu_memory_used_bytes / promMetrics.gpu_memory_total_bytes) * 100
    : null;

  const memoryUsedPercent = promMetrics?.memory_total_bytes && promMetrics?.memory_available_bytes
    ? ((promMetrics.memory_total_bytes - promMetrics.memory_available_bytes) / promMetrics.memory_total_bytes) * 100
    : null;

  const diskUsedPercent = promMetrics?.disk_total_bytes && promMetrics?.disk_free_bytes
    ? ((promMetrics.disk_total_bytes - promMetrics.disk_free_bytes) / promMetrics.disk_total_bytes) * 100
    : null;

  const cpuChart = useMemo(() => toChartData(rangeData.cpu || []), [rangeData.cpu]);
  const memoryChart = useMemo(() => toChartData(rangeData.memory || []), [rangeData.memory]);
  const diskChart = useMemo(() => toChartData(rangeData.disk || []), [rangeData.disk]);
  const netRxChart = useMemo(() => toChartData(rangeData.netRx || []), [rangeData.netRx]);
  const netTxChart = useMemo(() => toChartData(rangeData.netTx || []), [rangeData.netTx]);

  const evidentlyShiftImaging = promMetrics?.evidently_dataset_shift_imaging ?? 0;
  const evidentlyFeaturesImaging = promMetrics?.evidently_features_drifted_imaging ?? 0;
  const hasEvidentlyData = evidentlyShiftImaging > 0 || evidentlyFeaturesImaging > 0;

  if (loading && !snapshot) {
    return (
      <PageContainer>
        <div className='flex items-center justify-center h-[60vh]'>
          <div className='animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full' />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHeader
        title='MLOps Monitor'
        description='Live monitoring for model performance and system resources'
      />

      <Tabs value={activeTab} onValueChange={setActiveTab} className='w-full'>
        <TabsList>
          <TabsTrigger value='ml'>ML</TabsTrigger>
          <TabsTrigger value='drift'>Drift</TabsTrigger>
          <TabsTrigger value='system'>System</TabsTrigger>
          <TabsTrigger value='grafana'>Grafana</TabsTrigger>
        </TabsList>

        <TabsContent value='ml' className='mt-4'>
          <StatsRow columns={4}>
            <StatsCard
              title='Imaging Accuracy'
              value={metrics.imaging?.accuracy != null ? `${(metrics.imaging.accuracy * 100).toFixed(1)}%` : 'N/A'}
              icon={Brain}
              color='#3b82f6'
            />
            <StatsCard
              title='Val QWK'
              value={metrics.imaging?.quadratic_weighted_kappa != null ? metrics.imaging.quadratic_weighted_kappa.toFixed(2) : 'N/A'}
              icon={TrendingUp}
            />
            <StatsCard
              title='ROC-AUC'
              value={metrics.imaging?.roc_auc_macro != null ? `${(metrics.imaging.roc_auc_macro * 100).toFixed(1)}%` : 'N/A'}
              icon={Activity}
            />
            <StatsCard
              title='Drift Detected'
              value={promMetrics?.drift_detected_imaging === 1 ? 'Yes' : 'No'}
              icon={promMetrics?.drift_detected_imaging === 1 ? AlertTriangle : CheckCircle2}
              color={promMetrics?.drift_detected_imaging === 1 ? '#ef4444' : '#22c55e'}
            />
          </StatsRow>

          {promMetrics && (
            <div className='rounded-lg border bg-card p-4'>
              <h3 className='font-semibold mb-4'>Live Training Signals</h3>
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
                      <Database className='h-4 w-4 text-muted-foreground' />
                      <div>
                        <p className='text-sm text-muted-foreground'>Evidently Drift</p>
                        <p className='text-xl font-bold'>{hasEvidentlyData ? evidentlyShiftImaging.toFixed(3) : 'N/A'}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className='pt-4'>
                    <div className='flex items-center gap-2'>
                      <TrendingUp className='h-4 w-4 text-muted-foreground' />
                      <div>
                        <p className='text-sm text-muted-foreground'>Features Drifted</p>
                        <p className='text-xl font-bold'>{hasEvidentlyData ? evidentlyFeaturesImaging : 'N/A'}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {getMetricsRadarData.length > 0 && (
            <div className='rounded-lg border bg-card p-4'>
              <h3 className='font-semibold mb-4'>Model Performance</h3>
              <div className='h-[300px]'>
                <ResponsiveContainer width='100%' height='100%'>
                  <RadarChart data={getMetricsRadarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey='metric' />
                    <PolarRadiusAxis domain={[0, 100]} />
                    <Radar name='Imaging' dataKey='imaging' stroke='#2563eb' fill='#2563eb' fillOpacity={0.2} />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {getQwkHistoryData.length > 0 && (
            <div className='rounded-lg border bg-card p-4'>
              <div className='flex items-center justify-between mb-4'>
                <div className='flex items-center gap-2'>
                  <TrendingUp className='h-5 w-5' />
                  <h3 className='font-semibold'>QWK Over Epochs</h3>
                </div>
                {metrics.training_summary?.best_epoch && (
                  <Badge variant='secondary'>Best epoch #{metrics.training_summary.best_epoch}</Badge>
                )}
              </div>
              <div className='h-[220px]'>
                <ResponsiveContainer width='100%' height='100%'>
                  <LineChart data={getQwkHistoryData}>
                    <CartesianGrid strokeDasharray='3 3' />
                    <XAxis dataKey='epoch' />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, 'QWK']} />
                    <Line type='monotone' dataKey='qwk' stroke='#2563eb' strokeWidth={2} dot={false} name='Val QWK' />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className='text-xs text-muted-foreground mt-2'>Validation QWK by epoch (from training summary)</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value='drift' className='mt-4'>
          <StatsRow columns={3}>
            <StatsCard
              title='Current Status'
              value={driftHistory.length > 0 ? driftHistory[driftHistory.length - 1].status.replace('_', ' ') : 'Unknown'}
              icon={driftHistory.length > 0 && driftHistory[driftHistory.length - 1].status === 'drift_detected' ? AlertTriangle : CheckCircle2}
              color={driftHistory.length > 0 && driftHistory[driftHistory.length - 1].status === 'drift_detected' ? '#ef4444' : '#22c55e'}
            />
            <StatsCard
              title='Last Check'
              value={driftHistory.length > 0 ? new Date(driftHistory[driftHistory.length - 1].generated_at).toLocaleDateString() : '—'}
              icon={Activity}
            />
            <StatsCard
              title='Total Checks'
              value={String(driftHistory.length)}
              icon={Database}
            />
          </StatsRow>

          {driftHistory.length > 0 && (
            <div className='rounded-lg border bg-card p-4 mt-4'>
              <div className='flex items-center gap-2 mb-4'>
                <Waves className='h-5 w-5 text-blue-500' />
                <h3 className='font-semibold'>PSI Over Time</h3>
              </div>
              <div className='h-[250px]'>
                <ResponsiveContainer width='100%' height='100%'>
                  <LineChart
                    data={driftHistory.map((d) => ({
                      label: new Date(d.generated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                      psi: d.overall_psi,
                      samples: d.current_samples,
                      ref: d.reference_samples,
                    }))}
                  >
                    <CartesianGrid strokeDasharray='3 3' />
                    <XAxis dataKey='label' tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ fontSize: 12, borderRadius: 6 }}
                      formatter={(v: number, name: string) => {
                        if (name === 'psi') return [v.toFixed(6), 'PSI'];
                        return [v.toLocaleString(), name === 'samples' ? 'Current Samples' : 'Reference Samples'];
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type='monotone' dataKey='psi' stroke='#3b82f6' strokeWidth={2} dot={{ r: 3 }} name='psi' />
                    <Line type='monotone' dataKey='samples' stroke='#22c55e' strokeWidth={1} dot={false} name='samples' />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className='text-xs text-muted-foreground mt-2'>
                Population Stability Index (PSI) across {driftHistory.length} consecutive drift checks &middot; Lower is better
              </p>
            </div>
          )}

          {driftHistory.length === 0 && (
            <div className='rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground mt-4'>
              No drift history available. Run a drift check to populate.
            </div>
          )}
        </TabsContent>

        <TabsContent value='system' className='mt-4'>
          <StatsRow columns={4}>
            <StatsCard
              title='GPU Utilization'
              value={promMetrics?.gpu_utilization != null ? `${promMetrics.gpu_utilization.toFixed(0)}%` : 'N/A'}
              icon={Cpu}
              color='#22c55e'
            />
            <StatsCard
              title='GPU Memory'
              value={gpuMemoryPercent != null ? `${gpuMemoryPercent.toFixed(0)}%` : 'N/A'}
              icon={Gauge}
            />
            <StatsCard
              title='CPU Utilization'
              value={promMetrics?.cpu_utilization != null ? `${promMetrics.cpu_utilization.toFixed(0)}%` : 'N/A'}
              icon={Activity}
            />
            <StatsCard
              title='RAM Used'
              value={memoryUsedPercent != null ? `${memoryUsedPercent.toFixed(0)}%` : 'N/A'}
              icon={Database}
            />
          </StatsRow>

          <div className='flex items-center justify-end mb-4 gap-2'>
            <BarChart3 className='h-4 w-4 text-muted-foreground' />
            <span className='text-sm text-muted-foreground mr-1'>Range:</span>
            {TIME_RANGES.map((r) => (
              <Button
                key={r}
                variant={timeRange === r ? 'default' : 'outline'}
                size='sm'
                onClick={() => setTimeRange(r)}
                className='h-7 px-3 text-xs'
              >
                {r}
              </Button>
            ))}
          </div>

          {rangeLoading ? (
            <div className='flex items-center justify-center h-40'>
              <div className='animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full' />
            </div>
          ) : (
            <div className='grid grid-cols-1 lg:grid-cols-2 gap-5'>
              <div className='rounded-lg border bg-card p-4'>
                <h3 className='text-sm font-semibold mb-3 flex items-center gap-2'>
                  <Cpu className='h-4 w-4 text-blue-500' /> CPU Utilization
                </h3>
                <div className='h-[200px]'>
                  <ResponsiveContainer width='100%' height='100%'>
                    <LineChart data={cpuChart}>
                      <CartesianGrid strokeDasharray='3 3' />
                      <XAxis dataKey='time' tick={{ fontSize: 10 }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit='%' />
                      <Tooltip />
                      <Line type='monotone' dataKey='value' stroke='#2563eb' strokeWidth={2} dot={false} name='CPU' />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className='rounded-lg border bg-card p-4'>
                <h3 className='text-sm font-semibold mb-3 flex items-center gap-2'>
                  <Database className='h-4 w-4 text-orange-500' /> Memory Usage
                </h3>
                <div className='h-[200px]'>
                  <ResponsiveContainer width='100%' height='100%'>
                    <LineChart data={memoryChart}>
                      <CartesianGrid strokeDasharray='3 3' />
                      <XAxis dataKey='time' tick={{ fontSize: 10 }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit='%' />
                      <Tooltip />
                      <Line type='monotone' dataKey='value' stroke='#f97316' strokeWidth={2} dot={false} name='Memory' />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className='rounded-lg border bg-card p-4'>
                <h3 className='text-sm font-semibold mb-3 flex items-center gap-2'>
                  <HardDrive className='h-4 w-4 text-green-500' /> Disk Usage
                </h3>
                <div className='h-[200px]'>
                  <ResponsiveContainer width='100%' height='100%'>
                    <LineChart data={diskChart}>
                      <CartesianGrid strokeDasharray='3 3' />
                      <XAxis dataKey='time' tick={{ fontSize: 10 }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit='%' />
                      <Tooltip />
                      <Line type='monotone' dataKey='value' stroke='#22c55e' strokeWidth={2} dot={false} name='Disk' />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className='rounded-lg border bg-card p-4'>
                <h3 className='text-sm font-semibold mb-3 flex items-center gap-2'>
                  <Wifi className='h-4 w-4 text-purple-500' /> Network I/O
                </h3>
                <div className='h-[200px]'>
                  <ResponsiveContainer width='100%' height='100%'>
                    <LineChart>
                      <CartesianGrid strokeDasharray='3 3' />
                      <XAxis dataKey='time' tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} unit=' Mbps' />
                      <Tooltip />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {netRxChart.length > 0 && (
                        <Line type='monotone' dataKey='value' data={netRxChart} stroke='#a855f7' strokeWidth={2} dot={false} name='RX' />
                      )}
                      {netTxChart.length > 0 && (
                        <Line type='monotone' dataKey='value' data={netTxChart} stroke='#6366f1' strokeWidth={2} dot={false} name='TX' />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          <div className='grid grid-cols-1 md:grid-cols-2 gap-5 mt-5'>
            <div className='rounded-lg border bg-card p-4'>
              <h3 className='font-semibold mb-4'>Compute Trends</h3>
              <div className='h-[220px]'>
                <ResponsiveContainer width='100%' height='100%'>
                  <LineChart data={promSparkData}>
                    <CartesianGrid strokeDasharray='3 3' />
                    <XAxis dataKey='index' hide />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Line type='monotone' dataKey='gpu' stroke='#22c55e' strokeWidth={2} dot={false} name='GPU' />
                    <Line type='monotone' dataKey='cpu' stroke='#2563eb' strokeWidth={2} dot={false} name='CPU' />
                    <Line type='monotone' dataKey='mem' stroke='#f97316' strokeWidth={2} dot={false} name='RAM' />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className='rounded-lg border bg-card p-4'>
              <h3 className='font-semibold mb-4'>Storage & Network</h3>
              <div className='grid grid-cols-2 gap-4'>
                <Card>
                  <CardContent className='pt-4'>
                    <div className='flex items-center gap-2'>
                      <HardDrive className='h-4 w-4 text-muted-foreground' />
                      <div>
                        <p className='text-sm text-muted-foreground'>Disk Used</p>
                        <p className='text-xl font-bold'>{diskUsedPercent != null ? `${diskUsedPercent.toFixed(0)}%` : 'N/A'}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className='pt-4'>
                    <div className='flex items-center gap-2'>
                      <Wifi className='h-4 w-4 text-muted-foreground' />
                      <div>
                        <p className='text-sm text-muted-foreground'>Net Rx</p>
                        <p className='text-xl font-bold'>
                          {promMetrics?.network_receive_bytes_per_second != null
                            ? `${(promMetrics.network_receive_bytes_per_second / 1024 / 1024).toFixed(1)} MB/s`
                            : 'N/A'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className='pt-4'>
                    <div className='flex items-center gap-2'>
                      <Wifi className='h-4 w-4 text-muted-foreground' />
                      <div>
                        <p className='text-sm text-muted-foreground'>Net Tx</p>
                        <p className='text-xl font-bold'>
                          {promMetrics?.network_transmit_bytes_per_second != null
                            ? `${(promMetrics.network_transmit_bytes_per_second / 1024 / 1024).toFixed(1)} MB/s`
                            : 'N/A'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className='pt-4'>
                    <div className='flex items-center gap-2'>
                      <FileText className='h-4 w-4 text-muted-foreground' />
                      <div>
                        <p className='text-sm text-muted-foreground'>Snapshot</p>
                        <p className='text-xs font-medium'>
                          {snapshot?.generated_at ? new Date(snapshot.generated_at).toLocaleTimeString() : 'N/A'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value='grafana' className='mt-4'>
          <div className='rounded-lg border bg-card overflow-hidden'>
            <div className='flex items-center justify-between px-4 py-3 border-b bg-muted/20'>
              <div className='flex items-center gap-2'>
                <BarChart3 className='h-5 w-5 text-blue-500' />
                <h3 className='font-semibold'>RetinaXAI MLOps Dashboard</h3>
              </div>
              <div className='flex items-center gap-2'>
                <div className='hidden sm:flex items-center gap-1 flex-wrap'>
                  {GRAFANA_PANEL_LINKS.slice(0, 6).map(panel => (
                    <a
                      key={panel.id}
                      href={`${GRAFANA_VIEW_URL}&viewPanel=${panel.id}`}
                      target='_blank'
                      rel='noopener noreferrer'
                    >
                      <Button variant='ghost' size='sm' className='h-7 text-xs px-2'>
                        {panel.title}
                      </Button>
                    </a>
                  ))}
                </div>
                <a href={GRAFANA_KIOSK_URL} target='_blank' rel='noopener noreferrer'>
                  <Button variant='outline' size='sm'>
                    Open in Grafana <ExternalLink className='ml-1.5 h-3.5 w-3.5' />
                  </Button>
                </a>
              </div>
            </div>
            <iframe
              src={GRAFANA_VIEW_URL}
              className='w-full border-0'
              style={{ height: 'calc(100vh - 320px)', minHeight: 500 }}
              title='Grafana Dashboard'
              sandbox='allow-scripts allow-same-origin allow-forms'
            />
          </div>
        </TabsContent>
      </Tabs>
    </PageContainer>
  );
}
