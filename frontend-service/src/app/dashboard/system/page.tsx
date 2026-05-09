'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
import { StatsRow } from '@/components/ui/stats-row';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Server,
  Activity,
  RefreshCw,
  Users,
  FileText,
  Cpu,
  HardDrive,
  Wifi,
  Gauge,
  Thermometer,
  Zap,
  Wind,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from 'recharts';

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SystemMetrics {
  cpu: { usage_percent: number };
  memory: { total_gb: number; used_gb: number; available_gb: number; usage_percent: number };
  disk: { total_gb: number; used_gb: number; free_gb: number; usage_percent: number };
  load: number;
  network: { rx_mbps: number; tx_mbps: number };
}

interface GpuInfo {
  name: string;
  utilization_pct: number;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_pct: number;
  temperature_c: number;
  power_w: number;
  fan_speed_pct: number;
  clock_sm_mhz: number;
  clock_mem_mhz: number;
}

interface DashboardStats {
  totals: { patients: number; predictions: number; reports: number; scans: number };
  severity_distribution: Record<number, number>;
  predictions_timeline: Array<{ date: string; predictions: number }>;
  age_distribution: Record<string, number>;
  gender_distribution: Record<string, number>;
  recent_activity: { new_patients: number; new_predictions: number; new_reports: number };
  avg_confidence: number | null;
}

interface ServiceHealth {
  status: string;
  response_time?: number;
  status_code?: number;
  error?: string;
}

const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif DR'];
const GRADE_COLORS = ['#10b981', '#06b6d4', '#f59e0b', '#f97316', '#f43f5e'];

export default function SystemStatsPage() {
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo[]>([]);
  const [serviceHealth, setServiceHealth] = useState<Record<string, ServiceHealth>>({});
  const [loading, setLoading] = useState(true);
  const [healthLoading, setHealthLoading] = useState(true);

  const fetchSystemMetrics = async () => {
    try {
      const res = await fetch(`${BASE}/api/v1/system/metrics`);
      if (res.ok) setSystemMetrics(await res.json());
    } catch {}
    try {
      const res = await fetch(`${BASE}/api/v1/system/gpu`);
      if (res.ok) {
        const data = await res.json();
        setGpuInfo(data.gpus || []);
      }
    } catch {}
  };

  const fetchData = async () => {
    setHealthLoading(true);
    try {
      const dashboardRes = await fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' }).then(r => r.json()).catch(() => null);
      setDashboardStats(dashboardRes);

      const health: Record<string, ServiceHealth> = {};
      const services = [
        { name: 'backend', base: BASE, url: '/health' },
        { name: 'mlops', url: '/health', base: process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004' },
        { name: 'llmops', url: '/health', base: process.env.NEXT_PUBLIC_LLMOPS_URL || 'http://localhost:8002' },
      ];

      console.info('[System] probing services', services.map(svc => svc.name));
      for (const svc of services) {
        const base = svc.base || BASE;
        try {
          const start = Date.now();
          const isHealthProbe = svc.url === '/health' || svc.url.endsWith('/health');
          const res = await fetch(`${base}${svc.url}`, isHealthProbe ? {} : { credentials: 'include' });
          const responseTime = Date.now() - start;
          const status = res.ok ? 'healthy' : 'unhealthy';
          const error = res.ok ? undefined : `${res.status} ${res.statusText}`.trim();
          health[svc.name] = {
            status,
            response_time: responseTime,
            status_code: res.status,
            error,
          };
          console.info('[System] service health result', {
            service: svc.name,
            status,
            code: res.status,
            error,
            responseTime,
          });
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          health[svc.name] = { status: 'unhealthy', error: message };
          console.info('[System] service health failed', { service: svc.name, error: message });
        }
      }

      setServiceHealth(health);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    void fetchData();
    void fetchSystemMetrics();
    const interval = setInterval(() => { void fetchData(); void fetchSystemMetrics(); }, 30000);
    return () => clearInterval(interval);
  }, []);

  const getUsageColor = (pct: number) => {
    if (pct >= 90) return 'text-red-500';
    if (pct >= 70) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getBarColor = (pct: number) => {
    if (pct >= 90) return '#ef4444';
    if (pct >= 70) return '#f59e0b';
    return '#10b981';
  };

  const severityData = dashboardStats?.severity_distribution
    ? Object.entries(dashboardStats.severity_distribution).map(([k, v]) => ({ name: GRADE_LABELS[Number(k)], value: v, color: GRADE_COLORS[Number(k)] }))
    : [];

  if (loading && !dashboardStats && !systemMetrics) {
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
        title='System Monitor'
        description='Infrastructure, services, and database statistics'
        actions={
          <Button variant='outline' size='sm' onClick={() => { void fetchData(); void fetchSystemMetrics(); }}>
            <RefreshCw className='h-4 w-4 mr-2' />
            Refresh
          </Button>
        }
      />

      <div className='rounded-lg border bg-card p-4'>
        <div className='flex items-center gap-2 mb-4'>
          <Server className='h-5 w-5' />
          <h3 className='font-semibold'>Service Health</h3>
        </div>
        <div className='grid grid-cols-3 gap-4'>
          {Object.entries(serviceHealth).map(([service, health]) => (
            <div key={service} className='flex items-center justify-between p-3 rounded-lg border'>
              <span className='capitalize font-medium'>{service}</span>
              <div className='flex items-center gap-2'>
                {health.status === 'healthy' ? (
                  <CheckCircle2 className='h-4 w-4 text-green-500' />
                ) : (
                  <XCircle className='h-4 w-4 text-red-500' />
                )}
                  <span className='text-sm text-muted-foreground'>
                    {healthLoading
                      ? 'Checking...'
                      : health.status === 'healthy'
                        ? `${health.response_time ?? 0}ms`
                        : health.status_code
                          ? `HTTP ${health.status_code}${health.error ? `: ${health.error}` : ''}`
                          : health.error || 'Unavailable'}
                  </span>
                </div>
              </div>
            ))}
        </div>
      </div>

      {systemMetrics && (
        <div className='grid grid-cols-2 md:grid-cols-4 gap-5'>
          <Card>
            <CardContent className='pt-6'>
              <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
                <Cpu className='h-4 w-4' /> CPU
              </div>
              <div className={`text-2xl font-bold ${getUsageColor(systemMetrics.cpu.usage_percent)}`}>
                {systemMetrics.cpu.usage_percent.toFixed(1)}%
              </div>
              <div className='w-full h-2 bg-muted rounded-full overflow-hidden mt-2'>
                <div
                  className='h-full rounded-full transition-all'
                  style={{ width: `${systemMetrics.cpu.usage_percent}%`, backgroundColor: getBarColor(systemMetrics.cpu.usage_percent) }}
                />
              </div>
              <p className='text-xs text-muted-foreground mt-1'>Load: {systemMetrics.load.toFixed(2)}</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className='pt-6'>
              <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
                <Activity className='h-4 w-4' /> Memory
              </div>
              <div className={`text-2xl font-bold ${getUsageColor(systemMetrics.memory.usage_percent)}`}>
                {systemMetrics.memory.usage_percent.toFixed(1)}%
              </div>
              <div className='w-full h-2 bg-muted rounded-full overflow-hidden mt-2'>
                <div
                  className='h-full rounded-full transition-all'
                  style={{ width: `${systemMetrics.memory.usage_percent}%`, backgroundColor: getBarColor(systemMetrics.memory.usage_percent) }}
                />
              </div>
              <p className='text-xs text-muted-foreground mt-1'>{systemMetrics.memory.used_gb.toFixed(1)}/{systemMetrics.memory.total_gb.toFixed(1)} GB</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className='pt-6'>
              <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
                <HardDrive className='h-4 w-4' /> Disk
              </div>
              <div className={`text-2xl font-bold ${getUsageColor(systemMetrics.disk.usage_percent)}`}>
                {systemMetrics.disk.usage_percent.toFixed(1)}%
              </div>
              <div className='w-full h-2 bg-muted rounded-full overflow-hidden mt-2'>
                <div
                  className='h-full rounded-full transition-all'
                  style={{ width: `${systemMetrics.disk.usage_percent}%`, backgroundColor: getBarColor(systemMetrics.disk.usage_percent) }}
                />
              </div>
              <p className='text-xs text-muted-foreground mt-1'>{systemMetrics.disk.used_gb.toFixed(0)}/{systemMetrics.disk.total_gb.toFixed(0)} GB</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className='pt-6'>
              <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
                <Wifi className='h-4 w-4' /> Network
              </div>
              <div className='text-2xl font-bold text-blue-500'>↓{systemMetrics.network.rx_mbps.toFixed(1)}</div>
              <p className='text-xs text-muted-foreground'>↑{systemMetrics.network.tx_mbps.toFixed(1)} Mbps</p>
            </CardContent>
          </Card>
        </div>
      )}

      {gpuInfo.length > 0 && (
        <div className='rounded-lg border bg-card p-4'>
          <div className='flex items-center gap-2 mb-4'>
            <Gauge className='h-5 w-5' />
            <h3 className='font-semibold'>GPU</h3>
          </div>
          <div className='grid grid-cols-2 md:grid-cols-3 gap-4'>
            {gpuInfo.map((gpu, i) => (
              <Card key={i}>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm flex items-center justify-between'>
                    <span className='truncate'>{gpu.name}</span>
                    <span className={`font-bold ${getUsageColor(gpu.utilization_pct)}`}>{gpu.utilization_pct.toFixed(0)}%</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className='space-y-2'>
                  <div className='w-full h-2 bg-muted rounded-full overflow-hidden'>
                    <div
                      className='h-full rounded-full'
                      style={{ width: `${gpu.memory_pct}%`, backgroundColor: '#8b5cf6' }}
                    />
                  </div>
                  <div className='flex justify-between text-sm'>
                    <span className='text-muted-foreground'>VRAM</span>
                    <span>{gpu.memory_used_gb.toFixed(1)}/{gpu.memory_total_gb.toFixed(0)} GB</span>
                  </div>
                  <div className='flex justify-between text-sm'>
                    <span className='flex items-center gap-1 text-muted-foreground'>
                      <Thermometer className='h-3 w-3 text-orange-400' />{gpu.temperature_c.toFixed(0)}°C
                    </span>
                    <span className='flex items-center gap-1 text-muted-foreground'>
                      <Zap className='h-3 w-3 text-yellow-400' />{gpu.power_w.toFixed(1)}W
                    </span>
                    <span className='flex items-center gap-1 text-muted-foreground'>
                      <Wind className='h-3 w-3 text-cyan-400' />{gpu.clock_sm_mhz.toFixed(0)}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      <div className='grid grid-cols-2 md:grid-cols-4 gap-5'>
        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
              <Users className='h-4 w-4' /> Patients
            </div>
            <div className='text-2xl font-bold'>{dashboardStats?.totals?.patients || 0}</div>
            <p className='text-sm text-muted-foreground'>+{dashboardStats?.recent_activity?.new_patients || 0} this week</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
              <Activity className='h-4 w-4' /> Predictions
            </div>
            <div className='text-2xl font-bold'>{dashboardStats?.totals?.predictions || 0}</div>
            <p className='text-sm text-muted-foreground'>+{dashboardStats?.recent_activity?.new_predictions || 0} this week</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
              <FileText className='h-4 w-4' /> Reports
            </div>
            <div className='text-2xl font-bold'>{dashboardStats?.totals?.reports || 0}</div>
            <p className='text-sm text-muted-foreground'>+{dashboardStats?.recent_activity?.new_reports || 0} this week</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>
              <Gauge className='h-4 w-4' /> Avg Confidence
            </div>
            <div className='text-2xl font-bold'>
              {dashboardStats?.avg_confidence != null ? `${(dashboardStats.avg_confidence * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className='grid grid-cols-1 md:grid-cols-3 gap-5'>
        <div className='rounded-lg border bg-card p-4'>
          <h3 className='font-semibold mb-4'>DR Grades</h3>
          {severityData.length > 0 ? (
            <div className='h-[150px]'>
              <ResponsiveContainer width='100%' height='100%'>
                <PieChart>
                  <Pie
                    data={severityData}
                    dataKey='value'
                    nameKey='name'
                    cx='50%'
                    cy='50%'
                    outerRadius={60}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {severityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className='text-sm text-muted-foreground text-center py-8'>No data</p>
          )}
        </div>

        <div className='rounded-lg border bg-card p-4'>
          <h3 className='font-semibold mb-4'>Age Distribution</h3>
          {dashboardStats?.age_distribution && Object.keys(dashboardStats.age_distribution).length > 0 ? (
            <div className='space-y-2'>
              {Object.entries(dashboardStats.age_distribution).map(([age, count]) => (
                <div key={age} className='flex items-center justify-between text-sm'>
                  <span className='text-muted-foreground w-16'>{age}</span>
                  <div className='flex-1 mx-2'>
                    <div className='w-full h-2 bg-muted rounded-full overflow-hidden'>
                      <div
                        className='h-full bg-blue-500 rounded-full'
                        style={{ width: `${(count / (dashboardStats.totals?.patients || 1)) * 100}%` }}
                      />
                    </div>
                  </div>
                  <span className='text-muted-foreground w-8 text-right'>{count}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className='text-sm text-muted-foreground text-center py-8'>No data</p>
          )}
        </div>

        <div className='rounded-lg border bg-card p-4'>
          <h3 className='font-semibold mb-4'>Gender Distribution</h3>
          {dashboardStats?.gender_distribution && Object.keys(dashboardStats.gender_distribution).length > 0 ? (
            <div className='h-[150px]'>
              <ResponsiveContainer width='100%' height='100%'>
                <PieChart>
                  <Pie
                    data={Object.entries(dashboardStats.gender_distribution).map(([k, v]) => ({ name: k, value: v }))}
                    dataKey='value'
                    nameKey='name'
                    cx='50%'
                    cy='50%'
                    innerRadius={30}
                    outerRadius={60}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    <Cell fill='#8b5cf6' />
                    <Cell fill='#06b6d4' />
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className='text-sm text-muted-foreground text-center py-8'>No data</p>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
