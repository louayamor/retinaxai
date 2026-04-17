'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
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

const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif DR'];
const GRADE_COLORS = ['#10b981', '#06b6d4', '#f59e0b', '#f97316', '#f43f5e'];

const MiniBar = ({ value, max, color }: { value: number; max: number; color: string }) => (
  <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
    <div className="h-full rounded-full" style={{ width: `${Math.min(100, (value / max) * 100)}%`, backgroundColor: color }} />
  </div>
);

export default function SystemStatsPage() {
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo[]>([]);
  const [serviceHealth, setServiceHealth] = useState<Record<string, { status: string; response_time?: number }>>({});
  const [loading, setLoading] = useState(true);

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
    try {
      const dashboardRes = await fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' }).then(r => r.json()).catch(() => null);
      setDashboardStats(dashboardRes);

      const health: Record<string, { status: string; response_time?: number }> = {};
      const services = [
        { name: 'backend', url: `${BASE}/health` },
        { name: 'mlops', url: '/health', base: process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004' },
        { name: 'llmops', url: '/health', base: process.env.NEXT_PUBLIC_LLMOPS_URL || 'http://localhost:8002' },
      ];

      for (const svc of services) {
        const base = svc.base || BASE;
        try {
          const start = Date.now();
          const res = await fetch(`${base}${svc.url}`, { credentials: 'include' });
          health[svc.name] = { status: res.ok ? 'healthy' : 'unhealthy', response_time: Date.now() - start };
        } catch {
          health[svc.name] = { status: 'unhealthy' };
        }
      }

      setServiceHealth(health);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    fetchSystemMetrics();
    const interval = setInterval(() => { fetchData(); fetchSystemMetrics(); }, 30000);
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
        <div className="flex items-center justify-center h-[60vh]">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">System Monitor</h1>
            <p className="text-sm text-muted-foreground">Infrastructure, services, and database statistics</p>
          </div>
          <Button variant="outline" size="sm" onClick={() => { fetchData(); fetchSystemMetrics(); }}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Service Health */}
        <Card className="p-3">
          <CardHeader className="p-0 mb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Server className="h-4 w-4" />
              Service Health
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(serviceHealth).map(([service, health]) => (
                <div key={service} className="flex items-center justify-between p-2 rounded border text-xs">
                  <span className="capitalize font-medium">{service}</span>
                  <div className="flex items-center gap-1">
                    {health.status === 'healthy' ? (
                      <CheckCircle2 className="h-3 w-3 text-green-500" />
                    ) : (
                      <XCircle className="h-3 w-3 text-red-500" />
                    )}
                    <span className="text-muted-foreground">{health.response_time ? `${health.response_time}ms` : '-'}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System Resources */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {systemMetrics && (
            <>
              <Card className="p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <Cpu className="h-3 w-3" /> CPU
                </div>
                <div className={`text-xl font-bold ${getUsageColor(systemMetrics.cpu.usage_percent)}`}>
                  {systemMetrics.cpu.usage_percent.toFixed(1)}%
                </div>
                <div className="text-[10px] text-muted-foreground">Load: {systemMetrics.load.toFixed(2)}</div>
                <MiniBar value={systemMetrics.cpu.usage_percent} max={100} color={getBarColor(systemMetrics.cpu.usage_percent)} />
              </Card>
              <Card className="p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <Activity className="h-3 w-3" /> Memory
                </div>
                <div className={`text-xl font-bold ${getUsageColor(systemMetrics.memory.usage_percent)}`}>
                  {systemMetrics.memory.usage_percent.toFixed(1)}%
                </div>
                <div className="text-[10px] text-muted-foreground">{systemMetrics.memory.used_gb.toFixed(1)}/{systemMetrics.memory.total_gb.toFixed(1)} GB</div>
                <MiniBar value={systemMetrics.memory.usage_percent} max={100} color={getBarColor(systemMetrics.memory.usage_percent)} />
              </Card>
              <Card className="p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <HardDrive className="h-3 w-3" /> Disk
                </div>
                <div className={`text-xl font-bold ${getUsageColor(systemMetrics.disk.usage_percent)}`}>
                  {systemMetrics.disk.usage_percent.toFixed(1)}%
                </div>
                <div className="text-[10px] text-muted-foreground">{systemMetrics.disk.used_gb.toFixed(0)}/{systemMetrics.disk.total_gb.toFixed(0)} GB</div>
                <MiniBar value={systemMetrics.disk.usage_percent} max={100} color={getBarColor(systemMetrics.disk.usage_percent)} />
              </Card>
              <Card className="p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <Wifi className="h-3 w-3" /> Network
                </div>
                <div className="text-xl font-bold text-blue-500">↓{systemMetrics.network.rx_mbps.toFixed(1)}</div>
                <div className="text-[10px] text-muted-foreground">↑{systemMetrics.network.tx_mbps.toFixed(1)} Mbps</div>
              </Card>
            </>
          )}
        </div>

        {/* GPU */}
        {gpuInfo.length > 0 && (
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Gauge className="h-4 w-4" />
                GPU
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {gpuInfo.map((gpu, i) => (
                  <div key={i} className="p-2 rounded border text-xs">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-[10px] truncate">{gpu.name}</span>
                      <span className={`font-bold ${getUsageColor(gpu.utilization_pct)}`}>{gpu.utilization_pct.toFixed(0)}%</span>
                    </div>
                    <div className="space-y-0.5">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-muted-foreground">VRAM</span>
                        <span>{gpu.memory_used_gb.toFixed(1)}/{gpu.memory_total_gb.toFixed(0)} GB</span>
                      </div>
                      <MiniBar value={gpu.memory_pct} max={100} color="#8b5cf6" />
                      <div className="flex justify-between text-[10px] mt-1">
                        <span className="flex items-center gap-0.5"><Thermometer className="h-2.5 w-2.5 text-orange-400" />{gpu.temperature_c.toFixed(0)}°C</span>
                        <span className="flex items-center gap-0.5"><Zap className="h-2.5 w-2.5 text-yellow-400" />{gpu.power_w.toFixed(1)}W</span>
                        <span className="flex items-center gap-0.5"><Wind className="h-2.5 w-2.5 text-cyan-400" />{gpu.clock_sm_mhz.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Database Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Users className="h-3 w-3" /> Patients
            </div>
            <div className="text-xl font-bold">{dashboardStats?.totals?.patients || 0}</div>
            <div className="text-[10px] text-muted-foreground">+{dashboardStats?.recent_activity?.new_patients || 0} this week</div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Activity className="h-3 w-3" /> Predictions
            </div>
            <div className="text-xl font-bold">{dashboardStats?.totals?.predictions || 0}</div>
            <div className="text-[10px] text-muted-foreground">+{dashboardStats?.recent_activity?.new_predictions || 0} this week</div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <FileText className="h-3 w-3" /> Reports
            </div>
            <div className="text-xl font-bold">{dashboardStats?.totals?.reports || 0}</div>
            <div className="text-[10px] text-muted-foreground">+{dashboardStats?.recent_activity?.new_reports || 0} this week</div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Gauge className="h-3 w-3" /> Avg Confidence
            </div>
            <div className="text-xl font-bold">
              {dashboardStats?.avg_confidence != null ? `${(dashboardStats.avg_confidence * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </Card>
        </div>

        {/* DR Distribution & Demographics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* DR Distribution */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm">DR Grades</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {severityData.length > 0 ? (
                <div className="flex items-end justify-between gap-1 h-[80px]">
                  {severityData.map((item, i) => (
                    <div key={i} className="flex flex-col items-center gap-1 flex-1">
                      <div className="text-[10px] font-medium">{item.value}</div>
                      <div className="w-full rounded-t" style={{ height: `${Math.max(4, (item.value / Math.max(...severityData.map(d => d.value))) * 60)}px`, backgroundColor: item.color, opacity: 0.8 }} />
                      <div className="text-[8px] text-muted-foreground text-center leading-tight">{item.name}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground text-center py-4">No data</p>
              )}
            </CardContent>
          </Card>

          {/* Age Distribution */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm">Age Distribution</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {dashboardStats?.age_distribution && Object.keys(dashboardStats.age_distribution).length > 0 ? (
                <div className="space-y-1 max-h-[80px] overflow-y-auto">
                  {Object.entries(dashboardStats.age_distribution).map(([age, count]) => (
                    <div key={age} className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground w-12">{age}</span>
                      <div className="flex-1 mx-2">
                        <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500 rounded-full" style={{ width: `${(count / (dashboardStats.totals?.patients || 1)) * 100}%` }} />
                        </div>
                      </div>
                      <span className="text-muted-foreground w-4 text-right">{count}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground text-center py-4">No data</p>
              )}
            </CardContent>
          </Card>

          {/* Gender Distribution */}
          <Card className="p-3">
            <CardHeader className="p-0 mb-2">
              <CardTitle className="text-sm">Gender Distribution</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {dashboardStats?.gender_distribution && Object.keys(dashboardStats.gender_distribution).length > 0 ? (
                <div className="h-[80px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={Object.entries(dashboardStats.gender_distribution).map(([k, v]) => ({ name: k, value: v }))}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={20}
                        outerRadius={35}
                        label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                        labelLine={false}
                      >
                        <Cell fill="#8b5cf6" />
                        <Cell fill="#06b6d4" />
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground text-center py-4">No data</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </PageContainer>
  );
}
