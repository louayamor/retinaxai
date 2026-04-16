'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Database, 
  Server, 
  Activity, 
  Loader2, 
  RefreshCw,
  Users,
  FileText,
  Scan,
  CheckCircle2,
  XCircle,
  Clock
} from 'lucide-react';
import { toast } from 'sonner';

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';
const LLMOPS_BASE = process.env.NEXT_PUBLIC_LLMOPS_URL || 'http://localhost:8002';

interface ServiceHealth {
  status: 'healthy' | 'unhealthy';
  response_time?: number;
  error?: string;
}

interface DashboardStats {
  totals: {
    patients: number;
    predictions: number;
    reports: number;
    scans: number;
  };
  prediction_status: Record<string, number>;
  report_status: Record<string, number>;
  severity_distribution: Record<number, number>;
  predictions_timeline: Array<{ date: string; predictions: number }>;
  age_distribution: Record<string, number>;
  gender_distribution: Record<string, number>;
  recent_activity: {
    new_patients: number;
    new_predictions: number;
    new_reports: number;
  };
  avg_confidence: number | null;
}

export default function SystemStatsPage() {
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [serviceHealth, setServiceHealth] = useState<Record<string, ServiceHealth>>({});
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [dashboardRes] = await Promise.all([
        fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' }).then(r => r.json()).catch(() => null),
      ]);

      setDashboardStats(dashboardRes);

      const health: Record<string, ServiceHealth> = {};
      
      // Check backend
      try {
        const start = Date.now();
        const backendRes = await fetch(`${BASE}/health`, { credentials: 'include' });
        health.backend = { 
          status: backendRes.ok ? 'healthy' : 'unhealthy',
          response_time: Date.now() - start 
        };
      } catch {
        health.backend = { status: 'unhealthy', error: 'Connection failed' };
      }

      // Check MLOps
      try {
        const start = Date.now();
        const mlopsRes = await fetch(`${MLOPS_BASE}/api/health`);
        health.mlops = { 
          status: mlopsRes.ok ? 'healthy' : 'unhealthy',
          response_time: Date.now() - start 
        };
      } catch {
        health.mlops = { status: 'unhealthy', error: 'Connection failed' };
      }

      // Check LLMOps
      try {
        const start = Date.now();
        const llmopsRes = await fetch(`${LLMOPS_BASE}/health`);
        health.llmops = { 
          status: llmopsRes.ok ? 'healthy' : 'unhealthy',
          response_time: Date.now() - start 
        };
      } catch {
        health.llmops = { status: 'unhealthy', error: 'Connection failed' };
      }

      setServiceHealth(health);
    } catch (error) {
      console.error('Failed to fetch system stats:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getHealthIcon = (status?: string) => {
    if (status === 'healthy') return <CheckCircle2 className="h-5 w-5 text-green-500" />;
    if (status === 'unhealthy') return <XCircle className="h-5 w-5 text-red-500" />;
    return <Clock className="h-5 w-5 text-muted-foreground" />;
  };

  const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
  const GRADE_COLORS = ['bg-emerald-500', 'bg-cyan-500', 'bg-amber-500', 'bg-orange-500', 'bg-rose-500'];

  if (loading && !dashboardStats) {
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
            <h1 className="text-3xl font-bold tracking-tight">System Statistics</h1>
            <p className="text-muted-foreground">Database metrics and service health</p>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Service Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Service Health
            </CardTitle>
            <CardDescription>Real-time status of all microservices</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(serviceHealth).map(([service, health]) => (
                <div key={service} className="p-4 rounded-lg border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium capitalize">{service}</span>
                    {getHealthIcon(health.status)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {health.response_time ? `${health.response_time}ms` : health.error || 'Unknown'}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Database Totals */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Users className="h-4 w-4" />
                Total Patients
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{dashboardStats?.totals?.patients || 0}</div>
              <p className="text-xs text-muted-foreground">
                +{dashboardStats?.recent_activity?.new_patients || 0} this week
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Total Predictions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{dashboardStats?.totals?.predictions || 0}</div>
              <p className="text-xs text-muted-foreground">
                +{dashboardStats?.recent_activity?.new_predictions || 0} this week
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Total Reports
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{dashboardStats?.totals?.reports || 0}</div>
              <p className="text-xs text-muted-foreground">
                +{dashboardStats?.recent_activity?.new_reports || 0} this week
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Scan className="h-4 w-4" />
                Total OCT Scans
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{dashboardStats?.totals?.scans || 0}</div>
              <p className="text-xs text-muted-foreground">MRI scans in system</p>
            </CardContent>
          </Card>
        </div>

        {/* DR Grade Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>DR Grade Distribution</CardTitle>
            <CardDescription>Predictions by severity level</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {GRADE_LABELS.map((label, idx) => {
                const count = dashboardStats?.severity_distribution?.[idx] || 0;
                const total = Object.values(dashboardStats?.severity_distribution || {}).reduce((a, b) => a + b, 0);
                const pct = total > 0 ? (count / total * 100).toFixed(1) : '0';
                return (
                  <div key={idx} className="flex flex-col items-center gap-2 rounded-xl border p-4">
                    <div className={`h-3 w-full rounded-full ${GRADE_COLORS[idx]}`} style={{ opacity: 0.8 }} />
                    <div className="text-center">
                      <p className="text-2xl font-bold">{count}</p>
                      <p className="text-sm text-muted-foreground">{label}</p>
                      <Badge variant="secondary" className="mt-1">{pct}%</Badge>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Demographics */}
        <div className="grid gap-4 md:grid-cols-2">
          {/* Age Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Age Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              {dashboardStats?.age_distribution && Object.keys(dashboardStats.age_distribution).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(dashboardStats.age_distribution).map(([age, count]) => (
                    <div key={age} className="flex items-center justify-between">
                      <span className="text-sm">{age}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500"
                            style={{ width: `${(count / (dashboardStats.totals?.patients || 1)) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-muted-foreground w-8">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No age data available</p>
              )}
            </CardContent>
          </Card>

          {/* Gender Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Gender Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              {dashboardStats?.gender_distribution && Object.keys(dashboardStats.gender_distribution).length > 0 ? (
                <div className="space-y-2">
                  {Object.entries(dashboardStats.gender_distribution).map(([gender, count]) => (
                    <div key={gender} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{gender.toLowerCase()}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-purple-500"
                            style={{ width: `${(count / (dashboardStats.totals?.patients || 1)) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-muted-foreground w-8">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No gender data available</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Predictions Timeline */}
        {dashboardStats?.predictions_timeline && dashboardStats.predictions_timeline.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Predictions Over Time (Last 30 Days)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[200px]">
                <div className="grid grid-cols-7 gap-1">
                  {dashboardStats.predictions_timeline.slice(-14).map((day, idx) => (
                    <div key={idx} className="text-center">
                      <div 
                        className="mx-auto rounded bg-blue-500"
                        style={{ 
                          height: `${Math.max(4, (day.predictions / 10) * 40)}px`,
                          opacity: 0.5 + (day.predictions / 20)
                        }}
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </p>
                      <p className="text-xs font-medium">{day.predictions}</p>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Average Confidence */}
        {dashboardStats?.avg_confidence && (
          <Card>
            <CardHeader>
              <CardTitle>Average Model Confidence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <div className="text-4xl font-bold">{(dashboardStats.avg_confidence * 100).toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">
                  Average confidence score across all successful predictions
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  );
}