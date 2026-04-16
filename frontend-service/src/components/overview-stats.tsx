'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { IconEye, IconUsers, IconActivity, IconBrain } from '@tabler/icons-react';
import { Loader2 } from 'lucide-react';

interface OverviewStats {
  totals: {
    patients: number;
    predictions: number;
    reports: number;
    scans: number;
  };
  severity_distribution: Record<number, number>;
  avg_confidence: number | null;
}

interface MLOpsMetrics {
  imaging?: { accuracy?: number };
}

export interface GradeStat {
  grade: string;
  count: number;
  color: string;
  pct: string;
}

export function OverviewStats() {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [mlopsMetrics, setMlopsMetrics] = useState<MLOpsMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';

        const [dashboardRes, metricsRes] = await Promise.all([
          fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' }).then(r => r.json()).catch(() => null),
          fetch(`${MLOPS_BASE}/api/metrics`).then(r => r.json()).catch(() => null),
        ]);

        setStats(dashboardRes);
        setMlopsMetrics(metricsRes);
      } catch (error) {
        console.error('Failed to fetch overview stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="border-0 shadow-md">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
              <div className="h-4 w-24 bg-muted rounded animate-pulse" />
              <div className="h-5 w-5 bg-muted rounded animate-pulse" />
            </CardHeader>
            <CardContent>
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const totalPatients = stats?.totals?.patients || 0;
  const totalScans = stats?.totals?.scans || 0;
  const drDetected = Object.entries(stats?.severity_distribution || {})
    .filter(([grade]) => grade !== '0')
    .reduce((sum, [, count]) => sum + count, 0);
  const drPercentage = stats?.totals?.predictions ? ((drDetected / stats.totals.predictions) * 100).toFixed(1) : '0';
  const accuracy = mlopsMetrics?.imaging?.accuracy ? (mlopsMetrics.imaging.accuracy * 100).toFixed(1) : 'N/A';

  const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
  const GRADE_COLORS = ['bg-emerald-500', 'bg-cyan-500', 'bg-amber-500', 'bg-orange-500', 'bg-rose-500'];

  const gradeStats: GradeStat[] = GRADE_LABELS.map((label, idx) => {
    const count = stats?.severity_distribution?.[idx] || 0;
    const total = Object.values(stats?.severity_distribution || {}).reduce((a, b) => a + b, 0);
    const pct = total > 0 ? ((count / total) * 100).toFixed(1) : '0';
    return { grade: label, count, color: GRADE_COLORS[idx], pct: `${pct}%` };
  });

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
      <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-cyan-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-cyan-950/15">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <CardTitle className="text-base font-medium">Total OCT Reports</CardTitle>
          <IconEye className="h-5 w-5 text-[var(--brand-teal)]" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold">{totalScans}</div>
          <p className="text-muted-foreground text-sm mt-1">
            From {totalPatients} unique patients
          </p>
        </CardContent>
      </Card>
      <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-amber-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-amber-950/15">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <CardTitle className="text-base font-medium">DR Detected</CardTitle>
          <IconActivity className="h-5 w-5 text-[var(--brand-gold)]" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold">{drDetected}</div>
          <p className="text-muted-foreground text-sm mt-1">
            {drPercentage}% of predictions have DR
          </p>
        </CardContent>
      </Card>
      <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-cyan-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-cyan-950/15">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <CardTitle className="text-base font-medium">Active Patients</CardTitle>
          <IconUsers className="h-5 w-5 text-[var(--brand-teal)]" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold">{totalPatients}</div>
          <p className="text-muted-foreground text-sm mt-1">
            Registered in the system
          </p>
        </CardContent>
      </Card>
      <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-emerald-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-emerald-950/15">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <CardTitle className="text-base font-medium">Model Accuracy</CardTitle>
          <IconBrain className="h-5 w-5 text-[var(--brand-teal)]" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold">{accuracy !== 'N/A' ? `${accuracy}%` : 'N/A'}</div>
          <p className="text-muted-foreground text-sm mt-1">
            EfficientNet-B3 on EyePACS
          </p>
        </CardContent>
      </Card>
      </div>

      {/* Grade Distribution */}
      <Card className="animate-in-up border-0 shadow-md">
        <CardHeader>
          <CardTitle>DR Grade Distribution</CardTitle>
          <CardTitle className="text-sm font-normal text-muted-foreground">
            Breakdown of detected diabetic retinopathy severity levels
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {gradeStats.map((item) => (
              <div
                key={item.grade}
                className="flex flex-col items-center gap-2 rounded-xl border p-4 transition-colors duration-200 hover:bg-muted/40"
              >
                <div className={`h-3 w-full rounded-full ${item.color}`} style={{ opacity: 0.8 }} />
                <div className="text-center">
                  <p className="text-2xl font-bold">{item.count}</p>
                  <p className="text-sm text-muted-foreground">{item.grade}</p>
                  <Badge variant="secondary" className="mt-1">{item.pct}</Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}