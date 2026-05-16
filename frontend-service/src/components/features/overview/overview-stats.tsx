'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { IconUsers, IconScan, IconActivity } from '@tabler/icons-react';
import { Loader2 } from 'lucide-react';

interface OverviewStatsData {
  totals: {
    patients: number;
    predictions: number;
    reports: number;
    scans: number;
  };
  severity_distribution: Record<number, number>;
  recent_activity: {
    new_patients: number;
    new_predictions: number;
    new_reports: number;
  };
}

export interface GradeStat {
  grade: string;
  count: number;
  color: string;
  pct: string;
}

export function OverviewStats() {
  const [stats, setStats] = useState<OverviewStatsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const res = await fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' });
        const data = await res.json().catch(() => null);
        setStats(data);
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
      <div className="grid gap-6 md:grid-cols-3">
        {[1, 2, 3].map((i) => (
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
  const totalPredictions = stats?.totals?.predictions || 0;
  const predictionsToday = stats?.recent_activity?.new_predictions || 0;

  const distributionTotal = Object.values(stats?.severity_distribution || {}).reduce((a, b) => a + b, 0);
  const drDetected = Object.entries(stats?.severity_distribution || {})
    .filter(([grade]) => String(grade) !== '0')
    .reduce((sum, [, count]) => sum + count, 0);
  const drRate = distributionTotal > 0 ? (drDetected / distributionTotal) * 100 : 0;

  const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
  const GRADE_COLORS = ['bg-emerald-500', 'bg-cyan-500', 'bg-amber-500', 'bg-orange-500', 'bg-rose-500'];

  const gradeStats: GradeStat[] = GRADE_LABELS.map((label, idx) => {
    const count = stats?.severity_distribution?.[idx] || 0;
    const pct = distributionTotal > 0 ? ((count / distributionTotal) * 100).toFixed(1) : '0';
    return { grade: label, count, color: GRADE_COLORS[idx], pct: `${pct}%` };
  });

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-3">
        <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-cyan-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-cyan-950/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-base font-medium">Total Patients</CardTitle>
            <IconUsers className="h-5 w-5 text-[var(--brand-teal)]" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{totalPatients.toLocaleString()}</div>
            <p className="text-muted-foreground text-sm mt-1">
              Registered in the system
            </p>
          </CardContent>
        </Card>
        <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-cyan-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-cyan-950/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-base font-medium">Predictions Today</CardTitle>
            <IconScan className="h-5 w-5 text-[var(--brand-teal)]" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{predictionsToday}</div>
            <p className="text-muted-foreground text-sm mt-1">
              {totalPredictions} total predictions
            </p>
          </CardContent>
        </Card>
        <Card className="animate-in-up border-0 bg-gradient-to-br from-card to-amber-50/40 shadow-md transition-transform duration-300 hover:-translate-y-1 dark:to-amber-950/15">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-base font-medium">DR Detection Rate</CardTitle>
            <IconActivity className="h-5 w-5 text-[var(--brand-gold)]" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{drRate.toFixed(1)}%</div>
            <p className="text-muted-foreground text-sm mt-1">
              {drDetected} of {distributionTotal} predictions show DR
            </p>
          </CardContent>
        </Card>
      </div>

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
