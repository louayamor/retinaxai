'use client';

import { useEffect, useState } from 'react';
import { apiFetch } from '@/lib/auth';
import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { 
  RefreshCw, 
  Users, 
  FileText, 
  Scan, 
  TrendingUp,
  Calendar,
  BarChart3,
  Activity as ActivityIcon
} from 'lucide-react';

const SURFACE_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)', 'var(--chart-5)'];
const GRADE_COLORS: Record<string, string> = {
  no_dr: '#2ecc71',
  mild: 'var(--brand-teal)',
  moderate: 'var(--brand-gold)',
  severe: '#e67e22',
  proliferative: '#e74c3c',
  0: '#2ecc71',
  1: 'var(--brand-teal)',
  2: 'var(--brand-gold)',
  3: '#e67e22',
  4: '#e74c3c',
};

interface CombinedStats {
  summary: {
    total_patients: number;
    total_oct_reports: number;
    total_clinical_reports: number;
    total_predictions: number;
  };
  recent_activity: {
    patients_7d: number;
    patients_30d: number;
    predictions_7d: number;
    reports_7d: number;
  };
  patient_demographics: {
    gender: Record<string, number>;
    age_groups: Record<string, number>;
  };
  clinical_reports: {
    total: number;
    status: Record<string, number>;
  };
  predictions: {
    total: number;
    status: Record<string, number>;
    severity_distribution: Record<number, number>;
  };
  oct_reports: {
    total: number;
    grade_distribution: Record<string, number>;
    eye_distribution: Record<string, number>;
    edema: { present: number; absent: number };
    erm_distribution: Record<string, number>;
    thickness_averages: {
      center_fovea: number | null;
      average_thickness: number | null;
      total_volume_mm3: number | null;
    };
    avg_image_quality: number | null;
  };
}

export default function VisualisePage() {
  const [stats, setStats] = useState<CombinedStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const chartAxisClass = 'fill-muted-foreground';
  const chartGridStroke = 'hsl(var(--border) / 0.55)';

  const loadStats = async () => {
    try {
      setLoading(true);
      const data = await apiFetch<CombinedStats>('/api/v1/oct-stats/stats');
      setStats(data);
      setLastUpdated(new Date());
    } catch (err: unknown) {
      const error = err as { status?: number };
      if (error.status === 401) {
        window.location.href = '/auth/login';
      } else {
        console.error('[Visualise] Failed to fetch stats:', err);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const metadata = lastUpdated ? (
    <span className='text-[11px] text-muted-foreground'>Last updated: {lastUpdated.toLocaleTimeString()}</span>
  ) : null;

  if (loading) {
    return (
      <PageContainer>
        <div className='flex h-full items-center justify-center'>
          <p className='text-sm text-muted-foreground'>Loading analytics...</p>
        </div>
      </PageContainer>
    );
  }

  if (!stats) {
    return (
      <PageContainer>
        <div className="flex flex-col items-center justify-center gap-3 py-24">
          <p className="text-sm text-muted-foreground">No data available</p>
          <Button onClick={loadStats} variant="outline" size="sm">
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Load Data
          </Button>
        </div>
      </PageContainer>
    );
  }

  const { summary, recent_activity, patient_demographics, clinical_reports, predictions, oct_reports } = stats;

  const genderData = Object.entries(patient_demographics.gender).map(([key, val]) => ({
    name: key === 'M' ? 'Male' : key === 'F' ? 'Female' : 'Other',
    value: val,
  }));

  const ageData = Object.entries(patient_demographics.age_groups).map(([key, val]) => ({
    name: key,
    value: val,
  }));

  const severityData = Object.entries(predictions.severity_distribution).map(([key, val]) => ({
    name: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'][parseInt(key)] || key,
    value: val,
    fill: GRADE_COLORS[key] || 'var(--brand-teal)',
  }));

  const octGradeData = Object.entries(oct_reports.grade_distribution).map(([grade, count]) => ({
    name: grade.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    value: count,
    fill: GRADE_COLORS[grade] || 'var(--brand-teal)',
  }));

  const eyeData = Object.entries(oct_reports.eye_distribution).map(([eye, count]) => ({
    name: eye === 'OD' ? 'Right Eye (OD)' : 'Left Eye (OS)',
    value: count,
  }));

  const edemaData = [
    { name: 'Present', value: oct_reports.edema.present },
    { name: 'Absent', value: oct_reports.edema.absent },
  ];

  const reportStatusData = Object.entries(clinical_reports.status).map(([key, val]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    value: val,
  }));

  const thicknessData = [
    { name: 'Center Fovea', value: oct_reports.thickness_averages.center_fovea || 0 },
    { name: 'Average', value: oct_reports.thickness_averages.average_thickness || 0 },
    { name: 'Total Volume', value: oct_reports.thickness_averages.total_volume_mm3 || 0 },
  ];

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHeader
        title='Analytics Dashboard'
        description={`${summary.total_patients} patients, ${summary.total_predictions} predictions, ${summary.total_clinical_reports} reports`}
        metadata={metadata}
        actions={
          <Button variant="outline" size="sm" onClick={loadStats} disabled={loading}>
            <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        }
      />

      {/* Summary Stats Row */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-l-4 border-l-[var(--brand-teal)]">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
            <Users className="h-4 w-4 text-[var(--brand-teal)]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.total_patients}</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-0.5">
              <TrendingUp className="h-3 w-3 text-green-500" />
              +{recent_activity.patients_7d} this week
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-[var(--brand-gold)]">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">DR Predictions</CardTitle>
            <Scan className="h-4 w-4 text-[var(--brand-gold)]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.total_predictions}</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-0.5">
              <TrendingUp className="h-3 w-3 text-green-500" />
              +{recent_activity.predictions_7d} this week
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-blue-500">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Clinical Reports</CardTitle>
            <FileText className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{summary.total_clinical_reports}</div>
            <p className="text-xs text-muted-foreground flex items-center gap-1 mt-0.5">
              <TrendingUp className="h-3 w-3 text-green-500" />
              +{recent_activity.reports_7d} this week
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-purple-500">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">OCT Scans</CardTitle>
            <ActivityIcon className="h-4 w-4 text-purple-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{oct_reports.total}</div>
            <p className="text-xs text-muted-foreground mt-0.5">
              {oct_reports.avg_image_quality ? `Avg quality: ${oct_reports.avg_image_quality}%` : 'No scans'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Row 1: Patient Demographics & Predictions Severity */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Users className="h-4 w-4 text-[var(--brand-teal)]" />
              Patient Gender Distribution
            </CardTitle>
            <CardDescription className='text-xs'>Demographics of registered patients</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                M: { label: 'Male', color: SURFACE_COLORS[0] },
                F: { label: 'Female', color: SURFACE_COLORS[1] },
              }}
              className="mx-auto aspect-square h-[200px]"
            >
              <PieChart>
                <Pie
                  data={genderData.map((entry, index) => ({ ...entry, fill: SURFACE_COLORS[index % SURFACE_COLORS.length] }))}
                  cx="50%" cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={68} innerRadius={44}
                  dataKey="value"
                  stroke="var(--background)" strokeWidth={2}
                >
                  {genderData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={SURFACE_COLORS[index % SURFACE_COLORS.length]} />
                  ))}
                </Pie>
                <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              </PieChart>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <BarChart3 className="h-4 w-4 text-[var(--brand-teal)]" />
              DR Severity Distribution
            </CardTitle>
            <CardDescription className='text-xs'>Predictions by diabetic retinopathy grade</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{ predictions: { label: 'Predictions', color: 'var(--primary)' } }}
              className="aspect-auto h-[200px] w-full"
            >
              <BarChart data={severityData} margin={{ left: 8, right: 8 }}>
                <CartesianGrid vertical={false} stroke={chartGridStroke} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} />
                <ChartTooltip cursor={{ fill: 'var(--primary)', opacity: 0.1 }} content={<ChartTooltipContent indicator="dot" nameKey="value" />} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Row 2: Age Groups & Report Status */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Calendar className="h-4 w-4 text-[var(--brand-gold)]" />
              Patient Age Distribution
            </CardTitle>
            <CardDescription className='text-xs'>Age groups of registered patients</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{ age: { label: 'Patients', color: 'var(--chart-2)' } }}
              className="aspect-auto h-[200px] w-full"
            >
              <BarChart data={ageData} margin={{ left: 8, right: 8 }}>
                <CartesianGrid vertical={false} stroke={chartGridStroke} />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} />
                <ChartTooltip cursor={{ fill: 'var(--chart-2)', opacity: 0.1 }} content={<ChartTooltipContent indicator="dot" nameKey="value" />} />
                <Bar dataKey="value" fill="var(--chart-2)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <FileText className="h-4 w-4 text-blue-500" />
              Clinical Reports Status
            </CardTitle>
            <CardDescription className='text-xs'>Status breakdown of generated reports</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                completed: { label: 'Completed', color: 'var(--chart-1)' },
                pending: { label: 'Pending', color: 'var(--chart-3)' },
                running: { label: 'Running', color: 'var(--chart-2)' },
                failed: { label: 'Failed', color: 'var(--chart-5)' },
              }}
              className="mx-auto aspect-square h-[200px]"
            >
              <PieChart>
                <Pie
                  data={reportStatusData.map((entry, index) => ({ ...entry, fill: SURFACE_COLORS[index % SURFACE_COLORS.length] }))}
                  cx="50%" cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={68} innerRadius={44}
                  dataKey="value"
                  stroke="var(--background)" strokeWidth={2}
                >
                  {reportStatusData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={SURFACE_COLORS[index % SURFACE_COLORS.length]} />
                  ))}
                </Pie>
                <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              </PieChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* OCT Reports Section */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Scan className="h-4 w-4 text-[var(--brand-teal)]" />
          OCT Scan Analytics
        </h3>
        
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader className='pb-2'>
              <CardTitle className="text-sm">DR Grade Distribution (OCT)</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{ predictions: { label: 'Predictions', color: 'var(--primary)' } }}
                className="aspect-auto h-[240px] w-full"
              >
                <BarChart data={octGradeData} margin={{ left: 8, right: 8 }}>
                  <CartesianGrid vertical={false} stroke={chartGridStroke} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} />
                  <ChartTooltip cursor={{ fill: 'var(--primary)', opacity: 0.1 }} content={<ChartTooltipContent indicator="dot" nameKey="value" />} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {octGradeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className="text-sm">Eye Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{ od: { label: 'Right Eye', color: 'var(--chart-1)' }, os: { label: 'Left Eye', color: 'var(--chart-2)' } }}
                className="mx-auto aspect-square h-[240px]"
              >
                <PieChart>
                  <Pie
                    data={eyeData.map((entry, index) => ({ ...entry, fill: SURFACE_COLORS[index % SURFACE_COLORS.length] }))}
                    cx="50%" cy="50%"
                    labelLine={false}
                    label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                    outerRadius={72} innerRadius={48}
                    dataKey="value"
                    stroke="var(--background)" strokeWidth={2}
                  >
                    {eyeData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={SURFACE_COLORS[index % SURFACE_COLORS.length]} />
                    ))}
                  </Pie>
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className="text-sm">Macular Edema</CardTitle>
              <CardDescription className='text-xs'>Presence of macular edema</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={{ present: { label: 'Present', color: 'var(--chart-5)' }, absent: { label: 'Absent', color: 'var(--chart-1)' } }} className="mx-auto aspect-square h-[160px]">
                <PieChart>
                  <Pie
                    data={edemaData.map((entry, index) => ({ ...entry, fill: index === 0 ? 'var(--chart-5)' : 'var(--chart-1)' }))}
                    cx="50%" cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={48} innerRadius={32}
                    dataKey="value"
                    stroke="var(--background)" strokeWidth={2}
                  >
                    <Cell fill="var(--chart-5)" />
                    <Cell fill="var(--chart-1)" />
                  </Pie>
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className="text-sm">Epiretinal Membrane</CardTitle>
              <CardDescription className='text-xs'>ERM status distribution</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={{}} className="mx-auto aspect-square h-[160px]">
                <PieChart>
                  <Pie
                    data={Object.entries(oct_reports.erm_distribution).map(([k, v]) => ({ name: k || 'Unknown', value: v }))}
                    cx="50%" cy="50%"
                    labelLine={false}
                    label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                    outerRadius={48} innerRadius={32}
                    dataKey="value"
                    stroke="var(--background)" strokeWidth={2}
                  >
                    {Object.keys(oct_reports.erm_distribution).map((_, i) => (
                      <Cell key={i} fill={SURFACE_COLORS[i % SURFACE_COLORS.length]} />
                    ))}
                  </Pie>
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className="text-sm">Retinal Thickness</CardTitle>
              <CardDescription className='text-xs'>Average measurements (&mu;m)</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={{ thickness: { label: 'Thickness', color: 'var(--chart-1)' } }} className="aspect-auto h-[160px] w-full">
                <RadarChart data={thicknessData}>
                  <PolarGrid stroke={chartGridStroke} />
                  <PolarAngleAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} />
                  <PolarRadiusAxis tick={{ fill: 'var(--muted-foreground)', fontSize: 9 }} />
                  <Radar name="Thickness" dataKey="value" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.3} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                </RadarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>
      </div>
    </PageContainer>
  );
}
