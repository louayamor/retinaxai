'use client';

import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { apiFetch } from '@/lib/auth';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import Image from 'next/image';
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
  AreaChart,
  Area,
} from 'recharts';
import { fadeInUp, slideInUp, staggerItem } from '@/lib/animations';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { 
  RefreshCw, 
  Users, 
  FileText, 
  Scan, 
  Activity, 
  TrendingUp,
  TrendingDown,
  Calendar,
  BarChart3,
  PieChart as PieChartIcon,
  Activity as ActivityIcon
} from 'lucide-react';

const COLORS = ['var(--brand-teal)', 'var(--brand-gold)', '#e74c3c', '#3498db', '#9b59b6', '#2ecc71'];
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
  const shouldReduceMotion = useReducedMotion();

  const chartAxisClass = 'fill-muted-foreground';
  const chartGridStroke = 'hsl(var(--border) / 0.55)';

  const loadStats = async () => {
    try {
      setLoading(true);
      console.log('[Visualise] Fetching data...');
      const data = await apiFetch<CombinedStats>('/api/v1/oct-stats/stats');
      console.log('[Visualise] Received data:', JSON.stringify(data, null, 2));
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

  if (loading) {
    return (
      <div className='flex h-full items-center justify-center'>
        <p className='text-muted-foreground'>Loading analytics...</p>
      </div>
    );
  }

  if (!stats) {
    return (
      <PageContainer>
        <div className="flex h-96 items-center justify-center">
          <div className="text-center">
            <p className="text-muted-foreground mb-4">No data available</p>
            <Button onClick={loadStats} variant="outline">
              <RefreshCw className="mr-2 h-4 w-4" />
              Load Data
            </Button>
          </div>
        </div>
      </PageContainer>
    );
  }

  console.log('[Visualise] Rendering with stats:', stats);

  const { summary, recent_activity, patient_demographics, clinical_reports, predictions, oct_reports } = stats;

  // Transform data for charts
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

  const predStatusData = Object.entries(predictions.status).map(([key, val]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    value: val,
  }));

  const thicknessData = [
    { name: 'Center Fovea', value: oct_reports.thickness_averages.center_fovea || 0 },
    { name: 'Average', value: oct_reports.thickness_averages.average_thickness || 0 },
    { name: 'Total Volume', value: oct_reports.thickness_averages.total_volume_mm3 || 0 },
  ];

  const getSeverityColor = (level: number) => {
    const colors = ['#2ecc71', 'var(--brand-teal)', 'var(--brand-gold)', '#e67e22', '#e74c3c'];
    return colors[level] || 'var(--brand-teal)';
  };

  return (
    <PageContainer>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeInUp}
        initial="hidden"
        animate="visible"
        className="flex flex-col gap-8"
      >
        {/* Hero */}
        <motion.div
          variants={shouldReduceMotion ? {} : slideInUp}
          className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#0a2e3e] via-[#0d3a4c] to-[#104a5e] p-10 text-white"
        >
          <div className="absolute right-0 top-0 h-full w-1/3 opacity-15">
            <Image
              src="https://images.unsplash.com/photo-1551076805-e1869033e561?w=800&q=80"
              alt="Analytics"
              fill
              className="object-cover"
              unoptimized
            />
          </div>
          <div className="absolute -bottom-10 -left-10 h-32 w-32 rounded-full bg-[var(--brand-teal)]/10 blur-3xl" />
          <div className="absolute top-10 right-20 h-24 w-24 rounded-full bg-[var(--brand-gold)]/10 blur-2xl" />
          
          <div className="relative z-10 flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight mb-2">Analytics Dashboard</h1>
              <p className="text-white/70 text-lg max-w-xl">
                Comprehensive insights: {summary.total_patients} patients, {summary.total_predictions} predictions, {summary.total_clinical_reports} reports
              </p>
              {lastUpdated && (
                <p className="text-white/50 text-sm mt-2">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={loadStats}
              disabled={loading}
              className="bg-white/10 text-white hover:bg-white/20 border-white/20"
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </motion.div>

        {/* Summary Stats Row */}
        <motion.div
          variants={shouldReduceMotion ? {} : {
            hidden: { opacity: 0 },
            visible: { opacity: 1, transition: { staggerChildren: 0.08 } }
          }}
          initial="hidden"
          animate="visible"
          className="grid gap-4 md:grid-cols-2 lg:grid-cols-4"
        >
          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-[var(--brand-teal)]">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
                <Users className="h-4 w-4 text-[var(--brand-teal)]" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{summary.total_patients}</div>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  +{recent_activity.patients_7d} this week
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-[var(--brand-gold)]">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">DR Predictions</CardTitle>
                <Scan className="h-4 w-4 text-[var(--brand-gold)]" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{summary.total_predictions}</div>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  +{recent_activity.predictions_7d} this week
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-blue-500">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Clinical Reports</CardTitle>
                <FileText className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{summary.total_clinical_reports}</div>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  +{recent_activity.reports_7d} this week
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-purple-500">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">OCT Scans</CardTitle>
                <ActivityIcon className="h-4 w-4 text-purple-500" />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{oct_reports.total}</div>
                <p className="text-xs text-muted-foreground mt-1">
                  {oct_reports.avg_image_quality ? `Avg quality: ${oct_reports.avg_image_quality}%` : 'No scans'}
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        {/* Row 1: Patient Demographics & Predictions Severity */}
        <div className="grid gap-4 md:grid-cols-2">
          {/* Gender Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-[var(--brand-teal)]" />
                  Patient Gender Distribution
              </CardTitle>
              <CardDescription>Demographics of registered patients</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{
                  M: { label: 'Male', color: SURFACE_COLORS[0] },
                  F: { label: 'Female', color: SURFACE_COLORS[1] },
                  O: { label: 'Other', color: SURFACE_COLORS[2] },
                }}
                className="mx-auto aspect-square h-[250px]"
              >
                <PieChart>
                  <Pie
                    data={genderData.map((entry, index) => ({ ...entry, fill: SURFACE_COLORS[index % SURFACE_COLORS.length] }))}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={82}
                    innerRadius={54}
                    dataKey="value"
                    stroke="var(--background)"
                    strokeWidth={2}
                  >
                    {genderData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={SURFACE_COLORS[index % SURFACE_COLORS.length]} />
                    ))}
                  </Pie>
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>

          {/* DR Severity Distribution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-[var(--brand-teal)]" />
                DR Severity Distribution
              </CardTitle>
              <CardDescription>Predictions by diabetic retinopathy grade</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{ predictions: { label: 'Predictions', color: 'var(--primary)' } }}
                className="aspect-auto h-[250px] w-full"
              >
                <BarChart data={severityData} margin={{ left: 8, right: 8 }}>
                  <CartesianGrid vertical={false} stroke={chartGridStroke} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 12 }} />
                  <YAxis tickLine={false} axisLine={false} />
                  <ChartTooltip
                    cursor={{ fill: 'var(--primary)', opacity: 0.1 }}
                    content={<ChartTooltipContent indicator="dot" nameKey="value" />}
                  />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
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
          {/* Age Groups */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5 text-[var(--brand-gold)]" />
                Patient Age Distribution
              </CardTitle>
              <CardDescription>Age groups of registered patients</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{ age: { label: 'Patients', color: 'var(--chart-2)' } }}
                className="aspect-auto h-[250px] w-full"
              >
                <BarChart data={ageData} margin={{ left: 8, right: 8 }}>
                  <CartesianGrid vertical={false} stroke={chartGridStroke} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 12 }} />
                  <YAxis tickLine={false} axisLine={false} />
                  <ChartTooltip cursor={{ fill: 'var(--chart-2)', opacity: 0.1 }} content={<ChartTooltipContent indicator="dot" nameKey="value" />} />
                  <Bar dataKey="value" fill="var(--chart-2)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          {/* Report Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-blue-500" />
                Clinical Reports Status
              </CardTitle>
              <CardDescription>Status breakdown of generated reports</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer
                config={{
                  completed: { label: 'Completed', color: 'var(--chart-1)' },
                  pending: { label: 'Pending', color: 'var(--chart-3)' },
                  running: { label: 'Running', color: 'var(--chart-2)' },
                  failed: { label: 'Failed', color: 'var(--chart-5)' },
                }}
                className="mx-auto aspect-square h-[250px]"
              >
                <PieChart>
                  <Pie
                    data={reportStatusData.map((entry, index) => ({ ...entry, fill: SURFACE_COLORS[index % SURFACE_COLORS.length] }))}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={82}
                    innerRadius={54}
                    dataKey="value"
                    stroke="var(--background)"
                    strokeWidth={2}
                  >
                    {reportStatusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={SURFACE_COLORS[index % SURFACE_COLORS.length]} />
                    ))}
                  </Pie>
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>

        {/* Row 3: OCT Reports - keep existing charts */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold flex items-center gap-2">
            <Scan className="h-5 w-5 text-[var(--brand-teal)]" />
            OCT Scan Analytics
          </h3>
          
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {/* OCT Grade Distribution */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>DR Grade Distribution (OCT)</CardTitle>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{ predictions: { label: 'Predictions', color: 'var(--primary)' } }}
                  className="aspect-auto h-[280px] w-full"
                >
                  <BarChart data={octGradeData} margin={{ left: 8, right: 8 }}>
                    <CartesianGrid vertical={false} stroke={chartGridStroke} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
                    <YAxis tickLine={false} axisLine={false} />
                    <ChartTooltip cursor={{ fill: 'var(--primary)', opacity: 0.1 }} content={<ChartTooltipContent indicator="dot" nameKey="value" />} />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {octGradeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {/* Eye Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Eye Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{ od: { label: 'Right Eye', color: 'var(--chart-1)' }, os: { label: 'Left Eye', color: 'var(--chart-2)' } }}
                  className="mx-auto aspect-square h-[280px]"
                >
                  <PieChart>
                    <Pie
                      data={eyeData.map((entry, index) => ({ ...entry, fill: SURFACE_COLORS[index % SURFACE_COLORS.length] }))}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                      outerRadius={82}
                      innerRadius={54}
                      dataKey="value"
                      stroke="var(--background)"
                      strokeWidth={2}
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
            {/* Edema */}
            <Card>
              <CardHeader>
                <CardTitle>Macular Edema</CardTitle>
                <CardDescription>Presence of macular edema</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{ present: { label: 'Present', color: 'var(--chart-5)' }, absent: { label: 'Absent', color: 'var(--chart-1)' } }} className="mx-auto aspect-square h-[200px]">
                  <PieChart>
                    <Pie
                      data={edemaData.map((entry, index) => ({ ...entry, fill: index === 0 ? 'var(--chart-5)' : 'var(--chart-1)' }))}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={58}
                      innerRadius={40}
                      dataKey="value"
                      stroke="var(--background)"
                      strokeWidth={2}
                    >
                      <Cell fill="var(--chart-5)" />
                      <Cell fill="var(--chart-1)" />
                    </Pie>
                    <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                  </PieChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {/* ERM Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Epiretinal Membrane</CardTitle>
                <CardDescription>ERM status distribution</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{}} className="mx-auto aspect-square h-[200px]">
                  <PieChart>
                    <Pie
                      data={Object.entries(oct_reports.erm_distribution).map(([k, v]) => ({ name: k || 'Unknown', value: v }))}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                      outerRadius={58}
                      innerRadius={40}
                      dataKey="value"
                      stroke="var(--background)"
                      strokeWidth={2}
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

            {/* Thickness Radar */}
            <Card>
              <CardHeader>
                <CardTitle>Retinal Thickness</CardTitle>
                <CardDescription>Average measurements (μm)</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{ thickness: { label: 'Thickness', color: 'var(--chart-1)' } }} className="aspect-auto h-[200px] w-full">
                  <RadarChart data={thicknessData}>
                    <PolarGrid stroke={chartGridStroke} />
                    <PolarAngleAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} />
                    <PolarRadiusAxis tick={{ fill: 'var(--muted-foreground)', fontSize: 10 }} />
                    <Radar
                      name="Thickness"
                      dataKey="value"
                      stroke="var(--chart-1)"
                      fill="var(--chart-1)"
                      fillOpacity={0.3}
                    />
                    <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                  </RadarChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </div>
        </div>
      </motion.div>
    </PageContainer>
  );
}
