'use client';

import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import type { BiomarkerMetrics } from '@/types';

type EyeSide = 'left' | 'right';

const THRESHOLDS = {
  vessel_density: { low: 0.12, high: 0.32, label: '0.12-0.32' },
  tortuosity: { low: 1.0, high: 1.35, label: '1.00-1.35' },
  avr: { low: 0.55, high: 0.95, label: '0.55-0.95' },
  fractal_dimension: { low: 1.0, high: 1.65, label: '1.00-1.65' },
  bifurcation_count: { low: 5, high: 40, label: '5-40' },
};

const METRIC_ORDER: Array<keyof BiomarkerMetrics> = [
  'vessel_density',
  'tortuosity',
  'avr',
  'fractal_dimension',
  'bifurcation_count',
];

function formatValue(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Math.abs(value) < 10 && !Number.isInteger(value)) return value.toFixed(3);
  return value.toFixed(2);
}

function getFlag(metric: keyof typeof THRESHOLDS, value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'missing';
  const threshold = THRESHOLDS[metric];
  if (value < threshold.low) return 'low';
  if (value > threshold.high) return 'high';
  return 'normal';
}

function flagClass(flag: string) {
  if (flag === 'low' || flag === 'high') return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
  if (flag === 'missing') return 'text-muted-foreground border-border bg-muted/30';
  return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
}

function eyeLabel(side: EyeSide) {
  return side === 'left' ? 'Left Eye (OS)' : 'Right Eye (OD)';
}

function metricLabel(metric: keyof BiomarkerMetrics) {
  switch (metric) {
    case 'vessel_density': return 'Vessel Density';
    case 'tortuosity': return 'Tortuosity';
    case 'avr': return 'AVR';
    case 'fractal_dimension': return 'Fractal Dimension';
    case 'bifurcation_count': return 'Bifurcation Count';
    default: return metric;
  }
}

function extractSeries(predictions: Array<{ created_at: string; output_payload?: Record<string, unknown> | null }>, side: EyeSide, key: keyof BiomarkerMetrics) {
  return predictions
    .map((prediction) => {
      const biomarkerSet = (prediction.output_payload?.[side === 'left' ? 'vascular_biomarkers_left' : 'vascular_biomarkers_right'] as BiomarkerMetrics | undefined) || undefined;
      const value = biomarkerSet?.[key] as number | null | undefined;
      return {
        date: prediction.created_at,
        value: typeof value === 'number' ? value : null,
      };
    })
    .filter((point) => point.value !== null);
}

function extractComparisonSeries(
  predictions: Array<{ created_at: string; output_payload?: Record<string, unknown> | null }>,
  key: keyof BiomarkerMetrics,
) {
  return predictions
    .map((prediction) => {
      const payload = prediction.output_payload ?? {};
      const left = payload.vascular_biomarkers_left as BiomarkerMetrics | undefined;
      const right = payload.vascular_biomarkers_right as BiomarkerMetrics | undefined;
      const leftValue = left?.[key] as number | null | undefined;
      const rightValue = right?.[key] as number | null | undefined;
      const delta = typeof leftValue === 'number' && typeof rightValue === 'number' ? Math.abs(leftValue - rightValue) : null;
      return {
        date: prediction.created_at,
        left: typeof leftValue === 'number' ? leftValue : null,
        right: typeof rightValue === 'number' ? rightValue : null,
        delta,
      };
    })
    .filter((point) => point.left !== null || point.right !== null || point.delta !== null);
}

function computeStats(values: Array<number | null | undefined>) {
  const filtered = values.filter((value): value is number => typeof value === 'number' && !Number.isNaN(value));
  if (filtered.length === 0) {
    return { latest: null, min: null, max: null, mean: null, stddev: null };
  }
  const latest = filtered[filtered.length - 1];
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  const mean = filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
  const variance = filtered.reduce((sum, value) => sum + (value - mean) ** 2, 0) / filtered.length;
  return { latest, min, max, mean, stddev: Math.sqrt(variance) };
}

function StatBlock({ label, value, threshold, flag }: { label: string; value: number | null | undefined; threshold: string; flag: string }) {
  return (
    <div className={cn('rounded-md border px-3 py-2', flagClass(flag))}>
      <div className='flex items-center justify-between gap-2'>
        <span className='text-[11px] uppercase tracking-widest text-muted-foreground'>{label}</span>
        <Badge variant='outline' className='text-[10px] uppercase'>{threshold}</Badge>
      </div>
      <div className='mt-1 flex items-end justify-between gap-3'>
        <span className='text-lg font-semibold tabular-nums'>{formatValue(value)}</span>
        <span className='text-[11px] uppercase tracking-wide'>{flag}</span>
      </div>
    </div>
  );
}

function MetricTable({ title, biomarkers }: { title: string; biomarkers?: BiomarkerMetrics | null }) {
  return (
    <Card className='border-border/80 bg-card/80'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>{title}</CardTitle>
        <CardDescription className='text-xs'>Raw biomarker metrics with thresholds</CardDescription>
      </CardHeader>
      <CardContent className='space-y-3'>
        <div className='grid gap-2 sm:grid-cols-2 xl:grid-cols-3'>
          {METRIC_ORDER.map((metric) => {
            const value = biomarkers?.[metric] as number | null | undefined;
            const threshold = THRESHOLDS[metric as keyof typeof THRESHOLDS]?.label ?? 'n/a';
            const flag = metric in THRESHOLDS ? getFlag(metric as keyof typeof THRESHOLDS, value) : 'normal';
            return (
              <StatBlock key={metric as string} label={metricLabel(metric)} value={value} threshold={threshold} flag={flag} />
            );
          })}
        </div>

        <div className='grid gap-2 md:grid-cols-2'>
          <div className='rounded-md border border-border/70 bg-muted/20 p-3'>
            <p className='mb-2 text-[11px] uppercase tracking-widest text-muted-foreground'>CRE metrics</p>
            <div className='grid gap-2 sm:grid-cols-3'>
              <div>
                <p className='text-[11px] text-muted-foreground'>Artery CRE</p>
                <p className='text-sm font-semibold tabular-nums'>{formatValue(biomarkers?.cre?.artery_cre)}</p>
              </div>
              <div>
                <p className='text-[11px] text-muted-foreground'>Vein CRE</p>
                <p className='text-sm font-semibold tabular-nums'>{formatValue(biomarkers?.cre?.vein_cre)}</p>
              </div>
              <div>
                <p className='text-[11px] text-muted-foreground'>Samples</p>
                <p className='text-sm font-semibold tabular-nums'>{formatValue(biomarkers?.cre?.width_samples)}</p>
              </div>
            </div>
          </div>
          <div className='rounded-md border border-border/70 bg-muted/20 p-3'>
            <p className='mb-2 text-[11px] uppercase tracking-widest text-muted-foreground'>Bifurcation angles</p>
            <div className='flex flex-wrap gap-2'>
              {(biomarkers?.bifurcation_angles || []).slice(0, 10).map((angle, index) => (
                <Badge key={index} variant='outline' className='tabular-nums'>{Number(angle).toFixed(1)}°</Badge>
              ))}
              {(biomarkers?.bifurcation_angles || []).length === 0 && <p className='text-sm text-muted-foreground'>—</p>}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SideBySideChart({ left, right, metric }: { left?: BiomarkerMetrics | null; right?: BiomarkerMetrics | null; metric: keyof BiomarkerMetrics }) {
  const leftValue = typeof left?.[metric] === 'number' ? (left?.[metric] as number) : null;
  const rightValue = typeof right?.[metric] === 'number' ? (right?.[metric] as number) : null;
  const data = [
    { eye: 'Left', value: leftValue },
    { eye: 'Right', value: rightValue },
  ];
  const missingEyes = data.filter((entry) => entry.value === null).map((entry) => entry.eye);

  return (
    <Card className='border-border/80 bg-card/80'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>{metricLabel(metric)}</CardTitle>
        <CardDescription className='text-xs'>Left/right comparison</CardDescription>
      </CardHeader>
      <CardContent className='h-56 space-y-2'>
        <ResponsiveContainer width='100%' height='100%'>
          <BarChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray='3 3' opacity={0.25} />
            <XAxis dataKey='eye' tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value) => (typeof value === 'number' ? formatValue(value) : 'No data')} />
            <Bar dataKey='value' fill='#14b8a6' radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        {missingEyes.length > 0 && (
          <p className='text-xs text-muted-foreground'>No data for: {missingEyes.join(', ')}</p>
        )}
      </CardContent>
    </Card>
  );
}

function TrendChart({ title, series }: { title: string; series: Array<{ date: string; value: number | null }> }) {
  const data = series.map((point) => ({
    date: new Date(point.date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
    value: point.value,
  }));

  return (
    <Card className='border-border/80 bg-card/80'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>{title}</CardTitle>
        <CardDescription className='text-xs'>Historical trend across predictions</CardDescription>
      </CardHeader>
      <CardContent className='h-56'>
        <ResponsiveContainer width='100%' height='100%'>
          <LineChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray='3 3' opacity={0.25} />
            <XAxis dataKey='date' tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type='monotone' dataKey='value' stroke='#8b5cf6' strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function SummaryCard({ title, value, subtitle }: { title: string; value: string; subtitle: string }) {
  return (
    <div className='rounded-md border border-border/70 bg-muted/20 p-3'>
      <p className='text-[11px] uppercase tracking-widest text-muted-foreground'>{title}</p>
      <p className='mt-1 text-xl font-semibold tabular-nums'>{value}</p>
      <p className='text-xs text-muted-foreground'>{subtitle}</p>
    </div>
  );
}

function StatsGrid({ title, values, threshold }: { title: string; values: Array<number | null | undefined>; threshold: string }) {
  const stats = computeStats(values);
  return (
    <Card className='border-border/80 bg-card/80'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>{title}</CardTitle>
        <CardDescription className='text-xs'>Latest, range, mean, and variability</CardDescription>
      </CardHeader>
      <CardContent className='grid gap-2 sm:grid-cols-2 xl:grid-cols-4'>
        <SummaryCard title='Latest' value={formatValue(stats.latest)} subtitle={`Threshold ${threshold}`} />
        <SummaryCard title='Min / Max' value={`${formatValue(stats.min)} / ${formatValue(stats.max)}`} subtitle='Historical range' />
        <SummaryCard title='Mean' value={formatValue(stats.mean)} subtitle='Arithmetic average' />
        <SummaryCard title='Std Dev' value={formatValue(stats.stddev)} subtitle='Spread across scans' />
      </CardContent>
    </Card>
  );
}

function ComparisonTrend({ title, series }: { title: string; series: Array<{ date: string; left: number | null; right: number | null; delta: number | null }> }) {
  const data = series.map((point) => ({
    date: new Date(point.date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
    left: point.left,
    right: point.right,
    delta: point.delta,
  }));

  return (
    <Card className='border-border/80 bg-card/80'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>{title}</CardTitle>
        <CardDescription className='text-xs'>Bilateral metric asymmetry over time</CardDescription>
      </CardHeader>
      <CardContent className='h-56'>
        <ResponsiveContainer width='100%' height='100%'>
          <LineChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray='3 3' opacity={0.25} />
            <XAxis dataKey='date' tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <ReferenceLine y={0} stroke='rgba(255,255,255,0.15)' />
            <Line type='monotone' dataKey='delta' stroke='#ef4444' strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function ScatterMetric({ title, left, right }: { title: string; left: Array<{ x: number | null; y: number | null }>; right: Array<{ x: number | null; y: number | null }> }) {
  const data = [
    ...left.filter((point) => typeof point.x === 'number' && typeof point.y === 'number').map((point) => ({
      eye: 'Left',
      x: point.x,
      y: point.y,
    })),
    ...right.filter((point) => typeof point.x === 'number' && typeof point.y === 'number').map((point) => ({
      eye: 'Right',
      x: point.x,
      y: point.y,
    })),
  ];

  return (
    <Card className='border-border/80 bg-card/80'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>{title}</CardTitle>
        <CardDescription className='text-xs'>Density vs tortuosity relationship</CardDescription>
      </CardHeader>
      <CardContent className='h-56'>
        <ResponsiveContainer width='100%' height='100%'>
          <ScatterChart margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray='3 3' opacity={0.25} />
            <XAxis dataKey='x' name='Density' tick={{ fontSize: 11 }} />
            <YAxis dataKey='y' name='Tortuosity' tick={{ fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name='Values' data={data} fill='#14b8a6' />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

export function BiomarkerPanel({
  latestLeft,
  latestRight,
  predictions,
}: {
  latestLeft?: BiomarkerMetrics | null;
  latestRight?: BiomarkerMetrics | null;
  predictions: Array<{ created_at: string; output_payload: Record<string, unknown> | null }>;
}) {
  const latestComparison = METRIC_ORDER.map((metric) => {
    const leftValue = latestLeft?.[metric] as number | null | undefined;
    const rightValue = latestRight?.[metric] as number | null | undefined;
    const leftFlag = metric in THRESHOLDS ? getFlag(metric as keyof typeof THRESHOLDS, leftValue) : 'normal';
    const rightFlag = metric in THRESHOLDS ? getFlag(metric as keyof typeof THRESHOLDS, rightValue) : 'normal';
    return { metric, leftValue, rightValue, leftFlag, rightFlag };
  });

  const densitySeriesLeft = extractSeries(predictions, 'left', 'vessel_density');
  const densitySeriesRight = extractSeries(predictions, 'right', 'vessel_density');
  const tortuositySeriesLeft = extractSeries(predictions, 'left', 'tortuosity');
  const tortuositySeriesRight = extractSeries(predictions, 'right', 'tortuosity');
  const densityComparison = extractComparisonSeries(predictions, 'vessel_density');
  const tortuosityComparison = extractComparisonSeries(predictions, 'tortuosity');
  const tortuosityByDate = new Map(tortuosityComparison.map((point) => [point.date, point] as const));
  const buildScatterSeries = (eye: EyeSide) => densityComparison.flatMap((densityPoint) => {
    const tortuosityPoint = tortuosityByDate.get(densityPoint.date);
    if (!tortuosityPoint) return [];
    return [{ x: densityPoint[eye], y: tortuosityPoint[eye] }];
  });
  const scatterLeft = buildScatterSeries('left');
  const scatterRight = buildScatterSeries('right');
  const latestLeftDensity = latestLeft?.vessel_density ?? null;
  const latestRightDensity = latestRight?.vessel_density ?? null;
  const densityDelta = typeof latestLeftDensity === 'number' && typeof latestRightDensity === 'number'
    ? Math.abs(latestLeftDensity - latestRightDensity)
    : null;
  const densityValues = [latestLeftDensity, latestRightDensity].filter((v): v is number => typeof v === 'number');
  const densityMean = densityValues.length > 0 ? densityValues.reduce((a, b) => a + b, 0) / densityValues.length : null;

  return (
    <div className='space-y-4'>
      <Card className='border-border/80 bg-card/80'>
        <CardHeader className='pb-3'>
          <CardTitle className='text-sm'>Biomarker Overview</CardTitle>
          <CardDescription className='text-xs'>Raw metrics, thresholds, and per-eye comparison</CardDescription>
        </CardHeader>
        <CardContent className='grid gap-3 md:grid-cols-4'>
          <SummaryCard title='Latest Left Density' value={formatValue(latestLeftDensity)} subtitle={`Threshold ${THRESHOLDS.vessel_density.label}`} />
          <SummaryCard title='Latest Right Density' value={formatValue(latestRightDensity)} subtitle={`Threshold ${THRESHOLDS.vessel_density.label}`} />
          <SummaryCard title='Left-Right Delta' value={formatValue(densityDelta)} subtitle='Vessel density asymmetry' />
          <SummaryCard title='Mean Density' value={formatValue(densityMean)} subtitle='Latest bilateral mean' />
        </CardContent>
      </Card>

      <div className='grid gap-4 xl:grid-cols-2'>
        <MetricTable title={eyeLabel('left')} biomarkers={latestLeft} />
        <MetricTable title={eyeLabel('right')} biomarkers={latestRight} />
      </div>

      <div className='grid gap-4 xl:grid-cols-2'>
        <SideBySideChart left={latestLeft} right={latestRight} metric='vessel_density' />
        <SideBySideChart left={latestLeft} right={latestRight} metric='tortuosity' />
      </div>

      <div className='grid gap-4 xl:grid-cols-2'>
        <TrendChart title='Vessel Density Trend - Left Eye' series={densitySeriesLeft} />
        <TrendChart title='Vessel Density Trend - Right Eye' series={densitySeriesRight} />
      </div>

      <div className='grid gap-4 xl:grid-cols-2'>
        <TrendChart title='Tortuosity Trend - Left Eye' series={tortuositySeriesLeft} />
        <TrendChart title='Tortuosity Trend - Right Eye' series={tortuositySeriesRight} />
      </div>

      <div className='grid gap-4 xl:grid-cols-2'>
        <ComparisonTrend title='Density Asymmetry' series={densityComparison} />
        <ComparisonTrend title='Tortuosity Asymmetry' series={tortuosityComparison} />
      </div>

      <div className='grid gap-4 xl:grid-cols-2'>
        <StatsGrid title='Density Stats' values={[...densitySeriesLeft.map((point) => point.value), ...densitySeriesRight.map((point) => point.value)]} threshold={THRESHOLDS.vessel_density.label} />
        <StatsGrid title='Tortuosity Stats' values={[...tortuositySeriesLeft.map((point) => point.value), ...tortuositySeriesRight.map((point) => point.value)]} threshold={THRESHOLDS.tortuosity.label} />
      </div>

      <div className='grid gap-4 xl:grid-cols-2'>
        <StatsGrid title='AVR Stats' values={[
          ...predictions.flatMap((prediction) => {
            const payload = prediction.output_payload ?? {};
            const left = payload.vascular_biomarkers_left as BiomarkerMetrics | undefined;
            const right = payload.vascular_biomarkers_right as BiomarkerMetrics | undefined;
            return [left?.avr ?? null, right?.avr ?? null];
          }),
        ]} threshold={THRESHOLDS.avr.label} />
        <StatsGrid title='Fractal Dimension Stats' values={[
          ...predictions.flatMap((prediction) => {
            const payload = prediction.output_payload ?? {};
            const left = payload.vascular_biomarkers_left as BiomarkerMetrics | undefined;
            const right = payload.vascular_biomarkers_right as BiomarkerMetrics | undefined;
            return [left?.fractal_dimension ?? null, right?.fractal_dimension ?? null];
          }),
        ]} threshold={THRESHOLDS.fractal_dimension.label} />
      </div>

      <ScatterMetric
        title='Density vs Tortuosity Scatter'
        left={scatterLeft}
        right={scatterRight}
      />

      <Card className='border-border/80 bg-card/80'>
        <CardHeader className='pb-3'>
          <CardTitle className='text-sm'>Threshold Review</CardTitle>
          <CardDescription className='text-xs'>Latest values with normal range flags</CardDescription>
        </CardHeader>
        <CardContent className='overflow-x-auto'>
          <table className='w-full min-w-[760px] text-sm'>
            <thead className='text-left text-[11px] uppercase tracking-widest text-muted-foreground'>
              <tr className='border-b border-border/60'>
                <th className='py-2 pr-3'>Metric</th>
                <th className='py-2 pr-3'>Left</th>
                <th className='py-2 pr-3'>Left Flag</th>
                <th className='py-2 pr-3'>Right</th>
                <th className='py-2 pr-3'>Right Flag</th>
                <th className='py-2 pr-3'>Threshold</th>
                <th className='py-2 pr-3'>Delta</th>
              </tr>
            </thead>
            <tbody>
              {latestComparison.map((row) => {
                const delta = typeof row.leftValue === 'number' && typeof row.rightValue === 'number'
                  ? Math.abs(row.leftValue - row.rightValue)
                  : null;
                return (
                  <tr key={String(row.metric)} className='border-b border-border/40'>
                    <td className='py-2 pr-3 font-medium'>{metricLabel(row.metric)}</td>
                    <td className='py-2 pr-3 tabular-nums'>{formatValue(row.leftValue)}</td>
                    <td className='py-2 pr-3'><Badge variant='outline' className={cn('uppercase', flagClass(row.leftFlag))}>{row.leftFlag}</Badge></td>
                    <td className='py-2 pr-3 tabular-nums'>{formatValue(row.rightValue)}</td>
                    <td className='py-2 pr-3'><Badge variant='outline' className={cn('uppercase', flagClass(row.rightFlag))}>{row.rightFlag}</Badge></td>
                    <td className='py-2 pr-3 text-muted-foreground'>{THRESHOLDS[row.metric as keyof typeof THRESHOLDS]?.label ?? 'n/a'}</td>
                    <td className='py-2 pr-3 tabular-nums'>{formatValue(delta)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
