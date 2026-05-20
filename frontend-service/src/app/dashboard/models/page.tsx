'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader, RefreshButton } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AIChartRenderer } from '@/components/charts/ai-chart-renderer';
import { Skeleton } from '@/components/ui/skeleton';
import { startMLOpsTraining, stopMLOpsTraining } from '@/lib/api';
import { useLazyAnalytics } from '@/hooks/use-lazy-analytics';
import type { AnalyticsSection } from '@/lib/api';
import {
  Loader2,
  Play,
  RefreshCw,
  Square,
  CheckCircle2,
  WifiOff,
  AlertTriangle,
  AlertCircle,
  Bell,
  Sparkles,
  BarChart3,
  PauseCircle,
  Activity,
  TrendingUp,
  Target,
  Layers,
  ChevronDown,
  ChevronUp,
  Brain,
  Crosshair,
  BarChart2,
  Clock,
} from 'lucide-react';
import { useWebSocket } from '@/hooks/use-websocket';
import { TrainingProgress } from '@/components/training-progress';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';

interface SplitMetrics {
  split: string;
  accuracy: number;
  quadratic_weighted_kappa: number;
  roc_auc_macro: number;
  macro_f1: number;
  precision_macro: number;
  recall_macro: number;
  num_samples: number;
  confusion_matrix: number[][];
  classification_report: Record<string, unknown>;
  label_distribution: Record<string, number>;
  class_0_recall: number;
  class_1_recall: number;
  class_2_recall: number;
  class_3_recall: number;
  class_4_recall: number;
  class_0_f1: number;
  class_1_f1: number;
  class_2_f1: number;
  class_3_f1: number;
  class_4_f1: number;
  class_0_precision: number;
  class_1_precision: number;
  class_2_precision: number;
  class_3_precision: number;
  class_4_precision: number;
}

interface Metrics {
  imaging?: {
    accuracy?: number;
    quadratic_weighted_kappa?: number;
    roc_auc_macro?: number;
    num_samples?: number;
  };
  clinical?: {
    accuracy?: number;
    quadratic_weighted_kappa?: number;
    roc_auc_macro?: number;
    num_samples?: number;
  };
  imaging_detail?: Record<string, unknown>;
  training_summary?: Record<string, unknown>;
}

interface JobStatus {
  job_id: string | null;
  pipeline: string | null;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

interface AlertItem {
  labels: Record<string, string>;
  annotations: Record<string, string>;
  startsAt: string;
  endsAt: string | null;
  status: string | null;
  value: string | null;
  fingerprint?: string;
}

interface AlertsResponse {
  alerts: AlertItem[];
  total: number;
  firing: number;
  pending: number;
}

const DR_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'] as const;

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function formatVal(v: number): string {
  return v.toFixed(4);
}

/**
 * Normalize training errors into a user-friendly summary.
 */
function formatTrainingError(error: string | null | undefined): {
  title: string;
  summary: string;
  hint?: string;
  details?: string;
} | null {
  if (!error) return null;
  const text = String(error).trim();
  if (!text) return null;

  if (/cuda out of memory/i.test(text)) {
    return {
      title: 'GPU out of memory',
      summary: 'Training ran out of GPU memory during this run.',
      hint:
        'Try lowering batch size or gradient accumulation, or stop other GPU processes.',
      details: text,
    };
  }

  if (/connection|timeout|network/i.test(text)) {
    return {
      title: 'Network issue',
      summary: 'The training service could not reach a required dependency.',
      hint: 'Check MLOps/LLMOps services and network connectivity.',
      details: text,
    };
  }

  return {
    title: 'Training failed',
    summary: text.length > 160 ? `${text.slice(0, 160)}…` : text,
    details: text,
  };
}

function MetricCard({
  label,
  value,
  icon: Icon,
  accent,
  sub,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  accent: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border bg-card p-3 flex flex-col gap-1">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" style={{ color: accent }} />
        {label}
      </div>
      <div className="text-lg font-semibold tabular-nums" style={{ color: accent }}>
        {value}
      </div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

function SplitPanel({
  title,
  metrics,
  accent,
  confusionMatrixUrl,
  misclassifiedCount,
}: {
  title: string;
  metrics: SplitMetrics;
  accent: string;
  confusionMatrixUrl?: string;
  misclassifiedCount?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: accent }} />
          {title}
          <Badge variant="secondary" className="text-xs px-1.5 py-0 ml-auto">
            n={metrics.num_samples.toLocaleString()}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Accuracy"
            value={formatPct(metrics.accuracy)}
            icon={Target}
            accent={accent}
            sub={`${metrics.num_samples.toLocaleString()} samples`}
          />
          <MetricCard
            label="QWK"
            value={formatVal(metrics.quadratic_weighted_kappa)}
            icon={TrendingUp}
            accent={accent}
          />
          <MetricCard
            label="ROC-AUC (macro)"
            value={formatVal(metrics.roc_auc_macro)}
            icon={Crosshair}
            accent={accent}
          />
          <MetricCard
            label="Macro-F1"
            value={formatVal(metrics.macro_f1)}
            icon={BarChart2}
            accent={accent}
          />
        </div>

        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {expanded ? 'Hide' : 'Show'} per-class metrics
        </button>

        {expanded && (
          <div className="space-y-2">
            <div className="rounded-lg border overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-muted/50">
                    <th className="px-2 py-1 text-left font-medium">Grade</th>
                    <th className="px-2 py-1 text-right font-medium">Precision</th>
                    <th className="px-2 py-1 text-right font-medium">Recall</th>
                    <th className="px-2 py-1 text-right font-medium">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {DR_LABELS.map((label, i) => (
                    <tr key={i} className="border-t">
                      <td className="px-2 py-1 font-medium">{label}</td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatVal(metrics[`class_${i}_precision` as keyof SplitMetrics] as number)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatVal(metrics[`class_${i}_recall` as keyof SplitMetrics] as number)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatVal(metrics[`class_${i}_f1` as keyof SplitMetrics] as number)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="rounded-lg border overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-muted/50">
                    <th className="px-2 py-1 text-left font-medium">Grade</th>
                    {DR_LABELS.map((l) => (
                      <th key={l} className="px-2 py-1 text-right font-medium">{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.confusion_matrix.map((row, i) => (
                    <tr key={i} className="border-t">
                      <td className="px-2 py-1 font-medium">{DR_LABELS[i]}</td>
                      {row.map((val, j) => (
                        <td
                          key={j}
                          className={cn(
                            'px-2 py-1 text-right tabular-nums',
                            i === j ? 'font-semibold' : 'text-muted-foreground',
                          )}
                        >
                          {val.toLocaleString()}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {confusionMatrixUrl && (
          <div className="rounded-lg border bg-muted/30 p-2">
            <p className="text-xs text-muted-foreground mb-1.5">Confusion Matrix</p>
            <img
              src={confusionMatrixUrl}
              alt={`${title} confusion matrix`}
              className="w-full max-w-[400px] mx-auto rounded"
            />
          </div>
        )}

        {misclassifiedCount !== undefined && (
          <div className="text-xs text-muted-foreground flex items-center gap-1.5">
            <AlertCircle className="h-3 w-3" />
            {misclassifiedCount} misclassified samples
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function TrainingHistoryPanel({
  trainingSummary,
}: {
  trainingSummary: Record<string, unknown>;
}) {
  const epochLog = (trainingSummary.epoch_log ?? []) as Array<Record<string, unknown>>;
  const bestEpoch = trainingSummary.best_epoch as number;
  const bestValQwk = trainingSummary.best_val_qwk as number;
  const bestValAcc = trainingSummary.best_val_acc as number;
  const bestValF1 = trainingSummary.best_val_f1 as number;
  const bestValMae = trainingSummary.best_val_mae as number;
  const phase = trainingSummary.phase as string;
  const totalEpochs = trainingSummary.total_epochs as number;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Activity className="h-4 w-4 text-[var(--brand-teal)]" />
          Training History
          <Badge variant="secondary" className="text-xs px-1.5 py-0 ml-auto">
            {phase} — {totalEpochs} epochs
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <MetricCard label="Best Epoch" value={`#${bestEpoch}`} icon={Clock} accent="var(--brand-teal)" />
          <MetricCard label="Best Val QWK" value={formatVal(bestValQwk)} icon={TrendingUp} accent="var(--brand-teal)" />
          <MetricCard label="Best Val Acc" value={formatPct(bestValAcc)} icon={Target} accent="var(--brand-teal)" />
          <MetricCard label="Best Val MAE" value={formatVal(bestValMae)} icon={BarChart2} accent="var(--brand-teal)" />
        </div>

        {epochLog.length > 0 && (
          <div className="rounded-lg border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-muted/50">
                  <th className="px-2 py-1 text-left font-medium">Epoch</th>
                  <th className="px-2 py-1 text-right font-medium">Loss</th>
                  <th className="px-2 py-1 text-right font-medium">Val Acc</th>
                  <th className="px-2 py-1 text-right font-medium">Val QWK</th>
                  <th className="px-2 py-1 text-right font-medium">Val F1</th>
                  <th className="px-2 py-1 text-right font-medium">Val MAE</th>
                  <th className="px-2 py-1 text-right font-medium">LR</th>
                </tr>
              </thead>
              <tbody>
                {epochLog.map((ep) => {
                  const epoch = ep.epoch as number;
                  const isBest = epoch === bestEpoch;
                  return (
                    <tr
                      key={epoch}
                      className={cn('border-t', isBest && 'bg-[var(--brand-teal)]/10 font-semibold')}
                    >
                      <td className="px-2 py-1">
                        {epoch}
                        {isBest && (
                          <span className="ml-1 text-[var(--brand-teal)]">★</span>
                        )}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {(ep.loss as number).toFixed(4)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatPct(ep.val_acc as number)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatVal(ep.val_qwk as number)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatVal(ep.val_f1 as number)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {formatVal(ep.val_mae as number)}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums text-muted-foreground">
                        {(ep.lr as number).toExponential(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

type MetricKey = 'qwk' | 'macro_f1' | 'roc_auc' | 'per_class_recall';

const METRIC_OPTIONS: { value: MetricKey; label: string }[] = [
  { value: 'qwk', label: 'QWK' },
  { value: 'macro_f1', label: 'Macro F1' },
  { value: 'roc_auc', label: 'ROC-AUC' },
  { value: 'per_class_recall', label: 'Per-Class Recall' },
];

function fmt(v: number): string {
  return (v * 100).toFixed(1) + '%';
}

function MetricsComparisonChart({
  eyepacsMetrics,
  samayaMetrics,
}: {
  eyepacsMetrics: SplitMetrics | undefined;
  samayaMetrics: SplitMetrics | undefined;
}) {
  const [metric, setMetric] = useState<MetricKey>('qwk');

  const bars: { label: string; value: number; color: string }[] = [];

  if (metric === 'per_class_recall') {
    for (let g = 0; g < 5; g++) {
      const key = `class_${g}_recall` as keyof SplitMetrics;
      const ev = eyepacsMetrics?.[key] as number | undefined;
      const sv = samayaMetrics?.[key] as number | undefined;
      if (ev != null) bars.push({ label: `EyePACS G${g}`, value: ev, color: 'var(--brand-teal)' });
      if (sv != null) bars.push({ label: `Samaya  G${g}`, value: sv, color: '#E89020' });
    }
  } else {
    const vals: [string, number | undefined, string][] = [];
    if (eyepacsMetrics) {
      const v = metric === 'qwk' ? eyepacsMetrics.quadratic_weighted_kappa
        : metric === 'macro_f1' ? eyepacsMetrics.macro_f1
        : eyepacsMetrics.roc_auc_macro;
      vals.push(['EyePACS Test', v, 'var(--brand-teal)']);
    }
    if (samayaMetrics) {
      const v = metric === 'qwk' ? samayaMetrics.quadratic_weighted_kappa
        : metric === 'macro_f1' ? samayaMetrics.macro_f1
        : samayaMetrics.roc_auc_macro;
      vals.push(['Samaya Validation', v, '#E89020']);
    }
    for (const [label, value, color] of vals) {
      if (value != null) bars.push({ label, value, color });
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm">
            <BarChart3 className="h-4 w-4 text-[var(--brand-teal)]" />
            Metrics Comparison
          </CardTitle>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MetricKey)}
            className="text-xs rounded border bg-background px-2 py-1 focus:outline-none focus:ring-2 focus:ring-ring"
          >
            {METRIC_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <CardDescription className="text-xs">
          {metric === 'per_class_recall'
            ? 'Recall per DR grade across datasets'
            : `${METRIC_OPTIONS.find((o) => o.value === metric)?.label} across datasets`}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {bars.length === 0 && (
          <p className="text-sm text-muted-foreground">No metrics available.</p>
        )}
        {bars.map((b) => (
          <div key={b.label} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">{b.label}</span>
              <span className="font-mono tabular-nums font-medium">{fmt(b.value)}</span>
            </div>
            <div className="h-2.5 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.min(b.value * 100, 100)}%`, backgroundColor: b.color }}
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function PerClassMetricsGrid({
  eyepacsMetrics,
  samayaMetrics,
}: {
  eyepacsMetrics: SplitMetrics | undefined;
  samayaMetrics: SplitMetrics | undefined;
}) {
  const rows: { grade: number; label: string; ep: { precision: number; recall: number; f1: number } | null; sv: { precision: number; recall: number; f1: number } | null }[] = [];
  for (let g = 0; g < 5; g++) {
    rows.push({
      grade: g,
      label: DR_LABELS[g],
      ep: eyepacsMetrics ? {
        precision: eyepacsMetrics[`class_${g}_precision` as keyof SplitMetrics] as number,
        recall: eyepacsMetrics[`class_${g}_recall` as keyof SplitMetrics] as number,
        f1: eyepacsMetrics[`class_${g}_f1` as keyof SplitMetrics] as number,
      } : null,
      sv: samayaMetrics ? {
        precision: samayaMetrics[`class_${g}_precision` as keyof SplitMetrics] as number,
        recall: samayaMetrics[`class_${g}_recall` as keyof SplitMetrics] as number,
        f1: samayaMetrics[`class_${g}_f1` as keyof SplitMetrics] as number,
      } : null,
    });
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Layers className="h-4 w-4 text-[var(--brand-teal)]" />
          Per-Class Metrics
        </CardTitle>
        <CardDescription className="text-xs">Precision / Recall / F1 by DR grade</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1.5 pr-2 font-medium">Grade</th>
                <th className="text-right px-1 font-medium" colSpan={3}>EyePACS</th>
                <th className="text-right px-1 font-medium" colSpan={3}>Samaya</th>
              </tr>
              <tr className="border-b text-muted-foreground/70">
                <th />
                <th className="text-right px-1 font-medium text-[10px]">P</th>
                <th className="text-right px-1 font-medium text-[10px]">R</th>
                <th className="text-right px-1 font-medium text-[10px]">F1</th>
                <th className="text-right px-1 font-medium text-[10px]">P</th>
                <th className="text-right px-1 font-medium text-[10px]">R</th>
                <th className="text-right px-1 font-medium text-[10px]">F1</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.grade} className="border-b last:border-0">
                  <td className="py-2 pr-2 font-medium">{r.label}</td>
                  <td className="text-right px-1 py-2 tabular-nums">{r.ep ? fmt(r.ep.precision) : '—'}</td>
                  <td className="text-right px-1 py-2 tabular-nums">{r.ep ? fmt(r.ep.recall) : '—'}</td>
                  <td className="text-right px-1 py-2 tabular-nums">{r.ep ? fmt(r.ep.f1) : '—'}</td>
                  <td className="text-right px-1 py-2 tabular-nums">{r.sv ? fmt(r.sv.precision) : '—'}</td>
                  <td className="text-right px-1 py-2 tabular-nums">{r.sv ? fmt(r.sv.recall) : '—'}</td>
                  <td className="text-right px-1 py-2 tabular-nums">{r.sv ? fmt(r.sv.f1) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function ModelHealthCard({
  domainShift,
  trainingSummary,
}: {
  domainShift: { confidence_ece: number; embedding_mmd: number } | undefined;
  trainingSummary: Record<string, unknown> | undefined;
}) {
  const epochLog = (trainingSummary?.epoch_log ?? []) as Array<Record<string, unknown>>;
  const bestEpoch = trainingSummary?.best_epoch as number | undefined;
  const bestValQwk = trainingSummary?.best_val_qwk as number | undefined;
  const bestValAcc = trainingSummary?.best_val_acc as number | undefined;
  const bestValMae = trainingSummary?.best_val_mae as number | undefined;
  const maxQwk = epochLog.length > 0
    ? Math.max(...epochLog.map((e) => e.val_qwk as number))
    : 1;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Activity className="h-4 w-4 text-[var(--brand-teal)]" />
          Model Health
        </CardTitle>
        <CardDescription className="text-xs">
          Domain shift and training trajectory
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {domainShift && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Domain Shift</p>
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded border bg-muted/30 p-2">
                <p className="text-[10px] text-muted-foreground">ECE</p>
                <p className="text-sm font-mono font-semibold">{(domainShift.confidence_ece * 100).toFixed(1)}%</p>
                <div className="h-1.5 w-full rounded-full bg-muted mt-1 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-amber-500"
                    style={{ width: `${Math.min(domainShift.confidence_ece * 100, 100)}%` }}
                  />
                </div>
              </div>
              <div className="rounded border bg-muted/30 p-2">
                <p className="text-[10px] text-muted-foreground">MMD</p>
                <p className="text-sm font-mono font-semibold">{domainShift.embedding_mmd.toFixed(4)}</p>
                <div className="h-1.5 w-full rounded-full bg-muted mt-1 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500"
                    style={{ width: `${Math.min(domainShift.embedding_mmd * 100, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {epochLog.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">QWK over epochs</p>
            <div className="flex items-end gap-[2px] h-12">
              {epochLog.map((ep, i) => {
                const q = ep.val_qwk as number;
                const height = maxQwk > 0 ? (q / maxQwk) * 100 : 0;
                return (
                  <div
                    key={i}
                    className="flex-1 rounded-t transition-all hover:opacity-80"
                    style={{
                      height: `${Math.max(height, 5)}%`,
                      backgroundColor: (ep.epoch as number) === bestEpoch
                        ? 'var(--brand-teal)'
                        : 'hsl(var(--primary) / 0.4)',
                    }}
                    title={`Epoch ${ep.epoch}: QWK=${(q * 100).toFixed(1)}%`}
                  />
                );
              })}
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>Epoch 1</span>
              <span>Epoch {epochLog.length}</span>
            </div>
          </div>
        )}

        {bestEpoch != null && (
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="rounded border bg-muted/30 p-2 text-center">
              <p className="text-[10px] text-muted-foreground">Best Epoch</p>
              <p className="font-semibold">#{bestEpoch}</p>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-center">
              <p className="text-[10px] text-muted-foreground">Best QWK</p>
              <p className="font-semibold tabular-nums">{bestValQwk != null ? (bestValQwk * 100).toFixed(1) + '%' : '—'}</p>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-center">
              <p className="text-[10px] text-muted-foreground">Best MAE</p>
              <p className="font-semibold tabular-nums">{bestValMae != null ? bestValMae.toFixed(4) : '—'}</p>
            </div>
          </div>
        )}

        {!domainShift && !trainingSummary && (
          <p className="text-sm text-muted-foreground">No health or training data available.</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function ModelsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [alertLoading, setAlertLoading] = useState(true);
  const [alertsExpanded, setAlertsExpanded] = useState(false);
  const [training, setTraining] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<{
    stage: string;
    progress: number;
    status: 'started' | 'running' | 'completed' | 'failed';
    message?: string;
  } | null>(null);

  const { connected, subscribe } = useWebSocket();
  const formattedTrainingError = formatTrainingError(jobStatus?.error);

  useEffect(() => {
    const unsubStage = subscribe('training_stage', (data: unknown) => {
      const eventData = data as { stage: string; status: string; progress: number; message?: string; error?: string };
      const { stage, status, progress, message, error } = eventData;
      setTrainingProgress({ stage, progress, status: status as 'started' | 'running' | 'completed' | 'failed', message });

      if (status === 'completed') {
        toast.success(message || 'Training completed successfully!');
        setTraining(null);
        void fetchMetrics();
        void fetchStatus();
      } else if (status === 'failed') {
        toast.error(error || message || 'Training failed');
        setTraining(null);
      } else if (status === 'started' || status === 'running') {
        toast(message || `Stage: ${stage}`, { icon: '🔄' });
      }
    });

    return () => {
      unsubStage();
    };
  }, [subscribe]);

  // Live refresh on model registry events
  useEffect(() => {
    const unsubs = [
      subscribe('model.registered', () => { void fetchMetrics(); void fetchStatus(); }),
      subscribe('model.promoted', () => { void fetchMetrics(); void fetchStatus(); }),
      subscribe('model.rolled_back', () => { void fetchMetrics(); void fetchStatus(); }),
    ];
    return () => unsubs.forEach(u => u());
  }, [subscribe]);

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${MLOPS_BASE}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      toast.error(`MLOps metrics unavailable: ${String(err).slice(0, 120)}`);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${MLOPS_BASE}/metrics/alerts`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data);
      } else {
        setAlerts(null);
      }
    } catch (err) {
      toast.error(`Failed to fetch alerts: ${String(err).slice(0, 120)}`);
      setAlerts(null);
    } finally {
      setAlertLoading(false);
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${MLOPS_BASE}/status`);
      if (res.ok) {
        const data = await res.json();
        setJobStatus(data);
      }
    } catch (err) {
      toast.error(`MLOps status unavailable: ${String(err).slice(0, 120)}`);
    } finally {
      setLoading(false);
    }
  };

  const {
    sections: analyticsSections,
    llmopsOnline: analyticsOnline,
    lastUpdated: analyticsUpdated,
    paused: analyticsPaused,
    refresh: refreshAnalytics,
    retrySection: retryAnalyticsSection,
  } = useLazyAnalytics();

  const isTraining = jobStatus?.status === 'running' || jobStatus?.status === 'pending';
  const analyticsMetadata = analyticsUpdated ? (
    <span className="text-xs text-muted-foreground">
      Last updated: {analyticsUpdated.toLocaleTimeString()}
      {analyticsPaused && (
        <span className="ml-2 text-amber-500 flex items-center gap-1 inline-flex">
          <PauseCircle className="h-3 w-3" />
          Paused
        </span>
      )}
    </span>
  ) : null;

  const triggerTraining = async (pipeline: 'imaging') => {
    setTraining(pipeline);
    try {
      const data = await startMLOpsTraining(pipeline);
      toast.success(`Training started: ${data.job_id}`);
      await Promise.all([fetchMetrics(), fetchStatus(), fetchAlerts()]);
    } catch (err) {
      toast.error(`Failed to start training: ${String(err).slice(0, 120)}`);
    } finally {
      setTraining(null);
    }
  };

  const stopTraining = async () => {
    if (!jobStatus?.job_id) {
      toast.error('No active training job to stop');
      return;
    }
    try {
      await stopMLOpsTraining(jobStatus.job_id);
      toast.success('Stop requested');
      await fetchStatus();
    } catch (err) {
      toast.error(`Failed to stop training: ${String(err).slice(0, 120)}`);
    }
  };

  useEffect(() => {
    void fetchMetrics();
    void fetchStatus();
    void fetchAlerts();
  }, []);

  const eyepacsMetrics = metrics?.imaging_detail?.eyepacs_test as SplitMetrics | undefined;
  const samayaMetrics = metrics?.imaging_detail?.samaya_validation as SplitMetrics | undefined;
  const trainingSummary = metrics?.training_summary;
  const domainShift = metrics?.imaging_detail?.domain_shift as { confidence_ece: number; embedding_mmd: number } | undefined;

  if (loading && !metrics) {
    return (
      <PageContainer>
        <div className="flex items-center justify-center h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer className="flex flex-col gap-6">
      <PageHeader
        title="AI Models"
        description="Model performance, training controls, and AI-generated analytics"
      />

      {/* Top row: Training + Alerts side by side */}
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3 rounded-lg border bg-card p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Play className="h-5 w-5" />
              <h3 className="font-semibold">Training Pipeline</h3>
            </div>
            <RefreshButton
              onClick={() => {
                void Promise.all([fetchMetrics(), fetchStatus(), fetchAlerts()]);
              }}
              loading={loading}
            />
          </div>
          <div className="flex flex-wrap gap-4 items-center">
            <Button
              onClick={() => void triggerTraining('imaging')}
              disabled={isTraining || training === 'imaging'}
              className="bg-[var(--brand-teal)] hover:bg-[#1a9a9a]"
            >
              {training === 'imaging' ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              Train Imaging
            </Button>
            {isTraining && (
              <Button onClick={stopTraining} variant="destructive" size="sm">
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            )}
            {isTraining && (
              <Badge variant="secondary" className="ml-auto">
                <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                {jobStatus?.pipeline} — {jobStatus?.status}
              </Badge>
            )}
            {!connected && (
              <Badge variant="outline" className="ml-2 text-orange-500">
                <WifiOff className="mr-1 h-3 w-3" />
                Offline
              </Badge>
            )}
          </div>
          {trainingProgress && (
            <div className="mt-4 p-4 bg-muted/50 rounded-lg">
              <TrainingProgress
                stage={trainingProgress.stage}
                progress={trainingProgress.progress}
                status={trainingProgress.status}
                message={trainingProgress.message}
              />
            </div>
          )}
          {formattedTrainingError && (
            <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 p-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
                <AlertTriangle className="h-4 w-4" />
                {formattedTrainingError.title}
              </div>
              <p className="mt-1 text-sm text-destructive/90">
                {formattedTrainingError.summary}
              </p>
              {formattedTrainingError.hint && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {formattedTrainingError.hint}
                </p>
              )}
              {formattedTrainingError.details && (
                <details className="mt-2 text-xs text-muted-foreground">
                  <summary className="cursor-pointer">Show error details</summary>
                  <pre className="mt-2 whitespace-pre-wrap break-words rounded bg-muted/60 p-2 text-[11px]">
                    {formattedTrainingError.details}
                  </pre>
                </details>
              )}
            </div>
          )}
        </div>

        <Card className={cn(
          'lg:col-span-2',
          alerts && alerts.total > 0 ? 'border-destructive/50 bg-destructive/5' : 'border-border',
        )}>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Bell className="h-4 w-4" />
              Active Alerts
              {alerts && alerts.total > 0 && (
                <Badge variant={alerts.firing > 0 ? 'destructive' : 'secondary'} className="text-xs px-1.5 py-0">
                  {alerts.total}
                </Badge>
              )}
              {alerts && alerts.total > 0 && (
                <button
                  type="button"
                  onClick={() => setAlertsExpanded(!alertsExpanded)}
                  className="ml-auto text-muted-foreground hover:text-foreground transition-colors"
                >
                  {alertsExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alertLoading ? (
              <p className="text-sm text-muted-foreground">Loading...</p>
            ) : !alerts || alerts.total === 0 ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle2 className="h-4 w-4" />
                <p className="text-sm">All systems operational</p>
              </div>
            ) : alertsExpanded ? (
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {alerts.alerts.map((alert, idx) => (
                  <div
                    key={alert.fingerprint || idx}
                    className={cn(
                      'p-2 rounded-lg border text-xs',
                      alert.status === 'firing'
                        ? 'bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900'
                        : 'bg-yellow-50 border-yellow-200 dark:bg-yellow-950/30 dark:border-yellow-900',
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-1.5 flex-1 min-w-0">
                        {alert.status === 'firing' ? (
                          <AlertTriangle className="h-3 w-3 text-red-600 flex-shrink-0" />
                        ) : (
                          <AlertCircle className="h-3 w-3 text-yellow-600 flex-shrink-0" />
                        )}
                        <p className="font-medium truncate text-xs">
                          {alert.labels.alertname || 'Unknown Alert'}
                        </p>
                      </div>
                      <Badge
                        variant={alert.status === 'firing' ? 'destructive' : 'secondary'}
                        className="flex-shrink-0 text-xs px-1 py-0"
                      >
                        {alert.status || 'unknown'}
                      </Badge>
                    </div>
                    {alert.annotations.summary && (
                      <p className="text-xs text-muted-foreground mt-1 truncate">
                        {alert.annotations.summary}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {alerts.firing > 0 ? `${alerts.firing} firing` : `${alerts.total} pending`}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Model Performance */}
      {(eyepacsMetrics || samayaMetrics) && (
        <>
          <Separator />
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-[var(--brand-teal)]" />
            <h3 className="font-semibold">Model Performance</h3>
            {domainShift && (
              <Badge variant="outline" className="text-xs ml-2">
                ECE {formatVal(domainShift.confidence_ece)} · MMD {formatVal(domainShift.embedding_mmd)}
              </Badge>
            )}
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {eyepacsMetrics && (
              <SplitPanel
                title="EyePACS Test"
                metrics={eyepacsMetrics}
                accent="var(--brand-teal)"
                confusionMatrixUrl={`${MLOPS_BASE}/models/download/imaging/artifacts/confusion_matrix_eyepacs_test.png`}
                misclassifiedCount={58}
              />
            )}
            {samayaMetrics && (
              <SplitPanel
                title="Samaya Validation"
                metrics={samayaMetrics}
                accent="var(--brand-gold)"
                confusionMatrixUrl={`${MLOPS_BASE}/models/download/imaging/artifacts/confusion_matrix_samaya_validation.png`}
                misclassifiedCount={60}
              />
            )}
          </div>
        </>
      )}

      {/* Training History */}
      {trainingSummary && (
        <>
          <Separator />
          <TrainingHistoryPanel trainingSummary={trainingSummary} />
        </>
      )}

      {/* AI Analytics */}
      <Separator />

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-[var(--brand-teal)]" />
            Model Analytics
          </h3>
          {analyticsMetadata}
        </div>
        <Button variant="outline" size="sm" onClick={refreshAnalytics}>
          <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
          Refresh
        </Button>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {analyticsOnline !== false && analyticsSections.map((section) => (
          <div key={section.key} data-analytics-card={section.key}>
            <AnalyticsSectionCard
              section={section}
              onRetry={() => retryAnalyticsSection(section.key)}
            />
          </div>
        ))}
        {analyticsOnline === false && (
          <Card className="border-destructive/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <WifiOff className="h-4 w-4 text-destructive" />
                AI Analytics Unavailable
              </CardTitle>
              <CardDescription className="text-xs text-muted-foreground">
                LLMOps service offline.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" size="sm" onClick={refreshAnalytics}>
                <RefreshCw className="mr-1.5 h-3 w-3" />
                Retry
              </Button>
            </CardContent>
          </Card>
        )}

        <MetricsComparisonChart
          eyepacsMetrics={eyepacsMetrics}
          samayaMetrics={samayaMetrics}
        />
        <PerClassMetricsGrid
          eyepacsMetrics={eyepacsMetrics}
          samayaMetrics={samayaMetrics}
        />
        <ModelHealthCard
          domainShift={domainShift}
          trainingSummary={trainingSummary}
        />
      </div>
    </PageContainer>
  );
}

function AnalyticsSectionCard({
  section,
  onRetry,
}: {
  section: AnalyticsSection;
  onRetry: () => void;
}) {
  const { title, response, loading, error } = section;

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-48" />
          <Skeleton className="h-3 w-72 mt-1" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-[200px] w-full rounded-lg" />
        </CardContent>
      </Card>
    );
  }

  if (error || response?.error) {
    return (
      <Card className="border-destructive/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            {title}
          </CardTitle>
          <CardDescription className="text-xs text-destructive">
            {error || response?.error}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="mr-1.5 h-3 w-3" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const hasSources = response && response.sources.length > 0;
  const hasChart = response?.chart && response.chart.data.length > 0;

  if (response && !response.summary && !hasChart) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No data available. Run training and indexing to populate model metrics.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Sparkles className="h-4 w-4 text-[var(--brand-teal)]" />
          {title}
        </CardTitle>
        <CardDescription className="text-xs flex items-center gap-2">
          <span>AI-generated insight</span>
          {hasSources && (
            <Badge variant="secondary" className="text-xs px-1.5 py-0">
              {response!.sources.length} source{response!.sources.length !== 1 ? 's' : ''}
            </Badge>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {response?.summary && (
          <p className="text-sm leading-relaxed text-muted-foreground">
            {response.summary}
          </p>
        )}

        {hasChart && (
          <div className="rounded-lg border bg-card/50 p-3">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs font-medium">
                {response!.chart!.title}
              </span>
            </div>
            <AIChartRenderer spec={response!.chart!} height={220} />
            {response!.chart!.description && (
              <p className="text-xs text-muted-foreground mt-2">
                {response!.chart!.description}
              </p>
            )}
          </div>
        )}

        {hasSources && (
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
              View data sources ({response!.sources.length})
            </summary>
            <div className="mt-2 space-y-1.5">
              {response!.sources.map((src, i) => (
                <div
                  key={i}
                  className="rounded border bg-muted/30 px-2.5 py-1.5"
                >
                  <span className="font-mono text-xs text-[var(--brand-teal)]">
                    {src.artifact_id}
                  </span>
                  <p className="mt-0.5 text-muted-foreground leading-relaxed">
                    {src.snippet}
                  </p>
                </div>
              ))}
            </div>
          </details>
        )}
      </CardContent>
    </Card>
  );
}
