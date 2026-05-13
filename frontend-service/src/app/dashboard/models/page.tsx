'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader, RefreshButton } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AIChartRenderer } from '@/components/charts/ai-chart-renderer';
import { Skeleton } from '@/components/ui/skeleton';
import { getLLMOpsHealth, queryAnalytics } from '@/lib/api';
import type { AnalyticsQueryResponse, AnalyticsSection } from '@/lib/api';
import { MODEL_ANALYTIC_QUERIES } from '@/lib/api';
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
} from 'lucide-react';
import { useWebSocket } from '@/hooks/use-websocket';
import { TrainingProgress } from '@/components/training-progress';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

const MLOPS_BASE = process.env.NEXT_PUBLIC_MLOPS_URL || 'http://localhost:8004';
const REFRESH_INTERVAL_MS = 5 * 60 * 1000;

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

export default function ModelsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [alertLoading, setAlertLoading] = useState(true);
  const [training, setTraining] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<{
    stage: string;
    progress: number;
    status: 'started' | 'running' | 'completed' | 'failed';
    message?: string;
  } | null>(null);

  const [sections, setSections] = useState<AnalyticsSection[]>(() =>
    MODEL_ANALYTIC_QUERIES.map((q) => ({
      ...q,
      response: null,
      loading: true,
      error: null,
    })),
  );
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { connected, subscribe } = useWebSocket();

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

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${MLOPS_BASE}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.warn('MLOps service not available:', err);
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
      console.warn('Failed to fetch alerts:', err);
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
      console.warn('MLOps service not available:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadAnalytics = useCallback(async (silent = false) => {
    if (!silent) {
      setSections((prev) =>
        prev.map((s) => ({ ...s, loading: true, error: null })),
      );
    }

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    const online = await getLLMOpsHealth().then((h) => h?.status === 'ok').catch(() => false);
    if (!online) {
      setSections((prev) =>
        prev.map((s) => ({
          ...s,
          loading: false,
          error: 'Analytics engine unavailable',
        })),
      );
      return;
    }

    const results: Array<
      | { status: 'fulfilled'; value: { key: string; response: AnalyticsQueryResponse } }
      | { status: 'rejected'; reason: unknown }
    > = [];
    for (const q of MODEL_ANALYTIC_QUERIES) {
      if (abortRef.current?.signal.aborted) break;
      try {
        const response = await queryAnalytics(q.question);
        results.push({ status: 'fulfilled', value: { key: q.key, response } });
      } catch (err) {
        results.push({ status: 'rejected', reason: err });
      }
      await new Promise((r) => setTimeout(r, 500));
    }

    setSections((prev) =>
      prev.map((section) => {
        const result = results.find((r) => {
          if (r.status === 'fulfilled') return r.value.key === section.key;
          return false;
        });
        if (result && result.status === 'fulfilled') {
          return {
            ...section,
            response: result.value.response,
            loading: false,
            error: result.value.response.error || null,
          };
        }
        const rejected = results.find((r) => {
          if (r.status === 'rejected') {
            return MODEL_ANALYTIC_QUERIES.find(
              (q, i) => q.key === section.key && i === results.indexOf(r as never),
            );
          }
          return false;
        });
        return {
          ...section,
          response: null,
          loading: false,
          error: rejected
            ? String((rejected as PromiseRejectedResult).reason).slice(0, 200)
            : 'Query failed',
        };
      }),
    );

    if (!silent) {
      const successCount = results.filter((r) => r.status === 'fulfilled').length;
      if (successCount < MODEL_ANALYTIC_QUERIES.length) {
        toast.warning(`${successCount}/${MODEL_ANALYTIC_QUERIES.length} model sections loaded`);
      }
    }
  }, []);

  useEffect(() => {
    void fetchMetrics();
    void fetchStatus();
    void fetchAlerts();
    void loadAnalytics(true);

    intervalRef.current = setInterval(() => {
      void loadAnalytics(true);
    }, REFRESH_INTERVAL_MS);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      abortRef.current?.abort();
    };
  }, [loadAnalytics]);

  const triggerTraining = async (pipeline: 'imaging' | 'clinical') => {
    setTraining(pipeline);
    try {
      const res = await fetch(`${MLOPS_BASE}/api/train/${pipeline}`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        console.log(`Training started: ${data.job_id}`);
        setTimeout(fetchStatus, 2000);
      }
    } catch (err) {
      console.error(`Failed to start ${pipeline} training:`, err);
    } finally {
      setTraining(null);
    }
  };

  const stopTraining = async () => {
    if (!jobStatus?.job_id) return;
    try {
      const res = await fetch(`${MLOPS_BASE}/api/train/${jobStatus.job_id}/stop`, {
        method: 'POST',
      });
      if (res.ok) {
        toast.info('Training stop requested');
        setTraining(null);
        void fetchStatus();
      }
    } catch (err) {
      console.error('Failed to stop training:', err);
    }
  };

  const isTraining = jobStatus?.status === 'running' || jobStatus?.status === 'pending';

  if (loading && !metrics) {
    return (
      <PageContainer>
        <div className='flex items-center justify-center h-[60vh]'>
          <Loader2 className='h-8 w-8 animate-spin text-muted-foreground' />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHeader
        title='AI Models'
        description='Model performance, training controls, and AI-generated analytics'
      />

      <div className='rounded-lg border bg-card p-4'>
        <div className='flex items-center justify-between mb-4'>
          <div className='flex items-center gap-2'>
            <Play className='h-5 w-5' />
            <h3 className='font-semibold'>Training Pipeline</h3>
          </div>
          <RefreshButton
            onClick={() => {
              void Promise.all([fetchMetrics(), fetchStatus(), fetchAlerts()]);
            }}
            loading={loading}
          />
        </div>
        <div className='flex flex-wrap gap-4 items-center'>
          <Button
            onClick={() => void triggerTraining('imaging')}
            disabled={isTraining || training === 'imaging'}
            className='bg-[var(--brand-teal)] hover:bg-[#1a9a9a]'
          >
            {training === 'imaging' ? (
              <Loader2 className='mr-2 h-4 w-4 animate-spin' />
            ) : (
              <Play className='mr-2 h-4 w-4' />
            )}
            Train Imaging
          </Button>
          <Button
            onClick={() => void triggerTraining('clinical')}
            disabled={isTraining || training === 'clinical'}
            variant='outline'
          >
            {training === 'clinical' ? (
              <Loader2 className='mr-2 h-4 w-4 animate-spin' />
            ) : (
              <Play className='mr-2 h-4 w-4' />
            )}
            Train Clinical
          </Button>
          {isTraining && (
            <Button onClick={stopTraining} variant='destructive' size='sm'>
              <Square className='mr-2 h-4 w-4' />
              Stop
            </Button>
          )}
          {isTraining && (
            <Badge variant='secondary' className='ml-auto'>
              <Loader2 className='mr-1 h-3 w-3 animate-spin' />
              {jobStatus?.pipeline} — {jobStatus?.status}
            </Badge>
          )}
          {!connected && (
            <Badge variant='outline' className='ml-2 text-orange-500'>
              <WifiOff className='mr-1 h-3 w-3' />
              Offline
            </Badge>
          )}
        </div>
        {trainingProgress && (
          <div className='mt-4 p-4 bg-muted/50 rounded-lg'>
            <TrainingProgress
              stage={trainingProgress.stage}
              progress={trainingProgress.progress}
              status={trainingProgress.status}
              message={trainingProgress.message}
            />
          </div>
        )}
        {jobStatus?.error && (
          <p className='text-sm text-destructive mt-4'>{jobStatus.error}</p>
        )}
      </div>

      <Card className={cn(alerts && alerts.total > 0 ? 'border-destructive/50 bg-destructive/5' : 'border-border')}>
        <CardHeader>
          <CardTitle className='flex items-center gap-2'>
            <Bell className='h-5 w-5' />
            Active Alerts
            {alerts && alerts.total > 0 && (
              <Badge variant={alerts.firing > 0 ? 'destructive' : 'secondary'}>
                {alerts.total} total
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {alertLoading ? (
            <p className='text-muted-foreground'>Loading alerts...</p>
          ) : !alerts || alerts.total === 0 ? (
            <div className='flex items-center gap-2 text-green-600'>
              <CheckCircle2 className='h-5 w-5' />
              <p className='text-sm'>No active alerts - all systems operational</p>
            </div>
          ) : (
            <div className='space-y-3'>
              {alerts.alerts.map((alert, idx) => (
                <div
                  key={alert.fingerprint || idx}
                  className={cn(
                    'p-3 rounded-lg border text-sm',
                    alert.status === 'firing'
                      ? 'bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900'
                      : 'bg-yellow-50 border-yellow-200 dark:bg-yellow-950/30 dark:border-yellow-900'
                  )}
                >
                  <div className='flex items-start justify-between gap-2'>
                    <div className='flex items-center gap-2 flex-1'>
                      {alert.status === 'firing' ? (
                        <AlertTriangle className='h-4 w-4 text-red-600 flex-shrink-0' />
                      ) : (
                        <AlertCircle className='h-4 w-4 text-yellow-600 flex-shrink-0' />
                      )}
                      <div className='flex-1 min-w-0'>
                        <p className='font-medium truncate'>
                          {alert.labels.alertname || 'Unknown Alert'}
                        </p>
                        <p className='text-xs text-muted-foreground truncate'>
                          {alert.annotations.summary || alert.annotations.description || 'No description'}
                        </p>
                      </div>
                    </div>
                    <Badge
                      variant={alert.status === 'firing' ? 'destructive' : 'secondary'}
                      className='flex-shrink-0 text-xs'
                    >
                      {alert.status || 'unknown'}
                    </Badge>
                  </div>
                  <div className='mt-2 flex flex-wrap gap-2 text-xs'>
                    {alert.labels.severity && (
                      <Badge variant='outline' className='text-xs'>
                        Severity: {alert.labels.severity}
                      </Badge>
                    )}
                    {alert.labels.service && (
                      <Badge variant='outline' className='text-xs'>
                        {alert.labels.service}
                      </Badge>
                    )}
                    {alert.value && (
                      <span className='text-muted-foreground'>Value: {alert.value}</span>
                    )}
                  </div>
                  {alert.annotations.remediation && (
                    <p className='mt-2 text-xs text-muted-foreground'>
                      💡 {alert.annotations.remediation}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Separator />

      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-[var(--brand-teal)]" />
          AI Model Analytics
        </h3>
        <Button variant="outline" size="sm" onClick={() => { void loadAnalytics(false); }}>
          <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
          Refresh
        </Button>
      </div>

      <div className='grid gap-4 lg:grid-cols-2'>
        {sections.map((section) => (
          <AnalyticsSectionCard
            key={section.key}
            section={section}
            onRetry={() => {
              setSections((prev) =>
                prev.map((s) =>
                  s.key === section.key
                    ? { ...s, loading: true, error: null }
                    : s,
                ),
              );
              queryAnalytics(section.question)
                .then((response) => {
                  setSections((prev) =>
                    prev.map((s) =>
                      s.key === section.key
                        ? { ...s, response, loading: false, error: response.error || null }
                        : s,
                    ),
                  );
                })
                .catch((err) => {
                  setSections((prev) =>
                    prev.map((s) =>
                      s.key === section.key
                        ? {
                            ...s,
                            response: null,
                            loading: false,
                            error: String(err).slice(0, 200),
                          }
                        : s,
                    ),
                  );
                });
            }}
          />
        ))}
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
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
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
              <p className="text-[11px] text-muted-foreground mt-2">
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
                  <span className="font-mono text-[10px] text-[var(--brand-teal)]">
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
