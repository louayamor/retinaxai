'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AIChartRenderer } from '@/components/charts/ai-chart-renderer';
import { Skeleton } from '@/components/ui/skeleton';
import { getLLMOpsHealth, queryAnalytics } from '@/lib/api';
import type { AnalyticsQueryResponse, AnalyticsSection } from '@/lib/api';
import { MEDICAL_ANALYTIC_QUERIES } from '@/lib/api';
import {
  RefreshCw,
  Sparkles,
  AlertTriangle,
  BarChart3,
  WifiOff,
  FileQuestion,
} from 'lucide-react';
import { toast } from 'sonner';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000;

export default function AnalyticsPage() {
  const [sections, setSections] = useState<AnalyticsSection[]>(() =>
    MEDICAL_ANALYTIC_QUERIES.map((q) => ({
      ...q,
      response: null,
      loading: true,
      error: null,
    })),
  );
  const [llmopsOnline, setLlmopsOnline] = useState<boolean | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const health = await getLLMOpsHealth();
      setLlmopsOnline(health?.status === 'ok');
      return health?.status === 'ok';
    } catch {
      setLlmopsOnline(false);
      return false;
    }
  }, []);

  const loadAll = useCallback(async (silent = false) => {
    if (!silent) {
      setSections((prev) =>
        prev.map((s) => ({ ...s, loading: true, error: null })),
      );
    }

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    const online = await checkHealth();
    if (!online) {
      setSections((prev) =>
        prev.map((s) => ({
          ...s,
          loading: false,
          error: 'Analytics engine unavailable',
        })),
      );
      if (!silent) {
        toast.error('Analytics engine is not available');
      }
      return;
    }

    const results: Array<
      | { status: 'fulfilled'; value: { key: string; response: AnalyticsQueryResponse } }
      | { status: 'rejected'; reason: unknown }
    > = [];
    for (const q of MEDICAL_ANALYTIC_QUERIES) {
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
            return MEDICAL_ANALYTIC_QUERIES.find(
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
    setLastUpdated(new Date());

    if (!silent) {
      const successCount = results.filter((r) => r.status === 'fulfilled').length;
      if (successCount < MEDICAL_ANALYTIC_QUERIES.length) {
        toast.warning(`${successCount}/${MEDICAL_ANALYTIC_QUERIES.length} sections loaded`);
      }
    }
  }, [checkHealth]);

  const refresh = useCallback(() => {
    void loadAll(false);
  }, [loadAll]);

  useEffect(() => {
    void loadAll(true);

    intervalRef.current = setInterval(() => {
      void loadAll(true);
    }, REFRESH_INTERVAL_MS);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      abortRef.current?.abort();
    };
  }, [loadAll]);

  const metadata = lastUpdated ? (
    <span className="text-[11px] text-muted-foreground">
      Last updated: {lastUpdated.toLocaleTimeString()}
    </span>
  ) : null;

  if (llmopsOnline === false) {
    return (
      <PageContainer>
        <div className="flex flex-col items-center justify-center gap-4 py-24">
          <WifiOff className="h-12 w-12 text-muted-foreground opacity-50" />
          <h2 className="text-lg font-semibold">Analytics Engine Unavailable</h2>
          <p className="text-sm text-muted-foreground max-w-md text-center">
            The LLMOps service is not reachable. Start the service and ensure
            ChromaDB has been indexed with model metrics and patient data.
          </p>
          <Button onClick={refresh} variant="outline">
            <RefreshCw className="mr-2 h-4 w-4" />
            Retry
          </Button>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer className="flex flex-col gap-6">
      <PageHeader
        title="Medical Analytics"
        description="AI-generated insights from patient demographics, DR severity distribution, and clinical data"
        metadata={metadata}
        actions={
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Refresh
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
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
            <FileQuestion className="h-4 w-4 text-muted-foreground" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No data available for this section. Data may not exist yet — run
            training or indexing to populate metrics.
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
