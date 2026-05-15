'use client';

import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AIChartRenderer } from '@/components/charts/ai-chart-renderer';
import { Skeleton } from '@/components/ui/skeleton';
import { useLazyAnalytics } from '@/hooks/use-lazy-analytics';
import type { AnalyticsSection } from '@/lib/api';
import {
  RefreshCw,
  Sparkles,
  AlertTriangle,
  BarChart3,
  WifiOff,
  FileQuestion,
  PauseCircle,
} from 'lucide-react';

export default function AnalyticsPage() {
  const {
    sections,
    llmopsOnline,
    lastUpdated,
    paused,
    refresh,
    retrySection,
    containerRef,
  } = useLazyAnalytics();

  const metadata = lastUpdated ? (
    <span className="text-[11px] text-muted-foreground">
      Last updated: {lastUpdated.toLocaleTimeString()}
      {paused && (
        <span className="ml-2 text-amber-500 flex items-center gap-1 inline-flex">
          <PauseCircle className="h-3 w-3" />
          Paused
        </span>
      )}
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
        title="AI Analytics"
        description="AI-generated insights from patient data, model performance, and clinical findings"
        metadata={metadata}
        actions={
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Refresh
          </Button>
        }
      />

      <div ref={containerRef} className="grid gap-4 lg:grid-cols-2">
        {sections.map((section) => (
          <div key={section.key} data-analytics-card={section.key}>
            <AnalyticsSectionCard
              section={section}
              onRetry={() => retrySection(section.key)}
            />
          </div>
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

  if (error) {
    return (
      <Card className="border-destructive/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            {title}
          </CardTitle>
          <CardDescription className="text-xs text-destructive">
            {error}
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
            No data available for this section.
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
