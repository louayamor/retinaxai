import { ErrorBoundary } from '@/components/error-boundary';
import PageContainer from '@/components/layout/page-container';
import React from 'react';
import ScrollRestorer from '@/components/scroll-restorer';
import { OverviewStats } from '@/components/features/overview/overview-stats';
import { TriageAlerts } from '@/components/features/overview/triage-alerts';

export default function OverViewLayout({
  bar_stats,
  area_stats,
  pie_stats,
}: {
  bar_stats?: React.ReactNode;
  area_stats?: React.ReactNode;
  pie_stats?: React.ReactNode;
}) {
  return (
    <>
    <ScrollRestorer />
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>
              Welcome to RetinaXAI
            </h1>
            <p className='max-w-xl text-sm text-white/70'>
              AI-assisted diabetic retinopathy detection — Samaya Specialized Center
            </p>
          </div>
        </div>

        <ErrorBoundary>
          <OverviewStats />
        </ErrorBoundary>

        <div className='grid grid-cols-1 gap-6 lg:grid-cols-4'>
          <div className='lg:col-span-3 grid grid-cols-1 gap-6 md:grid-cols-3'>
          <ErrorBoundary>{bar_stats}</ErrorBoundary>
          <ErrorBoundary>{area_stats}</ErrorBoundary>
          <ErrorBoundary>{pie_stats}</ErrorBoundary>
          </div>
          <ErrorBoundary>
            <TriageAlerts />
          </ErrorBoundary>
        </div>
      </div>

      <p className='text-xs text-muted-foreground text-center pt-2'>
        RetinaXAI · Samaya Specialized Center
      </p>
    </PageContainer>
    </>
  );
}
