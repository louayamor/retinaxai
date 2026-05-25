import PageContainer from '@/components/layout/page-container';
import React from 'react';
import ScrollRestorer from '@/components/scroll-restorer';
import { OverviewStats } from '@/components/features/overview/overview-stats';

export default function ClinicalDashboard() {
  return (
    <>
    <ScrollRestorer />
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>
              Clinical Dashboard
            </h1>
            <p className='max-w-xl text-sm text-white/70'>
              AI-assisted diabetic retinopathy detection — Samaya Specialized Center
            </p>
          </div>
        </div>

        <OverviewStats />

        <p className='text-xs text-muted-foreground text-center pt-2'>
          RetinaXAI · Samaya Specialized Center
        </p>
      </div>
    </PageContainer>
    </>
  );
}
