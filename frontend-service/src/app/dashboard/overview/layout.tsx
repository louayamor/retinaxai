import PageContainer from '@/components/layout/page-container';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import React from 'react';
import ScrollRestorer from '@/components/scroll-restorer';
import { OverviewStats } from '@/components/features/overview/overview-stats';

const services = [
  'Contoura LASIK',
  'Cataract',
  'Keratoconus',
  'Glaucoma',
  'Diabetic Retinopathy',
  'Dry Eyes'
];

// Server component wrapper for client stats
async function OverviewStatsWrapper() {
  return <OverviewStats />;
}

export default function OverViewLayout({
  pie_stats,
  bar_stats,
  area_stats
}: {
  pie_stats: React.ReactNode;
  bar_stats: React.ReactNode;
  area_stats: React.ReactNode;
}) {
  return (
    <>
    <ScrollRestorer />
    <PageContainer>
      <div className='flex flex-1 flex-col gap-8 min-h-0'>
        {/* Welcome Header */}
        <div className='animate-in-up rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>
              Welcome to RetinaXAI
            </h1>
            <p className='max-w-xl text-sm text-white/70'>
              AI-assisted diabetic retinopathy detection — Samaya Specialized Center
            </p>
          </div>
        </div>

        {/* Real-time Stats from API */}
        <OverviewStats />

        {/* Charts */}
        <div className='grid gap-5 md:grid-cols-2 lg:grid-cols-7'>
          <div className='col-span-4'>{bar_stats}</div>
          <div className='col-span-4 md:col-span-3'>{pie_stats}</div>
          <div className='col-span-4 lg:col-span-7'>{area_stats}</div>
        </div>

        {/* Samaya Info */}
        <div className='grid gap-5 md:grid-cols-2'>
          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className='text-sm font-semibold'>About Samaya Specialized Center</CardTitle>
            </CardHeader>
            <CardContent className='space-y-3'>
              <p className='text-xs text-muted-foreground'>
                Leading ophthalmology hospital in Abu Dhabi, providing expert eye care with advanced
                technology and compassionate treatment.
              </p>
              <div className='grid gap-2 sm:grid-cols-2'>
                <div className='rounded-md bg-secondary p-3'>
                  <p className='text-xs font-medium'>Al Bateen Branch</p>
                  <p className='text-[11px] text-muted-foreground'>St. No. 6, Opp. Indonesian Embassy</p>
                  <p className='text-[11px] text-muted-foreground'>+971 2 885 3888</p>
                </div>
                <div className='rounded-md bg-secondary p-3'>
                  <p className='text-xs font-medium'>Khalifa City Branch</p>
                  <p className='text-[11px] text-muted-foreground'>Al Asayil Street, Khalifa City A</p>
                  <p className='text-[11px] text-muted-foreground'>+971 2 885 3888</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className='text-sm font-semibold'>Services</CardTitle>
            </CardHeader>
            <CardContent>
              <div className='grid gap-2 sm:grid-cols-2'>
                {services.map((s) => (
                  <div key={s} className='flex items-center gap-2 rounded-md border p-2.5'>
                    <div className='h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--brand-teal)]' />
                    <span className='text-xs'>{s}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </PageContainer>
    </>
  );
}
