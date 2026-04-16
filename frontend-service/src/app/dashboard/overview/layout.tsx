import PageContainer from '@/components/layout/page-container';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import Image from 'next/image';
import React from 'react';
import ScrollRestorer from '@/components/scroll-restorer';
import { OverviewStats } from '@/components/overview-stats';

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
        {/* Hero Section */}
        <div className='animate-in-up relative overflow-hidden rounded-2xl border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-10 text-white shadow-lg'>
          <div className='absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(45,212,191,0.18),transparent_45%)]' />
          <div className='absolute right-0 top-0 h-full w-1/3 opacity-15'>
            <Image
              src='https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=800&q=80'
              alt='Medical Technology'
              fill
              className='object-cover'
              priority
              sizes='(max-width: 768px) 0vw, 33vw'
            />
          </div>
          <div className='relative z-10 space-y-2'>
            <div className='flex items-center gap-3 mb-4'>
              <Image
                src='https://www.samayahospital.ae/home/images/logo.png'
                alt='Samaya Logo'
                width={160}
                height={40}
                className='h-10 w-auto object-contain brightness-0 invert'
                sizes='160px'
                quality={100}
                priority
                unoptimized
              />
              <span className='text-lg font-medium text-[var(--brand-teal)]'>|</span>
              <span className='text-lg font-medium'>RetinaXAI</span>
            </div>
            <h1 className='text-3xl font-bold tracking-tight md:text-4xl'>
              Welcome to RetinaXAI
            </h1>
            <p className='max-w-xl text-base text-white/80 md:text-lg'>
              AI-assisted diabetic retinopathy detection powered by Samaya Specialized Center.
              Love your Eyes. Love your Future.
            </p>
          </div>
        </div>

        {/* Real-time Stats from API */}
        <OverviewStats />

        {/* Charts */}
        <div className='grid gap-6 md:grid-cols-2 lg:grid-cols-7'>
          <div className='col-span-4'>{bar_stats}</div>
          <div className='col-span-4 md:col-span-3'>{pie_stats}</div>
          <div className='col-span-4 lg:col-span-7'>{area_stats}</div>
        </div>

        {/* Samaya Info */}
        <div className='grid gap-6 md:grid-cols-2'>
          <Card className='animate-in-up border-0 shadow-md'>
            <CardHeader>
              <CardTitle>About Samaya Specialized Center</CardTitle>
            </CardHeader>
            <CardContent className='space-y-4'>
              <p className='text-muted-foreground'>
                Samaya Specialized Center is a leading ophthalmology hospital in Abu Dhabi,
                providing expert eye care with advanced technology and compassionate treatment.
              </p>
              <div className='grid gap-3 sm:grid-cols-2'>
                <div className='rounded-lg bg-secondary p-4'>
                  <p className='text-sm font-medium'>Al Bateen Branch</p>
                  <p className='text-xs text-muted-foreground'>St. No. 6, Opp. Indonesian Embassy</p>
                  <p className='text-xs text-muted-foreground'>+971 2 885 3888</p>
                </div>
                <div className='rounded-lg bg-secondary p-4'>
                  <p className='text-sm font-medium'>Khalifa City Branch</p>
                  <p className='text-xs text-muted-foreground'>Al Asayil Street, Khalifa City A</p>
                  <p className='text-xs text-muted-foreground'>+971 2 885 3888</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className='animate-in-up border-0 shadow-md'>
            <CardHeader>
              <CardTitle>Services</CardTitle>
            </CardHeader>
            <CardContent>
              <div className='grid gap-3 sm:grid-cols-2'>
                {services.map((s) => (
                  <div key={s} className='flex items-center gap-2 rounded-lg border p-3'>
                    <div className='h-2 w-2 rounded-full bg-[var(--brand-teal)]' />
                    <span className='text-sm'>{s}</span>
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
