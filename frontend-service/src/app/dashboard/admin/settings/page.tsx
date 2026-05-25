'use client';

import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function AdminSettingsPage() {
  return (
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>Settings</h1>
            <p className='max-w-xl text-sm text-white/70'>
              Platform configuration and preferences
            </p>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className='text-base'>Platform Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <p className='text-sm text-muted-foreground'>
              Settings management coming soon. This page will allow configuring
              platform-wide options such as authentication policies, rate limits,
              notification defaults, and feature toggles.
            </p>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  );
}
