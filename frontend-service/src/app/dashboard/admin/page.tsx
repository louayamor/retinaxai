'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAdminStats, getAdminHealth } from '@/lib/api';
import type { AdminStats, AdminHealth } from '@/lib/api';

const SERVICE_LABELS: Record<string, string> = {
  backend: 'Backend',
  mlops: 'MLOps',
  llmops: 'LLMOps',
  redis: 'Redis',
  postgres: 'PostgreSQL',
};

export default function AdminDashboard() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [health, setHealth] = useState<AdminHealth | null>(null);

  useEffect(() => {
    getAdminStats().then(setStats).catch(() => {});
    getAdminHealth().then(setHealth).catch(() => {});
  }, []);

  return (
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>
              Admin Dashboard
            </h1>
            <p className='max-w-xl text-sm text-white/70'>
              Platform management, user administration, and analytics
            </p>
          </div>
        </div>

        {health && (
          <div>
            <h2 className='text-sm font-semibold text-muted-foreground mb-3 uppercase tracking-wider'>
              Service Health
            </h2>
            <div className='grid gap-3 grid-cols-5'>
              {Object.entries(SERVICE_LABELS).map(([key, label]) => {
                const status = health[key as keyof AdminHealth];
                return (
                  <Card key={key} className={status === 'healthy' ? 'border-green-500/30' : 'border-red-500/30'}>
                    <CardContent className='flex items-center gap-3 p-4'>
                      <span
                        className={`h-2.5 w-2.5 rounded-full shrink-0 ${
                          status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                        }`}
                      />
                      <div className='min-w-0'>
                        <p className='text-sm font-medium truncate'>{label}</p>
                        <p className='text-xs text-muted-foreground'>
                          {status === 'healthy' ? 'healthy' : 'unreachable'}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        )}

        {stats && (
          <>
            <div className='grid gap-4 md:grid-cols-3'>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Total Users</CardTitle>
                </CardHeader>
                <CardContent className='text-2xl font-bold'>{stats.users.total}</CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Active Users</CardTitle>
                </CardHeader>
                <CardContent className='text-2xl font-bold'>{stats.users.active}</CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Active Sessions</CardTitle>
                </CardHeader>
                <CardContent className='text-2xl font-bold'>{stats.platform.active_sessions}</CardContent>
              </Card>
            </div>

            <div className='grid gap-4 md:grid-cols-3'>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Patients</CardTitle>
                </CardHeader>
                <CardContent className='text-2xl font-bold'>{stats.platform.patients}</CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Predictions</CardTitle>
                </CardHeader>
                <CardContent className='text-2xl font-bold'>{stats.platform.predictions}</CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Users by Role</CardTitle>
                </CardHeader>
                <CardContent>
                  {Object.entries(stats.users.by_role).map(([role, count]) => (
                    <div key={role} className='flex justify-between text-sm'>
                      <span className='capitalize'>{role}</span>
                      <span className='font-medium'>{count}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </>
        )}

        <p className='text-xs text-muted-foreground text-center pt-2'>
          RetinaXAI · Administration
        </p>
      </div>
    </PageContainer>
  );
}
