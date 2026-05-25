'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface AdminStats {
  users: { total: number; by_role: Record<string, number>; active: number };
  platform: { patients: number; predictions: number; active_sessions: number };
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<AdminStats | null>(null);

  useEffect(() => {
    const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${API}/api/v1/admin/stats`, { credentials: 'include' })
      .then(r => r.ok ? r.json() : null)
      .then(setStats)
      .catch(() => {});
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
