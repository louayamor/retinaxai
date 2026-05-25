'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ServiceHealth {
  backend: 'healthy' | 'unhealthy';
  mlops: 'healthy' | 'unhealthy';
  llmops: 'healthy' | 'unhealthy';
}

interface SystemMetrics {
  cpu: { usage_percent: number };
  memory: { total_gb: number; used_gb: number; usage_percent: number };
  disk: { usage_percent: number };
}

export default function EngineeringDashboard() {
  const [services, setServices] = useState<ServiceHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    Promise.all([
      fetch(`${API}/api/v1/system/metrics`, { credentials: 'include' }).then(r => r.ok ? r.json() : null),
      fetch(`${API}/api/v1/stats`, { credentials: 'include' }).then(r => r.ok ? r.json() : null),
    ]).then(([sysMetrics, sysStats]) => {
      setMetrics(sysMetrics);
      if (sysStats?.services) {
        setServices(sysStats.services);
      }
    }).catch(() => {});
  }, []);

  return (
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>
              Engineering Dashboard
            </h1>
            <p className='max-w-xl text-sm text-white/70'>
              System monitoring, model performance, and infrastructure
            </p>
          </div>
        </div>

        <div className='grid gap-4 md:grid-cols-3'>
          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className='text-sm font-medium text-muted-foreground'>Backend</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant={services?.backend === 'healthy' ? 'default' : 'destructive'}>
                {services?.backend ?? 'checking'}
              </Badge>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className='text-sm font-medium text-muted-foreground'>MLOps</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant={services?.mlops === 'healthy' ? 'default' : 'destructive'}>
                {services?.mlops ?? 'checking'}
              </Badge>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className='pb-2'>
              <CardTitle className='text-sm font-medium text-muted-foreground'>LLMOps</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant={services?.llmops === 'healthy' ? 'default' : 'destructive'}>
                {services?.llmops ?? 'checking'}
              </Badge>
            </CardContent>
          </Card>
        </div>

        {metrics && (
          <div className='grid gap-4 md:grid-cols-3'>
            <Card>
              <CardHeader className='pb-2'>
                <CardTitle className='text-sm font-medium text-muted-foreground'>CPU Usage</CardTitle>
              </CardHeader>
              <CardContent className='text-2xl font-bold'>
                {metrics.cpu?.usage_percent ?? '-'}%
              </CardContent>
            </Card>
            <Card>
              <CardHeader className='pb-2'>
                <CardTitle className='text-sm font-medium text-muted-foreground'>Memory</CardTitle>
              </CardHeader>
              <CardContent className='text-2xl font-bold'>
                {metrics.memory ? `${metrics.memory.used_gb.toFixed(1)} / ${metrics.memory.total_gb.toFixed(1)} GB` : '-'}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className='pb-2'>
                <CardTitle className='text-sm font-medium text-muted-foreground'>Disk Usage</CardTitle>
              </CardHeader>
              <CardContent className='text-2xl font-bold'>
                {metrics.disk ? `${metrics.disk.usage_percent}%` : '-'}
              </CardContent>
            </Card>
          </div>
        )}

        <p className='text-xs text-muted-foreground text-center pt-2'>
          RetinaXAI · Infrastructure Monitoring
        </p>
      </div>
    </PageContainer>
  );
}
