'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHero } from '@/components/ui/page-hero';
import { PageSection, PageGrid } from '@/components/ui/page-section';
import { StatsRow } from '@/components/ui/stats-row';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { StatsCard } from '@/components/ui/stats-card';
import {
  Brain,
  Database,
  Loader2,
  RefreshCw,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Clock,
  Zap,
  FileText,
} from 'lucide-react';
import { toast } from 'sonner';

const LLMOPS_BASE = process.env.NEXT_PUBLIC_LLMOPS_URL || 'http://localhost:8002';
const LLMOPS_API_KEY = process.env.NEXT_PUBLIC_LLMOPS_API_KEY || '';

interface HealthStatus {
  status: string;
  llm_provider: string;
  model: string;
}

interface RagStatus {
  status: string;
  collection_name: string;
  total_documents: number;
  last_updated: string | null;
}

interface Operation {
  operation: string;
  status: string;
  progress: number;
  message: string;
  started_at: string;
}

export default function LLMOpsPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [ragStatus, setRagStatus] = useState<RagStatus | null>(null);
  const [operation, setOperation] = useState<Operation | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    try {
      const results = await Promise.allSettled([
        fetch(`${LLMOPS_BASE}/health`, { signal: controller.signal }),
        fetch(`${LLMOPS_BASE}/api/rag/status`, { signal: controller.signal }),
        fetch(`${LLMOPS_BASE}/api/operation`, { headers: { 'x-api-key': LLMOPS_API_KEY }, signal: controller.signal }),
      ]);

      const healthRes = results[0].status === 'fulfilled' ? await results[0].value.json().catch(() => ({ status: 'unavailable', llm_provider: 'unknown', model: 'unknown' })) : { status: 'unavailable', llm_provider: 'unknown', model: 'unknown' };
      const ragRes = results[1].status === 'fulfilled' ? await results[1].value.json().catch(() => ({})) : {};
      const opRes = results[2].status === 'fulfilled' && results[2].value.ok ? await results[2].value.json().catch(() => null) : null;

      setHealth(healthRes);
      setRagStatus(ragRes);
      setOperation(opRes);
    } catch (error) {
      console.error('Failed to fetch LLMOps data:', error);
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 5000);
    void fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, []);

  const handleReindex = async () => {
    setReindexing(true);
    try {
      const res = await fetch(`${LLMOPS_BASE}/api/rag/reindex`, {
        method: 'POST',
        headers: { 'x-api-key': LLMOPS_API_KEY },
      });
      const data = await res.json();
      toast.success(`Reindex triggered: ${data.job_id}`);
      setTimeout(fetchData, 2000);
    } catch (error) {
      toast.error('Failed to trigger reindex');
    } finally {
      setReindexing(false);
    }
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      default: return 'bg-red-500';
    }
  };

  if (loading && !health) {
    return (
      <PageContainer>
        <div className='flex items-center justify-center h-[60vh]'>
          <Loader2 className='h-8 w-8 animate-spin text-muted-foreground' />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHero
        title='LLMOps Monitor'
        description='LLM service, RAG pipeline, and explainability queue'
        actions={
          <Button variant='outline' size='sm' onClick={fetchData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        }
      />

      <StatsRow columns={3}>
        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>LLM Status</div>
            <div className='flex items-center gap-2'>
              <div className={`h-3 w-3 rounded-full ${getHealthColor(health?.status || 'unavailable')}`} />
              <span className='font-medium capitalize'>{health?.status || 'Unknown'}</span>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>Provider</div>
            <div className='font-medium'>{health?.llm_provider || 'Unknown'}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className='pt-6'>
            <div className='flex items-center gap-2 text-sm text-muted-foreground mb-2'>Model</div>
            <div className='font-medium font-mono text-sm'>{health?.model || 'Unknown'}</div>
          </CardContent>
        </Card>
      </StatsRow>

      <PageGrid columns={3}>
        <StatsCard
          title='RAG Documents'
          value={ragStatus?.total_documents || 0}
          icon={FileText}
          subtitle='in collection'
        />
        <StatsCard
          title='Active Operation'
          value={operation && operation.status !== 'idle' ? operation.operation : 'Idle'}
          icon={operation && operation.status !== 'idle' ? Loader2 : Sparkles}
          color={operation?.status === 'completed' ? '#22c55e' : '#3b82f6'}
          subtitle={operation?.message}
        />
        <StatsCard
          title='RAG Status'
          value={ragStatus?.status === 'ready' ? 'Ready' : ragStatus?.status === 'indexing' ? 'Indexing' : 'Unknown'}
          icon={ragStatus?.status === 'ready' ? CheckCircle2 : AlertCircle}
          color={ragStatus?.status === 'ready' ? '#22c55e' : ragStatus?.status === 'indexing' ? '#3b82f6' : '#6b7280'}
        />
      </PageGrid>

      <Tabs defaultValue='rag' className='space-y-4'>
        <TabsList>
          <TabsTrigger value='rag'>RAG Pipeline</TabsTrigger>
          <TabsTrigger value='xai'>XAI Queue</TabsTrigger>
        </TabsList>

        <TabsContent value='rag'>
          <PageSection
            title='RAG Vector Store'
            description='Document collection and indexing status'
            icon={<Database className='h-5 w-5' />}
            headerAction={
              <Button onClick={handleReindex} disabled={reindexing}>
                {reindexing ? (
                  <Loader2 className='h-4 w-4 mr-2 animate-spin' />
                ) : (
                  <RefreshCw className='h-4 w-4 mr-2' />
                )}
                Reindex
              </Button>
            }
          >
            <div className='grid gap-4 md:grid-cols-2'>
              <div className='p-4 rounded-lg border'>
                <p className='text-sm text-muted-foreground'>Collection Name</p>
                <p className='font-medium font-mono mt-1'>{ragStatus?.collection_name || 'N/A'}</p>
              </div>
              <div className='p-4 rounded-lg border'>
                <p className='text-sm text-muted-foreground'>Total Documents</p>
                <p className='font-medium mt-1 text-2xl'>{ragStatus?.total_documents || 0}</p>
              </div>
            </div>

            <div className='mt-4 p-4 rounded-lg border'>
              <div className='flex items-center justify-between'>
                <div>
                  <p className='font-medium'>Last Index Update</p>
                  <p className='text-sm text-muted-foreground'>
                    {ragStatus?.last_updated ? new Date(ragStatus.last_updated).toLocaleString() : 'Never'}
                  </p>
                </div>
                <Badge variant={ragStatus?.status === 'ready' ? 'default' : 'secondary'}>
                  {ragStatus?.status || 'Unknown'}
                </Badge>
              </div>
            </div>
          </PageSection>
        </TabsContent>

        <TabsContent value='xai'>
          <PageSection
            title='Explainability Queue'
            description='Current XAI operation status'
            icon={<Sparkles className='h-5 w-5' />}
          >
            {operation && operation.status !== 'idle' ? (
              <div className='space-y-4'>
                <div className='flex items-center justify-between p-4 rounded-lg border'>
                  <div className='space-y-1'>
                    <p className='font-medium capitalize'>{operation.operation}</p>
                    <p className='text-sm text-muted-foreground'>{operation.message}</p>
                  </div>
                  <Badge className={operation.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'}>
                    {operation.status}
                  </Badge>
                </div>

                {operation.status === 'running' && (
                  <div className='space-y-2'>
                    <div className='flex items-center justify-between text-sm'>
                      <span>Progress</span>
                      <span>{Math.round(operation.progress * 100)}%</span>
                    </div>
                    <div className='h-3 bg-muted rounded-full overflow-hidden'>
                      <div
                        className='h-full bg-blue-500 transition-all'
                        style={{ width: `${operation.progress * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className='text-sm text-muted-foreground'>
                  <Clock className='h-4 w-4 inline mr-1' />
                  Started: {new Date(operation.started_at).toLocaleString()}
                </div>
              </div>
            ) : (
              <div className='text-center py-8'>
                <Zap className='h-8 w-8 mx-auto mb-2 text-muted-foreground opacity-50' />
                <p className='text-muted-foreground'>No active XAI operations</p>
                <p className='text-sm text-muted-foreground mt-1'>XAI explanations are generated on-demand via predictions</p>
              </div>
            )}
          </PageSection>
        </TabsContent>
      </Tabs>
    </PageContainer>
  );
}
