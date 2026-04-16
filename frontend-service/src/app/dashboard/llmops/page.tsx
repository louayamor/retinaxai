'use client';

import { useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Brain, 
  Database, 
  FileText, 
  Loader2, 
  RefreshCw, 
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Clock,
  Zap
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
    try {
      const [healthRes, ragRes, opRes] = await Promise.all([
        fetch(`${LLMOPS_BASE}/health`).then(r => r.json()).catch(() => ({ status: 'unavailable', llm_provider: 'unknown', model: 'unknown' })),
        fetch(`${LLMOPS_BASE}/api/rag/status`).then(r => r.json()).catch(() => ({})),
        fetch(`${LLMOPS_BASE}/api/operation`, { headers: { 'x-api-key': LLMOPS_API_KEY } }).then(r => r.json()).catch(() => null),
      ]);
      setHealth(healthRes);
      setRagStatus(ragRes);
      setOperation(opRes);
    } catch (error) {
      console.error('Failed to fetch LLMOps data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
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
        <div className="flex items-center justify-center h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">LLMOps Monitor</h1>
            <p className="text-muted-foreground">LLM service, RAG pipeline, and explainability queue</p>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="rag">RAG Pipeline</TabsTrigger>
            <TabsTrigger value="xai">XAI Queue</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {/* LLM Service Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  LLM Service Status
                </CardTitle>
                <CardDescription>Current LLM provider and model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="p-4 rounded-lg border">
                    <p className="text-sm text-muted-foreground">Status</p>
                    <div className="flex items-center gap-2 mt-1">
                      <div className={`h-3 w-3 rounded-full ${getHealthColor(health?.status || 'unavailable')}`} />
                      <span className="font-medium capitalize">{health?.status || 'Unknown'}</span>
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border">
                    <p className="text-sm text-muted-foreground">Provider</p>
                    <p className="font-medium mt-1">{health?.llm_provider || 'Unknown'}</p>
                  </div>
                  <div className="p-4 rounded-lg border">
                    <p className="text-sm text-muted-foreground">Model</p>
                    <p className="font-medium mt-1 font-mono text-sm">{health?.model || 'Unknown'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">RAG Documents</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{ragStatus?.total_documents || 0}</div>
                  <p className="text-xs text-muted-foreground">in collection</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Active Operation</CardTitle>
                </CardHeader>
                <CardContent>
                  {operation && operation.status !== 'idle' ? (
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Loader2 className={`h-4 w-4 animate-spin ${operation.status === 'completed' ? 'text-green-500' : 'text-blue-500'}`} />
                        <span className="font-medium capitalize">{operation.operation}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">{operation.message}</p>
                    </div>
                  ) : (
                    <>
                      <div className="text-2xl font-bold">Idle</div>
                      <p className="text-xs text-muted-foreground">No active operations</p>
                    </>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">RAG Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {ragStatus?.status === 'ready' ? (
                      <>
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                        <span className="text-green-500 font-medium">Ready</span>
                      </>
                    ) : ragStatus?.status === 'indexing' ? (
                      <>
                        <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                        <span className="text-blue-500 font-medium">Indexing</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-5 w-5 text-muted-foreground" />
                        <span className="text-muted-foreground">Unknown</span>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* RAG Tab */}
          <TabsContent value="rag" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  RAG Vector Store
                </CardTitle>
                <CardDescription>Document collection and indexing status</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="p-4 rounded-lg border bg-gradient-to-r from-cyan-50/50 to-blue-50/30">
                    <p className="text-sm text-muted-foreground">Collection Name</p>
                    <p className="font-medium mt-1 font-mono">{ragStatus?.collection_name || 'N/A'}</p>
                  </div>
                  <div className="p-4 rounded-lg border bg-gradient-to-r from-purple-50/50 to-pink-50/30">
                    <p className="text-sm text-muted-foreground">Total Documents</p>
                    <p className="font-medium mt-1 text-2xl">{ragStatus?.total_documents || 0}</p>
                  </div>
                </div>

                <div className="p-4 rounded-lg border">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Last Index Update</p>
                      <p className="text-sm text-muted-foreground">
                        {ragStatus?.last_updated ? new Date(ragStatus.last_updated).toLocaleString() : 'Never'}
                      </p>
                    </div>
                    <Badge variant={ragStatus?.status === 'ready' ? 'default' : 'secondary'}>
                      {ragStatus?.status || 'Unknown'}
                    </Badge>
                  </div>
                </div>

                <Button onClick={handleReindex} disabled={reindexing} className="w-full">
                  {reindexing ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Reindexing...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Trigger Full Reindex
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* XAI Tab */}
          <TabsContent value="xai" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  Explainability Queue
                </CardTitle>
                <CardDescription>Current XAI operation status</CardDescription>
              </CardHeader>
              <CardContent>
                {operation && operation.status !== 'idle' ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-lg border">
                      <div className="space-y-1">
                        <p className="font-medium capitalize">{operation.operation}</p>
                        <p className="text-sm text-muted-foreground">{operation.message}</p>
                      </div>
                      <Badge className={operation.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'}>
                        {operation.status}
                      </Badge>
                    </div>
                    
                    {operation.status === 'running' && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Progress</span>
                          <span>{Math.round(operation.progress * 100)}%</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 transition-all duration-300"
                            style={{ width: `${operation.progress * 100}%` }}
                          />
                        </div>
                      </div>
                    )}

                    <div className="text-sm text-muted-foreground">
                      <Clock className="h-4 w-4 inline mr-1" />
                      Started: {new Date(operation.started_at).toLocaleString()}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Zap className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>No active XAI operations</p>
                    <p className="text-sm">XAI explanations are generated on-demand via predictions</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageContainer>
  );
}