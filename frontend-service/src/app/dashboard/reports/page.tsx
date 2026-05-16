'use client';

import { useEffect, useState } from 'react';
import {
  getPatients,
  getPatient,
  listAllReports,
  getReport,
  getRagStatus,
  triggerRagReindex,
  checkLlmoopsHealth,
  getOperationStatus,
  type OperationStatus
} from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { PageHeader } from '@/components/ui/page-header';
import { StatsRow } from '@/components/ui/stats-row';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Database,
  FileText,
  Loader2,
  RefreshCw,
  Sparkles,
  WifiOff,
  FileCheck,
  Clock,
  Activity,
  Search
} from 'lucide-react';
import { toast } from 'sonner';
import type { Report } from '@/types';
import { ReportCard } from '@/components/reports/ReportCard';
import { ReportFilters } from '@/components/reports/ReportFilters';
import { StatsCard } from '@/components/ui/stats-card';
import { useWebSocket } from '@/hooks/use-websocket';

type FilterStatus = 'all' | 'completed' | 'pending' | 'running' | 'failed';

export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [patientNames, setPatientNames] = useState<Record<string, string>>({});
  const [expandedReportId, setExpandedReportId] = useState<string | null>(null);

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<FilterStatus>('all');

  const [ragStatus, setRagStatus] = useState<{
    status: string;
    schema_version?: string;
    run_id?: string;
    artifact_count: number;
  } | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [llmopsDown, setLlmopsDown] = useState(false);
  const [operation, setOperation] = useState<OperationStatus | null>(null);

  const { connected, subscribe } = useWebSocket();

  useEffect(() => {
    const unsub = subscribe('llmops_event', (data) => {
      const event = data as { event: string; data: { status: string; progress: number; message: string; details?: Record<string, unknown> } };
      const { status, message, progress, details } = event.data;

      if (details?.state) {
        setOperation({
          state: details.state as string,
          message: message,
          progress: progress,
          started_at: new Date().toISOString(),
        });
      }

      if (status === 'completed') {
        toast.success(message || 'Operation completed');
      } else if (status === 'failed') {
        toast.error(message || 'Operation failed');
      } else if (status === 'running') {
        toast(message || 'Processing...', { icon: '🔄' });
      }
    });

    return () => {
      unsub();
    };
  }, [subscribe]);

  useEffect(() => {
    void loadReports();
    void loadRagStatus();
  }, []);

  const loadRagStatus = async () => {
    try {
      setRagLoading(true);
      setLlmopsDown(false);

      try {
        await checkLlmoopsHealth();
      } catch {
        setLlmopsDown(true);
        return;
      }

      const status = await getRagStatus();
      setRagStatus(status);

      try {
        const op = await getOperationStatus();
        setOperation(op);
      } catch {
        setOperation(null);
      }
    } catch (err) {
      console.error('Failed to load RAG status:', err);
      setRagStatus({ status: 'error', artifact_count: 0 });
    } finally {
      setRagLoading(false);
    }
  };

  const handleReindex = async () => {
    try {
      setReindexing(true);
      await triggerRagReindex();
      toast.success('RAG reindexing started');
      await loadRagStatus();
    } catch (err) {
      console.error('Failed to trigger reindex:', err);
      toast.error(err instanceof Error ? err.message : 'Failed to trigger reindex');
    } finally {
      setReindexing(false);
    }
  };

  const loadReports = async () => {
    try {
      setReportsLoading(true);
      const response = await listAllReports(1, 50);
      setReports(response.items);

      const patientIds = [...new Set(response.items.map((r) => r.patient_id))];
      const names: Record<string, string> = { ...patientNames };

      for (const pid of patientIds) {
        if (!names[pid]) {
          try {
            const patient = await getPatient(pid);
            names[pid] = `${patient.first_name} ${patient.last_name}`;
          } catch {
            names[pid] = 'Unknown';
          }
        }
      }
      setPatientNames(names);
    } catch (err) {
      console.error('Failed to load reports:', err);
      toast.error('Failed to load reports');
    } finally {
      setReportsLoading(false);
    }
  };

  const toggleReportExpand = async (report: Report) => {
    if (expandedReportId === report.id) {
      setExpandedReportId(null);
    } else {
      if (!selectedReport || selectedReport.id !== report.id) {
        try {
          const fullReport = await getReport(report.id);
          setSelectedReport(fullReport);
        } catch (err) {
          console.error('Failed to load report details:', err);
          toast.error('Failed to load report details');
        }
      }
      setExpandedReportId(report.id);
    }
  };

  const filteredReports = reports.filter((report) => {
    const matchesSearch = !search ||
      patientNames[report.patient_id]?.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === 'all' || report.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const totalReports = reports.length;
  const completedReports = reports.filter(r => r.status === 'completed').length;
  const pendingReports = reports.filter(r => r.status === 'pending' || r.status === 'running').length;
  const failedReports = reports.filter(r => r.status === 'failed').length;
  const successRate = totalReports > 0 ? Math.round((completedReports / totalReports) * 100) : 0;

  return (
    <PageContainer className='flex flex-col gap-6'>
      <PageHeader
        title='Clinical Reports'
        description='AI-generated clinical reports powered by LLM with retrieval-augmented generation'
      />

      <StatsRow columns={4}>
        <StatsCard
          title='Total Reports'
          value={totalReports}
          icon={FileText}
          subtitle='All time'
        />
        <StatsCard
          title='Completed'
          value={completedReports}
          icon={FileCheck}
          color='#22c55e'
        />
        <StatsCard
          title='In Progress'
          value={pendingReports}
          icon={Clock}
          color='#3b82f6'
        />
        <StatsCard
          title='Success Rate'
          value={`${successRate}%`}
          icon={Activity}
          color={successRate >= 80 ? '#22c55e' : successRate >= 50 ? '#eab308' : '#ef4444'}
        />
      </StatsRow>

      {/* Active Operation Banner */}
      {operation && operation.state !== 'idle' && (
        <Card className={operation.state === 'error' ? 'border-destructive' : 'border-primary'}>
          <CardContent className='p-3'>
            <div className='flex items-center gap-2 text-sm font-medium'>
              {operation.state === 'error' ? (
                <span className='text-destructive'>{operation.message}</span>
              ) : (
                <span className='text-muted-foreground'>{operation.message}</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* RAG Status / LLMOps Error */}
      {llmopsDown ? (
        <Card className='border-destructive'>
          <CardContent className='p-4'>
            <div className='flex items-center gap-2 text-sm font-medium text-destructive'>
              <WifiOff className='h-4 w-4' />
              LLMOps Service Unavailable
            </div>
            <p className='text-xs text-muted-foreground mt-1'>
              The LLMOps service is currently down or unreachable.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className='rounded-lg border bg-card p-4'>
          <div className='flex items-center justify-between mb-3'>
            <div className='flex items-center gap-2 text-sm font-semibold'>
              <Database className='h-4 w-4 text-[var(--brand-teal)]' />
              RAG Index
            </div>
            <div className='flex gap-2'>
              <Button variant='outline' size='sm' onClick={loadRagStatus} disabled={ragLoading}>
                <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${ragLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button size='sm' onClick={handleReindex} disabled={reindexing}>
                {reindexing ? (
                  <Loader2 className='mr-1.5 h-3.5 w-3.5 animate-spin' />
                ) : (
                  <Sparkles className='mr-1.5 h-3.5 w-3.5' />
                )}
                Reindex
              </Button>
            </div>
          </div>
          {ragLoading && !ragStatus ? (
            <div className='py-3 text-center'>
              <Loader2 className='mx-auto h-5 w-5 animate-spin text-muted-foreground' />
            </div>
          ) : ragStatus ? (
            <div className='grid gap-3 md:grid-cols-4'>
              <div>
                <p className='text-xs text-muted-foreground'>Status</p>
                <Badge
                  variant='outline'
                  className={`text-xs ${
                    ragStatus.status === 'ok'
                      ? 'bg-green-500 text-white'
                      : ragStatus.status === 'idle'
                      ? 'bg-yellow-500 text-white'
                      : 'bg-red-500 text-white'
                  }`}
                >
                  {ragStatus.status}
                </Badge>
              </div>
              <div>
                <p className='text-xs text-muted-foreground'>Artifacts</p>
                <p className='text-lg font-bold'>{ragStatus.artifact_count}</p>
              </div>
              <div>
                <p className='text-xs text-muted-foreground'>Schema Version</p>
                <p className='font-mono text-xs'>{ragStatus.schema_version || '—'}</p>
              </div>
              <div>
                <p className='text-xs text-muted-foreground'>Run ID</p>
                <p className='font-mono text-xs truncate'>{ragStatus.run_id || '—'}</p>
              </div>
            </div>
          ) : (
            <p className='text-xs text-muted-foreground'>Unable to load RAG status</p>
          )}
        </div>
      )}

      {/* Reports List */}
      <div className='rounded-lg border bg-card'>
        <div className='flex items-center justify-between p-4 pb-2'>
          <h3 className='text-sm font-semibold'>
            Clinical Reports
            <span className='ml-1.5 font-normal text-muted-foreground'>({filteredReports.length} of {reports.length})</span>
          </h3>
          <Button variant='outline' size='sm' onClick={loadReports}>
            <RefreshCw className='mr-1.5 h-3.5 w-3.5' />
            Refresh
          </Button>
        </div>
        <div className='px-4 pb-3'>
          <ReportFilters
            search={search}
            onSearchChange={setSearch}
            status={statusFilter}
            onStatusChange={setStatusFilter}
          />
        </div>
        {reportsLoading ? (
          <div className='py-8 text-center'>
            <Loader2 className='mx-auto h-6 w-6 animate-spin text-muted-foreground' />
            <p className='mt-1 text-xs text-muted-foreground'>Loading reports...</p>
          </div>
        ) : filteredReports.length === 0 ? (
          <div className='flex flex-col items-center justify-center py-10'>
            <FileText className='mb-3 h-10 w-10 text-muted-foreground/30' />
            <p className='text-xs text-muted-foreground text-center'>
              {search || statusFilter !== 'all'
                ? 'No reports match your filters'
                : 'No reports generated yet'}
            </p>
            <p className='text-xs text-muted-foreground/70 mt-0.5'>
              {search || statusFilter !== 'all'
                ? 'Try adjusting your search or filters'
                : 'Generate reports from completed predictions'}
            </p>
          </div>
        ) : (
          <div className='grid gap-2 p-4 pt-0'>
            {filteredReports.map((report) => (
              <ReportCard
                key={report.id}
                report={report}
                patientName={patientNames[report.patient_id] || 'Loading...'}
                expanded={expandedReportId === report.id}
                onExpand={() => toggleReportExpand(report)}
              />
            ))}
          </div>
        )}
      </div>
    </PageContainer>
  );
}
