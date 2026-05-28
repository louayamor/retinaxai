'use client';

import { useEffect, useState, useCallback } from 'react';
import PageContainer from '@/components/layout/page-container';
import { PageHeader, RefreshButton } from '@/components/ui/page-header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { StatsRow } from '@/components/ui/stats-row';
import {
  Loader2,
  Rocket,
  RotateCcw,
  CheckCircle2,
  Clock,
  Layers,
  ArrowUpCircle,
  Undo2,
  ExternalLink,
} from 'lucide-react';
import { toast } from 'sonner';
import {
  getModelRegistryList,
  getCurrentProductionModel,
  promoteModelVersion,
  rollbackModelVersion,
  getModelRegistryDetail,
  type ModelVersionInfo,
  type ModelDetailResponse,
  type ModelListResponse,
  type CurrentProductionResponse,
} from '@/lib/api';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

function fmtPct(v: number | undefined | null): string {
  if (v == null) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

function fmtVal(v: number | undefined | null): string {
  if (v == null) return '—';
  return v.toFixed(4);
}

function StageBadge({ stage }: { stage: string }) {
  switch (stage) {
    case 'production':
      return <Badge className="bg-green-500 hover:bg-green-600 text-white">production</Badge>;
    case 'staging':
      return <Badge variant="secondary">staging</Badge>;
    case 'archived':
      return <Badge variant="outline">archived</Badge>;
    default:
      return <Badge variant="outline">{stage}</Badge>;
  }
}

export default function ModelRegistryPage() {
  const [list, setList] = useState<ModelListResponse | null>(null);
  const [production, setProduction] = useState<CurrentProductionResponse | null>(null);
  const [detail, setDetail] = useState<ModelDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const fetchData = useCallback(async () => {
    const [listData, prodData] = await Promise.all([
      getModelRegistryList().catch(() => null),
      getCurrentProductionModel().catch(() => null),
    ]);
    if (listData) setList(listData);
    if (prodData) setProduction(prodData);
    setLoading(false);
  }, []);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  const handlePromote = async (version: string) => {
    setPromoting(version);
    try {
      await promoteModelVersion(version, 'Promoted from dashboard');
      toast.success(`Promoted ${version} to production`);
      void fetchData();
    } catch (err) {
      toast.error(`Promotion failed: ${String(err).slice(0, 120)}`);
    } finally {
      setPromoting(null);
    }
  };

  const handleRollback = async (version: string) => {
    setPromoting(version);
    try {
      await rollbackModelVersion(version, 'Rolled back from dashboard');
      toast.success(`Rolled back to ${version}`);
      void fetchData();
    } catch (err) {
      toast.error(`Rollback failed: ${String(err).slice(0, 120)}`);
    } finally {
      setPromoting(null);
    }
  };

  const openDetail = async (version: string) => {
    try {
      const data = await getModelRegistryDetail(version);
      setDetail(data);
      setDetailOpen(true);
    } catch (err) {
      toast.error(`Failed to load model detail: ${String(err).slice(0, 120)}`);
    }
  };

  if (loading && !list) {
    return (
      <PageContainer>
        <div className="flex items-center justify-center h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </PageContainer>
    );
  }

  const canPromoteFromStaging = list && list.staging_count > 0;

  return (
    <PageContainer className="flex flex-col gap-6">
      <PageHeader
        title="Model Registry"
        description="Manage model versions, promote to production, and rollback"
        actions={<RefreshButton onClick={fetchData} loading={loading} />}
      />

      <StatsRow columns={4}>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <Layers className="h-4 w-4" /> Total Versions
            </div>
            <p className="text-2xl font-bold">{list?.total ?? '—'}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <Clock className="h-4 w-4" /> Staging
            </div>
            <p className="text-2xl font-bold">{list?.staging_count ?? '—'}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <CheckCircle2 className="h-4 w-4" /> Production
            </div>
            <p className="text-2xl font-bold">{list?.production_count ?? '—'}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <RotateCcw className="h-4 w-4" /> Archived
            </div>
            <p className="text-2xl font-bold">{list?.archived_count ?? '—'}</p>
          </CardContent>
        </Card>
      </StatsRow>

      <Card className={production?.imaging ? 'border-green-500/30' : 'border-muted'}>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            Current Production Model
          </CardTitle>
        </CardHeader>
        <CardContent>
          {production?.imaging ? (
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <p className="text-lg font-semibold font-mono">{production.imaging.version}</p>
                <p className="text-sm text-muted-foreground">
                  Promoted: {production.promoted_at ? new Date(production.promoted_at).toLocaleString() : 'Unknown'}
                </p>
                {production.imaging.metrics && (
                  <div className="flex gap-4 mt-2 text-sm">
                    <span>Acc: {fmtPct(production.imaging.metrics.accuracy)}</span>
                    <span>QWK: {fmtVal(production.imaging.metrics.quadratic_weighted_kappa)}</span>
                    <span>AUC: {fmtVal(production.imaging.metrics.roc_auc_macro)}</span>
                    <span>F1: {fmtPct(production.imaging.metrics.macro_f1)}</span>
                  </div>
                )}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => openDetail(production.imaging!.version)}
              >
                <ExternalLink className="h-4 w-4 mr-1" /> Details
              </Button>
            </div>
          ) : (
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <p className="text-sm text-muted-foreground">No model deployed in production</p>
                {canPromoteFromStaging && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {list!.staging_count} version{list!.staging_count !== 1 ? 's' : ''} available in staging
                  </p>
                )}
              </div>
              {!canPromoteFromStaging && (
                <Badge variant="outline" className="text-xs">No versions to promote</Badge>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Version History</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {!list || list.models.length === 0 ? (
            <div className="p-6 text-center text-sm text-muted-foreground">
              No model versions registered.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Stage</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">QWK</TableHead>
                  <TableHead className="text-right">ROC-AUC</TableHead>
                  <TableHead className="text-right">F1</TableHead>
                  <TableHead className="text-right">Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {list.models.map((item) => {
                  const m = item.model;
                  const isProd = item.is_current_production;
                  return (
                    <TableRow
                      key={m.version}
                      className={isProd ? 'bg-green-500/5' : undefined}
                    >
                      <TableCell>
                        <button
                          type="button"
                          className="font-mono text-sm font-medium hover:text-[var(--brand-teal)] transition-colors cursor-pointer"
                          onClick={() => openDetail(m.version)}
                        >
                          {m.version}
                        </button>
                        {isProd && (
                          <Badge className="ml-2 bg-green-500 text-white text-[10px] px-1 py-0">PROD</Badge>
                        )}
                      </TableCell>
                      <TableCell><StageBadge stage={m.stage} /></TableCell>
                      <TableCell className="text-right tabular-nums">{fmtPct(m.metrics?.accuracy)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtVal(m.metrics?.quadratic_weighted_kappa)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtVal(m.metrics?.roc_auc_macro)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmtPct(m.metrics?.macro_f1)}</TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground">
                        {m.created_at ? new Date(m.created_at).toLocaleDateString() : '—'}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-1">
                          {m.stage === 'staging' && (
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 text-xs"
                              onClick={() => handlePromote(m.version)}
                              disabled={promoting === m.version}
                            >
                              {promoting === m.version ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <ArrowUpCircle className="h-3 w-3 mr-1" />
                              )}
                              Promote
                            </Button>
                          )}
                          {m.stage === 'production' && (
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 text-xs text-amber-600"
                              onClick={() => handleRollback(m.version)}
                              disabled={promoting === m.version}
                            >
                              {promoting === m.version ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <Undo2 className="h-3 w-3 mr-1" />
                              )}
                              Rollback
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className="max-w-2xl">
          {detail && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <span className="font-mono">{detail.model.version}</span>
                  <StageBadge stage={detail.model.stage} />
                  {detail.is_current_production && (
                    <Badge className="bg-green-500 text-white">PRODUCTION</Badge>
                  )}
                </DialogTitle>
                <DialogDescription>
                  Pipeline: {detail.model.pipeline || 'imaging'} &middot;
                  Created: {detail.model.created_at ? new Date(detail.model.created_at).toLocaleString() : 'Unknown'}
                  {detail.model.promoted_at && (
                    <> &middot; Promoted: {new Date(detail.model.promoted_at).toLocaleString()}</>
                  )}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4">
                {detail.model.metrics && (
                  <div>
                    <p className="text-sm font-medium mb-2">Metrics</p>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded border bg-muted/30 p-3">
                        <p className="text-xs text-muted-foreground">Accuracy</p>
                        <p className="text-lg font-semibold tabular-nums">{fmtPct(detail.model.metrics.accuracy)}</p>
                      </div>
                      <div className="rounded border bg-muted/30 p-3">
                        <p className="text-xs text-muted-foreground">Quadratic Weighted Kappa</p>
                        <p className="text-lg font-semibold tabular-nums">{fmtVal(detail.model.metrics.quadratic_weighted_kappa)}</p>
                      </div>
                      <div className="rounded border bg-muted/30 p-3">
                        <p className="text-xs text-muted-foreground">ROC-AUC (macro)</p>
                        <p className="text-lg font-semibold tabular-nums">{fmtVal(detail.model.metrics.roc_auc_macro)}</p>
                      </div>
                      <div className="rounded border bg-muted/30 p-3">
                        <p className="text-xs text-muted-foreground">Macro F1</p>
                        <p className="text-lg font-semibold tabular-nums">{fmtPct(detail.model.metrics.macro_f1)}</p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-xs text-muted-foreground">Artifact</p>
                    <p className="font-mono text-xs truncate">{detail.model.artifact_path || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Hash</p>
                    <p className="font-mono text-xs truncate">{detail.model.hash ? `${detail.model.hash.slice(0, 16)}...` : 'N/A'}</p>
                  </div>
                </div>

                {detail.promotion_history.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Promotion History</p>
                    <div className="text-xs text-muted-foreground space-y-1 max-h-24 overflow-y-auto">
                      {detail.promotion_history.map((h, i) => (
                        <p key={i} className="font-mono">{JSON.stringify(h)}</p>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  {detail.can_promote && (
                    <Button
                      size="sm"
                      onClick={() => {
                        void handlePromote(detail.model.version);
                        setDetailOpen(false);
                      }}
                      disabled={promoting === detail.model.version}
                    >
                      {promoting === detail.model.version ? (
                        <Loader2 className="h-4 w-4 animate-spin mr-1" />
                      ) : (
                        <ArrowUpCircle className="h-4 w-4 mr-1" />
                      )}
                      Promote to Production
                    </Button>
                  )}
                  {detail.can_rollback && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-amber-600"
                      onClick={() => {
                        void handleRollback(detail.model.version);
                        setDetailOpen(false);
                      }}
                      disabled={promoting === detail.model.version}
                    >
                      {promoting === detail.model.version ? (
                        <Loader2 className="h-4 w-4 animate-spin mr-1" />
                      ) : (
                        <Undo2 className="h-4 w-4 mr-1" />
                      )}
                      Rollback
                    </Button>
                  )}
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </PageContainer>
  );
}
