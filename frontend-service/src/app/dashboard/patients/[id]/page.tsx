'use client';

import { useEffect, useState, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { motion, useReducedMotion, AnimatePresence } from 'motion/react';
import PageContainer from '@/components/layout/page-container';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  ArrowLeft,
  Calendar,
  Scan,
  FileText,
  Activity,
  Eye,
  AlertCircle,
  CheckCircle,
  XCircle,
  Loader,
  BarChart3,
  ImageIcon,
  Brain,
  Layers,
  Sparkles,
  RefreshCw,
  User,
  Phone,
  Hash,
  Wifi,
  WifiOff,
  ChevronRight,
} from 'lucide-react';
import {
  getPatient,
  getPatientScans,
  getPatientOctReports,
  listPatientPredictions,
  listPatientReports,
  createReport,
  generateXAIExplanation,
  generateXAIGradCAM,
  generateXAISeverity,
  generateSHAPExplanation,
  storeXAIResults,
  ApiError,
  getXAIExplanations,
  type XAIResponse,
} from '@/lib/api';
import {
  usePatientWebSocket,
  type PredictionEventData,
  type XAIEventData,
  type SeverityEventData,
  type GradCAMEventData,
} from '@/hooks/use-patient-websocket';
import { toast } from 'sonner';
import type {
  Patient,
  MRIScan,
  OCTReport,
  Prediction,
  Report,
  PaginatedResponse,
} from '@/types';
import Image from 'next/image';
import MedicalReport from '@/components/features/reports/medical-report';
import XAICard from '@/components/features/xai/xai-card';

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];

const GRADE_META: Record<number, { label: string; color: string; bg: string; border: string }> = {
  0: { label: 'No DR',         color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
  1: { label: 'Mild',          color: 'text-cyan-400',    bg: 'bg-cyan-500/10',    border: 'border-cyan-500/30' },
  2: { label: 'Moderate',      color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/30' },
  3: { label: 'Severe',        color: 'text-orange-400',  bg: 'bg-orange-500/10',  border: 'border-orange-500/30' },
  4: { label: 'Proliferative', color: 'text-rose-400',    bg: 'bg-rose-500/10',    border: 'border-rose-500/30' },
};

const OCT_GRADE_COLORS: Record<string, string> = {
  no_dr: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
  mild: 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30',
  moderate: 'bg-amber-500/15 text-amber-400 border border-amber-500/30',
  severe: 'bg-orange-500/15 text-orange-400 border border-orange-500/30',
  proliferative: 'bg-rose-500/15 text-rose-400 border border-rose-500/30',
};

const RISK_META: Record<string, { color: string; bg: string; border: string }> = {
  low:      { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
  moderate: { color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/30' },
  high:     { color: 'text-orange-400',  bg: 'bg-orange-500/10',  border: 'border-orange-500/30' },
  critical: { color: 'text-rose-400',    bg: 'bg-rose-500/10',    border: 'border-rose-500/30' },
};

type TabId = 'scans' | 'analysis' | 'reports' | 'oct';

const TABS: { id: TabId; label: string; icon: typeof Scan }[] = [
  { id: 'scans',    label: 'MRI Scans',  icon: Scan },
  { id: 'analysis', label: 'AI Analysis', icon: Brain },
  { id: 'reports',  label: 'Reports',    icon: FileText },
  { id: 'oct',      label: 'OCT',        icon: Activity },
];

export default function PatientProfilePage() {
  const params = useParams();
  const router = useRouter();
  const patientId = params.id as string;
  const shouldReduceMotion = useReducedMotion();

  const [patient, setPatient] = useState<Patient | null>(null);
  const [scans, setScans] = useState<MRIScan[]>([]);
  const [octReports, setOctReports] = useState<OCTReport[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generatingXAI, setGeneratingXAI] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [xaiData, setXaiData] = useState<Record<string, XAIResponse>>({});
  const [activeTab, setActiveTab] = useState<TabId>('scans');
  const [needsRefresh, setNeedsRefresh] = useState({ predictions: false, reports: false, xai: false });

  const fetchXAIData = useCallback(async (predId: string) => {
    try {
      const xai = await getXAIExplanations(predId);
      if (xai.explanation || xai.severity_report || xai.gradcam_explanation) {
        setXaiData(prev => ({ ...prev, [predId]: xai }));
      }
    } catch {}
  }, []);

  const refreshPredictions = useCallback(async () => {
    try {
      const data = await listPatientPredictions(patientId, 1, 100);
      setPredictions((data as PaginatedResponse<Prediction>).items);
    } catch {}
  }, [patientId]);

  const refreshReports = useCallback(async () => {
    try {
      const data = await listPatientReports(patientId, 1, 100);
      setReports((data as PaginatedResponse<Report>).items);
    } catch {}
  }, [patientId]);

  const { connected } = usePatientWebSocket({
    patientId,
    onPredictionComplete: useCallback((_data: PredictionEventData) => {
      setNeedsRefresh(prev => ({ ...prev, predictions: true }));
    }, []),
    onXAIReady: useCallback(async (data: XAIEventData) => {
      if (data.prediction_id) {
        await fetchXAIData(data.prediction_id);
        setNeedsRefresh(prev => ({ ...prev, xai: true }));
      }
    }, [fetchXAIData]),
    onGradCAMReady: useCallback(async (data: GradCAMEventData) => {
      if (data.prediction_id) await fetchXAIData(data.prediction_id);
    }, [fetchXAIData]),
    onSeverityReady: useCallback(async (data: SeverityEventData) => {
      if (data.prediction_id) {
        await fetchXAIData(data.prediction_id);
        setNeedsRefresh(prev => ({ ...prev, xai: true }));
      }
    }, [fetchXAIData]),
  });

  useEffect(() => {
    const load = async () => {
      try {
        const [patientData, scansData, octData, predsData, repsData] = await Promise.all([
          getPatient(patientId),
          getPatientScans(patientId),
          getPatientOctReports(patientId),
          listPatientPredictions(patientId, 1, 100),
          listPatientReports(patientId, 1, 100),
        ]);
        setPatient(patientData);
        setScans(scansData);
        setOctReports(octData);
        setPredictions((predsData as PaginatedResponse<Prediction>).items);
        setReports((repsData as PaginatedResponse<Report>).items);
      } catch {
        setError('Failed to load patient data');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [patientId]);

  useEffect(() => {
    if (needsRefresh.predictions) { refreshPredictions(); setNeedsRefresh(p => ({ ...p, predictions: false })); }
  }, [needsRefresh.predictions, refreshPredictions]);

  useEffect(() => {
    if (needsRefresh.reports) { refreshReports(); setNeedsRefresh(p => ({ ...p, reports: false })); }
  }, [needsRefresh.reports, refreshReports]);

  const handleGenerateXAI = async (prediction: Prediction) => {
    if (!patient) return;
    setGeneratingXAI(prediction.id);
    try {
      const drGrade = String(prediction.output_payload?.combined_grade ?? prediction.output_payload?.predicted_class ?? 'Unknown');
      const confidence = prediction.confidence_score ?? 0;
      const clinicalFeatures = prediction.input_payload as Record<string, unknown>;

      let shapResult = null;
      try {
        toast.info('Generating SHAP explanations...');
        shapResult = await generateSHAPExplanation(prediction.id, clinicalFeatures || {});
      } catch {}

      toast.info('Generating AI explanation...');
      const xaiResult = await generateXAIExplanation(prediction.id, drGrade, confidence, clinicalFeatures);

      const leftRegions = ((prediction.output_payload?.gradcam_left_regions as any[]) || []).map((r: any) => r.name || r.region || '').filter(Boolean);
      const rightRegions = ((prediction.output_payload?.gradcam_right_regions as any[]) || []).map((r: any) => r.name || r.region || '').filter(Boolean);

      let gradcamResult = null;
      if (leftRegions.length > 0 || rightRegions.length > 0) {
        toast.info('Generating GradCAM interpretation...');
        gradcamResult = await generateXAIGradCAM(prediction.id, leftRegions, rightRegions);
      }

      toast.info('Generating severity assessment...');
      const severityResult = await generateXAISeverity(
        prediction.id,
        { name: `${patient.first_name} ${patient.last_name}`, age: patient.age, gender: patient.gender },
        drGrade,
        (clinicalFeatures?.risk_factors as string[]) || []
      );

      toast.info('Storing XAI results...');
      try {
        await storeXAIResults(prediction.id, {
          explanationContent: xaiResult?.content,
          explanationSummary: xaiResult?.summary,
          shapValues: shapResult ?? undefined,
          gradcamLeftExplanation: gradcamResult?.left_eye_explanation,
          gradcamRightExplanation: gradcamResult?.right_eye_explanation,
          severityContent: severityResult?.content,
          severitySummary: severityResult?.summary,
          severityRiskLevel: severityResult?.risk_level,
          severityRecommendations: severityResult?.recommendations,
          model: 'gpt-4o',
        });
        toast.success('XAI generation complete');
      } catch (e) {
        if (e instanceof ApiError && e.status === 409) toast.warning('XAI already exists for this prediction');
        else throw e;
      }

      await fetchXAIData(prediction.id);
    } catch (err) {
      toast.error('Failed to generate XAI', { description: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      setGeneratingXAI(null);
    }
  };

  const handleGenerateReport = async (predictionId: string) => {
    setGeneratingReport(true);
    try {
      toast.info('Generating clinical report...');
      await createReport(predictionId);
      toast.success('Report generation started');
      setNeedsRefresh(prev => ({ ...prev, reports: true }));
    } catch (err) {
      toast.error('Failed to generate report', { description: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      setGeneratingReport(false);
    }
  };

  const latestPrediction = predictions[0] ?? null;
  const latestGrade = latestPrediction?.output_payload?.combined_grade as number | undefined;
  const latestGradeMeta = latestGrade !== undefined ? GRADE_META[latestGrade] : null;
  const latestSeverity = latestPrediction?.output_payload?.overall_severity as string | undefined;
  const successPredictions = predictions.filter(p => p.status?.toLowerCase() === 'success');

  const tabCounts: Record<TabId, number> = {
    scans: scans.length,
    analysis: successPredictions.length,
    reports: reports.length,
    oct: octReports.length,
  };

  if (loading) {
    return (
      <PageContainer>
        <div className="flex h-96 items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader className="h-6 w-6 animate-spin text-[var(--brand-teal)]" />
            <span className="text-xs text-muted-foreground tracking-widest uppercase">Loading patient record</span>
          </div>
        </div>
      </PageContainer>
    );
  }

  if (error || !patient) {
    return (
      <PageContainer>
        <div className="flex flex-col items-center justify-center py-16 gap-4">
          <AlertCircle className="h-10 w-10 text-rose-500" />
          <p className="text-sm font-medium text-rose-500">{error || 'Patient not found'}</p>
          <Button variant="outline" size="sm" onClick={() => router.push('/dashboard/patients')}>
            <ArrowLeft className="h-3.5 w-3.5 mr-1.5" />
            Back to Patients
          </Button>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <motion.div
        initial={shouldReduceMotion ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
        className="flex flex-col gap-0 min-h-full"
      >
        {/* Top bar */}
        <div className="flex items-center justify-between py-3 border-b border-border mb-5">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-muted-foreground hover:text-foreground"
              onClick={() => router.push('/dashboard/patients')}
            >
              <ArrowLeft className="h-3.5 w-3.5 mr-1" />
              Patients
            </Button>
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/40" />
            <span className="text-sm font-medium">{patient.first_name} {patient.last_name}</span>
          </div>
          <div className="flex items-center gap-2">
            {connected
              ? <><Wifi className="h-3.5 w-3.5 text-emerald-500" /><span className="text-xs text-emerald-500">Live</span></>
              : <><WifiOff className="h-3.5 w-3.5 text-amber-500" /><span className="text-xs text-amber-500">Connecting</span></>
            }
          </div>
        </div>

        {/* Two-column layout: patient identity left, content right */}
        <div className="flex gap-5 items-start">

          {/* LEFT COLUMN — Patient identity panel */}
          <aside className="w-64 flex-shrink-0 sticky top-4 flex flex-col gap-3">

            {/* Identity card */}
            <div className="rounded-md border border-border bg-card overflow-hidden">
              <div className="bg-[var(--sidebar)] px-4 py-5 flex flex-col items-center gap-3 border-b border-border">
                <Avatar className="h-16 w-16 ring-2 ring-[var(--brand-teal)]/30">
                  <AvatarFallback
                    className="text-xl font-semibold text-white"
                    style={{
                      background: patient.gender === 'M'
                        ? 'linear-gradient(135deg, #20bdbe, #1a9a9a)'
                        : 'linear-gradient(135deg, #c8a951, #b09445)',
                    }}
                  >
                    {patient.first_name[0]}{patient.last_name[0]}
                  </AvatarFallback>
                </Avatar>
                <div className="text-center">
                  <p className="font-semibold text-sm text-sidebar-foreground">
                    {patient.first_name} {patient.last_name}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {patient.gender === 'M' ? 'Male' : 'Female'} · {patient.age} yrs
                  </p>
                </div>
              </div>

              <div className="divide-y divide-border">
                <DataRow icon={Hash} label="MRN" value={patient.medical_record_number} mono />
                <DataRow icon={Phone} label="Phone" value={patient.phone || '—'} />
                <DataRow
                  icon={Calendar}
                  label="Registered"
                  value={new Date(patient.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                />
              </div>
            </div>

            {/* Latest DR status */}
            {latestPrediction && (
              <div className="rounded-md border border-border bg-card overflow-hidden">
                <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-muted/30">
                  <Activity className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
                  <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Latest DR</span>
                </div>
                <div className="p-3 flex flex-col gap-3">
                  {latestGradeMeta && (
                    <div className={`rounded border px-3 py-2 ${latestGradeMeta.bg} ${latestGradeMeta.border}`}>
                      <p className="text-xs text-muted-foreground mb-0.5">Grade</p>
                      <p className={`text-base font-bold ${latestGradeMeta.color}`}>{latestGradeMeta.label}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-[var(--brand-teal)]"
                          style={{ width: `${(latestPrediction.confidence_score ?? 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono font-medium">
                        {latestPrediction.confidence_score ? `${(latestPrediction.confidence_score * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  </div>
                  {latestSeverity && (
                    <div>
                      <p className="text-xs text-muted-foreground mb-0.5">Severity</p>
                      <p className="text-sm font-semibold capitalize">{latestSeverity}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Summary stats */}
            <div className="rounded-md border border-border bg-card overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-muted/30">
                <Layers className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Summary</span>
              </div>
              <div className="grid grid-cols-2 divide-x divide-y divide-border">
                <StatCell label="MRI Scans" value={scans.length} accent="teal" />
                <StatCell label="Predictions" value={predictions.length} accent="gold" />
                <StatCell label="Reports" value={reports.length} accent="blue" />
                <StatCell label="OCT" value={octReports.length} accent="purple" />
              </div>
            </div>
          </aside>

          {/* RIGHT COLUMN — Tabbed clinical content */}
          <main className="flex-1 min-w-0 flex flex-col gap-4">

            {/* Tab bar */}
            <div className="flex items-center gap-0 border-b border-border">
              {TABS.map(tab => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      relative flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors
                      ${isActive
                        ? 'text-[var(--brand-teal)] border-b-2 border-[var(--brand-teal)] -mb-px'
                        : 'text-muted-foreground hover:text-foreground border-b-2 border-transparent -mb-px'
                      }
                    `}
                  >
                    <Icon className="h-3.5 w-3.5" />
                    {tab.label}
                    {tabCounts[tab.id] > 0 && (
                      <span className={`ml-1 text-xs rounded-full px-1.5 py-0 ${isActive ? 'bg-[var(--brand-teal)]/15 text-[var(--brand-teal)]' : 'bg-muted text-muted-foreground'}`}>
                        {tabCounts[tab.id]}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Tab content */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={shouldReduceMotion ? false : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.15 }}
              >

                {/* SCANS TAB */}
                {activeTab === 'scans' && (
                  <div className="flex flex-col gap-3">
                    {scans.length === 0
                      ? <EmptyState icon={Scan} title="No MRI Scans" description="No fundus scans have been uploaded for this patient." />
                      : scans.map(scan => (
                        <div key={scan.id} className="rounded-md border border-border bg-card overflow-hidden">
                          <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/20">
                            <div className="flex items-center gap-2">
                              <Scan className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
                              <span className="text-xs font-mono text-muted-foreground">SCAN</span>
                              <span className="text-xs font-mono font-medium">{scan.id.slice(0, 12).toUpperCase()}</span>
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {new Date(scan.uploaded_at).toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                          <div className="p-4 grid grid-cols-2 gap-3">
                            <EyeImagePanel
                              title="Left Eye (OS)"
                              src={scan.left_scan_path ? `${API_BASE}/${scan.left_scan_path}` : undefined}
                              eye="L"
                            />
                            <EyeImagePanel
                              title="Right Eye (OD)"
                              src={scan.right_scan_path ? `${API_BASE}/${scan.right_scan_path}` : undefined}
                              eye="R"
                            />
                          </div>
                        </div>
                      ))
                    }
                  </div>
                )}

                {/* ANALYSIS TAB */}
                {activeTab === 'analysis' && (
                  <div className="flex flex-col gap-4">
                    {/* GradCAM section */}
                    <SectionHeader
                      icon={Eye}
                      title="GradCAM Heatmaps"
                      iconColor="text-amber-500"
                      count={successPredictions.length}
                    />
                    {successPredictions.length === 0
                      ? <EmptyState icon={Eye} title="No GradCAM Available" description="Run a prediction to generate GradCAM heatmaps." />
                      : successPredictions.map(pred => {
                        const grade = pred.output_payload?.combined_grade as number | undefined;
                        const gradeMeta = grade !== undefined ? GRADE_META[grade] : null;
                        return (
                          <div key={pred.id} className="rounded-md border border-border bg-card overflow-hidden">
                            <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/20">
                              <div className="flex items-center gap-2">
                                <StatusDot status={pred.status} />
                                <span className="text-xs font-medium">{pred.model_name}</span>
                                <span className="text-xs text-muted-foreground">{new Date(pred.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                {gradeMeta && (
                                  <span className={`text-xs font-medium px-2 py-0.5 rounded border ${gradeMeta.bg} ${gradeMeta.border} ${gradeMeta.color}`}>
                                    {gradeMeta.label}
                                  </span>
                                )}
                                <span className="text-xs font-mono font-semibold">
                                  {pred.confidence_score ? `${(pred.confidence_score * 100).toFixed(1)}%` : '—'}
                                </span>
                              </div>
                            </div>
                            <div className="p-4 grid grid-cols-2 gap-3">
                              <GradCAMPanel title="Left Eye (OS)" gradcamBase64={pred.output_payload?.gradcam_left as string | undefined} />
                              <GradCAMPanel title="Right Eye (OD)" gradcamBase64={pred.output_payload?.gradcam_right as string | undefined} />
                            </div>
                          </div>
                        );
                      })
                    }

                    {/* XAI Explanations section */}
                    <div className="flex items-center justify-between mt-2">
                      <SectionHeader icon={Brain} title="AI Explanations" iconColor="text-purple-500" count={successPredictions.length} />
                      {latestPrediction && !xaiData[latestPrediction.id] && (
                        <Button
                          onClick={() => handleGenerateXAI(latestPrediction)}
                          disabled={generatingXAI === latestPrediction.id}
                          size="sm"
                          className="h-7 gap-1.5 bg-[var(--brand-teal)] text-white hover:bg-[var(--brand-teal-dark)] text-xs"
                        >
                          {generatingXAI === latestPrediction.id
                            ? <><Loader className="h-3 w-3 animate-spin" />Generating...</>
                            : <><Sparkles className="h-3 w-3" />Generate XAI</>
                          }
                        </Button>
                      )}
                    </div>

                    {successPredictions.length === 0
                      ? <EmptyState icon={Brain} title="No AI Explanations" description="Run a prediction to generate XAI explanations." />
                      : successPredictions.map(pred => {
                        const xai = xaiData[pred.id];
                        const outputPayload = pred.output_payload as Record<string, unknown> | null;
                        const grade = outputPayload?.combined_grade as number | undefined;
                        const gradeMeta = grade !== undefined ? GRADE_META[grade] : null;
                        const shapValues = xai?.explanation?.shap_values || outputPayload?.shap_values as { top_positive: Array<{ name: string; contribution: number }> } | undefined;
                        const explanation = xai?.explanation?.content || outputPayload?.explanation as string | undefined;
                        const gradcamExp = xai?.gradcam_explanation;
                        const severityReport = xai?.severity_report || (
                          outputPayload?.severity_risk_level ? {
                            risk_level: outputPayload.severity_risk_level as string,
                            summary: outputPayload.severity_summary as string | null,
                            recommendations: outputPayload.severity_recommendations as string[] | null,
                          } : null
                        );
                        const hasXAI = shapValues || explanation || severityReport || gradcamExp;
                        const riskMeta = severityReport ? RISK_META[severityReport.risk_level] ?? RISK_META.critical : null;

                        return (
                          <div key={pred.id} className="rounded-md border border-border bg-card overflow-hidden">
                            <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/20">
                              <span className="text-xs font-medium">
                                Prediction — {new Date(pred.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                              </span>
                              <div className="flex items-center gap-2">
                                {riskMeta && severityReport && (
                                  <span className={`text-xs font-medium px-2 py-0.5 rounded border uppercase tracking-wide ${riskMeta.bg} ${riskMeta.border} ${riskMeta.color}`}>
                                    {severityReport.risk_level.replace('_', ' ')} Risk
                                  </span>
                                )}
                                {gradeMeta && (
                                  <span className={`text-xs font-medium px-2 py-0.5 rounded border ${gradeMeta.bg} ${gradeMeta.border} ${gradeMeta.color}`}>
                                    {gradeMeta.label}
                                  </span>
                                )}
                              </div>
                            </div>

                            <div className="p-4 flex flex-col gap-4">
                              {/* Confidence bar */}
                              <div>
                                <div className="flex items-center justify-between mb-1">
                                  <span className="text-xs text-muted-foreground">Confidence Score</span>
                                  <span className="text-xs font-mono font-semibold">
                                    {pred.confidence_score ? `${(pred.confidence_score * 100).toFixed(1)}%` : '—'}
                                  </span>
                                </div>
                                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(pred.confidence_score ?? 0) * 100}%` }}
                                    transition={{ duration: 0.6, ease: 'easeOut' }}
                                    className="h-full bg-[var(--brand-teal)] rounded-full"
                                  />
                                </div>
                              </div>

                              {/* SHAP values */}
                              {shapValues?.top_positive && shapValues.top_positive.length > 0 && (
                                <div>
                                  <div className="flex items-center gap-1.5 mb-2">
                                    <BarChart3 className="h-3.5 w-3.5 text-muted-foreground" />
                                    <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Feature Contributions</span>
                                  </div>
                                  <div className="flex flex-col gap-1.5">
                                    {shapValues.top_positive.slice(0, 5).map((feature, i) => (
                                      <div key={i} className="flex items-center gap-2">
                                        <span className="text-xs w-28 truncate text-muted-foreground">{feature.name}</span>
                                        <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                                          <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${Math.min(Math.abs(feature.contribution) * 100, 100)}%` }}
                                            transition={{ delay: i * 0.07, duration: 0.5 }}
                                            className="h-full bg-emerald-500 rounded-full"
                                          />
                                        </div>
                                        <span className="text-xs font-mono text-muted-foreground w-12 text-right">
                                          {feature.contribution.toFixed(3)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* XAI Card or generate prompt */}
                              {hasXAI ? (
                                <XAICard
                                  predictionId={pred.id}
                                  createdAt={pred.created_at}
                                  data={{
                                    diagnosis: explanation ? (() => { try { return JSON.parse(explanation).diagnosis; } catch { return {}; } })() : undefined,
                                    clinical_findings: explanation ? (() => { try { return JSON.parse(explanation).clinical_findings; } catch { return undefined; } })() : undefined,
                                    severity_report: severityReport ? {
                                      patient: { name: patient?.first_name, age: patient?.age, gender: patient?.gender },
                                      diagnosis: severityReport.risk_level ? { dr_grade: grade, severity_label: severityReport.risk_level, risk_level: severityReport.risk_level } : undefined,
                                      recommendations: severityReport.recommendations?.map((r: string) => ({ action: r })),
                                      summary: severityReport.summary,
                                    } : undefined,
                                    gradcam_explanation: gradcamExp ? {
                                      left_eye_explanation: gradcamExp.left_eye_explanation,
                                      right_eye_explanation: gradcamExp.right_eye_explanation,
                                      highlighted_regions: gradcamExp.highlighted_regions,
                                    } : undefined,
                                    feature_importance: shapValues,
                                    summary: explanation,
                                  }}
                                />
                              ) : (
                                <div className="flex items-center justify-between py-3 px-4 rounded border border-dashed border-border bg-muted/20">
                                  <p className="text-xs text-muted-foreground">No XAI explanations generated yet</p>
                                  <Button
                                    onClick={() => handleGenerateXAI(pred)}
                                    disabled={generatingXAI === pred.id}
                                    size="sm"
                                    variant="outline"
                                    className="h-7 gap-1.5 text-xs"
                                  >
                                    {generatingXAI === pred.id
                                      ? <><Loader className="h-3 w-3 animate-spin" />Generating...</>
                                      : <><Sparkles className="h-3 w-3" />Generate XAI</>
                                    }
                                  </Button>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })
                    }
                  </div>
                )}

                {/* REPORTS TAB */}
                {activeTab === 'reports' && (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <SectionHeader icon={FileText} title="Clinical Reports" iconColor="text-blue-500" count={reports.length} />
                      {latestPrediction && (
                        <Button
                          onClick={() => handleGenerateReport(latestPrediction.id)}
                          disabled={generatingReport}
                          size="sm"
                          variant="outline"
                          className="h-7 gap-1.5 text-xs"
                        >
                          {generatingReport
                            ? <><Loader className="h-3 w-3 animate-spin" />Generating...</>
                            : <><RefreshCw className="h-3 w-3" />Generate Report</>
                          }
                        </Button>
                      )}
                    </div>

                    {reports.length === 0 ? (
                      <EmptyState icon={FileText} title="No Clinical Reports" description="Generate a report from the latest prediction." />
                    ) : (
                      <>
                        {reports.filter(r => r.status?.toLowerCase() === 'completed').map(report => (
                          <MedicalReport key={report.id} report={report} />
                        ))}
                        {reports.filter(r => r.status?.toLowerCase() !== 'completed').length > 0 && (
                          <div className="flex flex-col gap-2">
                            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground px-0.5">
                              Pending ({reports.filter(r => r.status?.toLowerCase() !== 'completed').length})
                            </p>
                            {reports.filter(r => r.status?.toLowerCase() !== 'completed').map(report => (
                              <div key={report.id} className="flex items-center justify-between px-4 py-3 rounded-md border border-border bg-card opacity-60">
                                <div className="flex items-center gap-2.5">
                                  <StatusDot status={report.status} />
                                  <div>
                                    <p className="text-xs font-medium">{report.llm_model}</p>
                                    <p className="text-xs text-muted-foreground">{new Date(report.created_at).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}</p>
                                  </div>
                                </div>
                                <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground capitalize">{report.status}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}

                {/* OCT TAB */}
                {activeTab === 'oct' && (
                  <div className="flex flex-col gap-3">
                    <SectionHeader icon={Activity} title="OCT Analysis" iconColor="text-purple-500" count={octReports.length} />
                    {octReports.length === 0 ? (
                      <EmptyState icon={Activity} title="No OCT Reports" description="Process OCT scans to see results here." />
                    ) : (
                      <div className="rounded-md border border-border bg-card overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-border bg-muted/30">
                              {['Eye', 'DR Grade', 'Edema', 'ERM', 'Image Quality', 'Center Fovea'].map(h => (
                                <th key={h} className="text-left px-4 py-2.5 font-semibold uppercase tracking-widest text-muted-foreground text-[10px]">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {octReports.map(oct => (
                              <tr key={oct.id} className="hover:bg-muted/10 transition-colors">
                                <td className="px-4 py-3 font-medium">{oct.eye}</td>
                                <td className="px-4 py-3">
                                  <span className={`text-xs px-2 py-0.5 rounded font-medium ${OCT_GRADE_COLORS[oct.dr_grade || ''] ?? 'bg-muted text-muted-foreground'}`}>
                                    {oct.dr_grade?.replace('_', ' ') || 'N/A'}
                                  </span>
                                </td>
                                <td className="px-4 py-3">
                                  <span className={`text-xs font-medium ${oct.edema ? 'text-amber-400' : 'text-muted-foreground'}`}>
                                    {oct.edema ? 'Yes' : 'No'}
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-muted-foreground">{oct.erm_status || '—'}</td>
                                <td className="px-4 py-3">
                                  {oct.image_quality != null ? (
                                    <div className="flex items-center gap-2">
                                      <div className="w-16 h-1 bg-muted rounded-full overflow-hidden">
                                        <div
                                          className="h-full bg-[var(--brand-teal)] rounded-full"
                                          style={{ width: `${oct.image_quality}%` }}
                                        />
                                      </div>
                                      <span className="font-mono">{oct.image_quality}%</span>
                                    </div>
                                  ) : '—'}
                                </td>
                                <td className="px-4 py-3 font-mono text-muted-foreground">{oct.thickness_center_fovea ?? '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {/* Activity timeline */}
                    <div className="mt-2">
                      <SectionHeader icon={Layers} title="Recent Activity" iconColor="text-muted-foreground" />
                      {predictions.length === 0 && reports.length === 0 && scans.length === 0
                        ? <EmptyState icon={Activity} title="No Activity" description="Patient activity will appear here." />
                        : (
                          <div className="rounded-md border border-border bg-card divide-y divide-border">
                            {[
                              ...predictions.map(p => ({ type: 'prediction' as const, date: p.created_at, data: p })),
                              ...reports.map(r => ({ type: 'report' as const, date: r.created_at, data: r })),
                              ...scans.map(s => ({ type: 'scan' as const, date: s.uploaded_at, data: s })),
                            ]
                              .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                              .slice(0, 8)
                              .map((activity, i) => (
                                <div key={i} className="flex items-center gap-3 px-4 py-2.5">
                                  <div className={`p-1.5 rounded-full flex-shrink-0 ${
                                    activity.type === 'prediction' ? 'bg-amber-500/10 text-amber-500'
                                    : activity.type === 'report' ? 'bg-blue-500/10 text-blue-500'
                                    : 'bg-[var(--brand-teal)]/10 text-[var(--brand-teal)]'
                                  }`}>
                                    {activity.type === 'prediction' ? <Eye className="h-3 w-3" />
                                      : activity.type === 'report' ? <FileText className="h-3 w-3" />
                                      : <Scan className="h-3 w-3" />
                                    }
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <p className="text-xs font-medium">
                                      {activity.type === 'prediction' ? 'DR Screening'
                                        : activity.type === 'report' ? 'Clinical Report'
                                        : 'MRI Scan Upload'}
                                    </p>
                                    <p className="text-xs text-muted-foreground truncate">
                                      {activity.type === 'prediction' && (activity.data as Prediction).model_name}
                                      {activity.type === 'report' && (activity.data as Report).llm_model}
                                      {activity.type === 'scan' && `ID: ${(activity.data as MRIScan).id.slice(0, 12)}`}
                                    </p>
                                  </div>
                                  <span className="text-xs text-muted-foreground flex-shrink-0">
                                    {new Date(activity.date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })}
                                  </span>
                                </div>
                              ))
                            }
                          </div>
                        )
                      }
                    </div>
                  </div>
                )}

              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </motion.div>
    </PageContainer>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function DataRow({ icon: Icon, label, value, mono }: { icon: typeof Hash; label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center gap-3 px-3 py-2.5">
      <Icon className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</p>
        <p className={`text-xs font-medium truncate ${mono ? 'font-mono' : ''}`}>{value}</p>
      </div>
    </div>
  );
}

function StatCell({ label, value, accent }: { label: string; value: number; accent: 'teal' | 'gold' | 'blue' | 'purple' }) {
  const colors: Record<string, string> = {
    teal: 'text-[var(--brand-teal)]',
    gold: 'text-[var(--brand-gold)]',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
  };
  return (
    <div className="flex flex-col items-center py-3 gap-0.5">
      <span className={`text-xl font-bold ${colors[accent]}`}>{value}</span>
      <span className="text-[10px] text-muted-foreground uppercase tracking-widest">{label}</span>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, iconColor, count }: { icon: typeof Scan; title: string; iconColor: string; count?: number }) {
  return (
    <div className="flex items-center gap-2 mb-1">
      <Icon className={`h-3.5 w-3.5 ${iconColor}`} />
      <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">{title}</span>
      {count !== undefined && <span className="text-xs text-muted-foreground">({count})</span>}
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  if (status === 'success' || status === 'completed') return <CheckCircle className="h-3.5 w-3.5 text-emerald-500 flex-shrink-0" />;
  if (status === 'failed') return <XCircle className="h-3.5 w-3.5 text-rose-500 flex-shrink-0" />;
  if (status === 'pending' || status === 'running' || status === 'generating') return <Loader className="h-3.5 w-3.5 text-amber-500 animate-spin flex-shrink-0" />;
  return <AlertCircle className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />;
}

function EmptyState({ icon: Icon, title, description }: { icon: typeof Scan; title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-10 rounded-md border border-dashed border-border bg-muted/10">
      <Icon className="h-7 w-7 mb-2 text-muted-foreground/40" />
      <p className="text-sm font-medium text-muted-foreground">{title}</p>
      <p className="text-xs text-muted-foreground/60 mt-0.5">{description}</p>
    </div>
  );
}

function EyeImagePanel({ title, src, eye }: { title: string; src?: string; eye: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-xs font-medium text-muted-foreground">{title}</p>
      <div className="relative aspect-square rounded bg-black overflow-hidden">
        {src ? (
          <Image src={src} alt={title} fill className="object-cover" unoptimized />
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-1 text-muted-foreground">
            <ImageIcon className="h-5 w-5 opacity-40" />
            <span className="text-xs opacity-40">No image</span>
          </div>
        )}
        <span className="absolute bottom-2 left-2 text-xs font-mono font-bold bg-black/60 text-white px-1.5 py-0.5 rounded">
          {eye}
        </span>
      </div>
    </div>
  );
}

function GradCAMPanel({ title, gradcamBase64 }: { title: string; gradcamBase64?: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-xs font-medium text-muted-foreground">{title}</p>
      <div className="relative aspect-square rounded overflow-hidden bg-black">
        {gradcamBase64 ? (
          <img
            src={`data:image/png;base64,${gradcamBase64}`}
            alt={title}
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-1 text-muted-foreground">
            <ImageIcon className="h-5 w-5 opacity-40" />
            <span className="text-xs opacity-40">No GradCAM</span>
          </div>
        )}
      </div>
    </div>
  );
}