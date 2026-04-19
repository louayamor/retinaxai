'use client';

import { useEffect, useState, useCallback } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import PageContainer from '@/components/layout/page-container';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  ArrowLeft,
  User,
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
import { usePatientWebSocket, type PredictionEventData, type XAIEventData, type SeverityEventData, type GradCAMEventData } from '@/hooks/use-patient-websocket';
import { toast } from 'sonner';
import type { Patient, MRIScan, OCTReport, Prediction, Report, PaginatedResponse } from '@/types';
import { fadeInUp, slideInUp, staggerItem } from '@/lib/animations';
import Image from 'next/image';
import MedicalReport from '@/components/features/reports/medical-report';
import XAICard from '@/components/features/xai/xai-card';

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
const GRADE_COLORS: Record<string, string> = {
  no_dr: 'bg-emerald-500',
  mild: 'bg-cyan-500',
  moderate: 'bg-amber-500',
  severe: 'bg-orange-500',
  proliferative: 'bg-rose-500',
};
const GRADE_COLORS_NUM: Record<number, string> = {
  0: 'bg-emerald-500',
  1: 'bg-cyan-500',
  2: 'bg-amber-500',
  3: 'bg-orange-500',
  4: 'bg-rose-500',
};

export default function PatientProfilePage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
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
  const [needsRefresh, setNeedsRefresh] = useState({
    predictions: false,
    reports: false,
    xai: false,
  });

  const fetchXAIData = useCallback(async (predId: string) => {
    try {
      const xai = await getXAIExplanations(predId);
      if (xai.explanation || xai.severity_report || xai.gradcam_explanation) {
        setXaiData(prev => ({ ...prev, [predId]: xai }));
      }
    } catch {
      // XAI not found
    }
  }, []);

  const refreshPredictions = useCallback(async () => {
    try {
      const predsData = await listPatientPredictions(patientId, 1, 100);
      setPredictions((predsData as PaginatedResponse<Prediction>).items);
    } catch (err) {
      console.error('Failed to refresh predictions:', err);
    }
  }, [patientId]);

  const refreshReports = useCallback(async () => {
    try {
      const repsData = await listPatientReports(patientId, 1, 100);
      setReports((repsData as PaginatedResponse<Report>).items);
    } catch (err) {
      console.error('Failed to refresh reports:', err);
    }
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
      if (data.prediction_id) {
        await fetchXAIData(data.prediction_id);
      }
    }, [fetchXAIData]),
    onSeverityReady: useCallback(async (data: SeverityEventData) => {
      if (data.prediction_id) {
        await fetchXAIData(data.prediction_id);
        setNeedsRefresh(prev => ({ ...prev, xai: true }));
      }
    }, [fetchXAIData]),
  });

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setError(null);
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
      } catch (err) {
        console.error('Failed to load patient data:', err);
        setError('Failed to load patient data');
      } finally {
        setLoading(false);
      }
    };
    loadInitialData();
  }, [patientId]);

  useEffect(() => {
    if (needsRefresh.predictions) {
      refreshPredictions();
      setNeedsRefresh(prev => ({ ...prev, predictions: false }));
    }
  }, [needsRefresh, refreshPredictions]);

  useEffect(() => {
    if (needsRefresh.reports) {
      refreshReports();
      setNeedsRefresh(prev => ({ ...prev, reports: false }));
    }
  }, [needsRefresh, refreshReports]);

  const getInitials = (firstName: string, lastName: string) => {
    return `${firstName[0]}${lastName[0]}`.toUpperCase();
  };

  const getGenderColor = (gender: string) => {
    return gender === 'M' ? 'from-blue-500 to-blue-700' : 'from-pink-500 to-pink-700';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-emerald-500" />;
      case 'pending':
      case 'running':
      case 'generating':
        return <Loader className="h-4 w-4 text-amber-500 animate-spin" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-rose-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const EmptyState = ({ icon: Icon, title, description }: { icon: typeof Scan; title: string; description: string }) => (
    <div className="flex flex-col items-center justify-center py-6 bg-muted/30 rounded-lg">
      <Icon className="h-8 w-8 mb-2 opacity-50 text-muted-foreground" />
      <p className="text-sm text-muted-foreground font-medium">{title}</p>
      <p className="text-xs text-muted-foreground/70">{description}</p>
    </div>
  );

  const handleGenerateXAI = async (prediction: Prediction) => {
    if (!patient) return;
    
    setGeneratingXAI(prediction.id);
    try {
      const drGrade = String(prediction.output_payload?.combined_grade ?? prediction.output_payload?.predicted_class ?? 'Unknown');
      const confidence = prediction.confidence_score ?? 0;
      const clinicalFeatures = prediction.input_payload as Record<string, unknown>;

      let shapResult = null;
      let xaiResult = null;
      let gradcamResult = null;
      let severityResult = null;

      try {
        toast.info('Generating SHAP explanations...');
        shapResult = await generateSHAPExplanation(prediction.id, clinicalFeatures || {});
      } catch (shapError) {
        console.warn('SHAP generation failed:', shapError);
      }

      toast.info('Generating AI explanation...');
      xaiResult = await generateXAIExplanation(prediction.id, drGrade, confidence, clinicalFeatures);

      const leftRegions = (prediction.output_payload?.gradcam_left_regions as string[]) || [];
      const rightRegions = (prediction.output_payload?.gradcam_right_regions as string[]) || [];
      if (leftRegions.length > 0 || rightRegions.length > 0) {
        toast.info('Generating GradCAM interpretation...');
        gradcamResult = await generateXAIGradCAM(prediction.id, leftRegions, rightRegions);
      }

      toast.info('Generating severity assessment...');
      severityResult = await generateXAISeverity(
        prediction.id,
        {
          name: `${patient.first_name} ${patient.last_name}`,
          age: patient.age,
          gender: patient.gender,
        },
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
        toast.success('XAI generation complete!');
      } catch (storeError) {
        if (storeError instanceof ApiError && storeError.status === 409) {
          toast.warning('XAI explanation already exists for this prediction');
        } else {
          throw storeError;
        }
      }

      await fetchXAIData(prediction.id);
    } catch (err) {
      console.error('XAI generation failed:', err);
      toast.error('Failed to generate XAI', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    } finally {
      setGeneratingXAI(null);
    }
  };

  const handleGenerateReport = async (predictionId: string) => {
    setGeneratingReport(true);
    try {
      toast.info('Generating clinical report...');
      await createReport(predictionId);
      toast.success('Report generation started!');
      setNeedsRefresh(prev => ({ ...prev, reports: true }));
    } catch (err) {
      console.error('Report generation failed:', err);
      toast.error('Failed to generate report', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    } finally {
      setGeneratingReport(false);
    }
  };

  const latestPrediction = predictions.length > 0 ? predictions[0] : null;
  const latestGrade = latestPrediction?.output_payload?.combined_grade as number | undefined;
  const latestGradeLabel = latestGrade !== undefined ? GRADE_LABELS[latestGrade] : null;
  const latestSeverity = latestPrediction?.output_payload?.overall_severity as string | undefined;

  if (loading) {
    return (
      <PageContainer>
        <div className="flex h-96 items-center justify-center">
          <Loader className="h-8 w-8 animate-spin text-[var(--brand-teal)]" />
        </div>
      </PageContainer>
    );
  }

  if (error || !patient) {
    return (
      <PageContainer>
        <div className="flex flex-col items-center justify-center py-12">
          <AlertCircle className="h-12 w-12 mb-3 text-rose-500" />
          <p className="text-rose-500 font-medium">{error || 'Patient not found'}</p>
          <Button variant="outline" className="mt-4" onClick={() => router.push('/dashboard/patients')}>
            Back to Patients
          </Button>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeInUp}
        initial="hidden"
        animate="visible"
        className="flex flex-col gap-4"
      >
        {/* Header */}
        <motion.div variants={shouldReduceMotion ? {} : slideInUp} className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={() => router.push('/dashboard/patients')}>
              <ArrowLeft className="h-4 w-4 mr-1" />
              Back
            </Button>
            <h1 className="text-xl font-bold">Patient Profile</h1>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            <span className="text-xs text-muted-foreground">
              {connected ? 'Live' : 'Connecting...'}
            </span>
          </div>
        </motion.div>

        {/* Patient Info Card */}
        <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
          <Card>
            <CardHeader className="pb-2 bg-gradient-to-r from-muted/30 to-transparent">
              <div className="flex items-center gap-3">
                <Avatar className="h-12 w-12">
                  <AvatarFallback className={`text-lg bg-gradient-to-br ${getGenderColor(patient.gender)} text-white`}>
                    {getInitials(patient.first_name, patient.last_name)}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <CardTitle className="text-lg">
                    {patient.first_name} {patient.last_name}
                  </CardTitle>
                  <div className="flex items-center gap-2 mt-0.5">
                    <Badge variant="outline" className="text-xs">{patient.gender === 'M' ? 'Male' : 'Female'}</Badge>
                    <span className="text-xs text-muted-foreground">{patient.age} years old</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-2">
              <div className="bg-muted/50 rounded p-2">
                <p className="text-xs text-muted-foreground">Medical Record #</p>
                <p className="font-mono font-medium text-sm">{patient.medical_record_number}</p>
              </div>
              <div className="bg-muted/50 rounded p-2">
                <p className="text-xs text-muted-foreground">Phone</p>
                <p className="font-medium text-sm">{patient.phone || 'N/A'}</p>
              </div>
              <div className="bg-muted/50 rounded p-2">
                <p className="text-xs text-muted-foreground">Registered</p>
                <p className="font-medium text-sm flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {new Date(patient.created_at).toLocaleDateString()}
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="border-l-4 border-l-[var(--brand-teal)]">
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">MRI Scans</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{scans.length}</p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="border-l-4 border-l-amber-500">
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{predictions.length}</p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="border-l-4 border-l-blue-500">
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">Reports</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{reports.length}</p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="border-l-4 border-l-purple-500">
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">OCT Scans</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{octReports.length}</p>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Latest Prediction Summary */}
        {latestPrediction && (
          <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
            <Card className="border-l-4 border-l-emerald-500">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="h-4 w-4 text-emerald-500" />
                  Latest Prediction
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <p className="text-xs text-muted-foreground">DR Grade</p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {latestGradeLabel && (
                        <Badge className={GRADE_COLORS_NUM[latestGrade!] || 'bg-muted'}>
                          {latestGradeLabel}
                        </Badge>
                      )}
                      {getStatusIcon(latestPrediction.status)}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Confidence</p>
                    <p className="text-xl font-bold mt-0.5">
                      {latestPrediction.confidence_score
                        ? `${(latestPrediction.confidence_score * 100).toFixed(1)}%`
                        : 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Risk Level</p>
                    <p className="text-sm font-semibold mt-0.5 capitalize">{latestSeverity || 'Unknown'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* MRI Scans Section */}
        <section>
          <h2 className="text-base font-semibold mb-2 flex items-center gap-2">
            <Scan className="h-4 w-4 text-[var(--brand-teal)]" />
            MRI Scans ({scans.length})
          </h2>
          {scans.length === 0 ? (
            <EmptyState icon={Scan} title="No MRI Scans" description="Add scans to see them here" />
          ) : (
            <div className="grid gap-2 md:grid-cols-2">
              {scans.map((scan) => (
                <Card key={scan.id}>
                  <CardHeader className="pb-1">
                    <CardTitle className="text-xs flex items-center gap-1.5">
                      <Scan className="h-3 w-3 text-[var(--brand-teal)]" />
                      Scan {scan.id.slice(0, 8)}
                    </CardTitle>
                    <CardDescription className="text-xs">
                      {new Date(scan.uploaded_at).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid grid-cols-2 gap-1">
                    <div className="relative aspect-square rounded bg-muted overflow-hidden">
                      {scan.left_scan_path ? (
                        <Image
                          src={`${API_BASE}/` + scan.left_scan_path}
                          alt="Left eye"
                          fill
                          className="object-cover"
                          unoptimized
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
                          No image
                        </div>
                      )}
                      <span className="absolute bottom-1 left-1 bg-black/50 text-white text-xs px-1 rounded">
                        L
                      </span>
                    </div>
                    <div className="relative aspect-square rounded bg-muted overflow-hidden">
                      {scan.right_scan_path ? (
                        <Image
                          src={`${API_BASE}/` + scan.right_scan_path}
                          alt="Right eye"
                          fill
                          className="object-cover"
                          unoptimized
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
                          No image
                        </div>
                      )}
                      <span className="absolute bottom-1 right-1 bg-black/50 text-white text-xs px-1 rounded">
                        R
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </section>

        {/* GradCAM Analysis Section */}
        <section>
          <h2 className="text-base font-semibold mb-2 flex items-center gap-2">
            <Eye className="h-4 w-4 text-amber-500" />
            GradCAM Analysis ({predictions.filter((p) => p.status?.toLowerCase() === 'success').length})
          </h2>
          {predictions.filter((p) => p.status?.toLowerCase() === 'success').length === 0 ? (
            <EmptyState icon={Eye} title="No GradCAM Available" description="Run predictions to generate GradCAM analysis" />
          ) : (
            <div className="space-y-2">
              {predictions
                .filter((p) => p.status?.toLowerCase() === 'success')
                .map((pred) => {
                  const grade = pred.output_payload?.combined_grade as number | undefined;
                  const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : null;
                  return (
                    <Card key={pred.id}>
                      <CardHeader className="pb-1">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(pred.status)}
                            <div>
                              <CardTitle className="text-xs">{pred.model_name}</CardTitle>
                              <CardDescription className="text-xs">
                                {new Date(pred.created_at).toLocaleString()}
                              </CardDescription>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-base font-bold">
                              {pred.confidence_score
                                ? `${(pred.confidence_score * 100).toFixed(1)}%`
                                : 'N/A'}
                            </p>
                            {gradeLabel && (
                              <Badge className={GRADE_COLORS_NUM[grade!] || 'bg-muted'}>
                                {gradeLabel}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-2">
                          <GradCAMDisplay
                            title="Left Eye (OS)"
                            gradcamBase64={pred.output_payload?.gradcam_left as string | undefined}
                          />
                          <GradCAMDisplay
                            title="Right Eye (OD)"
                            gradcamBase64={pred.output_payload?.gradcam_right as string | undefined}
                          />
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
            </div>
          )}
        </section>

        {/* AI Explanations Section */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <Brain className="h-4 w-4 text-purple-500" />
              AI Explanations
            </h2>
            {latestPrediction && !xaiData[latestPrediction.id] && (
              <Button
                onClick={() => handleGenerateXAI(latestPrediction)}
                disabled={generatingXAI === latestPrediction.id}
                size="sm"
              >
                {generatingXAI === latestPrediction.id ? (
                  <>
                    <Loader className="h-3 w-3 mr-1 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3 w-3 mr-1" />
                    Generate XAI
                  </>
                )}
              </Button>
            )}
          </div>
          {predictions.filter((p) => p.status?.toLowerCase() === 'success').length === 0 ? (
            <EmptyState
              icon={Brain}
              title="No XAI Explanations Available"
              description="Run predictions to generate AI explanations"
            />
          ) : (
            <div className="space-y-2">
              {predictions
                .filter((p) => p.status?.toLowerCase() === 'success')
                .map((pred) => {
                  const xai = xaiData[pred.id];
                  const outputPayload = pred.output_payload as Record<string, unknown> | null;
                  const grade = outputPayload?.combined_grade as number | undefined;
                  const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : null;
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

                  return (
                    <Card key={pred.id}>
                      <CardHeader className="pb-1">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm">
                            Prediction - {new Date(pred.created_at).toLocaleDateString()}
                          </CardTitle>
                          <div className="flex items-center gap-1.5">
                            {severityReport && (
                              <Badge className={
                                severityReport.risk_level === 'low' ? 'bg-emerald-500' :
                                severityReport.risk_level === 'moderate' ? 'bg-amber-500' :
                                severityReport.risk_level === 'high' ? 'bg-orange-500' :
                                'bg-rose-500'
                              }>
                                {severityReport.risk_level.replace('_', ' ').toUpperCase()} Risk
                              </Badge>
                            )}
                            {grade !== undefined && gradeLabel && (
                              <Badge className={GRADE_COLORS_NUM[grade] || 'bg-muted'}>
                                {gradeLabel}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        {/* Confidence Score */}
                        <div>
                          <p className="text-xs text-muted-foreground mb-1">Confidence Score</p>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{
                                  width: `${(pred.confidence_score || 0) * 100}%`,
                                }}
                                className="h-full bg-[var(--brand-teal)]"
                              />
                            </div>
                            <span className="text-xs font-medium">
                              {pred.confidence_score
                                ? `${(pred.confidence_score * 100).toFixed(1)}%`
                                : 'N/A'}
                            </span>
                          </div>
                        </div>

                        {/* SHAP Values */}
                        {shapValues && (
                          <div>
                            <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                              <BarChart3 className="h-3 w-3" />
                              Top Contributing Features
                            </p>
                            <div className="space-y-1">
                              {shapValues?.top_positive?.slice(0, 3).map(
                                (feature, i) => (
                                  <div key={i} className="flex items-center gap-1.5">
                                    <span className="text-xs w-24 truncate">{feature.name}</span>
                                    <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                                      <motion.div
                                        initial={{ width: 0 }}
                                        animate={{
                                          width: `${Math.min(Math.abs(feature.contribution) * 100, 100)}%`,
                                        }}
                                        transition={{ delay: i * 0.1 }}
                                        className="h-full bg-emerald-500"
                                      />
                                    </div>
                                    <span className="text-xs text-muted-foreground w-10 text-right">
                                      {feature.contribution.toFixed(3)}
                                    </span>
                                  </div>
                                )
                              )}
                            </div>
                          </div>
                        )}

                        {/* XAI Card */}
                        {hasXAI && (
                          <XAICard
                            predictionId={pred.id}
                            createdAt={pred.created_at}
                            data={{
                              diagnosis: explanation ? (() => {
                                try {
                                  const parsed = JSON.parse(explanation);
                                  return {
                                    condition: parsed.diagnosis?.condition,
                                    severity: parsed.diagnosis?.severity,
                                    overall_grade: parsed.diagnosis?.overall_grade,
                                    confidence: parsed.diagnosis?.confidence,
                                    risk_level: parsed.diagnosis?.risk_level,
                                  };
                                } catch {
                                  return {};
                                }
                              })() : undefined,
                              clinical_findings: explanation ? (() => {
                                try {
                                  const parsed = JSON.parse(explanation);
                                  return parsed.clinical_findings;
                                } catch {
                                  return undefined;
                                }
                              })() : undefined,
                              severity_report: severityReport ? {
                                patient: { name: patient?.first_name, age: patient?.age, gender: patient?.gender },
                                diagnosis: severityReport.risk_level ? {
                                  dr_grade: grade,
                                  severity_label: severityReport.risk_level,
                                  risk_level: severityReport.risk_level,
                                } : undefined,
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
                        )}

                        {!hasXAI && (
                          <div className="flex flex-col items-center gap-1.5 py-2">
                            <p className="text-xs text-muted-foreground italic">
                              No XAI explanations generated yet
                            </p>
                            <Button
                              onClick={() => handleGenerateXAI(pred)}
                              disabled={generatingXAI === pred.id}
                              size="sm"
                              variant="outline"
                            >
                              {generatingXAI === pred.id ? (
                                <>
                                  <Loader className="h-3 w-3 mr-1 animate-spin" />
                                  Generating...
                                </>
                              ) : (
                                <>
                                  <Sparkles className="h-3 w-3 mr-1" />
                                  Generate XAI
                                </>
                              )}
                            </Button>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
            </div>
          )}
        </section>

        {/* Clinical Reports Section */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <FileText className="h-4 w-4 text-blue-500" />
              Clinical Reports ({reports.length})
            </h2>
            {latestPrediction && (
              <Button
                onClick={() => handleGenerateReport(latestPrediction.id)}
                disabled={generatingReport}
                size="sm"
                variant="outline"
              >
                {generatingReport ? (
                  <>
                    <Loader className="h-3 w-3 mr-1 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Generate Report
                  </>
                )}
              </Button>
            )}
          </div>
          {reports.length === 0 ? (
            <EmptyState
              icon={FileText}
              title="No Clinical Reports"
              description="Generate reports to see them here"
            />
          ) : (
            <div className="space-y-2">
              {reports
                .filter((r) => r.status?.toLowerCase() === 'completed')
                .map((report) => (
                  <MedicalReport key={report.id} report={report} />
                ))}
              {reports.filter((r) => r.status !== 'completed').length > 0 && (
                <div className="space-y-1">
                  <h4 className="text-xs font-medium text-muted-foreground">
                    Pending ({reports.filter((r) => r.status !== 'completed').length})
                  </h4>
                  {reports
                    .filter((r) => r.status !== 'completed')
                    .map((report) => (
                      <Card key={report.id} className="opacity-60">
                        <CardContent className="flex items-center justify-between py-2">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(report.status)}
                            <div>
                              <p className="font-medium text-sm">{report.llm_model}</p>
                              <p className="text-xs text-muted-foreground">
                                {new Date(report.created_at).toLocaleString()}
                              </p>
                            </div>
                          </div>
                          <Badge variant="secondary">{report.status}</Badge>
                        </CardContent>
                      </Card>
                    ))}
                </div>
              )}
            </div>
          )}
        </section>

        {/* OCT Reports Section */}
        <section>
          <h2 className="text-base font-semibold mb-2 flex items-center gap-2">
            <Activity className="h-4 w-4 text-purple-500" />
            OCT Analysis ({octReports.length})
          </h2>
          {octReports.length === 0 ? (
            <EmptyState
              icon={Activity}
              title="No OCT Reports"
              description="Process OCT scans to see them here"
            />
          ) : (
            <Card>
              <CardContent className="p-0">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Eye</th>
                      <th className="text-left p-2">DR</th>
                      <th className="text-left p-2">Edema</th>
                      <th className="text-left p-2">ERM</th>
                      <th className="text-left p-2">Quality</th>
                      <th className="text-left p-2">Fovea</th>
                    </tr>
                  </thead>
                  <tbody>
                    {octReports.map((oct) => (
                      <tr key={oct.id} className="border-b">
                        <td className="p-2">{oct.eye}</td>
                        <td className="p-2">
                          <Badge className={GRADE_COLORS[oct.dr_grade || ''] || 'bg-muted'}>
                            {oct.dr_grade || 'N/A'}
                          </Badge>
                        </td>
                        <td className="p-2">{oct.edema ? 'Yes' : 'No'}</td>
                        <td className="p-2">{oct.erm_status || 'N/A'}</td>
                        <td className="p-2">{oct.image_quality ? `${oct.image_quality}%` : 'N/A'}</td>
                        <td className="p-2">{oct.thickness_center_fovea || 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Recent Activity */}
        <section>
          <h2 className="text-base font-semibold mb-2 flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue-500" />
            Recent Activity
          </h2>
          {predictions.length === 0 && reports.length === 0 && scans.length === 0 ? (
            <EmptyState
              icon={Activity}
              title="No Recent Activity"
              description="Patient activity will appear here"
            />
          ) : (
            <Card>
              <CardContent>
                <div className="space-y-2">
                  {[
                    ...predictions.map((p) => ({ type: 'prediction', date: p.created_at, data: p })),
                    ...reports.map((r) => ({ type: 'report', date: r.created_at, data: r })),
                    ...scans.map((s) => ({ type: 'scan', date: s.uploaded_at, data: s })),
                  ]
                    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                    .slice(0, 5)
                    .map((activity, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <div className={`p-1.5 rounded-full ${
                          activity.type === 'prediction'
                            ? 'bg-amber-100 text-amber-600'
                            : activity.type === 'report'
                            ? 'bg-blue-100 text-blue-600'
                            : 'bg-teal-100 text-teal-600'
                        }`}>
                          {activity.type === 'prediction' ? (
                            <Eye className="h-3 w-3" />
                          ) : activity.type === 'report' ? (
                            <FileText className="h-3 w-3" />
                          ) : (
                            <Scan className="h-3 w-3" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="font-medium capitalize">
                            {activity.type === 'prediction'
                              ? 'DR Screening'
                              : activity.type === 'report'
                              ? 'Clinical Report'
                              : 'MRI Scan'} - {new Date(activity.date).toLocaleDateString()}
                          </p>
                          <p className="text-muted-foreground text-xs">
                            {activity.type === 'prediction' && (activity.data as Prediction).model_name}
                            {activity.type === 'report' && (activity.data as Report).llm_model}
                            {activity.type === 'scan' && `ID: ${(activity.data as MRIScan).id.slice(0, 8)}`}
                          </p>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}
        </section>
      </motion.div>
    </PageContainer>
  );
}

function GradCAMDisplay({
  title,
  gradcamBase64,
}: {
  title: string;
  gradcamBase64?: string;
}) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-muted-foreground">{title}</p>
      {gradcamBase64 ? (
        <div className="relative aspect-square rounded overflow-hidden bg-black">
          <img
            src={`data:image/png;base64,${gradcamBase64}`}
            alt={title}
            className="w-full h-full object-contain"
          />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center aspect-square bg-muted rounded">
          <ImageIcon className="h-6 w-6 text-muted-foreground" />
          <p className="text-xs text-muted-foreground mt-0.5">No GradCAM</p>
        </div>
      )}
    </div>
  );
}