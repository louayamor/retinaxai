'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import PageContainer from '@/components/layout/page-container';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
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
  Download,
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
  getPrediction,
  createReport,
  generateXAIExplanation,
  generateXAIGradCAM,
  generateXAISeverity,
  generateSHAPExplanation,
  storeXAIResults,
} from '@/lib/api';
import { toast } from 'sonner';
import type { Patient, MRIScan, OCTReport, Prediction, Report, PaginatedResponse } from '@/types';
import { fadeInUp, slideInUp, staggerItem } from '@/lib/animations';
import Image from 'next/image';
import MedicalReport from '@/components/medical-report';

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
  const initialTab = searchParams.get('tab') || 'overview';

  const [patient, setPatient] = useState<Patient | null>(null);
  const [scans, setScans] = useState<MRIScan[]>([]);
  const [octReports, setOctReports] = useState<OCTReport[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [activeTab, setActiveTab] = useState(initialTab);
  const [generatingXAI, setGeneratingXAI] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);

  useEffect(() => {
    loadPatientData();
  }, [patientId]);

  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && ['overview', 'scans', 'xai', 'reports'].includes(tab)) {
      setActiveTab(tab);
    }
  }, [searchParams]);

  const loadPatientData = async () => {
    try {
      setLoading(true);
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

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    const url = new URL(window.location.href);
    url.searchParams.set('tab', tab);
    router.replace(url.toString(), { scroll: false });
  };

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
    <div className="flex flex-col items-center justify-center py-12 bg-muted/30 rounded-lg">
      <Icon className="h-12 w-12 mb-3 opacity-50 text-muted-foreground" />
      <p className="text-muted-foreground font-medium">{title}</p>
      <p className="text-sm text-muted-foreground/70">{description}</p>
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
      let severityResult = null;

      // Generate SHAP explanation (may fail if model features don't match)
      try {
        toast.info('Generating SHAP explanations...');
        shapResult = await generateSHAPExplanation(prediction.id, clinicalFeatures || {});
      } catch (shapError) {
        console.warn('SHAP generation failed (expected with limited features):', shapError);
      }

      // Generate XAI explanation with LLM
      toast.info('Generating AI explanation...');
      xaiResult = await generateXAIExplanation(prediction.id, drGrade, confidence, clinicalFeatures);

      // Generate GradCAM interpretation (only if we have regions)
      const leftRegions = (prediction.output_payload?.gradcam_left_regions as string[]) || [];
      const rightRegions = (prediction.output_payload?.gradcam_right_regions as string[]) || [];
      if (leftRegions.length > 0 || rightRegions.length > 0) {
        toast.info('Generating GradCAM interpretation...');
        await generateXAIGradCAM(prediction.id, leftRegions, rightRegions);
      }

      // Generate severity report
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

      // Store all XAI results in backend
      toast.info('Storing XAI results...');
      await storeXAIResults(prediction.id, {
        explanationContent: xaiResult?.content,
        explanationSummary: xaiResult?.summary,
        shapValues: shapResult ?? undefined,
        severityContent: severityResult?.content,
        severitySummary: severityResult?.summary,
        severityRiskLevel: severityResult?.risk_level,
        severityRecommendations: severityResult?.recommendations,
        model: 'gpt-4.1-mini',
      });

      toast.success('XAI generation complete!');

      // Refresh prediction data to show new results
      await loadPatientData();
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
      
      // Refresh reports
      const repsData = await listPatientReports(patientId, 1, 100);
      setReports((repsData as PaginatedResponse<Report>).items);
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
        className="flex flex-col gap-6"
      >
        {/* Header */}
        <motion.div variants={shouldReduceMotion ? {} : slideInUp} className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={() => router.push('/dashboard/patients')}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <h1 className="text-2xl font-bold">Patient Profile</h1>
        </motion.div>

        {/* Patient Info Card */}
        <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
          <Card>
            <CardHeader className="bg-gradient-to-r from-muted/30 to-transparent">
              <div className="flex items-center gap-4">
                <Avatar className="h-16 w-16">
                  <AvatarFallback className={`text-xl bg-gradient-to-br ${getGenderColor(patient.gender)} text-white`}>
                    {getInitials(patient.first_name, patient.last_name)}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <CardTitle className="text-2xl">
                    {patient.first_name} {patient.last_name}
                  </CardTitle>
                  <div className="flex items-center gap-3 mt-1">
                    <Badge variant="outline">{patient.gender === 'M' ? 'Male' : 'Female'}</Badge>
                    <span className="text-muted-foreground">{patient.age} years old</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Medical Record #</p>
                <p className="font-mono font-medium">{patient.medical_record_number}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Phone</p>
                <p className="font-medium">{patient.phone || 'Not Available'}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Registered</p>
                <p className="font-medium flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {new Date(patient.created_at).toLocaleDateString()}
                </p>
              </div>
              {patient.address && (
                <div className="bg-muted/50 rounded-lg p-3 md:col-span-2">
                  <p className="text-xs text-muted-foreground">Address</p>
                  <p className="font-medium">{patient.address}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Tabbed Interface */}
        <Tabs value={activeTab} onValueChange={handleTabChange}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview" className="gap-2">
              <User className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="scans" className="gap-2">
              <Scan className="h-4 w-4" />
              Scans & GradCAM
            </TabsTrigger>
            <TabsTrigger value="xai" className="gap-2">
              <Brain className="h-4 w-4" />
              XAI Explanations
            </TabsTrigger>
            <TabsTrigger value="reports" className="gap-2">
              <FileText className="h-4 w-4" />
              Reports
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-6 space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
                <Card className="border-l-4 border-l-[var(--brand-teal)]">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">MRI Scans</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold">{scans.length}</p>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
                <Card className="border-l-4 border-l-amber-500">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Predictions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold">{predictions.length}</p>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
                <Card className="border-l-4 border-l-blue-500">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Reports</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold">{reports.length}</p>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
                <Card className="border-l-4 border-l-purple-500">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">OCT Scans</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold">{octReports.length}</p>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Latest Prediction Summary */}
            {latestPrediction && (
              <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
                <Card className="border-l-4 border-l-emerald-500">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Activity className="h-5 w-5 text-emerald-500" />
                      Latest Prediction
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid md:grid-cols-3 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">DR Grade</p>
                        <div className="flex items-center gap-2 mt-1">
                          {latestGradeLabel && (
                            <Badge className={GRADE_COLORS_NUM[latestGrade!] || 'bg-muted'}>
                              {latestGradeLabel}
                            </Badge>
                          )}
                          {getStatusIcon(latestPrediction.status)}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Confidence</p>
                        <p className="text-2xl font-bold mt-1">
                          {latestPrediction.confidence_score
                            ? `${(latestPrediction.confidence_score * 100).toFixed(1)}%`
                            : 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Risk Level</p>
                        <p className="text-lg font-semibold mt-1 capitalize">{latestSeverity || 'Unknown'}</p>
                      </div>
                    </div>
                    <div className="flex gap-2 mt-4">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleTabChange('scans')}
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        View GradCAM
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleTabChange('xai')}
                      >
                        <BarChart3 className="h-4 w-4 mr-1" />
                        View XAI Details
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Recent Activity Timeline */}
            <motion.div variants={shouldReduceMotion ? {} : staggerItem}>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Layers className="h-5 w-5 text-blue-500" />
                    Recent Activity
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {predictions.length === 0 && reports.length === 0 && scans.length === 0 ? (
                    <EmptyState
                      icon={Activity}
                      title="No Recent Activity"
                      description="Patient activity will appear here"
                    />
                  ) : (
                    <div className="space-y-4">
                      {/* Merge and sort activities */}
                      {[
                        ...predictions.map((p) => ({ type: 'prediction', date: p.created_at, data: p })),
                        ...reports.map((r) => ({ type: 'report', date: r.created_at, data: r })),
                        ...scans.map((s) => ({ type: 'scan', date: s.uploaded_at, data: s })),
                      ]
                        .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                        .slice(0, 5)
                        .map((activity, i) => (
                          <div key={i} className="flex items-center gap-3 text-sm">
                            <div className={`p-2 rounded-full ${
                              activity.type === 'prediction'
                                ? 'bg-amber-100 text-amber-600'
                                : activity.type === 'report'
                                ? 'bg-blue-100 text-blue-600'
                                : 'bg-teal-100 text-teal-600'
                            }`}>
                              {activity.type === 'prediction' ? (
                                <Eye className="h-4 w-4" />
                              ) : activity.type === 'report' ? (
                                <FileText className="h-4 w-4" />
                              ) : (
                                <Scan className="h-4 w-4" />
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
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Scans & GradCAM Tab */}
          <TabsContent value="scans" className="mt-6 space-y-6">
            {/* MRI Scans Section */}
            <div>
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Scan className="h-5 w-5 text-[var(--brand-teal)]" />
                MRI Scans ({scans.length})
              </h3>
              {scans.length === 0 ? (
                <EmptyState icon={Scan} title="No MRI Scans" description="Add scans to see them here" />
              ) : (
                <div className="grid gap-4 md:grid-cols-2">
                  {scans.map((scan) => (
                    <Card key={scan.id}>
                      <CardHeader>
                        <CardTitle className="text-sm flex items-center gap-2">
                          <Scan className="h-4 w-4 text-[var(--brand-teal)]" />
                          Scan {scan.id.slice(0, 8)}
                        </CardTitle>
                        <CardDescription>
                          {new Date(scan.uploaded_at).toLocaleString()}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="grid grid-cols-2 gap-2">
                        <div className="relative aspect-square rounded-lg bg-muted overflow-hidden">
                          {scan.left_scan_path ? (
                            <Image
                              src={`${API_BASE}/` + scan.left_scan_path}
                              alt="Left eye"
                              fill
                              className="object-cover"
                              unoptimized
                            />
                          ) : (
                            <div className="flex items-center justify-center h-full text-muted-foreground">
                              No image
                            </div>
                          )}
                          <span className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                            Left
                          </span>
                        </div>
                        <div className="relative aspect-square rounded-lg bg-muted overflow-hidden">
                          {scan.right_scan_path ? (
                            <Image
                              src={`${API_BASE}/` + scan.right_scan_path}
                              alt="Right eye"
                              fill
                              className="object-cover"
                              unoptimized
                            />
                          ) : (
                            <div className="flex items-center justify-center h-full text-muted-foreground">
                              No image
                            </div>
                          )}
                          <span className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                            Right
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>

            {/* GradCAM Analysis Section */}
            <div>
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Eye className="h-5 w-5 text-amber-500" />
                GradCAM Analysis ({predictions.filter((p) => p.status === 'success').length})
              </h3>
              {predictions.filter((p) => p.status === 'success').length === 0 ? (
                <EmptyState icon={Eye} title="No GradCAM Available" description="Run predictions to generate GradCAM analysis" />
              ) : (
                <div className="space-y-4">
                  {predictions
                    .filter((p) => p.status === 'success')
                    .map((pred) => {
                      const grade = pred.output_payload?.combined_grade as number | undefined;
                      const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : null;
                      return (
                        <Card key={pred.id}>
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                {getStatusIcon(pred.status)}
                                <div>
                                  <CardTitle className="text-sm">{pred.model_name}</CardTitle>
                                  <CardDescription>
                                    {new Date(pred.created_at).toLocaleString()}
                                  </CardDescription>
                                </div>
                              </div>
                              <div className="text-right">
                                <p className="text-lg font-bold">
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
                            <div className="grid grid-cols-2 gap-4">
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
            </div>
          </TabsContent>

          {/* XAI Explanations Tab */}
          <TabsContent value="xai" className="mt-6 space-y-6">
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-500" />
                  AI Explanations
                </h3>
                {latestPrediction && !latestPrediction.output_payload?.shap_values && (
                  <Button
                    onClick={() => handleGenerateXAI(latestPrediction)}
                    disabled={generatingXAI === latestPrediction.id}
                    size="sm"
                  >
                    {generatingXAI === latestPrediction.id ? (
                      <>
                        <Loader className="h-4 w-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Generate XAI
                      </>
                    )}
                  </Button>
                )}
              </div>
              {predictions.filter((p) => p.status === 'success').length === 0 ? (
                <EmptyState
                  icon={Brain}
                  title="No XAI Explanations Available"
                  description="Run predictions to generate AI explanations"
                />
              ) : (
                <div className="space-y-4">
                  {predictions
                    .filter((p) => p.status === 'success')
                    .map((pred) => {
                      const outputPayload = pred.output_payload as Record<string, unknown> | null;
                      const grade = outputPayload?.combined_grade as number | undefined;
                      const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : null;
                      const shapValues = outputPayload?.shap_values as { top_positive: Array<{ name: string; contribution: number }> } | undefined;
                      const explanation = outputPayload?.explanation as string | undefined;

                      return (
                        <Card key={pred.id}>
                          <CardHeader>
                            <div className="flex items-center justify-between">
                              <CardTitle className="text-base">
                                Prediction - {new Date(pred.created_at).toLocaleDateString()}
                              </CardTitle>
                              {grade !== undefined && gradeLabel && (
                                <Badge className={GRADE_COLORS_NUM[grade] || 'bg-muted'}>
                                  {gradeLabel}
                                </Badge>
                              )}
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            {/* Confidence Score */}
                            <div>
                              <p className="text-sm text-muted-foreground mb-1">Confidence Score</p>
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{
                                      width: `${(pred.confidence_score || 0) * 100}%`,
                                    }}
                                    className="h-full bg-[var(--brand-teal)]"
                                  />
                                </div>
                                <span className="text-sm font-medium">
                                  {pred.confidence_score
                                    ? `${(pred.confidence_score * 100).toFixed(1)}%`
                                    : 'N/A'}
                                </span>
                              </div>
                            </div>

                            {/* SHAP Values */}
                            {shapValues && (
                              <div>
                                <p className="text-sm text-muted-foreground mb-2 flex items-center gap-1">
                                  <BarChart3 className="h-4 w-4" />
                                  Top Contributing Features
                                </p>
                                <div className="space-y-2">
                                  {(shapValues as any)?.top_positive?.slice(0, 5).map(
                                    (feature: { name: string; contribution: number }, i: number) => (
                                      <div key={i} className="flex items-center gap-2">
                                        <span className="text-sm w-32 truncate">{feature.name}</span>
                                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                          <motion.div
                                            initial={{ width: 0 }}
                                            animate={{
                                              width: `${Math.min(Math.abs(feature.contribution) * 100, 100)}%`,
                                            }}
                                            transition={{ delay: i * 0.1 }}
                                            className="h-full bg-emerald-500"
                                          />
                                        </div>
                                        <span className="text-xs text-muted-foreground w-12 text-right">
                                          {feature.contribution.toFixed(3)}
                                        </span>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>
                            )}

                            {/* LLM Explanation */}
                            {explanation && (
                              <div>
                                <p className="text-sm text-muted-foreground mb-2 flex items-center gap-1">
                                  <FileText className="h-4 w-4" />
                                  AI Interpretation
                                </p>
                                <div className="bg-muted/50 rounded-lg p-3 text-sm">
                                  {typeof explanation === 'string'
                                    ? explanation
                                    : JSON.stringify(explanation, null, 2)}
                                </div>
                              </div>
                            )}

                            {!shapValues && !explanation && (
                              <div className="flex flex-col items-center gap-2 py-4">
                                <p className="text-sm text-muted-foreground italic">
                                  No XAI explanations generated for this prediction
                                </p>
                                <Button
                                  onClick={() => handleGenerateXAI(pred)}
                                  disabled={generatingXAI === pred.id}
                                  size="sm"
                                  variant="outline"
                                >
                                  {generatingXAI === pred.id ? (
                                    <>
                                      <Loader className="h-4 w-4 mr-2 animate-spin" />
                                      Generating...
                                    </>
                                  ) : (
                                    <>
                                      <Sparkles className="h-4 w-4 mr-2" />
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
            </div>
          </TabsContent>

          {/* Reports Tab */}
          <TabsContent value="reports" className="mt-6 space-y-6">
            {/* Clinical Reports */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <FileText className="h-5 w-5 text-blue-500" />
                  Clinical Reports ({reports.length})
                </h3>
                {latestPrediction && (
                  <Button
                    onClick={() => handleGenerateReport(latestPrediction.id)}
                    disabled={generatingReport}
                    size="sm"
                    variant="outline"
                  >
                    {generatingReport ? (
                      <>
                        <Loader className="h-4 w-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2" />
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
                <div className="space-y-6">
                  {reports
                    .filter((r) => r.status === 'completed')
                    .map((report) => (
                      <MedicalReport key={report.id} report={report} />
                    ))}
                  {reports.filter((r) => r.status !== 'completed').length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium text-muted-foreground">
                        Pending Reports ({reports.filter((r) => r.status !== 'completed').length})
                      </h4>
                      {reports
                        .filter((r) => r.status !== 'completed')
                        .map((report) => (
                          <Card key={report.id} className="opacity-60">
                            <CardContent className="flex items-center justify-between py-3">
                              <div className="flex items-center gap-3">
                                {getStatusIcon(report.status)}
                                <div>
                                  <p className="font-medium">{report.llm_model}</p>
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
            </div>

            {/* OCT Reports */}
            <div>
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="h-5 w-5 text-purple-500" />
                OCT Analysis ({octReports.length})
              </h3>
              {octReports.length === 0 ? (
                <EmptyState
                  icon={Activity}
                  title="No OCT Reports"
                  description="Process OCT scans to see them here"
                />
              ) : (
                <Card>
                  <CardContent className="p-0">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-3">Eye</th>
                          <th className="text-left p-3">DR Grade</th>
                          <th className="text-left p-3">Edema</th>
                          <th className="text-left p-3">ERM</th>
                          <th className="text-left p-3">Quality</th>
                          <th className="text-left p-3">Center Fovea (μm)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {octReports.map((oct) => (
                          <tr key={oct.id} className="border-b">
                            <td className="p-3">{oct.eye}</td>
                            <td className="p-3">
                              <Badge className={GRADE_COLORS[oct.dr_grade || ''] || 'bg-muted'}>
                                {oct.dr_grade || 'N/A'}
                              </Badge>
                            </td>
                            <td className="p-3">{oct.edema ? 'Yes' : 'No'}</td>
                            <td className="p-3">{oct.erm_status || 'N/A'}</td>
                            <td className="p-3">{oct.image_quality ? `${oct.image_quality}%` : 'N/A'}</td>
                            <td className="p-3">{oct.thickness_center_fovea || 'N/A'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
        </Tabs>
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
    <div className="space-y-2">
      <p className="text-sm font-medium text-muted-foreground">{title}</p>
      {gradcamBase64 ? (
        <div className="relative aspect-square rounded-lg overflow-hidden bg-black">
          <img
            src={`data:image/png;base64,${gradcamBase64}`}
            alt={title}
            className="w-full h-full object-contain"
          />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center aspect-square bg-muted rounded-lg">
          <ImageIcon className="h-8 w-8 text-muted-foreground" />
          <p className="text-xs text-muted-foreground mt-1">No GradCAM</p>
        </div>
      )}
    </div>
  );
}
