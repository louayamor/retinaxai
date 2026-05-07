'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import {
  createPrediction,
  uploadScans,
  getPatients,
  listAllPredictions,
  getPatient,
  createReport,
  listAllReports,
  PredictionRequest
} from '@/lib/api';
import { usePatientWebSocket, type LogMessageData, type BiomarkerEventData } from '@/hooks/use-patient-websocket';
import PageContainer from '@/components/layout/page-container';
import { PageHero } from '@/components/ui/page-hero';
import { PageSection } from '@/components/ui/page-section';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/dialog';
import {
  Upload,
  FileImage,
  FileText,
  X,
  Loader2,
  Eye,
  RefreshCw,
  User,
  Activity,
  AlertCircle,
  CheckCircle2,
  ImageIcon
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { toast } from 'sonner';
import type { Patient, Prediction, PredictionStatus, DRSeverity, Report } from '@/types';
import { fadeInUp, slideInUp, borderPulse, scaleIn } from '@/lib/animations';
import { StatusBadge } from '@/components/ui/status-badge';
import { PredictionProgress } from '@/components/prediction-progress';

const SEVERITY_COLORS: Record<DRSeverity, string> = {
  no_dr: 'bg-green-500',
  mild: 'bg-blue-500',
  moderate: 'bg-yellow-500',
  severe: 'bg-orange-500',
  proliferative: 'bg-red-500'
};

const SEVERITY_LABELS: Record<DRSeverity, string> = {
  no_dr: 'No DR',
  mild: 'Mild',
  moderate: 'Moderate',
  severe: 'Severe',
  proliferative: 'Proliferative'
};

const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];

const GRADE_COLORS: Record<string, string> = {
  0: 'bg-emerald-500',
  1: 'bg-cyan-500',
  2: 'bg-amber-500',
  3: 'bg-orange-500',
  4: 'bg-rose-500',
};

interface FileUpload {
  file: File;
  preview: string;
}

interface PredictionWorkflowState {
  status: 'idle' | 'uploading' | 'predicting' | 'biomarker' | 'xai' | 'reporting' | 'completed' | 'failed';
  stage: string;
  progress: number;
  message: string;
}

interface BiomarkerStageState {
  status: 'idle' | 'started' | 'completed' | 'failed';
  message: string;
  progress: number;
}

const INITIAL_WORKFLOW: PredictionWorkflowState = {
  status: 'idle',
  stage: 'upload',
  progress: 0,
  message: 'Upload scans to start prediction',
};

export default function PredictionsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const shouldReduceMotion = useReducedMotion();
  const initialTab = searchParams.get('tab') || 'screening';

  const [activeTab, setActiveTab] = useState(initialTab);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [lockedPatientId, setLockedPatientId] = useState<string | null>(null);
  const [leftEyeFile, setLeftEyeFile] = useState<FileUpload | null>(null);
  const [rightEyeFile, setRightEyeFile] = useState<FileUpload | null>(null);
  const [uploading, setUploading] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionsLoading, setPredictionsLoading] = useState(true);
  const [reports, setReports] = useState<Report[]>([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [patientNames, setPatientNames] = useState<Record<string, string>>({});
  const [logMessages, setLogMessages] = useState<LogMessageData[]>([]);
  const [workflow, setWorkflow] = useState<PredictionWorkflowState>(INITIAL_WORKFLOW);
  const [biomarkerStages, setBiomarkerStages] = useState<Record<string, BiomarkerStageState>>({
    left: { status: 'idle', message: 'Waiting for left-eye biomarkers', progress: 0 },
    right: { status: 'idle', message: 'Waiting for right-eye biomarkers', progress: 0 },
  });

  const appendLog = (step: string, status: LogMessageData['status'], message: string) => {
    setLogMessages((prev) => [
      ...prev,
      {
        prediction_id: selectedPatientId || 'pending',
        patient_id: selectedPatientId || 'pending',
        step,
        status,
        message,
        timestamp: new Date().toISOString(),
      },
    ]);
  };

  const { connected: wsConnected, send: wsSend } = usePatientWebSocket({
    patientId: lockedPatientId || selectedPatientId || 'global',
    onLogMessage: (data) => {
      setLogMessages(prev => [...prev, data]);
    },
    onPredictionComplete: (data) => {
      appendLog('prediction', 'success', `Prediction completed: Grade ${data.dr_grade}, ${data.overall_severity}`);
      setWorkflow({
        status: 'xai',
        stage: 'xai',
        progress: 60,
        message: 'Prediction complete. Generating explanations...',
      });
    },
    onPredictionFailed: (data) => {
      appendLog('prediction', 'error', data.error || 'Prediction failed');
      setWorkflow({
        status: 'failed',
        stage: 'failed',
        progress: 100,
        message: data.error || 'Prediction failed',
      });
    },
    onBiomarkerUpdate: (data: BiomarkerEventData) => {
      setBiomarkerStages((prev) => ({
        ...prev,
        [data.eye_side]: {
          status: data.status === 'completed' ? 'completed' : data.status === 'failed' ? 'failed' : 'started',
          message: data.message,
          progress: data.progress,
        },
      }));

      appendLog(
        `biomarker:${data.eye_side}`,
        data.status === 'failed' ? 'error' : data.status === 'completed' ? 'success' : 'info',
        data.message,
      );
      setWorkflow((prev) => ({
        ...prev,
        status: data.status === 'failed' ? 'failed' : prev.status === 'completed' ? 'completed' : 'biomarker',
        stage: data.status === 'failed' ? 'failed' : prev.stage === 'completed' ? 'completed' : 'biomarker',
        progress: data.status === 'failed' ? 100 : Math.max(prev.progress, 65),
        message: data.message,
      }));
    },
    onGradCAMReady: (data) => {
      appendLog('xai', 'success', data.message || 'GradCAM analysis complete');
      setWorkflow((prev) => ({
        ...prev,
        status: 'xai',
        stage: 'xai',
        progress: Math.max(prev.progress, 75),
        message: data.message || 'GradCAM analysis complete',
      }));
    },
    onXAIReady: (data) => {
      appendLog('xai', 'success', data.message || 'Explanation ready');
      setWorkflow((prev) => ({
        ...prev,
        status: 'xai',
        stage: 'xai',
        progress: Math.max(prev.progress, 85),
        message: data.message || 'Explanation ready',
      }));
    },
    onSeverityReady: (data) => {
      appendLog('xai', 'success', data.message || 'Risk assessment complete');
      setWorkflow((prev) => ({
        ...prev,
        status: 'completed',
        stage: 'completed',
        progress: 100,
        message: data.message || 'Workflow completed',
      }));
      setBiomarkerStages((prev) => ({
        left: prev.left.status !== 'completed' ? { ...prev.left, status: 'completed', progress: 100 } : prev.left,
        right: prev.right.status !== 'completed' ? { ...prev.right, status: 'completed', progress: 100 } : prev.right,
      }));
      void loadPredictions();
    },
  });

  useEffect(() => {
    if (wsConnected && wsSend) {
      wsSend('subscribe', { room: 'training_stage' });
      if (selectedPatientId) {
        wsSend('subscribe', { room: `prediction:${selectedPatientId}` });
      }
    }
  }, [wsConnected, wsSend, selectedPatientId]);

  useEffect(() => {
    void loadPatients();
    void loadPredictions();
    void loadReports();
  }, []);

  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && ['screening', 'reports', 'gradcam'].includes(tab)) {
      setActiveTab(tab);
    }
  }, [searchParams]);

  const loadReports = async () => {
    try {
      setReportsLoading(true);
      const response = await listAllReports(1, 50);
      setReports(response.items);
    } catch (err) {
      console.error('Failed to load reports:', err);
    } finally {
      setReportsLoading(false);
    }
  };

  const loadPatients = async () => {
    try {
      const data = await getPatients();
      setPatients(data);
    } catch (err) {
      console.error('Failed to load patients:', err);
      toast.error('Failed to load patients');
    }
  };

  const loadPredictions = async () => {
    try {
      const response = await listAllPredictions(1, 50);
      setPredictions(response.items);

      const patientIds = [...new Set(response.items.map((p) => p.patient_id))];
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
      console.error('Failed to load predictions:', err);
      toast.error('Failed to load predictions');
    } finally {
      setPredictionsLoading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, eye: 'left' | 'right') => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const upload = { file, preview: e.target?.result as string };
      if (eye === 'left') {
        setLeftEyeFile(upload);
      } else {
        setRightEyeFile(upload);
      }
    };
    reader.readAsDataURL(file);
  };

  const clearFile = (eye: 'left' | 'right') => {
    if (eye === 'left') {
      setLeftEyeFile(null);
    } else {
      setRightEyeFile(null);
    }
  };

  const handleUploadAndPredict = async () => {
    if (!selectedPatientId) {
      toast.error('Please select a patient');
      return;
    }

    if (!leftEyeFile || !rightEyeFile) {
      toast.error('Please upload both left and right eye images');
      return;
    }

    setUploading(true);
    setLogMessages([]);
      setWorkflow({
        status: 'uploading',
        stage: 'upload',
        progress: 5,
        message: 'Uploading scans...',
      });
      setBiomarkerStages({
        left: { status: 'idle', message: 'Waiting for left-eye biomarkers', progress: 0 },
        right: { status: 'idle', message: 'Waiting for right-eye biomarkers', progress: 0 },
      });

    try {
      const formData = new FormData();
      formData.append('left_scan', leftEyeFile.file);
      formData.append('right_scan', rightEyeFile.file);
      formData.append('modality', 'fundus');

      const scan = await uploadScans(selectedPatientId, formData);
      toast.success('Scans uploaded successfully');
      appendLog('upload', 'success', 'Scans uploaded successfully');
      setLockedPatientId(selectedPatientId);
      if (wsConnected && wsSend) {
        wsSend('subscribe', { room: `prediction:${selectedPatientId}` });
      }
      setWorkflow({
        status: 'predicting',
        stage: 'prediction',
        progress: 35,
        message: 'Scans uploaded. Starting prediction...',
      });
      const predictionData: PredictionRequest = {
        patient_id: selectedPatientId,
        mri_scan_id: scan.id,
        model_name: 'efficientnet_b4',
        model_version: '1.0.0',
        input_payload: {
          left_eye_path: scan.left_scan_path,
          right_eye_path: scan.right_scan_path
        }
      };

      await createPrediction(predictionData);
      toast.success('Prediction started');
      appendLog('prediction', 'info', 'Prediction request submitted');
      setWorkflow((prev) => ({
        ...prev,
        status: 'predicting',
        stage: 'prediction',
        progress: 50,
        message: 'Prediction request accepted. Waiting for backend events...',
      }));

      void loadPredictions();
    } catch (err) {
      console.error('Failed to process:', err);
      toast.error(err instanceof Error ? err.message : 'Failed to process');
      setWorkflow({
        status: 'failed',
        stage: 'failed',
        progress: 100,
        message: err instanceof Error ? err.message : 'Failed to process',
      });
    } finally {
      setUploading(false);
    }
  };

  const getSeverityFromPrediction = (prediction: Prediction): DRSeverity | null => {
    if (!prediction.output_payload?.severity) return null;
    return prediction.output_payload.severity as DRSeverity;
  };

  const viewPredictionDetails = (prediction: Prediction) => {
    setSelectedPrediction(prediction);
    setDetailOpen(true);
  };

  const handleGenerateReport = async () => {
    if (!selectedPrediction) return;
    setGeneratingReport(true);
    try {
      await createReport(selectedPrediction.id);
      toast.success('Report generated successfully');
      setDetailOpen(false);
    } catch (err) {
      console.error('Failed to generate report:', err);
      toast.error(err instanceof Error ? err.message : 'Failed to generate report');
    } finally {
      setGeneratingReport(false);
    }
  };

  useEffect(() => {
    if (!wsConnected || !selectedPatientId) return;

    const handleVisibilityRefresh = () => {
      if (document.visibilityState === 'visible') {
        appendLog('prediction', 'info', 'Tab is visible. Refreshing prediction list');
        void loadPredictions();
      }
    };

    const timer = window.setTimeout(() => {
      if (workflow.status === 'predicting' || workflow.status === 'xai') {
        appendLog('prediction', 'warning', 'Waiting for backend events... refreshing prediction list');
        void loadPredictions();
      }
    }, 30000);

    window.addEventListener('visibilitychange', handleVisibilityRefresh);

    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('visibilitychange', handleVisibilityRefresh);
    };
  }, [wsConnected, selectedPatientId, workflow.status]);

  return (
    <PageContainer className='flex flex-col gap-6'>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeInUp}
        initial='hidden'
        animate='visible'
        className='flex flex-col gap-6'
      >
        <PageHero
          title='Diagnostics'
          description='AI-assisted diabetic retinopathy screening, clinical reports, and explainability'
        />

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="screening">Screening</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
            <TabsTrigger value="gradcam">GradCAM</TabsTrigger>
          </TabsList>

          <TabsContent value="screening" className="space-y-6">
            <PageSection
              title='New Prediction'
              description='Select a patient and upload fundus images to run AI prediction'
              icon={<Activity className='h-5 w-5' />}
            >
          <div className='space-y-6'>
            <div className='space-y-2'>
              <label className='text-sm font-medium'>
                <User className='mr-1 inline-block h-4 w-4' />
                Select Patient
              </label>
              <Select value={selectedPatientId} onValueChange={setSelectedPatientId}>
                <SelectTrigger className='w-full max-w-md'>
                  <SelectValue placeholder='Choose a patient...' />
                </SelectTrigger>
                <SelectContent>
                  {patients.map((patient) => (
                    <SelectItem key={patient.id} value={patient.id}>
                      {patient.first_name} {patient.last_name} (MRN: {patient.medical_record_number})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <PredictionProgress
              status={workflow.status}
              stage={workflow.stage}
              progress={workflow.progress}
              message={workflow.message}
            />

            <div className='grid gap-3 md:grid-cols-2'>
              <PredictionProgress
                status={biomarkerStages.left.status === 'failed' ? 'failed' : biomarkerStages.left.status === 'started' || biomarkerStages.left.status === 'completed' ? 'biomarker' : 'idle'}
                stage='biomarker_left'
                progress={biomarkerStages.left.progress}
                message={biomarkerStages.left.message}
              />
              <PredictionProgress
                status={biomarkerStages.right.status === 'failed' ? 'failed' : biomarkerStages.right.status === 'started' || biomarkerStages.right.status === 'completed' ? 'biomarker' : 'idle'}
                stage='biomarker_right'
                progress={biomarkerStages.right.progress}
                message={biomarkerStages.right.message}
              />
            </div>

            <div className='grid gap-6 md:grid-cols-2 lg:grid-cols-3'>
              <div className='space-y-2'>
                <label className='text-sm font-medium'>Left Eye (OS)</label>
                <div
                  className={`relative flex h-48 flex-col items-center justify-center rounded-lg border-2 border-dashed ${
                    leftEyeFile
                      ? 'border-primary bg-primary/5'
                      : 'border-muted-foreground/25 hover:border-primary'
                  }`}
                >
                  {leftEyeFile ? (
                    <div className='relative h-full w-full p-4'>
                      <img
                        src={leftEyeFile.preview}
                        alt='Left eye preview'
                        className='h-full w-full rounded object-contain'
                      />
                      <button
                        onClick={() => clearFile('left')}
                        className='absolute right-2 top-2 rounded-full bg-destructive p-1 text-white hover:bg-destructive/90'
                        title='Remove left eye image'
                      >
                        <X className='h-4 w-4' />
                      </button>
                    </div>
                  ) : (
                    <label className='flex cursor-pointer flex-col items-center justify-center gap-2 p-6'>
                      <FileImage className='h-10 w-10 text-muted-foreground' />
                      <span className='text-sm text-muted-foreground'>
                        Click to upload left eye image
                      </span>
                      <Input
                        type='file'
                        accept='image/*'
                        className='hidden'
                        onChange={(e) => handleFileSelect(e, 'left')}
                      />
                    </label>
                  )}
                </div>
              </div>

              <div className='space-y-2'>
                <label className='text-sm font-medium'>Right Eye (OD)</label>
                <div
                  className={`relative flex h-48 flex-col items-center justify-center rounded-lg border-2 border-dashed ${
                    rightEyeFile
                      ? 'border-primary bg-primary/5'
                      : 'border-muted-foreground/25 hover:border-primary'
                  }`}
                >
                  {rightEyeFile ? (
                    <div className='relative h-full w-full p-4'>
                      <img
                        src={rightEyeFile.preview}
                        alt='Right eye preview'
                        className='h-full w-full rounded object-contain'
                      />
                      <button
                        onClick={() => clearFile('right')}
                        className='absolute right-2 top-2 rounded-full bg-destructive p-1 text-white hover:bg-destructive/90'
                        title='Remove right eye image'
                      >
                        <X className='h-4 w-4' />
                      </button>
                    </div>
                  ) : (
                    <label className='flex cursor-pointer flex-col items-center justify-center gap-2 p-6'>
                      <FileImage className='h-10 w-10 text-muted-foreground' />
                      <span className='text-sm text-muted-foreground'>
                        Click to upload right eye image
                      </span>
                      <Input
                        type='file'
                        accept='image/*'
                        className='hidden'
                        onChange={(e) => handleFileSelect(e, 'right')}
                      />
                    </label>
                  )}
                </div>
              </div>

              <div className='space-y-2 lg:col-span-1'>
                <label className='text-sm font-medium'>Status</label>
                <div className='h-48 space-y-2 overflow-y-auto rounded-lg border bg-muted/30 p-3'>
                  {logMessages.length === 0 ? (
                    <p className='text-xs text-muted-foreground'>Upload images to start prediction</p>
                  ) : (
                    logMessages.map((msg, idx) => (
                      <div key={idx} className='flex items-start gap-2 text-xs'>
                        <span className={`mt-0.5 inline-block h-2 w-2 rounded-full flex-shrink-0 ${
                          msg.status === 'info' ? 'bg-blue-500' :
                          msg.status === 'success' ? 'bg-emerald-500' :
                          msg.status === 'warning' ? 'bg-amber-500' :
                          'bg-rose-500'
                        }`} />
                        <span className={
                          msg.status === 'success' ? 'text-emerald-600 dark:text-emerald-400' :
                          msg.status === 'error' ? 'text-rose-600 dark:text-rose-400' :
                          msg.status === 'warning' ? 'text-amber-600 dark:text-amber-400' :
                          'text-muted-foreground'
                        }>
                          {msg.message}
                        </span>
                      </div>
                    ))
                  )}
                </div>
                <div className='flex items-center gap-2 text-xs text-muted-foreground'>
                  <span className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                  {wsConnected ? 'Live' : 'Connecting...'}
                </div>
              </div>
            </div>

            <div className='flex justify-end'>
              <Button
                onClick={handleUploadAndPredict}
                disabled={!selectedPatientId || !leftEyeFile || !rightEyeFile || uploading || workflow.status === 'uploading' || workflow.status === 'predicting' || workflow.status === 'xai'}
                className='min-w-[200px]'
              >
                {uploading ? (
                  <>
                    <Loader2 className='mr-2 h-4 w-4 animate-spin' />
                    Processing...
                  </>
                ) : (
                  <>
                        <Upload className='mr-2 h-4 w-4' />
                        Upload & Predict
                      </>
                )}
              </Button>
            </div>
          </div>
        </PageSection>

        {predictions.length > 0 && (
          <PageSection title='Prediction Timeline' description='DR grade progression over time across all patients'>
            <div className='h-[300px]'>
              <ResponsiveContainer width='100%' height='100%'>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray='3 3' className='opacity-30' />
                  <XAxis
                    dataKey='date'
                    type='category'
                    tick={{ fontSize: 11 }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  />
                  <YAxis
                    dataKey='gradeLabel'
                    type='category'
                    domain={['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']}
                    tick={{ fontSize: 11 }}
                  />
                  <ZAxis dataKey='confidence' range={[50, 200]} name='Confidence' />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className='bg-background border rounded-lg p-3 shadow-lg'>
                            <p className='font-medium'>{data.patientName}</p>
                            <p className='text-sm text-muted-foreground'>
                              {new Date(data.date).toLocaleDateString()}
                            </p>
                            <p className='text-sm'>Grade: {data.gradeLabel}</p>
                            <p className='text-sm'>Confidence: {(data.confidence * 100).toFixed(1)}%</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter
                    name='Predictions'
                    data={predictions.map(p => {
                      const payload = p.output_payload as Record<string, unknown> | null;
                      const grade = payload?.combined_grade as number | undefined;
                      const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
                      return {
                        date: p.created_at,
                        gradeLabel: GRADE_LABELS[grade ?? 2] || 'Moderate',
                        grade: grade ?? 2,
                        confidence: p.confidence_score ?? 0.8,
                        patientName: patientNames[p.patient_id] || 'Unknown',
                      };
                    })}
                    fill='#2563eb'
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </PageSection>
        )}

        <PageSection
          title='Recent Predictions'
          padding='none'
          headerAction={
            <Button variant='outline' size='sm' onClick={loadPredictions}>
              <RefreshCw className='mr-2 h-4 w-4' />
              Refresh
            </Button>
          }
        >
          {predictionsLoading ? (
            <div className='py-8 text-center'>
              <Loader2 className='mx-auto h-8 w-8 animate-spin text-muted-foreground' />
              <p className='mt-2 text-sm text-muted-foreground'>Loading predictions...</p>
            </div>
          ) : predictions.length === 0 ? (
            <div className='flex flex-col items-center justify-center py-12'>
              <Activity className='mb-4 h-16 w-16 text-muted-foreground' />
              <p className='text-muted-foreground'>
                No predictions yet. Upload scans to get started.
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Patient</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead className='text-right'>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {predictions.map((prediction) => {
                  const severity = getSeverityFromPrediction(prediction);
                  return (
                    <TableRow key={prediction.id}>
                      <TableCell className='font-medium'>
                        {patientNames[prediction.patient_id] || 'Loading...'}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={prediction.status} />
                      </TableCell>
                      <TableCell>
                        {severity ? (
                          <Badge className={`${SEVERITY_COLORS[severity]} text-white`}>
                            {SEVERITY_LABELS[severity]}
                          </Badge>
                        ) : (
                          <span className='text-muted-foreground'>—</span>
                        )}
                      </TableCell>
                      <TableCell>
                        {prediction.confidence_score != null
                          ? `${(prediction.confidence_score * 100).toFixed(1)}%`
                          : '—'}
                      </TableCell>
                      <TableCell className='text-sm text-muted-foreground'>
                        {new Date(prediction.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell className='text-right'>
                        <Button
                          variant='outline'
                          size='sm'
                          onClick={() => router.push(`/dashboard/predictions/${prediction.id}/gradcam`)}
                          disabled={prediction.status !== 'success'}
                        >
                          <Eye className='mr-2 h-4 w-4' />
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </PageSection>
          </TabsContent>

          <TabsContent value="reports" className="space-y-6">
            <PageSection
              title='Clinical Reports'
              description='AI-generated clinical reports from DR screening results'
              padding="none"
              headerAction={
                <Button variant="outline" size="sm" onClick={loadReports}>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </Button>
              }
            >
              {reportsLoading ? (
                <div className="py-8 text-center">
                  <Loader2 className="mx-auto h-8 w-8 animate-spin text-muted-foreground" />
                  <p className="mt-2 text-sm text-muted-foreground">Loading reports...</p>
                </div>
              ) : reports.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12">
                  <FileText className="mb-4 h-16 w-16 text-muted-foreground" />
                  <p className="text-muted-foreground">
                    No clinical reports yet. Generate a report from a prediction.
                  </p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Patient</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Date</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {reports.map((report) => (
                      <TableRow key={report.id}>
                        <TableCell className="font-medium">
                          {report.patient_id ? patientNames[report.patient_id] || 'Loading...' : 'N/A'}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{report.report_type || 'LLM'}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={report.status === 'completed' ? 'default' : 'secondary'}>
                            {report.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {new Date(report.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => router.push(`/dashboard/reports?id=${report.id}`)}
                          >
                            <Eye className="mr-1 h-4 w-4" />
                            View
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </PageSection>
          </TabsContent>

          <TabsContent value="gradcam" className="space-y-6">
            <PageSection
              title='GradCAM Visualizations'
              description='AI explainability heatmaps for DR predictions'
              padding="none"
              headerAction={
                <Button variant="outline" size="sm" onClick={loadPredictions}>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </Button>
              }
            >
              {predictionsLoading ? (
                <div className="py-8 text-center">
                  <Loader2 className="mx-auto h-8 w-8 animate-spin text-muted-foreground" />
                  <p className="mt-2 text-sm text-muted-foreground">Loading...</p>
                </div>
              ) : (
                (() => {
                  const withGradCAM = predictions.filter(
                    (p) => p.output_payload?.gradcam_left || p.output_payload?.gradcam_right
                  );
                  if (withGradCAM.length === 0) {
                    return (
                      <div className="flex flex-col items-center justify-center py-12">
                        <ImageIcon className="mb-4 h-16 w-16 text-muted-foreground" />
                        <p className="text-muted-foreground font-medium">
                          No GradCAM Visualizations Available
                        </p>
                        <p className="text-sm text-muted-foreground mt-1">
                          Run predictions with GradCAM enabled to see heatmaps here.
                        </p>
                      </div>
                    );
                  }
                  return (
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 p-4">
                      {withGradCAM.map((prediction) => {
                        const grade = prediction.output_payload?.combined_grade as number | undefined;
                        const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : null;
                        return (
                          <Card
                            key={prediction.id}
                            className="cursor-pointer hover:shadow-lg transition-shadow"
                            onClick={() => router.push(`/dashboard/predictions/${prediction.id}/gradcam`)}
                          >
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm flex items-center justify-between">
                                <span>{patientNames[prediction.patient_id] || 'Loading...'}</span>
                                {gradeLabel && (
                                  <Badge className={GRADE_COLORS[String(grade)] || 'bg-muted'}>
                                    {gradeLabel}
                                  </Badge>
                                )}
                              </CardTitle>
                            </CardHeader>
                            <CardContent>
                              <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                                <span>{new Date(prediction.created_at).toLocaleDateString()}</span>
                                <span>|</span>
                                <span>
                                  {prediction.confidence_score
                                    ? `${(prediction.confidence_score * 100).toFixed(1)}%`
                                    : 'N/A'}
                                </span>
                              </div>
                              <div className="grid grid-cols-2 gap-2 mb-3">
                                {prediction.output_payload?.gradcam_left ? (
                                  <div className="relative aspect-square rounded-lg overflow-hidden bg-black">
                                    <img
                                      src={`data:image/png;base64,${prediction.output_payload.gradcam_left}`}
                                      alt="Left eye"
                                      className="w-full h-full object-cover"
                                    />
                                    <span className="absolute bottom-1 left-1 text-[10px] text-white bg-black/50 px-1 rounded">
                                      OS
                                    </span>
                                  </div>
                                ) : (
                                  <div className="relative aspect-square rounded-lg bg-muted flex items-center justify-center">
                                    <span className="text-xs text-muted-foreground">No Left</span>
                                  </div>
                                )}
                                {prediction.output_payload?.gradcam_right ? (
                                  <div className="relative aspect-square rounded-lg overflow-hidden bg-black">
                                    <img
                                      src={`data:image/png;base64,${prediction.output_payload.gradcam_right}`}
                                      alt="Right eye"
                                      className="w-full h-full object-cover"
                                    />
                                    <span className="absolute bottom-1 right-1 text-[10px] text-white bg-black/50 px-1 rounded">
                                      OD
                                    </span>
                                  </div>
                                ) : (
                                  <div className="relative aspect-square rounded-lg bg-muted flex items-center justify-center">
                                    <span className="text-xs text-muted-foreground">No Right</span>
                                  </div>
                                )}
                              </div>
                              <Button variant="outline" size="sm" className="w-full">
                                <Eye className="mr-2 h-4 w-4" />
                                View Analysis
                              </Button>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  );
                })()
              )}
            </PageSection>
          </TabsContent>
        </Tabs>
      </motion.div>

      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className='max-w-3xl'>
          <DialogHeader>
            <DialogTitle className='flex items-center gap-2'>
              {selectedPrediction?.status === 'success' ? (
                <CheckCircle2 className='h-5 w-5 text-green-500' />
              ) : selectedPrediction?.status === 'failed' ? (
                <AlertCircle className='h-5 w-5 text-red-500' />
              ) : (
                <Loader2 className='h-5 w-5 animate-spin' />
              )}
              Prediction Details
            </DialogTitle>
            <DialogDescription>
              {selectedPrediction && (
                <>
                  Patient: {patientNames[selectedPrediction.patient_id] || 'Unknown'} |
                  Date: {new Date(selectedPrediction.created_at).toLocaleString()}
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          {selectedPrediction && (
            <div className='space-y-6'>
              <div className='flex items-center gap-4'>
                <span className='text-sm font-medium'>Status:</span>
                <StatusBadge status={selectedPrediction.status} />
              </div>

              {selectedPrediction.status === 'success' && (
                <div className='flex justify-end'>
                  <Button
                    onClick={handleGenerateReport}
                    disabled={generatingReport}
                  >
                    {generatingReport ? (
                      <>
                        <Loader2 className='mr-2 h-4 w-4 animate-spin' />
                        Generating...
                      </>
                    ) : (
                      <>
                        <FileText className='mr-2 h-4 w-4' />
                        Generate Report
                      </>
                    )}
                  </Button>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </PageContainer>
  );
}
