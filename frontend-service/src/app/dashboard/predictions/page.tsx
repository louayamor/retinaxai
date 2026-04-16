'use client';

import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import {
  createPrediction,
  uploadScans,
  getPatients,
  listAllPredictions,
  getPatient,
  createReport,
  PredictionRequest
} from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
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
  CheckCircle2
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
import Image from 'next/image';
import { toast } from 'sonner';
import { useWebSocket } from '@/hooks/use-websocket';
import type { Patient, Prediction, PredictionStatus, DRSeverity } from '@/types';
import { fadeInUp, slideInUp, staggerContainer, staggerItemFast, buttonTap, scaleIn, dialogOverlay, dialogContent, borderPulse } from '@/lib/animations';

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

const STATUS_COLORS: Record<PredictionStatus, string> = {
  pending: 'bg-yellow-500',
  success: 'bg-green-500',
  failed: 'bg-red-500'
};

const STATUS_LABELS: Record<PredictionStatus, string> = {
  pending: 'Processing',
  success: 'Completed',
  failed: 'Failed'
};

interface FileUpload {
  file: File;
  preview: string;
}

export default function PredictionsPage() {
  const router = useRouter();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [leftEyeFile, setLeftEyeFile] = useState<FileUpload | null>(null);
  const [rightEyeFile, setRightEyeFile] = useState<FileUpload | null>(null);
  const [uploading, setUploading] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionsLoading, setPredictionsLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [patientNames, setPatientNames] = useState<Record<string, string>>({});
  const shouldReduceMotion = useReducedMotion();

  const { connected, subscribe } = useWebSocket();

  // WebSocket subscription for real-time updates
  useEffect(() => {
    const unsub = subscribe('training_stage', () => {
      // When any training event occurs, refresh predictions to get updated statuses
      loadPredictions();
    });

    return () => {
      unsub();
    };
  }, [subscribe]);

  useEffect(() => {
    loadPatients();
    loadPredictions();
  }, []);

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

    try {
      const formData = new FormData();
      formData.append('left_scan', leftEyeFile.file);
      formData.append('right_scan', rightEyeFile.file);
      formData.append('modality', 'fundus');

      const scan = await uploadScans(selectedPatientId, formData);
      toast.success('Scans uploaded successfully');

      const predictionData: PredictionRequest = {
        patient_id: selectedPatientId,
        mri_scan_id: scan.id,
        model_name: 'efficientnet_b3',
        model_version: '1.0.0',
        input_payload: {
          left_eye_path: scan.left_scan_path,
          right_eye_path: scan.right_scan_path
        }
      };

      await createPrediction(predictionData);
      toast.success('Prediction started');

      setLeftEyeFile(null);
      setRightEyeFile(null);
      setSelectedPatientId('');

      await loadPredictions();
    } catch (err) {
      console.error('Failed to process:', err);
      toast.error(err instanceof Error ? err.message : 'Failed to process');
    } finally {
      setUploading(false);
    }
  };

  const getSeverityFromPrediction = (prediction: Prediction): DRSeverity | null => {
    if (!prediction.output_payload?.severity) return null;
    return prediction.output_payload.severity as DRSeverity;
  };

  const getConfidenceDisplay = (prediction: Prediction): string => {
    if (prediction.confidence_score === null || prediction.confidence_score === undefined) {
      return '—';
    }
    return `${(prediction.confidence_score * 100).toFixed(1)}%`;
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

  return (
    <PageContainer>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeInUp}
        initial="hidden"
        animate="visible"
        className="flex flex-col gap-8"
      >
        {/* Hero */}
        <motion.div
          variants={shouldReduceMotion ? {} : slideInUp}
          className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#0a2e3e] via-[#0d3a4c] to-[#104a5e] p-10 text-white"
        >
          <div className="absolute right-0 top-0 h-full w-1/3 opacity-10">
            <Image
              src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&q=80"
              alt="AI Predictions"
              fill
              className="object-cover"
              unoptimized
            />
          </div>
          <div className="relative z-10">
            <h1 className="mb-2 text-3xl font-bold tracking-tight">DR Screening</h1>
            <p className="max-w-xl text-lg text-white/70">
              Upload scans and run DR grading predictions — AI-assisted retinal analysis
            </p>
          </div>
        </motion.div>

        {/* Upload Section */}
        <motion.div variants={shouldReduceMotion ? {} : slideInUp}>
          <Card className="border-dashed">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                New Prediction
              </CardTitle>
              <CardDescription>
                Select a patient and upload fundus images to run AI prediction
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Patient Selector */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  <User className="mr-1 inline-block h-4 w-4" />
                  Select Patient
                </label>
                <Select value={selectedPatientId} onValueChange={setSelectedPatientId}>
                  <SelectTrigger className="w-full max-w-md">
                    <SelectValue placeholder="Choose a patient..." />
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

              {/* File Upload Area */}
              <div className="grid gap-6 md:grid-cols-2">
                {/* Left Eye */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Left Eye (OS)</label>
                  <motion.div
                    variants={shouldReduceMotion ? {} : borderPulse}
                    className={`relative flex h-48 flex-col items-center justify-center rounded-lg border-2 border-dashed ${
                      leftEyeFile
                        ? "border-primary bg-primary/5"
                        : "border-muted-foreground/25 hover:border-primary"
                    }`}
                  >
                    {leftEyeFile ? (
                      <motion.div
                        variants={shouldReduceMotion ? {} : scaleIn}
                        className="relative h-full w-full p-4"
                      >
                        <img
                          src={leftEyeFile.preview}
                          alt="Left eye preview"
                          className="h-full w-full rounded object-contain"
                        />
                        <button
                          onClick={() => clearFile("left")}
                          className="absolute right-2 top-2 rounded-full bg-destructive p-1 text-white hover:bg-destructive/90"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </motion.div>
                    ) : (
                      <label className="flex cursor-pointer flex-col items-center justify-center gap-2 p-6">
                        <FileImage className="h-10 w-10 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">
                          Click to upload left eye image
                        </span>
                        <Input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => handleFileSelect(e, "left")}
                        />
                      </label>
                    )}
                  </motion.div>
                </div>

                {/* Right Eye */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Right Eye (OD)</label>
                  <motion.div
                    variants={shouldReduceMotion ? {} : borderPulse}
                    className={`relative flex h-48 flex-col items-center justify-center rounded-lg border-2 border-dashed ${
                      rightEyeFile
                        ? "border-primary bg-primary/5"
                        : "border-muted-foreground/25 hover:border-primary"
                    }`}
                  >
                    {rightEyeFile ? (
                      <motion.div
                        variants={shouldReduceMotion ? {} : scaleIn}
                        className="relative h-full w-full p-4"
                      >
                        <img
                          src={rightEyeFile.preview}
                          alt="Right eye preview"
                          className="h-full w-full rounded object-contain"
                        />
                        <button
                          onClick={() => clearFile("right")}
                          className="absolute right-2 top-2 rounded-full bg-destructive p-1 text-white hover:bg-destructive/90"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </motion.div>
                    ) : (
                      <label className="flex cursor-pointer flex-col items-center justify-center gap-2 p-6">
                        <FileImage className="h-10 w-10 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">
                          Click to upload right eye image
                        </span>
                        <Input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => handleFileSelect(e, "right")}
                        />
                      </label>
                    )}
                  </motion.div>
                </div>
              </div>

              {/* Submit Button */}
              <div className="flex justify-end">
                <motion.div variants={shouldReduceMotion ? {} : buttonTap}>
                  <Button
                    onClick={handleUploadAndPredict}
                    disabled={!selectedPatientId || !leftEyeFile || !rightEyeFile || uploading}
                    className="min-w-[200px]"
                  >
                    {uploading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="mr-2 h-4 w-4" />
                        Upload & Predict
                      </>
                    )}
                  </Button>
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Prediction Timeline */}
        {predictions.length > 0 && (
          <motion.div variants={shouldReduceMotion ? {} : fadeInUp}>
            <Card>
              <CardHeader>
                <CardTitle>Prediction Timeline</CardTitle>
                <CardDescription>DR grade progression over time across all patients</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis 
                        dataKey="date" 
                        type="category" 
                        tick={{ fontSize: 11 }}
                        tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      />
                      <YAxis 
                        dataKey="gradeLabel" 
                        type="category" 
                        domain={['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']}
                        tick={{ fontSize: 11 }}
                      />
                      <ZAxis dataKey="confidence" range={[50, 200]} name="Confidence" />
                      <Tooltip 
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-background border rounded-lg p-3 shadow-lg">
                                <p className="font-medium">{data.patientName}</p>
                                <p className="text-sm text-muted-foreground">
                                  {new Date(data.date).toLocaleDateString()}
                                </p>
                                <p className="text-sm">Grade: {data.gradeLabel}</p>
                                <p className="text-sm">Confidence: {(data.confidence * 100).toFixed(1)}%</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter 
                        name="Predictions" 
                        data={predictions.map(p => {
                          const payload = p.output_payload as Record<string, unknown> | null;
                          const grade = payload?.combined_grade as number | undefined;
                          const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
                          const pred = p as unknown as { patient?: { first_name?: string; last_name?: string } };
                          const patient = pred.patient;
                          return {
                            date: p.created_at,
                            gradeLabel: GRADE_LABELS[grade ?? 2] || 'Moderate',
                            grade: grade ?? 2,
                            confidence: p.confidence_score ?? 0.8,
                            patientName: patient?.first_name ? `${patient.first_name} ${patient.last_name || ''}` : 'Unknown',
                          };
                        })} 
                        fill="#2563eb"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Predictions List */}
        <motion.div variants={shouldReduceMotion ? {} : fadeInUp}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Recent Predictions</CardTitle>
              <Button variant="outline" size="sm" onClick={loadPredictions}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </CardHeader>
          <CardContent>
            {predictionsLoading ? (
              <div className="py-8 text-center">
                <Loader2 className="mx-auto h-8 w-8 animate-spin text-muted-foreground" />
                <p className="mt-2 text-muted-foreground">Loading predictions...</p>
              </div>
            ) : predictions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Activity className="mb-4 h-16 w-16 text-muted-foreground" />
                <p className="text-muted-foreground">
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
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {predictions.map((prediction) => {
                    const severity = getSeverityFromPrediction(prediction);
                    return (
                      <TableRow key={prediction.id}>
                        <TableCell className="font-medium">
                          {patientNames[prediction.patient_id] || "Loading..."}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={`${STATUS_COLORS[prediction.status]} text-white`}
                          >
                            {STATUS_LABELS[prediction.status]}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {severity ? (
                            <Badge
                              variant="outline"
                              className={`${SEVERITY_COLORS[severity]} text-white`}
                            >
                              {SEVERITY_LABELS[severity]}
                            </Badge>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </TableCell>
                        <TableCell>{getConfidenceDisplay(prediction)}</TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {new Date(prediction.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => router.push(`/dashboard/predictions/${prediction.id}/gradcam`)}
                            disabled={prediction.status !== "success"}
                          >
                            <Eye className="mr-2 h-4 w-4" />
                            View
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
        </motion.div>

        {/* Prediction Detail Dialog */}
        <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
          <DialogContent className="max-w-3xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                {selectedPrediction?.status === "success" ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : selectedPrediction?.status === "failed" ? (
                  <AlertCircle className="h-5 w-5 text-red-500" />
                ) : (
                  <Loader2 className="h-5 w-5 animate-spin" />
                )}
                Prediction Details
              </DialogTitle>
              <DialogDescription>
                {selectedPrediction && (
                  <>
                    Patient: {patientNames[selectedPrediction.patient_id] || "Unknown"} | 
                    Date: {new Date(selectedPrediction.created_at).toLocaleString()}
                  </>
                )}
              </DialogDescription>
            </DialogHeader>
            {selectedPrediction && (
              <div className="space-y-6">
                {/* Status Badge */}
                <div className="flex items-center gap-4">
                  <span className="text-sm font-medium">Status:</span>
                  <Badge
                    variant="outline"
                    className={`${STATUS_COLORS[selectedPrediction.status]} text-white`}
                  >
                    {STATUS_LABELS[selectedPrediction.status]}
                  </Badge>
                </div>

                {/* Prediction Results */}
                {selectedPrediction.status === "success" && (
                  <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Diabetic Retinopathy Severity</CardTitle>
                      </CardHeader>
                      <CardContent>
                        {getSeverityFromPrediction(selectedPrediction) ? (
                          <div className="space-y-2">
                            <Badge
                              variant="outline"
                              className={`${
                                SEVERITY_COLORS[getSeverityFromPrediction(selectedPrediction)!]
                              } text-white text-lg px-4 py-1`}
                            >
                              {SEVERITY_LABELS[getSeverityFromPrediction(selectedPrediction)!]}
                            </Badge>
                            <p className="text-sm text-muted-foreground">
                              Confidence: {getConfidenceDisplay(selectedPrediction)}
                            </p>
                          </div>
                        ) : (
                          <p className="text-muted-foreground">No severity data available</p>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Model Information</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-1">
                        <p className="text-sm">
                          <span className="text-muted-foreground">Model:</span>{" "}
                          {selectedPrediction.model_name}
                        </p>
                        <p className="text-sm">
                          <span className="text-muted-foreground">Version:</span>{" "}
                          {selectedPrediction.model_version}
                        </p>
                      </CardContent>
                    </Card>

                    {/* Generate Report Button */}
                    <div className="flex justify-end">
                      <Button
                        onClick={handleGenerateReport}
                        disabled={generatingReport}
                        className="mt-4"
                      >
                        {generatingReport ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <FileText className="mr-2 h-4 w-4" />
                            Generate Report
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {selectedPrediction.status === "failed" && selectedPrediction.error_message && (
                  <Card className="border-destructive">
                    <CardHeader>
                      <CardTitle className="text-sm text-destructive">Error</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-destructive">{selectedPrediction.error_message}</p>
                    </CardContent>
                  </Card>
                )}

                {/* Technical Details */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Technical Details</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <p>
                        <span className="text-muted-foreground">Prediction ID:</span>{" "}
                        <code className="rounded bg-muted px-1 py-0.5">{selectedPrediction.id}</code>
                      </p>
                      <p>
                        <span className="text-muted-foreground">Scan ID:</span>{" "}
                        <code className="rounded bg-muted px-1 py-0.5">{selectedPrediction.mri_scan_id}</code>
                      </p>
                      {selectedPrediction.output_payload && (
                        <details>
                          <summary className="cursor-pointer text-muted-foreground">
                            View Output Payload
                          </summary>
                          <pre className="mt-2 max-h-48 overflow-auto rounded bg-muted p-2 text-xs">
                            {JSON.stringify(selectedPrediction.output_payload, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </motion.div>
    </PageContainer>
  );
}
