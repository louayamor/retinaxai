'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import { getPrediction, getPatient, listPatientPredictions, getScan } from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  ArrowLeft,
  ArrowRight,
  Eye,
  ImageIcon,
  Loader,
  Download,
  User,
  Calendar,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  ArrowRightLeft,
  ExternalLink,
  Info,
  ZoomIn,
  ZoomOut,
  X,
} from 'lucide-react';
import type { Prediction, Patient } from '@/types';

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
const GRADE_COLORS: Record<string, string> = {
  'No DR': 'bg-emerald-500',
  'Mild': 'bg-cyan-500',
  'Moderate': 'bg-amber-500',
  'Severe': 'bg-orange-500',
  'Proliferative': 'bg-rose-500',
};

const GRADE_SEVERITY: Record<string, { level: string; color: string; bg: string }> = {
  '0': { level: 'None', color: 'text-emerald-500', bg: 'bg-emerald-500' },
  '1': { level: 'Mild', color: 'text-cyan-500', bg: 'bg-cyan-500' },
  '2': { level: 'Moderate', color: 'text-amber-500', bg: 'bg-amber-500' },
  '3': { level: 'Severe', color: 'text-orange-500', bg: 'bg-orange-500' },
  '4': { level: 'Proliferative', color: 'text-rose-500', bg: 'bg-rose-500' },
};

const CLINICAL_RECOMMENDATIONS: Record<string, { title: string; description: string; urgency: 'low' | 'medium' | 'high' | 'critical' }> = {
  '0': {
    title: 'No Diabetic Retinopathy',
    description: 'No signs of DR detected. Continue routine annual eye exams.',
    urgency: 'low',
  },
  '1': {
    title: 'Mild Non-Proliferative DR',
    description: 'Recommend annual follow-up. Monitor for progression.',
    urgency: 'medium',
  },
  '2': {
    title: 'Moderate Non-Proliferative DR',
    description: 'Referral to ophthalmologist within 3 months. Consider treatment evaluation.',
    urgency: 'medium',
  },
  '3': {
    title: 'Severe Non-Proliferative DR',
    description: 'Urgent referral to ophthalmologist within weeks. High risk of progression.',
    urgency: 'high',
  },
  '4': {
    title: 'Proliferative DR',
    description: 'Immediate ophthalmology consultation. Risk of vision loss without treatment.',
    urgency: 'critical',
  },
};

const GRADCAM_INTERPRETATIONS: Record<string, string> = {
  '0': 'No significant abnormal regions detected. The model focused on normal retinal structures.',
  '1': 'Few microaneurysms detected in peripheral retina. Early signs of diabetic changes.',
  '2': 'Multiple microaneurysms, hemorrhages, and cotton wool spots visible. Moderate DR indicators.',
  '3': 'Extensive hemorrhages, venous beading, and IRMA detected. High-risk DR features present.',
  '4': 'Neovascularization visible on optic disc and elsewhere. Advanced proliferative changes.',
};

interface ImageProps {
  title: string;
  gradcamBase64?: string;
  originalBase64?: string;
  fundusScore?: number;
}

function ZoomableImage({ src, alt, isActive }: { src?: string; alt: string; isActive?: boolean }) {
  const [zoom, setZoom] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  if (!src) {
    return (
      <div className="flex flex-col items-center justify-center aspect-square bg-black rounded-lg">
        <ImageIcon className="h-16 w-16 text-slate-600" />
        <p className="mt-2 text-sm text-slate-400">No image available</p>
      </div>
    );
  }

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-black/95 flex items-center justify-center p-8' : ''}`}>
      <div 
        className="relative aspect-square bg-black rounded-lg overflow-hidden cursor-zoom-in"
        style={{ 
          transform: `scale(${zoom})`,
          transition: 'transform 0.2s ease'
        }}
        onClick={() => {
          if (zoom === 1) setZoom(1.5);
          else if (zoom === 1.5) setZoom(2);
          else setZoom(1);
        }}
      >
        <img
          src={`data:image/png;base64,${src}`}
          alt={alt}
          className="w-full h-full object-contain"
        />
      </div>
      
      {/* Zoom Controls */}
      {isActive && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2 bg-black/70 rounded-lg p-2">
          <Button variant="ghost" size="sm" onClick={() => setZoom(Math.max(0.5, zoom - 0.5))}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="flex items-center text-sm text-white px-2">
            {Math.round(zoom * 100)}%
          </span>
          <Button variant="ghost" size="sm" onClick={() => setZoom(Math.min(3, zoom + 0.5))}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          {isFullscreen && (
            <Button variant="ghost" size="sm" onClick={() => setIsFullscreen(false)}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

function ImageCard({ title, gradcamBase64, originalBase64, fundusScore }: ImageProps) {
  const hasGradCAM = !!gradcamBase64;
  const isFundus = fundusScore !== undefined && fundusScore >= 0.3;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-slate-900/50">
        <CardTitle className="text-lg text-white">{title}</CardTitle>
        {fundusScore !== undefined && (
          <div className={`text-xs mt-1 ${isFundus ? 'text-green-400' : 'text-red-400'}`}>
            {isFundus ? '\u2705 Valid fundus' : '\u274c Rejected'}
            <span className="ml-1 text-slate-300">({fundusScore.toFixed(3)})</span>
          </div>
        )}
      </CardHeader>
      <CardContent className="p-0 space-y-2">
        {/* GradCAM Heatmap - always available when prediction has gradcam */}
        {hasGradCAM && (
          <div>
            <div className="flex items-center justify-between px-3 py-1 bg-slate-800/50">
              <span className="text-xs text-slate-400">AI Heatmap (GradCAM)</span>
              <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => {
                const link = document.createElement('a');
                link.href = `data:image/png;base64,${gradcamBase64}`;
                link.download = `${title.toLowerCase().replace(/\s+/g, '_')}_gradcam.png`;
                link.click();
              }}>
                <Download className="h-3 w-3 mr-1" />
                Download
              </Button>
            </div>
            <ZoomableImage src={gradcamBase64} alt={`${title} - GradCAM`} isActive={true} />
          </div>
        )}
        
        {/* No GradCAM available */}
        {!hasGradCAM && (
          <div className="flex flex-col items-center justify-center aspect-square bg-black rounded-lg">
            <ImageIcon className="h-16 w-16 text-slate-600" />
            <p className="mt-2 text-sm text-slate-400">No GradCAM available</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProbabilityBars({ probabilities, title }: { probabilities?: Record<string, number>; title: string }) {
  if (!probabilities) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {Object.entries(probabilities).map(([label, prob]) => {
          const percentage = (prob as number) * 100;
          const labelKey = label.trim();
          const barColor = GRADE_COLORS[labelKey] || 'bg-slate-500';
          
          return (
            <div key={label} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="font-medium">{label}</span>
                <span className="text-muted-foreground">{percentage.toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                  className={`h-full ${barColor}`}
                />
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function RecommendationPanel({ grade }: { grade: number }) {
  const gradeKey = String(grade);
  const rec = CLINICAL_RECOMMENDATIONS[gradeKey];
  const urgencyColors = {
    low: 'bg-emerald-500/20 border-emerald-500 text-emerald-400',
    medium: 'bg-cyan-500/20 border-cyan-500 text-cyan-400',
    high: 'bg-orange-500/20 border-orange-500 text-orange-400',
    critical: 'bg-rose-500/20 border-rose-500 text-rose-400',
  };

  const urgencyIcons = {
    low: CheckCircle,
    medium: Clock,
    high: AlertCircle,
    critical: AlertCircle,
  };
  const UrgencyIcon = urgencyIcons[rec.urgency];

  return (
    <Card className={`border-l-4 ${urgencyColors[rec.urgency]}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <UrgencyIcon className="h-5 w-5" />
          Clinical Recommendation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h4 className="font-semibold text-lg">{rec.title}</h4>
          <p className="text-muted-foreground mt-1">{rec.description}</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge className={urgencyColors[rec.urgency]}>
            {rec.urgency.toUpperCase()} PRIORITY
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

export default function GradCAMPage() {
  const params = useParams();
  const router = useRouter();
  const shouldReduceMotion = useReducedMotion();

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [patient, setPatient] = useState<Patient | null>(null);
  const [patientPredictions, setPatientPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [originalImages, setOriginalImages] = useState<{ left?: string; right?: string }>({});

  useEffect(() => {
    if (params.id) loadData();
  }, [params.id]);

  const loadData = async () => {
    try {
      setLoading(true);
      const pred = await getPrediction(params.id as string);
      setPrediction(pred);

      const patientData = await getPatient(pred.patient_id);
      setPatient(patientData);

      // Fetch original images from MRI scan
      if (pred.mri_scan_id) {
        try {
          const scan = await getScan(pred.mri_scan_id);
          setOriginalImages({
            left: scan.left_image,
            right: scan.right_image,
          });
        } catch (scanErr) {
          console.error('Failed to load scan images:', scanErr);
        }
      }

      const history = await listPatientPredictions(pred.patient_id, 1, 50);
      setPatientPredictions(history.items.filter(p => p.id !== pred.id));
    } catch (err) {
      console.error('Failed to load prediction:', err);
    } finally {
      setLoading(false);
    }
  };

  const gradcamLeft = prediction?.output_payload?.gradcam_left as string | undefined;
  const gradcamRight = prediction?.output_payload?.gradcam_right as string | undefined;

  const fundusLeft = prediction?.output_payload?.fundus_score_left as number | undefined;
  const fundusRight = prediction?.output_payload?.fundus_score_right as number | undefined;

  const leftEye = prediction?.output_payload?.left_eye as Record<string, unknown> | undefined;
  const rightEye = prediction?.output_payload?.right_eye as Record<string, unknown> | undefined;

  const grade = prediction?.output_payload?.combined_grade as number | undefined;
  const gradeKey = grade !== undefined ? String(grade) : '2';
  const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : 'Unknown';
  const gradeInfo = GRADE_SEVERITY[gradeKey];

  const interpretation = GRADCAM_INTERPRETATIONS[gradeKey] || '';

  if (loading) {
    return (
      <PageContainer>
        <div className="flex h-96 items-center justify-center">
          <Loader className="h-8 w-8 animate-spin text-[var(--brand-teal)]" />
        </div>
      </PageContainer>
    );
  }

  if (!prediction) {
    return (
      <PageContainer>
        <div className="flex flex-col items-center justify-center py-12">
          <p className="text-muted-foreground">Prediction not found</p>
          <Button variant="outline" className="mt-4" onClick={() => router.back()}>
            Back to Diagnostics
          </Button>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <motion.div
        variants={shouldReduceMotion ? {} : { hidden: { opacity: 0 }, visible: { opacity: 1 } }}
        initial="hidden"
        animate="visible"
        className="flex flex-col gap-6"
      >
        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <Button variant="ghost" onClick={() => router.back()}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Diagnostics
          </Button>
          <div className="flex items-center gap-3">
            {patient && (
              <Button variant="outline" onClick={() => router.push(`/dashboard/patients/${patient.id}`)}>
                <User className="mr-2 h-4 w-4" />
                {patient.first_name} {patient.last_name}
                <ExternalLink className="ml-2 h-3 w-3" />
              </Button>
            )}
          </div>
        </div>

        {/* Patient Info Bar */}
        <Card className="bg-gradient-to-r from-slate-900 to-slate-800 border-slate-700">
          <CardContent className="py-4">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-6 text-white">
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-slate-400" />
                  <span className="font-medium">{patient?.first_name} {patient?.last_name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-slate-400" />
                  <span>{new Date(prediction.created_at).toLocaleDateString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-slate-400" />
                  <span>Model: {prediction.model_name}</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-slate-400">Severity:</span>
                <Badge className={`${gradeInfo.bg} text-white text-lg px-3 py-1`}>
                  {gradeLabel}
                </Badge>
                <span className="text-slate-400">
                  {(prediction.confidence_score != null) 
                    ? `${(prediction.confidence_score * 100).toFixed(1)}% confidence`
                    : ''}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - GradCAM Images */}
          <div className="lg:col-span-2 space-y-6">
            {/* Image Grid */}
            <div className="grid md:grid-cols-2 gap-6">
              <ImageCard
                title="Left Eye (OS)"
                gradcamBase64={gradcamLeft}
                originalBase64={originalImages.left}
                fundusScore={fundusLeft}
              />
              <ImageCard
                title="Right Eye (OD)"
                gradcamBase64={gradcamRight}
                originalBase64={originalImages.right}
                fundusScore={fundusRight}
              />
            </div>

            {/* Clinical Interpretation */}
            <Card className="border-l-4 border-[var(--brand-teal)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  AI Interpretation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{interpretation}</p>
                <p className="text-sm text-muted-foreground mt-4">
                  The highlighted regions indicate areas the AI model focused on when making this prediction. 
                  Warmer colors (red/yellow) indicate regions with higher activation.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Details */}
          <div className="space-y-6">
            {/* Recommendations */}
            {grade !== undefined && <RecommendationPanel grade={grade} />}

            {/* Left Eye Probabilities */}
            <ProbabilityBars
              probabilities={leftEye?.probabilities as Record<string, number> | undefined}
              title="Left Eye (OS) Probabilities"
            />

            {/* Right Eye Probabilities */}
            <ProbabilityBars
              probabilities={rightEye?.probabilities as Record<string, number> | undefined}
              title="Right Eye (OD) Probabilities"
            />

            {/* Patient History Timeline */}
            {patientPredictions.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Patient DR History</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {patientPredictions.slice(0, 5).map((p) => {
                    const pGrade = p.output_payload?.combined_grade as number | undefined;
                    const pLabel = pGrade !== undefined ? GRADE_LABELS[pGrade] : 'Unknown';
                    return (
                      <div key={p.id} className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">
                          {new Date(p.created_at).toLocaleDateString()}
                        </span>
                        <Badge className={pGrade !== undefined ? GRADE_COLORS[pGrade] : 'bg-muted'}>
                          {pLabel}
                        </Badge>
                      </div>
                    );
                  })}
                  {patientPredictions.length > 5 && (
                    <p className="text-xs text-muted-foreground text-center">
                      +{patientPredictions.length - 5} more historical predictions
                    </p>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex justify-center pt-4 border-t">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              onClick={() => router.push('/dashboard/predictions?tab=gradcam')}
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to GradCAM List
            </Button>
            {patient && (
              <Button
                variant="default"
                onClick={() => router.push(`/dashboard/patients/${patient.id}`)}
              >
                <User className="mr-2 h-4 w-4" />
                View Patient Profile
              </Button>
            )}
          </div>
        </div>
      </motion.div>
    </PageContainer>
  );
}