'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import { listAllPredictions, getPatient } from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ImageIcon, Eye, RefreshCw, Loader } from 'lucide-react';
import type { Prediction, Patient } from '@/types';

const GRADE_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
const GRADE_COLORS: Record<string, string> = {
  0: 'bg-emerald-500',
  1: 'bg-cyan-500',
  2: 'bg-amber-500',
  3: 'bg-orange-500',
  4: 'bg-rose-500',
};

export default function GradCAMListPage() {
  const router = useRouter();
  const shouldReduceMotion = useReducedMotion();

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [patientNames, setPatientNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      setLoading(true);
      const response = await listAllPredictions(1, 100);
      
      const withGradCAM = response.items.filter(
        (p) => p.output_payload?.gradcam_left || p.output_payload?.gradcam_right
      );
      setPredictions(withGradCAM);

      const patientIds = [...new Set(withGradCAM.map((p) => p.patient_id))];
      const names: Record<string, string> = {};
      for (const pid of patientIds) {
        try {
          const patient = await getPatient(pid);
          names[pid] = `${patient.first_name} ${patient.last_name}`;
        } catch {
          names[pid] = 'Unknown';
        }
      }
      setPatientNames(names);
    } catch (err) {
      console.error('Failed to load predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <PageContainer>
        <div className="flex h-96 items-center justify-center">
          <Loader className="h-8 w-8 animate-spin text-[var(--brand-teal)]" />
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
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">GradCAM Visualizations</h1>
          <Button variant="outline" size="sm" onClick={loadPredictions}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>

        {/* Description */}
        <Card>
          <CardContent className="py-4">
            <p className="text-muted-foreground">
              View AI explainability heatmaps for predictions. GradCAM highlights which regions
              of the retinal image influenced the diabetic retinopathy classification.
            </p>
          </CardContent>
        </Card>

        {/* Predictions with GradCAM */}
        {predictions.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <ImageIcon className="h-12 w-12 text-muted-foreground mb-3" />
              <p className="text-muted-foreground font-medium">
                No GradCAM Visualizations Available
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Run predictions with GradCAM enabled to see heatmaps here.
              </p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => router.push('/dashboard/predictions')}
              >
                Go to DR Screening
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {predictions.map((prediction) => {
              const grade = prediction.output_payload?.combined_grade as number | undefined;
              const gradeLabel = grade !== undefined ? GRADE_LABELS[grade] : null;
              const hasGradCAM = prediction.output_payload?.gradcam_left || prediction.output_payload?.gradcam_right;

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

                    {/* Mini GradCAM preview */}
                    <div className="grid grid-cols-2 gap-2 mb-3">
                      {prediction.output_payload?.gradcam_left ? (
                        <div className="relative aspect-square rounded-lg overflow-hidden bg-black">
                          <img
                            src={`data:image/png;base64,${prediction.output_payload.gradcam_left}`}
                            alt="Left eye"
                            className="w-full h-full object-cover"
                          />
                          <span className="absolute bottom-1 left-1 text-xs text-white bg-black/50 px-1 rounded">
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
                          <span className="absolute bottom-1 right-1 text-xs text-white bg-black/50 px-1 rounded">
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
        )}
      </motion.div>
    </PageContainer>
  );
}
