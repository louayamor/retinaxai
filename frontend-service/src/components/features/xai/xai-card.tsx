'use client';

import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Brain, 
  Eye, 
  AlertTriangle, 
  CheckCircle, 
  Activity,
  TrendingUp,
  Heart,
  Calendar,
  Stethoscope,
  Scan
} from 'lucide-react';
import { motion } from 'framer-motion';

interface XAIExplanationsData {
  diagnosis?: {
    condition?: string;
    severity?: string;
    overall_grade?: number | string;
    confidence?: number;
    risk_level?: string;
  };
  clinical_findings?: {
    left_eye?: {
      grade?: string | number;
      severity?: string;
      confidence?: number;
      description?: string;
    };
    right_eye?: {
      grade?: string | number;
      severity?: string;
      confidence?: number;
      description?: string;
    };
  };
  severity_report?: {
    patient?: { name?: string; age?: string | number; gender?: string };
    diagnosis?: { dr_grade?: number; severity_label?: string; risk_level?: string };
    risk_stratification?: { overall_risk?: string; progression_risk?: string; vision_loss_risk?: string };
    recommendations?: Array<{ action?: string; timeframe?: string }>;
    summary?: string | null;
  };
  gradcam_explanation?: {
    left_eye_explanation?: string;
    right_eye_explanation?: string;
    highlighted_regions?: { left_eye?: string[]; right_eye?: string[] };
  };
  feature_importance?: {
    top_contributors?: Array<{ feature_name?: string; contribution?: number }>;
    top_positive?: Array<{ name?: string; contribution?: number }>;
  };
  summary?: string;
}

interface XAICardProps {
  predictionId: string;
  createdAt: string;
  data: XAIExplanationsData;
}

const GRADE_LABELS: Record<string, string> = {
  '0': 'No DR',
  '1': 'Mild',
  '2': 'Moderate',
  '3': 'Severe',
  '4': 'Proliferative DR',
  'No DR': 'No DR',
  'Mild': 'Mild',
  'Moderate': 'Moderate',
  'Severe': 'Severe',
  'Proliferative DR': 'Proliferative DR',
};

const GRADE_COLORS: Record<string, string> = {
  '0': 'bg-emerald-500',
  '1': 'bg-cyan-500',
  '2': 'bg-amber-500',
  '3': 'bg-orange-500',
  '4': 'bg-rose-500',
  'No DR': 'bg-emerald-500',
  'Mild': 'bg-cyan-500',
  'Moderate': 'bg-amber-500',
  'Severe': 'bg-orange-500',
  'Proliferative DR': 'bg-rose-500',
};

const RISK_COLORS: Record<string, string> = {
  low: 'bg-emerald-500',
  moderate: 'bg-amber-500',
  elevated: 'bg-amber-500',
  high: 'bg-orange-500',
  severe: 'bg-red-600',
  very_high: 'bg-red-600',
};

export default function XAICard({ predictionId, createdAt, data }: XAICardProps) {
  const { diagnosis, clinical_findings, severity_report, gradcam_explanation, feature_importance, summary } = data;

  const getGradeKey = (grade: string | number | undefined) => {
    if (!grade) return 'N/A';
    return GRADE_LABELS[String(grade)] || String(grade);
  };

  const getGradeColor = (grade: string | number | undefined) => {
    if (!grade) return 'bg-gray-500';
    return GRADE_COLORS[String(grade)] || 'bg-gray-500';
  };

  const getRiskColor = (risk?: string) => {
    const lowerRisk = risk?.toLowerCase() || '';
    if (lowerRisk.includes('high') || lowerRisk.includes('severe') || lowerRisk.includes('very')) return 'bg-red-600';
    if (lowerRisk.includes('moderate') || lowerRisk.includes('elevated')) return 'bg-amber-500';
    if (lowerRisk.includes('low')) return 'bg-emerald-500';
    return 'bg-gray-500';
  };

  const drGrade = severity_report?.diagnosis?.dr_grade ?? diagnosis?.overall_grade;
  const riskLevel = severity_report?.diagnosis?.risk_level ?? diagnosis?.risk_level;

  return (
    <Card className="overflow-hidden border-2 border-violet-200 dark:border-violet-800 shadow-lg">
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-6 py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-white/20 p-3 rounded-xl backdrop-blur-sm">
              <Brain className="h-7 w-7 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">XAI Diabetic Retinopathy Explanation</h2>
              <p className="text-violet-100 text-sm flex items-center gap-2">
                <Activity className="h-4 w-4" />
                AI Explainable Analysis
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-1 text-violet-100 text-sm">
              <Calendar className="h-4 w-4" />
              Generated
            </div>
            <p className="font-medium text-white">{new Date(createdAt).toLocaleDateString()}</p>
            <p className="text-xs text-violet-200">{new Date(createdAt).toLocaleTimeString()}</p>
          </div>
        </div>
      </div>

      <CardContent className="p-6 space-y-6">
        {diagnosis && (
          <>
            <div>
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Stethoscope className="h-5 w-5 text-blue-600" />
                Diagnosis Summary
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-muted/50 rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">Condition</p>
                  <p className="font-semibold text-sm">{diagnosis.condition || 'Diabetic Retinopathy'}</p>
                </div>
                <div className="text-center p-4 bg-muted/50 rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">Severity</p>
                  <p className="font-semibold text-sm">{diagnosis.severity || severity_report?.diagnosis?.severity_label || 'N/A'}</p>
                </div>
                <div className="text-center p-4 bg-muted/50 rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">DR Grade</p>
                  <Badge className={getGradeColor(drGrade)}>
                    {getGradeKey(drGrade)}
                  </Badge>
                </div>
                <div className="text-center p-4 bg-muted/50 rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">Risk Level</p>
                  <Badge className={getRiskColor(riskLevel)}>
                    {riskLevel || 'N/A'}
                  </Badge>
                </div>
              </div>
            </div>
            <Separator />
          </>
        )}

        {(clinical_findings?.left_eye || clinical_findings?.right_eye) && (
          <div>
            <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <Eye className="h-5 w-5 text-purple-600" />
              Clinical Findings
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              {clinical_findings?.left_eye && (
                <div className="border-2 border-blue-200 dark:border-blue-800 rounded-lg p-4 bg-blue-50/50 dark:bg-blue-950/30">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100 flex items-center gap-2">
                      <Scan className="h-4 w-4" />
                      Left Eye (OS)
                    </h4>
                    <Badge className={getGradeColor(clinical_findings.left_eye.grade)}>
                      {getGradeKey(clinical_findings.left_eye.grade)}
                    </Badge>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Severity:</span>
                      <span className="font-medium capitalize">{clinical_findings.left_eye.severity || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {clinical_findings.left_eye.confidence ? `${(clinical_findings.left_eye.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    {clinical_findings.left_eye.description && (
                      <p className="text-xs text-muted-foreground mt-2 pt-2 border-t">
                        {clinical_findings.left_eye.description}
                      </p>
                    )}
                  </div>
                </div>
              )}
              {clinical_findings?.right_eye && (
                <div className="border-2 border-purple-200 dark:border-purple-800 rounded-lg p-4 bg-purple-50/50 dark:bg-purple-950/30">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-100 flex items-center gap-2">
                      <Scan className="h-4 w-4" />
                      Right Eye (OD)
                    </h4>
                    <Badge className={getGradeColor(clinical_findings.right_eye.grade)}>
                      {getGradeKey(clinical_findings.right_eye.grade)}
                    </Badge>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Severity:</span>
                      <span className="font-medium capitalize">{clinical_findings.right_eye.severity || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {clinical_findings.right_eye.confidence ? `${(clinical_findings.right_eye.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    {clinical_findings.right_eye.description && (
                      <p className="text-xs text-muted-foreground mt-2 pt-2 border-t">
                        {clinical_findings.right_eye.description}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {(gradcam_explanation?.highlighted_regions?.left_eye?.length || gradcam_explanation?.highlighted_regions?.right_eye?.length) && (
          <>
            <Separator />
            <div>
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-emerald-600" />
                Grad-CAM Anatomical Regions
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                {gradcam_explanation?.highlighted_regions?.left_eye?.length ? (
                  <div className="border-2 border-blue-200 dark:border-blue-800 rounded-lg p-4 bg-blue-50/50 dark:bg-blue-950/30">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-3 flex items-center gap-2">
                      <Eye className="h-4 w-4" />
                      Left Eye (OS)
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {gradcam_explanation.highlighted_regions?.left_eye?.map((region: string, idx: number) => (
                        <Badge key={idx} variant="outline" className="bg-blue-100/50 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200">
                          {region.replace(/_/g, ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ) : null}
                {gradcam_explanation?.highlighted_regions?.right_eye?.length ? (
                  <div className="border-2 border-purple-200 dark:border-purple-800 rounded-lg p-4 bg-purple-50/50 dark:bg-purple-950/30">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-3 flex items-center gap-2">
                      <Eye className="h-4 w-4" />
                      Right Eye (OD)
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {gradcam_explanation.highlighted_regions?.right_eye?.map((region: string, idx: number) => (
                        <Badge key={idx} variant="outline" className="bg-purple-100/50 dark:bg-purple-900/50 text-purple-800 dark:text-purple-200">
                          {region.replace(/_/g, ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </>
        )}

        {severity_report?.recommendations && severity_report.recommendations.length > 0 && (
          <>
            <Separator />
            <div>
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Heart className="h-5 w-5 text-red-600" />
                Recommendations
              </h3>
              <div className="space-y-2">
                {severity_report.recommendations.slice(0, 5).map((rec, idx) => (
                  rec?.action && (
                    <div key={idx} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                      <div className="bg-red-100 dark:bg-red-900/30 p-2 rounded-full">
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-sm">{rec.action}</p>
                        {rec.timeframe && (
                          <p className="text-xs text-muted-foreground">{rec.timeframe}</p>
                        )}
                      </div>
                    </div>
                  )
                ))}
              </div>
            </div>
          </>
        )}

        {(gradcam_explanation?.left_eye_explanation || gradcam_explanation?.right_eye_explanation) && (
          <>
            <Separator />
            <div>
              <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                <Eye className="h-5 w-5 text-violet-600" />
                Detailed Analysis
              </h3>
              <div className="space-y-3">
                {gradcam_explanation?.left_eye_explanation && (
                  <div className="p-4 bg-blue-50/50 dark:bg-blue-950/30 rounded-lg">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Left Eye (OS)</h4>
                    <p className="text-sm text-muted-foreground">{gradcam_explanation.left_eye_explanation}</p>
                  </div>
                )}
                {gradcam_explanation?.right_eye_explanation && (
                  <div className="p-4 bg-purple-50/50 dark:bg-purple-950/30 rounded-lg">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Right Eye (OD)</h4>
                    <p className="text-sm text-muted-foreground">{gradcam_explanation.right_eye_explanation}</p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}