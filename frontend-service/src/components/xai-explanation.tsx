'use client';

import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Brain, 
  Activity, 
  AlertTriangle, 
  TrendingUp, 
  Eye,
  Heart,
  Clock,
  Shield,
  Info
} from 'lucide-react';
import { motion } from 'framer-motion';

interface DiagnosisData {
  condition?: string;
  severity?: string;
  overall_grade?: number | string;
  confidence?: number;
  risk_level?: string;
}

interface ClinicalFindingsEye {
  grade?: number | string;
  severity?: string;
  confidence?: number;
  description?: string;
}

interface ClinicalFindings {
  left_eye?: ClinicalFindingsEye;
  right_eye?: ClinicalFindingsEye;
}

interface FeatureImportance {
  top_contributors?: Array<{ feature_name: string; contribution: number }>;
  key_insights?: string[];
}

interface ClinicalContext {
  risk_factors?: string[];
  visual_indicators?: string[];
  recommendations?: string[];
}

interface XAIExplanationData {
  diagnosis?: DiagnosisData;
  clinical_findings?: ClinicalFindings;
  feature_importance?: FeatureImportance;
  clinical_context?: ClinicalContext;
  summary?: string;
}

interface SeverityData {
  patient?: { name?: string; age?: number | string; gender?: string };
  diagnosis?: { dr_grade?: number; severity_label?: string; risk_level?: string };
  clinical_assessment?: { findings?: string; visual_indicators?: string[] };
  risk_factors?: string[];
  risk_stratification?: { overall_risk?: string; progression_risk?: string; vision_loss_risk?: string };
  recommendations?: Array<{ action: string; timeframe?: string; rationale?: string }> | string[];
  follow_up?: { next_appointment?: number; frequency?: number; tests_required?: string[] };
  summary?: string;
  risk_level?: string;
}

interface XAIExplanationProps {
  explanation?: string | XAIExplanationData | null;
  severityReport?: string | SeverityData | null;
  shapValues?: {
    top_positive?: Array<{ name: string; contribution: number }>;
    top_negative?: Array<{ name: string; contribution: number }>;
  } | null;
}

const RISK_COLORS: Record<string, string> = {
  low: 'bg-emerald-500',
  moderate: 'bg-amber-500',
  high: 'bg-orange-500',
  severe: 'bg-red-600',
};

const GRADE_LABELS: Record<number, string> = {
  0: 'No DR',
  1: 'Mild',
  2: 'Moderate',
  3: 'Severe',
  4: 'Proliferative DR',
};

const GRADE_COLORS: Record<number, string> = {
  0: 'bg-emerald-500',
  1: 'bg-cyan-500',
  2: 'bg-amber-500',
  3: 'bg-orange-500',
  4: 'bg-rose-500',
};

export default function XAIExplanation({ 
  explanation, 
  severityReport, 
  shapValues 
}: XAIExplanationProps) {
  // Parse explanation if it's a string
  let parsedExplanation: XAIExplanationData | null = null;
  if (explanation && typeof explanation === 'string') {
    try {
      parsedExplanation = JSON.parse(explanation);
    } catch {
      // If parsing fails, treat as plain text
      parsedExplanation = { summary: explanation };
    }
  } else if (explanation && typeof explanation === 'object') {
    parsedExplanation = explanation as XAIExplanationData;
  }

  // Parse severity if it's a string
  let parsedSeverity: SeverityData | null = null;
  if (severityReport && typeof severityReport === 'string') {
    try {
      parsedSeverity = JSON.parse(severityReport);
    } catch {
      parsedSeverity = { summary: severityReport };
    }
  } else if (severityReport && typeof severityReport === 'object') {
    parsedSeverity = severityReport as SeverityData;
  }

  const diagnosis = parsedExplanation?.diagnosis;
  const clinicalFindings = parsedExplanation?.clinical_findings;
  const featureImportance = parsedExplanation?.feature_importance;
  const clinicalContext = parsedExplanation?.clinical_context;
  const expSummary = parsedExplanation?.summary;

  const getRiskBadge = (risk?: string) => {
    const level = risk?.toLowerCase() || 'moderate';
    const color = RISK_COLORS[level] || 'bg-gray-500';
    return <Badge className={`${color} text-white`}>{risk || 'Unknown'} Risk</Badge>;
  };

  return (
    <div className="space-y-4">
      {/* Diagnosis Section */}
      {diagnosis && (
        <Card className="overflow-hidden border-l-4 border-l-blue-500">
          <CardHeader className="pb-2 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-blue-600" />
                <span className="font-semibold text-blue-900 dark:text-blue-100">AI Diagnosis</span>
              </div>
              {getRiskBadge(diagnosis.risk_level)}
            </div>
          </CardHeader>
          <CardContent className="pt-4 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground mb-1">Condition</p>
                <p className="font-semibold text-sm">{diagnosis.condition || 'Diabetic Retinopathy'}</p>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground mb-1">Severity</p>
                <p className="font-semibold text-sm">{diagnosis.severity || 'N/A'}</p>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground mb-1">Grade</p>
                <Badge className={GRADE_COLORS[Number(diagnosis.overall_grade) || 0] || 'bg-gray-500'}>
                  {GRADE_LABELS[Number(diagnosis.overall_grade) as keyof typeof GRADE_LABELS] || diagnosis.overall_grade || 'N/A'}
                </Badge>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground mb-1">Confidence</p>
                <p className="font-semibold text-sm">
                  {diagnosis.confidence ? `${(diagnosis.confidence * 100).toFixed(1)}%` : 'N/A'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Clinical Findings - Left/Right Eye */}
      {clinicalFindings && (clinicalFindings.left_eye || clinicalFindings.right_eye) && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Eye className="h-5 w-5 text-purple-600" />
              <span className="font-semibold">Clinical Findings</span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              {clinicalFindings.left_eye && (
                <div className="border rounded-lg p-4 bg-blue-50/50 dark:bg-blue-950/30">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100">Left Eye (OS)</h4>
                    <Badge className={GRADE_COLORS[Number(clinicalFindings.left_eye.grade) as keyof typeof GRADE_COLORS] || 'bg-gray-500'}>
                      {GRADE_LABELS[Number(clinicalFindings.left_eye.grade) as keyof typeof GRADE_LABELS] || clinicalFindings.left_eye.grade || 'N/A'}
                    </Badge>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Severity:</span>
                      <span className="font-medium capitalize">{clinicalFindings.left_eye.severity || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {clinicalFindings.left_eye.confidence ? `${(clinicalFindings.left_eye.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    {clinicalFindings.left_eye.description && (
                      <p className="text-xs text-muted-foreground mt-2 pt-2 border-t">
                        {clinicalFindings.left_eye.description}
                      </p>
                    )}
                  </div>
                </div>
              )}
              {clinicalFindings.right_eye && (
                <div className="border rounded-lg p-4 bg-purple-50/50 dark:bg-purple-950/30">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-purple-900 dark:text-purple-100">Right Eye (OD)</h4>
                    <Badge className={GRADE_COLORS[Number(clinicalFindings.right_eye.grade) as keyof typeof GRADE_COLORS] || 'bg-gray-500'}>
                      {GRADE_LABELS[Number(clinicalFindings.right_eye.grade) as keyof typeof GRADE_LABELS] || clinicalFindings.right_eye.grade || 'N/A'}
                    </Badge>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Severity:</span>
                      <span className="font-medium capitalize">{clinicalFindings.right_eye.severity || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {clinicalFindings.right_eye.confidence ? `${(clinicalFindings.right_eye.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    {clinicalFindings.right_eye.description && (
                      <p className="text-xs text-muted-foreground mt-2 pt-2 border-t">
                        {clinicalFindings.right_eye.description}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* SHAP / Feature Importance */}
      {(shapValues?.top_positive?.length || featureImportance?.top_contributors?.length) && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-emerald-600" />
              <span className="font-semibold">Top Contributing Features</span>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {(shapValues?.top_positive || featureImportance?.top_contributors)?.slice(0, 5).map((feature, i) => {
              const featureAny = feature as unknown as { name?: string; feature_name?: string; contribution: number };
              const name = featureAny.name || featureAny.feature_name || 'Unknown';
              const contribution = featureAny.contribution;
              return (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-sm w-32 truncate font-medium">{name}</span>
                  <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(Math.abs(contribution) * 100, 100)}%` }}
                      transition={{ delay: i * 0.1, duration: 0.5 }}
                      className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600"
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-16 text-right">
                    {contribution.toFixed(3)}
                  </span>
                </div>
              );
            })}
            {featureImportance?.key_insights && featureImportance.key_insights.length > 0 && (
              <div className="mt-4 pt-3 border-t">
                <p className="text-xs text-muted-foreground mb-2">Key Insights:</p>
                <ul className="text-sm space-y-1">
                  {featureImportance.key_insights.slice(0, 3).map((insight, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <Info className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Clinical Context / Recommendations */}
      {(clinicalContext?.recommendations?.length || parsedSeverity?.recommendations?.length) && (
        <Card className="border-amber-200 dark:border-amber-800">
          <CardHeader className="pb-2 bg-amber-50 dark:bg-amber-950/30">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-amber-600" />
              <span className="font-semibold text-amber-900 dark:text-amber-100">Recommendations</span>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {(clinicalContext?.recommendations || parsedSeverity?.recommendations)?.map((rec, i) => {
              const action = typeof rec === 'string' ? rec : rec.action;
              const timeframe = typeof rec === 'object' ? rec.timeframe : null;
              return (
                <div key={i} className="flex items-start gap-3 p-3 bg-amber-50/50 dark:bg-amber-950/20 rounded-lg">
                  <div className="bg-amber-500 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                    {i + 1}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm">{action}</p>
                    {timeframe && (
                      <Badge variant="outline" className="mt-1 text-xs">
                        {timeframe}
                      </Badge>
                    )}
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      {/* Summary */}
      {(expSummary || parsedSeverity?.summary) && (
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-blue-200 dark:border-blue-800">
          <CardContent className="pt-4">
            <div className="flex items-start gap-3">
              <Activity className="h-5 w-5 text-blue-600 mt-0.5 shrink-0" />
              <div>
                <p className="font-semibold text-sm text-blue-900 dark:text-blue-100 mb-1">Summary</p>
                <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
                  {expSummary || parsedSeverity?.summary}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Risk Factors */}
      {(clinicalContext?.risk_factors?.length || parsedSeverity?.risk_factors?.length) && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Heart className="h-5 w-5 text-red-500" />
              <span className="font-semibold">Risk Factors</span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {(clinicalContext?.risk_factors || parsedSeverity?.risk_factors)?.map((factor, i) => (
                <Badge key={i} variant="outline" className="border-red-200 text-red-700 dark:border-red-800 dark:text-red-400">
                  {factor}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Follow-up (from severity) */}
      {parsedSeverity?.follow_up && (
        <Card className="bg-muted/30">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-blue-500" />
              <span className="font-semibold">Follow-up Schedule</span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xs text-muted-foreground">Next Appointment</p>
                <p className="font-semibold">{parsedSeverity.follow_up.next_appointment || 'N/A'} weeks</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Frequency</p>
                <p className="font-semibold">{parsedSeverity.follow_up.frequency || 'N/A'} months</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Tests Required</p>
                <p className="font-semibold text-sm">
                  {parsedSeverity.follow_up.tests_required?.join(', ') || 'None'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Risk Stratification (from severity) */}
      {parsedSeverity?.risk_stratification && (
        <div className="grid md:grid-cols-3 gap-3">
          {[
            { label: 'Overall Risk', value: parsedSeverity.risk_stratification.overall_risk, icon: AlertTriangle },
            { label: 'Progression Risk', value: parsedSeverity.risk_stratification.progression_risk, icon: TrendingUp },
            { label: 'Vision Loss Risk', value: parsedSeverity.risk_stratification.vision_loss_risk, icon: Eye },
          ].map((item, i) => {
            const Icon = item.icon;
            const riskLevel = item.value?.toLowerCase() || 'moderate';
            const bgColor = riskLevel === 'high' || riskLevel === 'severe' ? 'bg-red-100 dark:bg-red-950/30 border-red-200' :
                           riskLevel === 'moderate' ? 'bg-amber-100 dark:bg-amber-950/30 border-amber-200' :
                           'bg-emerald-100 dark:bg-emerald-950/30 border-emerald-200';
            return (
              <div key={i} className={`p-3 rounded-lg border ${bgColor} text-center`}>
                <Icon className="h-4 w-4 mx-auto mb-1 text-muted-foreground" />
                <p className="text-xs text-muted-foreground">{item.label}</p>
                <p className="font-semibold capitalize">{item.value || 'N/A'}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}