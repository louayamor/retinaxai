'use client';

import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  FileText, 
  User, 
  Eye, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Activity,
  TrendingUp,
  Heart,
  Pill,
  Calendar,
  Monitor
} from 'lucide-react';
import { motion } from 'framer-motion';

interface ReportData {
  patient_info?: {
    name?: string;
    age?: string | number;
    gender?: string;
    mrn?: string;
  } | null;
  clinical_findings?: {
    left_eye?: {
      grade?: string;
      severity?: string;
      confidence?: number;
      description?: string;
    };
    right_eye?: {
      grade?: string;
      severity?: string;
      confidence?: number;
      description?: string;
    };
  } | null;
  diagnosis?: {
    condition?: string;
    severity?: string;
    overall_grade?: string;
    risk_level?: string;
  } | null;
  recommendations?: string[];
  summary?: string;
  report_metadata?: {
    generated_date?: string;
    model?: string;
    model_version?: string;
  } | null;
}

interface MedicalReportProps {
  report: {
    id: string;
    content: string | null;
    summary: string | null;
    llm_model: string;
    created_at: string;
  };
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

const SEVERITY_COLORS: Record<string, string> = {
  low: 'bg-emerald-500',
  moderate: 'bg-amber-500',
  high: 'bg-orange-500',
  severe: 'bg-red-600',
  'very high': 'bg-red-600',
};

export default function MedicalReport({ report }: MedicalReportProps) {
  let reportData: ReportData = {};
  
  try {
    if (report.content) {
      const parsed = JSON.parse(report.content);
      if (typeof parsed === 'object') {
        reportData = parsed;
      }
    }
  } catch {
    reportData = { summary: report.summary ?? undefined };
  }

  const { patient_info, clinical_findings, diagnosis, recommendations, summary, report_metadata } = reportData;

  const getRiskBadgeColor = (risk?: string) => {
    const lowerRisk = risk?.toLowerCase() || '';
    if (lowerRisk.includes('high') || lowerRisk.includes('severe') || lowerRisk.includes('very')) return 'bg-red-600';
    if (lowerRisk.includes('moderate')) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  const getGradeKey = (grade: string | undefined) => {
    if (!grade) return 'Moderate';
    return GRADE_LABELS[grade] || grade;
  };

  return (
    <Card className="overflow-hidden border-2 border-blue-200 dark:border-blue-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-600 px-6 py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-white/20 p-3 rounded-xl backdrop-blur-sm">
              <FileText className="h-7 w-7 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Clinical Diabetic Retinopathy Report</h2>
              <p className="text-blue-100 text-sm flex items-center gap-2">
                <Activity className="h-4 w-4" />
                AI-Generated Medical Assessment
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-1 text-blue-100 text-sm">
              <Calendar className="h-4 w-4" />
              Generated
            </div>
            <p className="font-medium text-white">{new Date(report.created_at).toLocaleDateString()}</p>
            <p className="text-xs text-blue-200">{new Date(report.created_at).toLocaleTimeString()}</p>
          </div>
        </div>
      </div>

      <CardContent className="p-6 space-y-6">
        {/* Patient Information */}
        {patient_info && (
          <Card className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/50 dark:to-gray-900/50 border-slate-200 dark:border-slate-700">
            <CardHeader className="pb-3 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-2">
                <User className="h-5 w-5 text-slate-600 dark:text-slate-400" />
                <span className="font-semibold text-slate-900 dark:text-slate-100">Patient Information</span>
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {patient_info.name && (
                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                    <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                      <User className="h-3 w-3" />
                      Patient Name
                    </p>
                    <p className="font-semibold text-sm">{patient_info.name}</p>
                  </div>
                )}
                {patient_info.age && (
                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                    <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      Age
                    </p>
                    <p className="font-semibold text-sm">{patient_info.age}</p>
                  </div>
                )}
                {patient_info.gender && (
                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                    <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                      <Heart className="h-3 w-3" />
                      Gender
                    </p>
                    <p className="font-semibold text-sm">{patient_info.gender}</p>
                  </div>
                )}
                {patient_info.mrn && (
                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg shadow-sm">
                    <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                      <Monitor className="h-3 w-3" />
                      MRN
                    </p>
                    <p className="font-semibold text-sm font-mono">{patient_info.mrn}</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        <Separator className="my-4" />

        {/* Clinical Findings */}
        {clinical_findings && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Eye className="h-5 w-5 text-purple-600" />
              <h3 className="font-semibold text-lg">Clinical Findings</h3>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              {clinical_findings.left_eye && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="border-2 border-l-4 border-l-blue-500 border-blue-200 dark:border-blue-800 rounded-xl p-5 bg-gradient-to-br from-blue-50/50 to-indigo-50/30 dark:from-blue-950/30 dark:to-indigo-950/20"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-blue-900 dark:text-blue-100">Left Eye (OS)</span>
                      {clinical_findings.left_eye.severity && (
                        <Badge className={SEVERITY_COLORS[clinical_findings.left_eye.severity.toLowerCase()] || 'bg-blue-500'}>
                          {clinical_findings.left_eye.severity}
                        </Badge>
                      )}
                    </div>
                    <Badge className={GRADE_COLORS[getGradeKey(clinical_findings.left_eye.grade)] || 'bg-gray-500'}>
                      {getGradeKey(clinical_findings.left_eye.grade)}
                    </Badge>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-2 bg-white/60 dark:bg-black/20 rounded-lg">
                      <span className="text-sm text-muted-foreground">Confidence</span>
                      <span className="font-semibold text-blue-700 dark:text-blue-300">
                        {clinical_findings.left_eye.confidence 
                          ? `${(clinical_findings.left_eye.confidence * 100).toFixed(1)}%` 
                          : 'N/A'}
                      </span>
                    </div>
                    {clinical_findings.left_eye.description && (
                      <div className="pt-3 border-t border-blue-200 dark:border-blue-800">
                        <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
                          {clinical_findings.left_eye.description}
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
              {clinical_findings.right_eye && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="border-2 border-l-4 border-l-purple-500 border-purple-200 dark:border-purple-800 rounded-xl p-5 bg-gradient-to-br from-purple-50/50 to-pink-50/30 dark:from-purple-950/30 dark:to-pink-950/20"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-purple-900 dark:text-purple-100">Right Eye (OD)</span>
                      {clinical_findings.right_eye.severity && (
                        <Badge className={SEVERITY_COLORS[clinical_findings.right_eye.severity.toLowerCase()] || 'bg-purple-500'}>
                          {clinical_findings.right_eye.severity}
                        </Badge>
                      )}
                    </div>
                    <Badge className={GRADE_COLORS[getGradeKey(clinical_findings.right_eye.grade)] || 'bg-gray-500'}>
                      {getGradeKey(clinical_findings.right_eye.grade)}
                    </Badge>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-2 bg-white/60 dark:bg-black/20 rounded-lg">
                      <span className="text-sm text-muted-foreground">Confidence</span>
                      <span className="font-semibold text-purple-700 dark:text-purple-300">
                        {clinical_findings.right_eye.confidence 
                          ? `${(clinical_findings.right_eye.confidence * 100).toFixed(1)}%` 
                          : 'N/A'}
                      </span>
                    </div>
                    {clinical_findings.right_eye.description && (
                      <div className="pt-3 border-t border-purple-200 dark:border-purple-800">
                        <p className="text-sm text-purple-800 dark:text-purple-200 leading-relaxed">
                          {clinical_findings.right_eye.description}
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        )}

        {/* Diagnosis */}
        {diagnosis && (
          <Card className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-950/30 dark:to-orange-950/30 border-2 border-red-200 dark:border-red-800">
            <CardHeader className="pb-3 border-b border-red-200 dark:border-red-800">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                  <span className="font-semibold text-lg text-red-900 dark:text-red-100">Diagnosis</span>
                </div>
                {diagnosis.risk_level && (
                  <Badge className={`${getRiskBadgeColor(diagnosis.risk_level)} text-white px-3 py-1`}>
                    {diagnosis.risk_level.toUpperCase()} RISK
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-white/70 dark:bg-black/30 rounded-xl border border-red-100 dark:border-red-900/50">
                  <p className="text-xs text-red-600 dark:text-red-400 mb-1">Condition</p>
                  <p className="font-bold text-lg text-red-900 dark:text-red-100">
                    {diagnosis.condition || 'Diabetic Retinopathy'}
                  </p>
                </div>
                <div className="p-4 bg-white/70 dark:bg-black/30 rounded-xl border border-red-100 dark:border-red-900/50">
                  <p className="text-xs text-red-600 dark:text-red-400 mb-1">Severity</p>
                  <p className="font-bold text-lg text-red-900 dark:text-red-100">
                    {diagnosis.severity || 'N/A'}
                  </p>
                </div>
                <div className="p-4 bg-white/70 dark:bg-black/30 rounded-xl border border-red-100 dark:border-red-900/50">
                  <p className="text-xs text-red-600 dark:text-red-400 mb-1">Overall Grade</p>
                  <div className="flex items-center gap-2">
                    <Badge className={GRADE_COLORS[getGradeKey(diagnosis.overall_grade)] || 'bg-gray-500'}>
                      {getGradeKey(diagnosis.overall_grade)}
                    </Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Executive Summary */}
        {(summary || report.summary) && (
          <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-2 border-blue-200 dark:border-blue-800">
            <CardContent className="pt-5">
              <div className="flex items-start gap-4">
                <div className="bg-blue-100 dark:bg-blue-900/50 p-3 rounded-xl">
                  <CheckCircle className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-blue-900 dark:text-blue-100 mb-2">
                    Executive Summary
                  </h3>
                  <p className="text-blue-800 dark:text-blue-200 leading-relaxed">
                    {summary || report.summary}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <Card className="border-2 border-amber-200 dark:border-amber-800">
            <CardHeader className="pb-3 bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-950/30 dark:to-yellow-950/30 border-b border-amber-200 dark:border-amber-800">
              <div className="flex items-center gap-2">
                <Pill className="h-5 w-5 text-amber-600" />
                <span className="font-semibold text-lg text-amber-900 dark:text-amber-100">
                  Recommendations
                </span>
              </div>
            </CardHeader>
            <CardContent className="pt-4 space-y-3">
              {recommendations.map((rec, index) => (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-start gap-4 p-4 bg-gradient-to-r from-amber-50/50 to-yellow-50/30 dark:from-amber-950/20 dark:to-yellow-950/10 rounded-xl border border-amber-100 dark:border-amber-900/30"
                >
                  <div className="bg-gradient-to-br from-amber-500 to-yellow-500 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shrink-0">
                    {index + 1}
                  </div>
                  <p className="text-sm leading-relaxed text-amber-900 dark:text-amber-100">
                    {rec}
                  </p>
                </motion.div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Metadata Footer */}
        <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
          <div className="flex flex-wrap items-center justify-between text-sm text-muted-foreground gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Monitor className="h-4 w-4" />
                <span>Model: <span className="font-medium text-foreground">{report_metadata?.model || report.llm_model}</span></span>
              </div>
              {report_metadata?.model_version && (
                <div className="flex items-center gap-2">
                  <span>Version: <span className="font-medium text-foreground">{report_metadata.model_version}</span></span>
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              <span>
                Generated: {report_metadata?.generated_date || new Date(report.created_at).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}