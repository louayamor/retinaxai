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

const GRADE_META: Record<string, { color: string; bg: string; border: string }> = {
  '0': { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
  '1': { color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/30' },
  '2': { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
  '3': { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30' },
  '4': { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/30' },
  'No DR': { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
  'Mild': { color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/30' },
  'Moderate': { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
  'Severe': { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30' },
  'Proliferative DR': { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/30' },
};

const RISK_META: Record<string, { color: string; bg: string; border: string }> = {
  low: { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
  moderate: { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
  elevated: { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
  high: { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30' },
  severe: { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/30' },
  very_high: { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/30' },
};

function getGradeKey(grade: string | number | undefined) {
  if (!grade) return 'N/A';
  return GRADE_LABELS[String(grade)] || String(grade);
}

function getGradeMeta(grade: string | number | undefined) {
  if (!grade) return null;
  return GRADE_META[String(grade)] ?? GRADE_META['2'];
}

function getRiskMeta(risk?: string) {
  if (!risk) return null;
  const lower = risk.toLowerCase();
  return RISK_META[lower] ?? RISK_META['moderate'];
}

interface ExplanationSection {
  title: string;
  body: string;
}

function parseExplanationSections(text: string): ExplanationSection[] {
  const sections: ExplanationSection[] = [];
  const boldHeaderRegex = /\*\*(.+?)\*\*\s*:?\s*/g;
  const parts = text.split(boldHeaderRegex);

  if (parts.length > 1) {
    for (let i = 1; i < parts.length; i += 2) {
      const title = parts[i].trim();
      const body = (parts[i + 1] || '').trim();
      if (title) {
        sections.push({
          title,
          body: body.replace(/^\n+/, '').trim(),
        });
      }
    }
  }

  if (sections.length === 0) {
    const lines = text.split(/\n{2,}/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length > 10) {
        sections.push({ title: '', body: trimmed });
      }
    }
  }

  return sections;
}

export default function XAICard({ predictionId, createdAt, data }: XAICardProps) {
  const { diagnosis, clinical_findings, severity_report, gradcam_explanation, feature_importance, summary } = data;

  const drGrade = severity_report?.diagnosis?.dr_grade ?? diagnosis?.overall_grade;
  const riskLevel = severity_report?.diagnosis?.risk_level ?? diagnosis?.risk_level;
  const gradeMeta = getGradeMeta(drGrade);
  const riskMeta = getRiskMeta(riskLevel);

  return (
    <Card className="overflow-hidden border border-border bg-card">
      <div className="bg-[var(--sidebar)] px-5 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-[var(--brand-teal)]/20 p-2 rounded-lg">
            <Brain className="h-5 w-5 text-[var(--brand-teal)]" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-sidebar-foreground">XAI Clinical Explanation</h2>
            <p className="text-xs text-sidebar-foreground/60 flex items-center gap-1.5 mt-0.5">
              <Activity className="h-3 w-3" />
              AI Explainable Analysis
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs text-sidebar-foreground/60 flex items-center gap-1.5 justify-end">
            <Calendar className="h-3 w-3" />
            Generated
          </p>
          <p className="text-xs font-medium text-sidebar-foreground">{new Date(createdAt).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</p>
        </div>
      </div>

      <CardContent className="p-5 space-y-5">
        {diagnosis && (
          <>
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Stethoscope className="h-4 w-4 text-[var(--brand-teal)]" />
                  <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Diagnosis</span>
                </div>
                {riskMeta && riskLevel && (
                  <span className={`text-xs font-medium px-2 py-0.5 rounded border uppercase tracking-wide ${riskMeta.bg} ${riskMeta.border} ${riskMeta.color}`}>
                    {riskLevel.replace('_', ' ')} Risk
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="rounded-md bg-muted/30 px-3 py-2">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Condition</p>
                  <p className="text-sm font-semibold mt-0.5">{diagnosis.condition || 'Diabetic Retinopathy'}</p>
                </div>
                <div className="rounded-md bg-muted/30 px-3 py-2">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Severity</p>
                  <p className="text-sm font-semibold mt-0.5">{diagnosis.severity || severity_report?.diagnosis?.severity_label || '—'}</p>
                </div>
                <div className="rounded-md bg-muted/30 px-3 py-2">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground">DR Grade</p>
                  {gradeMeta ? (
                    <span className={`text-xs font-medium px-2 py-0.5 rounded border mt-0.5 inline-block ${gradeMeta.bg} ${gradeMeta.border} ${gradeMeta.color}`}>
                      {getGradeKey(drGrade)}
                    </span>
                  ) : (
                    <p className="text-sm font-semibold mt-0.5">{getGradeKey(drGrade)}</p>
                  )}
                </div>
                <div className="rounded-md bg-muted/30 px-3 py-2">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Confidence</p>
                  <p className="text-sm font-semibold mt-0.5">{diagnosis.confidence ? `${(diagnosis.confidence * 100).toFixed(1)}%` : '—'}</p>
                </div>
              </div>
            </div>
          </>
        )}

        {(clinical_findings?.left_eye || clinical_findings?.right_eye) && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Eye className="h-4 w-4 text-[var(--brand-teal)]" />
              <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Clinical Findings</span>
            </div>
            <div className="grid md:grid-cols-2 gap-3">
              {clinical_findings?.left_eye && (
                <div className="rounded-lg border border-[var(--brand-teal)]/30 bg-[var(--brand-teal)]/5 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Scan className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
                      <h4 className="text-sm font-semibold text-foreground">Left Eye (OS)</h4>
                    </div>
                    {gradeMeta && (
                      <span className={`text-xs font-medium px-2 py-0.5 rounded border ${gradeMeta.bg} ${gradeMeta.border} ${gradeMeta.color}`}>
                        {getGradeKey(clinical_findings.left_eye.grade)}
                      </span>
                    )}
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Severity:</span>
                      <span className="font-medium capitalize">{clinical_findings.left_eye.severity || '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {clinical_findings.left_eye.confidence ? `${(clinical_findings.left_eye.confidence * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    {clinical_findings.left_eye.description && (
                      <p className="text-xs text-muted-foreground mt-2 pt-2 border-t border-border">
                        {clinical_findings.left_eye.description}
                      </p>
                    )}
                  </div>
                </div>
              )}
              {clinical_findings?.right_eye && (
                <div className="rounded-lg border border-[var(--brand-gold)]/30 bg-[var(--brand-gold)]/5 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Scan className="h-3.5 w-3.5 text-[var(--brand-gold)]" />
                      <h4 className="text-sm font-semibold text-foreground">Right Eye (OD)</h4>
                    </div>
                    {gradeMeta && (
                      <span className={`text-xs font-medium px-2 py-0.5 rounded border ${gradeMeta.bg} ${gradeMeta.border} ${gradeMeta.color}`}>
                        {getGradeKey(clinical_findings.right_eye.grade)}
                      </span>
                    )}
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Severity:</span>
                      <span className="font-medium capitalize">{clinical_findings.right_eye.severity || '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {clinical_findings.right_eye.confidence ? `${(clinical_findings.right_eye.confidence * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    {clinical_findings.right_eye.description && (
                      <p className="text-xs text-muted-foreground mt-2 pt-2 border-t border-border">
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
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="h-4 w-4 text-[var(--brand-teal)]" />
                <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">GradCAM Anatomical Regions</span>
              </div>
              <div className="grid md:grid-cols-2 gap-3">
                {gradcam_explanation?.highlighted_regions?.left_eye?.length ? (
                  <div className="rounded-md border border-[var(--brand-teal)]/20 bg-[var(--brand-teal)]/5 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
                      <h4 className="text-sm font-semibold text-foreground">Left Eye (OS)</h4>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {gradcam_explanation.highlighted_regions?.left_eye?.map((region: string, idx: number) => (
                        <Badge key={idx} variant="outline" className="text-xs border-[var(--brand-teal)]/30 text-[var(--brand-teal)]">
                          {region.replace(/_/g, ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ) : null}
                {gradcam_explanation?.highlighted_regions?.right_eye?.length ? (
                  <div className="rounded-md border border-[var(--brand-gold)]/20 bg-[var(--brand-gold)]/5 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="h-3.5 w-3.5 text-[var(--brand-gold)]" />
                      <h4 className="text-sm font-semibold text-foreground">Right Eye (OD)</h4>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {gradcam_explanation.highlighted_regions?.right_eye?.map((region: string, idx: number) => (
                        <Badge key={idx} variant="outline" className="text-xs border-[var(--brand-gold)]/30 text-[var(--brand-gold)]">
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
            <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
              <div className="flex items-center gap-2 mb-3">
                <Heart className="h-4 w-4 text-amber-500" />
                <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Recommendations</span>
              </div>
              <div className="space-y-2">
                {severity_report.recommendations.slice(0, 5).map((rec, idx) => (
                  rec?.action && (
                    <div key={idx} className="flex items-start gap-3 p-2.5 rounded-md bg-card border border-border">
                      <div className="bg-amber-500 text-white w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">
                        {idx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium">{rec.action}</p>
                        {rec.timeframe && (
                          <p className="text-xs text-muted-foreground mt-0.5">{rec.timeframe}</p>
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
              <div className="flex items-center gap-2 mb-3">
                <Eye className="h-4 w-4 text-[var(--brand-teal)]" />
                <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Detailed GradCAM Analysis</span>
              </div>
              <div className="grid md:grid-cols-2 gap-3">
                {gradcam_explanation?.left_eye_explanation && (
                  <div className="rounded-lg border border-[var(--brand-teal)]/30 bg-[var(--brand-teal)]/5 p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Eye className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
                        <h4 className="text-sm font-semibold text-foreground">Left Eye (OS)</h4>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-white px-1.5 py-0.5 rounded bg-[var(--brand-teal)]">
                        {gradcam_explanation.highlighted_regions?.left_eye?.length || 0} regions
                      </span>
                    </div>
                    <div className="space-y-3">
                      {parseExplanationSections(gradcam_explanation.left_eye_explanation).map((section, idx) => (
                        <div key={idx}>
                          {section.title && (
                            <p className="text-xs font-semibold text-foreground mb-1">{section.title}</p>
                          )}
                          <p className="text-sm text-muted-foreground leading-relaxed">{section.body}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {gradcam_explanation?.right_eye_explanation && (
                  <div className="rounded-lg border border-[var(--brand-gold)]/30 bg-[var(--brand-gold)]/5 p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Eye className="h-3.5 w-3.5 text-[var(--brand-gold)]" />
                        <h4 className="text-sm font-semibold text-foreground">Right Eye (OD)</h4>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-white px-1.5 py-0.5 rounded bg-[var(--brand-gold)]">
                        {gradcam_explanation.highlighted_regions?.right_eye?.length || 0} regions
                      </span>
                    </div>
                    <div className="space-y-3">
                      {parseExplanationSections(gradcam_explanation.right_eye_explanation).map((section, idx) => (
                        <div key={idx}>
                          {section.title && (
                            <p className="text-xs font-semibold text-foreground mb-1">{section.title}</p>
                          )}
                          <p className="text-sm text-muted-foreground leading-relaxed">{section.body}</p>
                        </div>
                      ))}
                    </div>
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