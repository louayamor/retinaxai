'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  Calendar, 
  Bot, 
  User,
  Copy,
  Printer,
  Activity,
  AlertCircle,
  CheckCircle,
  Stethoscope,
  ClipboardList
} from 'lucide-react';
import type { Report, ReportStatus } from '@/types';
import { slideInUp } from '@/lib/animations';

interface ReportCardProps {
  report: Report;
  patientName: string;
  onExpand?: () => void;
  expanded?: boolean;
}

interface ParsedReport {
  patient_info?: { name?: string; age?: string; gender?: string; mrn?: string };
  clinical_findings?: {
    left_eye?: { grade?: string; severity?: string; confidence?: string; description?: string };
    right_eye?: { grade?: string; severity?: string; confidence?: string; description?: string };
  };
  diagnosis?: { condition?: string; severity?: string; overall_grade?: string; risk_level?: string };
  recommendations?: string[];
  summary?: string;
  report_metadata?: { generated_date?: string; model?: string; model_version?: string };
  content?: string;
}

const STATUS_CONFIG: Record<ReportStatus, { label: string; color: string; icon: string }> = {
  pending: { label: 'Pending', color: 'bg-yellow-500', icon: '⏳' },
  running: { label: 'Generating', color: 'bg-blue-500 animate-pulse', icon: '🔄' },
  completed: { label: 'Completed', color: 'bg-green-500', icon: '✓' },
  failed: { label: 'Failed', color: 'bg-red-500', icon: '✗' }
};

const DEFAULT_STATUS = { label: 'Unknown', color: 'bg-gray-500', icon: '?' };

function parseReportContent(content: string | null): ParsedReport | null {
  if (!content) return null;
  
  // Try parsing as JSON first (new format)
  try {
    const parsed = JSON.parse(content);
    if (parsed.patient_info || parsed.clinical_findings || parsed.diagnosis) {
      return parsed;
    }
  } catch {
    // Not JSON, try to extract structured content from plain text
  }
  
  // Return as legacy format
  return { content, summary: content.slice(0, 200) };
}

function getGradeColor(grade: string | number | undefined): string {
  const gradeStr = String(grade || '').toLowerCase();
  if (gradeStr === '0' || gradeStr === 'no dr' || gradeStr === 'none') return 'text-green-600';
  if (gradeStr === '1' || gradeStr === 'mild') return 'text-cyan-600';
  if (gradeStr === '2' || gradeStr === 'moderate') return 'text-amber-600';
  if (gradeStr === '3' || gradeStr === 'severe') return 'text-orange-600';
  if (gradeStr === '4' || gradeStr === 'proliferative') return 'text-red-600';
  return 'text-gray-600';
}

export function ReportCard({ report, patientName, onExpand, expanded }: ReportCardProps) {
  const [copied, setCopied] = useState(false);
  const status = STATUS_CONFIG[report.status] ?? DEFAULT_STATUS;
  const parsedReport = parseReportContent(report.content);

  const handleCopy = () => {
    if (report.content) {
      navigator.clipboard.writeText(report.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getInitials = (name: string) => {
    const parts = name.split(' ');
    return parts.length > 1 
      ? `${parts[0][0]}${parts[1][0]}`.toUpperCase() 
      : name.slice(0, 2).toUpperCase();
  };

  return (
    <motion.div variants={slideInUp}>
      <Card className="overflow-hidden hover:shadow-lg transition-shadow duration-300 border-l-4 border-l-[var(--brand-teal)]">
        {/* Card Header */}
        <CardHeader className="pb-3 bg-gradient-to-r from-muted/30 to-transparent">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              {/* Patient Avatar */}
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-[var(--brand-teal)] to-[#0d3a4c] text-white font-semibold">
                {getInitials(patientName)}
              </div>
              <div>
                <CardTitle className="text-lg">{patientName}</CardTitle>
                <p className="text-sm text-muted-foreground flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {new Date(report.created_at).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                  })}
                </p>
              </div>
            </div>
            <Badge className={`${status.color} text-white`}>
              {status.label}
            </Badge>
          </div>
        </CardHeader>

        {/* Card Content - Summary Preview */}
        <CardContent className="pb-3">
          <div className="space-y-3">
            {/* Show structured summary if available */}
            {parsedReport?.summary && (
              <div className="bg-gradient-to-r from-[var(--brand-teal)]/10 to-transparent rounded-lg p-3 border-l-2 border-[var(--brand-teal)]">
                <p className="text-sm font-medium text-[var(--brand-teal)] mb-1">Summary</p>
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {parsedReport.summary}
                </p>
              </div>
            )}
            
            {/* Show diagnosis if available */}
            {parsedReport?.diagnosis?.condition && (
              <div className="flex items-center gap-2 text-sm">
                <Stethoscope className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">{parsedReport.diagnosis.condition}</span>
                <Badge variant="outline" className={`${getGradeColor(parsedReport.diagnosis.overall_grade)} border-current`}>
                  Grade {parsedReport.diagnosis.overall_grade || 'N/A'}
                </Badge>
              </div>
            )}

            {/* Fallback to legacy summary */}
            {!parsedReport?.summary && report.summary && (
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {report.summary}
                </p>
              </div>
            )}

            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-1 text-muted-foreground">
                <Bot className="h-4 w-4" />
                <span>{report.llm_model || 'gpt-4o'}</span>
              </div>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={onExpand}
                disabled={report.status !== 'completed'}
                className="text-[var(--brand-teal)] hover:text-[#1a9a9a] hover:bg-[var(--brand-teal)]/10"
              >
                {expanded ? (
                  <>
                    <ChevronUp className="h-4 w-4 mr-1" />
                    Collapse
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-4 w-4 mr-1" />
                    View Report
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>

        {/* Expanded Report Detail */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="border-t bg-muted/20"
            >
              <CardContent className="pt-4">
                <div className="space-y-4">
                  {/* Patient Info Section */}
                  {parsedReport?.patient_info && (
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border-l-4 border-blue-500">
                      <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                        <User className="h-4 w-4 text-blue-500" />
                        Patient Information
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        {parsedReport.patient_info.name && (
                          <div>
                            <p className="text-xs text-muted-foreground">Name</p>
                            <p className="font-medium">{parsedReport.patient_info.name}</p>
                          </div>
                        )}
                        {parsedReport.patient_info.age && (
                          <div>
                            <p className="text-xs text-muted-foreground">Age</p>
                            <p className="font-medium">{parsedReport.patient_info.age}</p>
                          </div>
                        )}
                        {parsedReport.patient_info.gender && (
                          <div>
                            <p className="text-xs text-muted-foreground">Gender</p>
                            <p className="font-medium">{parsedReport.patient_info.gender}</p>
                          </div>
                        )}
                        {parsedReport.patient_info.mrn && (
                          <div>
                            <p className="text-xs text-muted-foreground">MRN</p>
                            <p className="font-mono">{parsedReport.patient_info.mrn}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Clinical Findings Section */}
                  {parsedReport?.clinical_findings && (
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border-l-4 border-[var(--brand-teal)]">
                      <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                        <Activity className="h-4 w-4 text-[var(--brand-teal)]" />
                        Clinical Findings
                      </h4>
                      <div className="grid md:grid-cols-2 gap-4">
                        {/* Left Eye */}
                        {parsedReport.clinical_findings.left_eye && (
                          <div className="bg-muted/50 rounded-lg p-3">
                            <p className="text-sm font-semibold mb-2">Left Eye (OS)</p>
                            <div className="space-y-1 text-sm">
                              <p>
                                <span className="text-muted-foreground">Grade: </span>
                                <span className={getGradeColor(parsedReport.clinical_findings.left_eye.grade)}>
                                  {parsedReport.clinical_findings.left_eye.grade || 'N/A'}
                                </span>
                              </p>
                              <p>
                                <span className="text-muted-foreground">Severity: </span>
                                {parsedReport.clinical_findings.left_eye.severity || 'N/A'}
                              </p>
                              <p>
                                <span className="text-muted-foreground">Confidence: </span>
                                {parsedReport.clinical_findings.left_eye.confidence || 'N/A'}
                              </p>
                              {parsedReport.clinical_findings.left_eye.description && (
                                <p className="text-muted-foreground mt-2">{parsedReport.clinical_findings.left_eye.description}</p>
                              )}
                            </div>
                          </div>
                        )}
                        {/* Right Eye */}
                        {parsedReport.clinical_findings.right_eye && (
                          <div className="bg-muted/50 rounded-lg p-3">
                            <p className="text-sm font-semibold mb-2">Right Eye (OD)</p>
                            <div className="space-y-1 text-sm">
                              <p>
                                <span className="text-muted-foreground">Grade: </span>
                                <span className={getGradeColor(parsedReport.clinical_findings.right_eye.grade)}>
                                  {parsedReport.clinical_findings.right_eye.grade || 'N/A'}
                                </span>
                              </p>
                              <p>
                                <span className="text-muted-foreground">Severity: </span>
                                {parsedReport.clinical_findings.right_eye.severity || 'N/A'}
                              </p>
                              <p>
                                <span className="text-muted-foreground">Confidence: </span>
                                {parsedReport.clinical_findings.right_eye.confidence || 'N/A'}
                              </p>
                              {parsedReport.clinical_findings.right_eye.description && (
                                <p className="text-muted-foreground mt-2">{parsedReport.clinical_findings.right_eye.description}</p>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Diagnosis Section */}
                  {parsedReport?.diagnosis && (
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border-l-4 border-amber-500">
                      <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                        <AlertCircle className="h-4 w-4 text-amber-500" />
                        Diagnosis
                      </h4>
                      <div className="space-y-2 text-sm">
                        {parsedReport.diagnosis.condition && (
                          <p><span className="text-muted-foreground">Condition: </span><span className="font-medium">{parsedReport.diagnosis.condition}</span></p>
                        )}
                        {parsedReport.diagnosis.overall_grade && (
                          <p>
                            <span className="text-muted-foreground">Overall Grade: </span>
                            <Badge variant="outline" className={`${getGradeColor(parsedReport.diagnosis.overall_grade)} border-current`}>
                              {parsedReport.diagnosis.overall_grade}
                            </Badge>
                          </p>
                        )}
                        {parsedReport.diagnosis.risk_level && (
                          <p><span className="text-muted-foreground">Risk Level: </span>{parsedReport.diagnosis.risk_level}</p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Recommendations Section */}
                  {parsedReport?.recommendations && parsedReport.recommendations.length > 0 && (
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border-l-4 border-green-500">
                      <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Recommendations
                      </h4>
                      <ul className="space-y-2">
                        {parsedReport.recommendations.map((rec, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm">
                            <span className="text-[var(--brand-teal)] mt-1">•</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Legacy Content (fallback) */}
                  {parsedReport?.content && !parsedReport.patient_info && (
                    <div className="bg-white dark:bg-slate-900 rounded-lg p-4">
                      <h4 className="font-semibold text-sm mb-2 flex items-center gap-2">
                        <FileText className="h-4 w-4 text-[var(--brand-teal)]" />
                        Report Content
                      </h4>
                      <div className="text-sm whitespace-pre-wrap">
                        {parsedReport.content}
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {report.status === 'failed' && report.error_message && (
                    <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200">
                      <h4 className="font-semibold text-sm text-red-600 mb-2">Error</h4>
                      <p className="text-sm text-red-600">{report.error_message}</p>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-2 pt-2">
                    <Button variant="outline" size="sm" onClick={handleCopy}>
                      {copied ? (
                        <>
                          <Copy className="h-4 w-4 mr-1" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="h-4 w-4 mr-1" />
                          Copy
                        </>
                      )}
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => window.print()}
                      disabled={report.status !== 'completed'}
                    >
                      <Printer className="h-4 w-4 mr-1" />
                      Print
                    </Button>
                  </div>
                </div>
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </motion.div>
  );
}