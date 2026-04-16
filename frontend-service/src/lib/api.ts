import { clearTokens } from './auth';
import type { TokenPair, AuthUser } from './auth';
import type { Patient, MRIScan, Prediction, Report, PaginatedResponse, OCTReport } from '@/types';

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // Cookie-based auth: cookies sent automatically with credentials: 'include'
  const isFormData = init.body instanceof FormData;
  const headers: Record<string, string> = isFormData
    ? {}
    : { 'Content-Type': 'application/json', ...(init.headers as Record<string, string> || {}) };

  const res = await fetch(`${BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers
  });

  // Handle 401 by clearing tokens and redirecting to login
  if (res.status === 401) {
    clearTokens();
    if (typeof window !== 'undefined') {
      window.location.href = '/auth/login';
    }
    const body = await res.json().catch(() => ({ detail: 'Session expired. Please login again.' }));
    throw new ApiError(body.detail ?? 'Session expired', 401);
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(body.detail ?? 'Request failed', res.status);
  }

  return res.json();
}

export interface RegisterPayload {
  username: string;
  email: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export async function registerUser(payload: RegisterPayload): Promise<AuthUser> {
  return request<AuthUser>('/api/v1/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

export async function loginUser(payload: LoginPayload): Promise<TokenPair> {
  const form = new URLSearchParams();
  form.append('username', payload.email);
  form.append('password', payload.password);
  return request<TokenPair>('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form.toString()
  });
}

export async function getPatients() {
  return request<Patient[]>('/api/v1/patients/');
}

export interface PatientStats {
  total: number;
  avg_age: number;
  male_count: number;
  female_count: number;
  this_month: number;
}

export async function getPatientStats() {
  return request<PatientStats>('/api/v1/patients/stats');
}

export async function searchPatients(query: string) {
  const params = new URLSearchParams({ q: query });
  return request<Patient[]>(`/api/v1/patients/?${params.toString()}`);
}

export async function getPatient(id: string) {
  return request<Patient>(`/api/v1/patients/${id}`);
}

export async function createPatient(payload: Omit<Patient, 'id' | 'created_at'>) {
  return request<Patient>('/api/v1/patients/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

export async function updatePatient(
  id: string,
  payload: Partial<
    Pick<
      Patient,
      | 'first_name'
      | 'last_name'
      | 'age'
      | 'gender'
      | 'phone'
      | 'address'
      | 'medical_record_number'
      | 'ocr_patient_id'
    >
  >
) {
  return request<Patient>(`/api/v1/patients/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

export async function deletePatient(id: string) {
  return request<{ message: string }>(`/api/v1/patients/${id}`, {
    method: 'DELETE'
  });
}

export async function uploadScans(patientId: string, form: FormData) {
  return request<MRIScan>(`/api/v1/patients/${patientId}/scans`, {
    method: 'POST',
    body: form
  });
}

export async function getPatientScans(patientId: string) {
  return request<MRIScan[]>(`/api/v1/patients/${patientId}/scans`);
}

export async function getPatientOctReports(patientId: string) {
  return request<OCTReport[]>(`/api/v1/oct-reports/patients/${patientId}`);
}

export interface PredictionRequest {
  patient_id: string;
  mri_scan_id: string;
  model_name: string;
  model_version: string;
  input_payload: Record<string, unknown>;
}

export async function createPrediction(data: PredictionRequest) {
  return request<Prediction>('/api/v1/predictions/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
}

export async function getPrediction(id: string) {
  return request<Prediction>(`/api/v1/predictions/${id}`);
}

export async function listPatientPredictions(
  patientId: string,
  page = 1,
  size = 20
): Promise<PaginatedResponse<Prediction>> {
  return request<PaginatedResponse<Prediction>>(
    `/api/v1/predictions/patient/${patientId}?page=${page}&size=${size}`
  );
}

export async function listAllPredictions(page = 1, size = 20): Promise<PaginatedResponse<Prediction>> {
  return request<PaginatedResponse<Prediction>>(`/api/v1/predictions/?page=${page}&size=${size}`);
}

export async function createReport(predictionId: string) {
  return request<Report>('/api/v1/reports/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prediction_id: predictionId })
  });
}

export async function getReport(id: string) {
  return request<Report>(`/api/v1/reports/${id}`);
}

export async function listPatientReports(
  patientId: string,
  page = 1,
  size = 20
): Promise<PaginatedResponse<Report>> {
  return request<PaginatedResponse<Report>>(
    `/api/v1/reports/patient/${patientId}?page=${page}&size=${size}`
  );
}

export async function listAllReports(page = 1, size = 20): Promise<PaginatedResponse<Report>> {
  return request<PaginatedResponse<Report>>(`/api/v1/reports/?page=${page}&size=${size}`);
}

// Async report generation (LLMOps)
interface AsyncReportResponse {
  job_id: string;
  status: string;
}

interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: Report;
  error?: string;
  progress?: number;
  created_at: string;
  updated_at: string;
}

export async function createAsyncReport(predictionId: string): Promise<AsyncReportResponse> {
  return request<AsyncReportResponse>('/api/v1/reports/async', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prediction_id: predictionId })
  });
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/api/v1/reports/jobs/${jobId}`);
}

export async function listJobs(): Promise<JobStatusResponse[]> {
  return request<JobStatusResponse[]>('/api/v1/reports/jobs');
}

export async function healthCheck() {
  return request<{ status: string }>('/health');
}

// Dashboard stats
export interface DashboardStats {
  totals: {
    patients: number;
    predictions: number;
    reports: number;
    scans: number;
  };
  prediction_status: Record<string, number>;
  report_status: Record<string, number>;
  severity_distribution: Record<number, number>;
  predictions_timeline: Array<{ date: string; predictions: number }>;
  age_distribution: Record<string, number>;
  gender_distribution: Record<string, number>;
  recent_activity: {
    new_patients: number;
    new_predictions: number;
    new_reports: number;
  };
  avg_confidence: number | null;
}

export async function getDashboardStats(): Promise<DashboardStats> {
  return request<DashboardStats>('/api/v1/dashboard/stats');
}

// LLMOps service API (runs on port 8002)
const LLMOPS_BASE = process.env.NEXT_PUBLIC_LLMOPS_URL ?? 'http://localhost:8002';

interface RagStatusResponse {
  status: string;
  schema_version?: string;
  run_id?: string;
  artifact_count: number;
  collection_name?: string;
  persist_directory?: string;
}

interface RagReindexResponse {
  status: string;
  result?: {
    schema_version: string;
    run_id: string;
    artifact_count: number;
    pipeline: string;
    document_count: number;
    chunk_count: number;
  };
}

const LLMOPS_API_KEY = process.env.NEXT_PUBLIC_LLMOPS_API_KEY ?? 'dev-api-key';

async function _handleLlmoopsResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body?.detail || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return res.json();
}

export async function getRagStatus(): Promise<RagStatusResponse> {
  const res = await fetch(`${LLMOPS_BASE}/api/rag/status`, {
    headers: { 'x-api-key': LLMOPS_API_KEY },
  });
  return _handleLlmoopsResponse(res);
}

export async function triggerRagReindex(): Promise<RagReindexResponse> {
  const res = await fetch(`${LLMOPS_BASE}/api/rag/reindex`, {
    method: 'POST',
    headers: { 'x-api-key': LLMOPS_API_KEY },
  });
  return _handleLlmoopsResponse(res);
}

export async function checkLlmoopsHealth(): Promise<{ status: string }> {
  const res = await fetch(`${LLMOPS_BASE}/health`);
  return _handleLlmoopsResponse(res);
}

export interface OperationStatus {
  state: string;
  message: string;
  progress: number | null;
  started_at: string | null;
}

export async function getOperationStatus(): Promise<OperationStatus> {
  const res = await fetch(`${LLMOPS_BASE}/api/operation/status`);
  return _handleLlmoopsResponse(res);
}

// Notification API
export interface NotificationItem {
  id: string;
  type: string;
  title: string;
  message: string;
  read: boolean;
  created_at: string;
  user_id: string | null;
}

export async function fetchNotifications(unreadOnly = false, limit = 50, offset = 0): Promise<NotificationItem[]> {
  return request<NotificationItem[]>(
    `/api/v1/notifications/?unread_only=${unreadOnly}&limit=${limit}&offset=${offset}`,
    { method: 'GET' }
  );
}

export async function getUnreadCount(): Promise<{ unread_count: number }> {
  return request<{ unread_count: number }>('/api/v1/notifications/unread-count', { method: 'GET' });
}

export async function markNotificationsRead(notificationIds: string[]): Promise<{ status: string; marked_count: number }> {
  return request<{ status: string; marked_count: number }>('/api/v1/notifications/read', {
    method: 'PUT',
    body: JSON.stringify({ notification_ids: notificationIds })
  });
}

export async function markAllNotificationsRead(): Promise<{ status: string; marked_count: number }> {
  return request<{ status: string; marked_count: number }>('/api/v1/notifications/read-all', {
    method: 'PUT'
  });
}

export async function clearNotifications(readOnly = true): Promise<{ status: string; deleted_count: number }> {
  return request<{ status: string; deleted_count: number }>(
    `/api/v1/notifications/clear?read_only=${readOnly}`,
    { method: 'DELETE' }
  );
}

// XAI Generation APIs (calls LLMOps service directly)
interface XAIExplanationResponse {
  prediction_id: string;
  content: string;
  summary: string;
  model_used: string;
  shap_values?: {
    top_positive: Array<{ name: string; contribution: number }>;
    top_negative: Array<{ name: string; contribution: number }>;
    features: Array<{ name: string; contribution: number }>;
  };
}

interface XAIGradCAMResponse {
  prediction_id: string;
  left_eye_explanation: string;
  right_eye_explanation: string;
  highlighted_regions: {
    left_eye: string[];
    right_eye: string[];
  };
  model_used: string;
}

interface XAISeverityResponse {
  prediction_id: string;
  content: string;
  summary: string;
  risk_level: 'low' | 'moderate' | 'high' | 'very_high';
  recommendations: string[];
  model_used: string;
}

export async function generateXAIExplanation(
  predictionId: string,
  drGrade: string,
  confidence: number,
  clinicalFeatures?: Record<string, unknown>
): Promise<XAIExplanationResponse> {
  const res = await fetch(`${LLMOPS_BASE}/api/xai/explain`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': LLMOPS_API_KEY,
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      dr_grade: drGrade,
      confidence,
      clinical_features: clinicalFeatures,
    }),
  });
  return _handleLlmoopsResponse(res);
}

export async function generateXAIGradCAM(
  predictionId: string,
  leftEyeRegions: string[] = [],
  rightEyeRegions: string[] = []
): Promise<XAIGradCAMResponse> {
  const res = await fetch(`${LLMOPS_BASE}/api/xai/gradcam`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': LLMOPS_API_KEY,
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      left_eye_regions: leftEyeRegions,
      right_eye_regions: rightEyeRegions,
    }),
  });
  return _handleLlmoopsResponse(res);
}

export async function generateXAISeverity(
  predictionId: string,
  patientData: { name: string; age: number; gender: string },
  drGrade: string,
  riskFactors: string[] = []
): Promise<XAISeverityResponse> {
  const res = await fetch(`${LLMOPS_BASE}/api/xai/severity`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': LLMOPS_API_KEY,
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      patient_data: patientData,
      dr_grade: drGrade,
      risk_factors: riskFactors,
    }),
  });
  return _handleLlmoopsResponse(res);
}

// SHAP-specific API
interface SHAPExplainResponse {
  model_type: string;
  expected_value: number;
  pipeline: string;
  features: Array<{
    name: string;
    contribution: number;
    base_value: number;
    value: number;
  }>;
  top_positive: Array<{ name: string; contribution: number }>;
  top_negative: Array<{ name: string; contribution: number }>;
}

export async function generateSHAPExplanation(
  predictionId: string,
  clinicalFeatures: Record<string, unknown>,
  pipeline = 'clinical'
): Promise<SHAPExplainResponse> {
  const res = await fetch(`${LLMOPS_BASE}/api/xai/shap/explain`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': LLMOPS_API_KEY,
    },
    body: JSON.stringify({
      prediction_id: predictionId,
      features: clinicalFeatures,
      pipeline,
    }),
  });
  return _handleLlmoopsResponse(res);
}

export async function storeXAIShallowResults(
  predictionId: string,
  shapValues: SHAPExplainResponse,
  model: string = 'shap'
): Promise<{ status: string }> {
  return request<{ status: string }>('/api/v1/explanations/store/shap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prediction_id: predictionId,
      shap_values: shapValues,
      content: `SHAP analysis: ${shapValues.features?.length || 0} features analyzed`,
      summary: `Top positive: ${shapValues.top_positive?.map(f => f.name).join(', ') || 'none'}`,
      model,
    }),
  });
}

export async function storeXAIResults(
  predictionId: string,
  options: {
    explanationContent?: string;
    explanationSummary?: string;
    shapValues?: SHAPExplainResponse;
    severityContent?: string;
    severitySummary?: string;
    severityRiskLevel?: string;
    severityRecommendations?: string[];
    model?: string;
  }
): Promise<{ status: string }> {
  return request<{ status: string }>('/api/v1/explanations/store', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prediction_id: predictionId,
      explanation_content: options.explanationContent,
      explanation_summary: options.explanationSummary,
      explanation_model: options.model || 'gpt-4.1-mini',
      shap_values: options.shapValues,
      severity_content: options.severityContent,
      severity_summary: options.severitySummary,
      severity_risk_level: options.severityRiskLevel || 'moderate',
      severity_recommendations: options.severityRecommendations || [],
    }),
  });
}

export class ApiConflictError extends ApiError {
  constructor(message: string) {
    super(message, 409);
    this.name = 'ApiConflictError';
  }
}

export interface XAIExplanation {
  id: string;
  content: string;
  summary: string | null;
  model_used: string;
  status: string;
  shap_values?: {
    top_positive: Array<{ name: string; contribution: number }>;
    top_negative: Array<{ name: string; contribution: number }>;
    features: Array<{ name: string; contribution: number }>;
  };
}

export interface SeverityReport {
  id: string;
  content: string;
  summary: string | null;
  risk_level: 'low' | 'moderate' | 'high' | 'severe';
  recommendations: string[];
  model_used: string;
}

export interface XAIResponse {
  prediction_id: string;
  explanation: XAIExplanation | null;
  severity_report: SeverityReport | null;
  gradcam_explanation: null;
}

export async function getXAIExplanations(predictionId: string): Promise<XAIResponse> {
  return request<XAIResponse>(`/api/v1/explanations/${predictionId}`);
}