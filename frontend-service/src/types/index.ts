import { Icons } from '@/components/icons';

// ─── Existing template types (keep as-is) ────────────────────────────────────

export interface PermissionCheck {
  permission?: string;
  plan?: string;
  feature?: string;
  role?: string;
  requireOrg?: boolean;
}

export interface NavItem {
  title: string;
  url: string;
  disabled?: boolean;
  external?: boolean;
  shortcut?: [string, string];
  icon?: keyof typeof Icons;
  label?: string;
  description?: string;
  isActive?: boolean;
  items?: NavItem[];
  access?: PermissionCheck;
}

export interface NavItemWithChildren extends NavItem {
  items: NavItemWithChildren[];
}

export interface NavItemWithOptionalChildren extends NavItem {
  items?: NavItemWithChildren[];
}

export interface FooterItem {
  title: string;
  items: {
    title: string;
    href: string;
    external?: boolean;
  }[];
}

export type MainNavItem = NavItemWithOptionalChildren;
export type SidebarNavItem = NavItemWithChildren;

// ─── RetinaXAI domain types ───────────────────────────────────────────────────

export interface AuthUser {
  id: string;
  username: string;
  email: string;
}

export interface Patient {
  id: string;
  first_name: string;
  last_name: string;
  age: number;
  gender: 'M' | 'F';
  phone: string | null;
  address: string | null;
  medical_record_number: string;
  ocr_patient_id: string | null;
  created_at: string;
}

export interface MRIScan {
  id: string;
  patient_id: string;
  left_scan_path: string;
  right_scan_path: string;
  scanned_at: string;
  created_at: string;
  uploaded_at: string;
}

export interface BiomarkerCreMetrics {
  artery_cre?: number | null;
  vein_cre?: number | null;
  width_samples?: number | null;
}

export interface BiomarkerMetrics {
  tortuosity?: number | null;
  avr?: number | null;
  fractal_dimension?: number | null;
  vessel_density?: number | null;
  bifurcation_count?: number | null;
  bifurcation_angles?: number[] | null;
  cre?: BiomarkerCreMetrics | null;
  raw_feature_vector?: number[] | null;
}

export interface PredictionBiomarkers {
  left_eye?: BiomarkerMetrics | null;
  right_eye?: BiomarkerMetrics | null;
}

export type DRSeverity =
  | 'no_dr'
  | 'mild'
  | 'moderate'
  | 'severe'
  | 'proliferative';

export type PredictionStatus = 'pending' | 'success' | 'failed' | 'SUCCESS' | 'FAILED' | 'PENDING';

export interface Prediction {
  id: string;
  patient_id: string;
  mri_scan_id: string;
  requested_by: string;
  model_name: string;
  model_version: string;
  input_payload: Record<string, unknown>;
  output_payload: Record<string, unknown> | null;
  confidence_score: number | null;
  status: PredictionStatus;
  error_message: string | null;
  created_at: string;
}

export type ReportStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface Report {
  id: string;
  patient_id: string;
  prediction_id: string;
  generated_by: string;
  llm_model: string;
  content: string | null;
  summary: string | null;
  status: ReportStatus;
  error_message: string | null;
  report_type?: string;
  created_at: string;
}

export interface PaginatedResponse<T> {
  total: number;
  page: number;
  size: number;
  pages: number;
  items: T[];
}

export interface ApiError {
  detail: string;
  status: number;
}

export interface OCTReport {
  id: string;
  patient_id: string;
  eye: string;
  dr_grade: string | null;
  edema: boolean;
  erm_status: string | null;
  image_quality: number | null;
  thickness_center_fovea: number | null;
  thickness_average_thickness: number | null;
  thickness_total_volume_mm3: number | null;
  thickness_inner_superior?: number | null;
  thickness_inner_nasal?: number | null;
  thickness_inner_inferior?: number | null;
  thickness_inner_temporal?: number | null;
  thickness_outer_superior?: number | null;
  thickness_outer_nasal?: number | null;
  thickness_outer_inferior?: number | null;
  thickness_outer_temporal?: number | null;
  source_file?: string | null;
  created_at: string;
}
