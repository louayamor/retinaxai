export const DESIGN_TOKENS = {
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
    '2xl': '1.5rem',
  },
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
  },
  transitions: {
    fast: '150ms',
    normal: '200ms',
    slow: '300ms',
  },
} as const;

export const CARD_PADDING = {
  compact: 'p-3 md:p-4',
  default: 'p-4 md:p-6',
  comfortable: 'p-6 md:p-8',
  spacious: 'p-8 md:p-10',
} as const;

export const TEXT_SIZES = {
  xs: 'text-xs',
  sm: 'text-sm',
  base: 'text-base',
  lg: 'text-lg',
  xl: 'text-xl',
  '2xl': 'text-2xl',
  '3xl': 'text-3xl',
} as const;

export const GRID_BREAKPOINTS = {
  sm: 'grid-cols-1',
  md: 'md:grid-cols-2',
  lg: 'lg:grid-cols-3',
  xl: 'xl:grid-cols-4',
} as const;

export const DR_GRADE_COLORS = {
  no_dr: '#10b981',
  mild: '#06b6d4',
  moderate: '#f59e0b',
  severe: '#f97316',
  proliferative: '#f43f5e',
} as const;

export const DR_GRADE_LABELS = [
  'No DR',
  'Mild',
  'Moderate',
  'Severe',
  'Proliferative',
] as const;

export const STATUS_COLORS = {
  success: 'bg-green-500',
  warning: 'bg-yellow-500',
  error: 'bg-red-500',
  info: 'bg-blue-500',
  neutral: 'bg-gray-500',
} as const;

export const BRAND_COLORS = {
  primary: 'var(--brand-teal)',
  secondary: 'var(--brand-gold)',
} as const;

export const MIN_FONT_SIZE = {
  label: 'text-xs',
  body: 'text-sm',
  title: 'text-base',
  heading: 'text-lg',
  page: 'text-2xl',
} as const;

export function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ');
}
