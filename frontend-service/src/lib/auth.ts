const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export type UserRole = 'doctor' | 'engineer' | 'admin';

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  role: UserRole;
}

function deleteCookie(name: string): void {
  if (typeof document === 'undefined') return;
  const isProduction = typeof window !== 'undefined' && window.location.hostname !== 'localhost';
  const secure = isProduction ? '; secure' : '';
  document.cookie = name + '=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; samesite=lax' + secure;
}

export function clearTokens(): void {
  deleteCookie('rxa_access_token');
  deleteCookie('rxa_refresh_token');
}

let refreshing = false;
let refreshPromise: Promise<boolean> | null = null;

async function doRefresh(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function tryRefresh(): Promise<boolean> {
  if (refreshing && refreshPromise) {
    return refreshPromise;
  }
  refreshing = true;
  refreshPromise = doRefresh();
  try {
    return await refreshPromise;
  } finally {
    refreshing = false;
    refreshPromise = null;
  }
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const makeRequest = () =>
    fetch(`${API_BASE_URL}${path}`, {
      ...init,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers as Record<string, string>),
      },
    });

  let res = await makeRequest();

  if (res.status === 401) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      res = await makeRequest();
    } else {
      window.location.href = '/auth/login';
      throw new Error('Not authenticated');
    }
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(body.detail ?? 'Request failed', res.status);
  }

  return res.json();
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
