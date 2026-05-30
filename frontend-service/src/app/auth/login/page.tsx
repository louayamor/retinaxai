'use client';

import { useAuth } from '@/providers/auth-context';
import SignInView from '@/features/auth/components/sign-in-view';
import type { UserRole } from '@/lib/auth';

const ROLE_REDIRECT: Record<UserRole, string> = {
  doctor: '/dashboard/clinical',
  engineer: '/dashboard/engineering',
  admin: '/dashboard/admin',
};

export default function LoginPage() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--brand-teal)]" />
      </div>
    );
  }

  if (user) {
    if (typeof window !== 'undefined') {
      window.location.href = ROLE_REDIRECT[user.role] || '/dashboard/overview';
    }
    return null;
  }

  return <SignInView />;
}
