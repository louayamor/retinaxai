'use client';

import { useAuth } from '@/providers/auth-context';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

const ROLE_ROUTE: Record<string, string> = {
  doctor: '/dashboard/clinical',
  engineer: '/dashboard/engineering',
  admin: '/dashboard/admin',
};

export default function DashboardPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace('/auth/login');
      return;
    }
    router.replace(ROLE_ROUTE[user.role] ?? '/dashboard/clinical');
  }, [user, loading, router]);

  return null;
}
