import KBar from '@/components/kbar';
import { SessionTimeoutDialog } from '@/components/session-timeout-dialog';
import AppSidebar from '@/components/layout/app-sidebar';
import { Breadcrumbs } from '@/components/breadcrumbs';
import Header from '@/components/layout/header';
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar';
import type { Metadata } from 'next';
import { cookies } from 'next/headers';

export const metadata: Metadata = {
  title: 'RetinaXAI Dashboard',
  description: 'RetinaXAI clinical dashboard',
  robots: {
    index: false,
    follow: false
  }
};

export default async function DashboardLayout({
  children
}: {
  children: React.ReactNode;
}) {
  const cookieStore = await cookies();
  const defaultOpen = true;

  return (
    <KBar>
      <SidebarProvider defaultOpen={defaultOpen}>
        <AppSidebar />
        <SidebarInset>
          <Header />
          <main id='main-content' className='flex-1 overflow-auto page-transition'>
            <div className='px-6 pt-3 pb-1'>
              <Breadcrumbs />
            </div>
            <SessionTimeoutDialog />
            {children}
          </main>
        </SidebarInset>
      </SidebarProvider>
    </KBar>
  );
}
