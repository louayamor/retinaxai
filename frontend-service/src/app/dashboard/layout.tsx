import KBar from '@/components/kbar';
import AppSidebar from '@/components/layout/app-sidebar';
import { Breadcrumbs } from '@/components/breadcrumbs';
import Header from '@/components/layout/header';
import { InfoSidebar } from '@/components/layout/info-sidebar';
import { InfobarProvider } from '@/components/ui/infobar';
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
        <InfobarProvider defaultOpen={false}>
          <AppSidebar />
          <SidebarInset>
            <Header />
            <main className='flex-1 overflow-auto page-transition'>
              <div className='px-6 pt-3 pb-1'>
                <Breadcrumbs />
              </div>
              {children}
            </main>
          </SidebarInset>
          <InfoSidebar side='right' />
        </InfobarProvider>
      </SidebarProvider>
    </KBar>
  );
}
