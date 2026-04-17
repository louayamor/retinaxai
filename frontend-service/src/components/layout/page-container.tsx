'use client';

import React from 'react';
import { Heading } from '@/components/ui/heading';
import { LoadingState } from '@/components/ui/empty-state';
import { EmptyState } from '@/components/ui/empty-state';
import { Button } from '@/components/ui/button';
import type { InfobarContent } from '@/components/ui/infobar';
import { cn } from '@/lib/utils';

interface PageContainerProps {
  children: React.ReactNode;
  isLoading?: boolean;
  access?: boolean;
  accessFallback?: React.ReactNode;
  pageTitle?: string;
  pageDescription?: string;
  infoContent?: InfobarContent;
  pageHeaderAction?: React.ReactNode;
  className?: string;
  loadingMessage?: string;
  empty?: {
    show: boolean;
    icon?: React.ComponentType<{ className?: string }>;
    title?: string;
    description?: string;
    action?: {
      label: string;
      onClick: () => void;
    };
  };
}

function PageSkeleton() {
  return (
    <div className='flex flex-1 flex-col gap-4 p-4 md:px-6'>
      <div className='flex items-center justify-between'>
        <div>
          <div className='bg-muted mb-2 h-8 w-48 rounded animate-pulse' />
          <div className='bg-muted h-4 w-64 rounded animate-pulse' />
        </div>
      </div>
      <div className='bg-muted mt-6 h-40 w-full rounded-lg animate-pulse' />
      <div className='bg-muted h-40 w-full rounded-lg animate-pulse' />
    </div>
  );
}

export default function PageContainer({
  children,
  isLoading = false,
  access = true,
  accessFallback,
  pageTitle,
  pageDescription,
  infoContent,
  pageHeaderAction,
  className,
  loadingMessage = 'Loading...',
  empty,
}: PageContainerProps) {
  if (!access) {
    return (
      <div className='flex flex-1 items-center justify-center p-4 md:px-6'>
        {accessFallback ?? (
          <div className='text-muted-foreground text-center text-lg'>
            You do not have access to this page.
          </div>
        )}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={cn('flex flex-col p-4 md:px-6 w-full', className)}>
        {pageTitle && (
          <div className='mb-4 flex items-start justify-between'>
            <Heading
              title={pageTitle}
              description={pageDescription}
              infoContent={infoContent}
            />
            {pageHeaderAction && <div>{pageHeaderAction}</div>}
          </div>
        )}
        <PageSkeleton />
      </div>
    );
  }

  if (empty?.show) {
    const EmptyIcon = empty.icon;
    return (
      <div className={cn('flex flex-col p-4 md:px-6 w-full', className)}>
        {pageTitle && (
          <div className='mb-4 flex items-start justify-between'>
            <Heading
              title={pageTitle}
              description={pageDescription}
              infoContent={infoContent}
            />
            {pageHeaderAction && <div>{pageHeaderAction}</div>}
          </div>
        )}
        <div className='flex-1'>
          <EmptyState
            icon={EmptyIcon}
            title={empty.title ?? 'No data'}
            description={empty.description}
            action={empty.action}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col p-4 md:px-6 w-full', className)}>
      {pageTitle && (
        <div className='mb-4 flex items-start justify-between'>
          <Heading
            title={pageTitle}
            description={pageDescription}
            infoContent={infoContent}
          />
          {pageHeaderAction && <div>{pageHeaderAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
