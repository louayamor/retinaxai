'use client';

import React from 'react';
import { Heading } from '@/components/ui/heading';
import { LoadingState } from '@/components/ui/empty-state';
import { EmptyState } from '@/components/ui/empty-state';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface PageContainerProps {
  children: React.ReactNode;
  isLoading?: boolean;
  access?: boolean;
  accessFallback?: React.ReactNode;
  pageTitle?: string;
  pageDescription?: string;
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
        <div className='space-y-2'>
          <Skeleton className='h-5 w-40' />
          <Skeleton className='h-3 w-56' />
        </div>
      </div>
      <div className='grid gap-4 md:grid-cols-2 lg:grid-cols-4'>
        <Skeleton className='h-24' />
        <Skeleton className='h-24' />
        <Skeleton className='h-24' />
        <Skeleton className='h-24' />
      </div>
      <Skeleton className='h-64 w-full' />
      <Skeleton className='h-48 w-full' />
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
  pageHeaderAction,
  className,
  loadingMessage = 'Loading...',
  empty,
}: PageContainerProps) {
  if (!access) {
    return (
      <div className='flex flex-1 items-center justify-center p-3 md:p-5'>
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
      <div className={cn('flex flex-col p-3 md:p-5 w-full', className)}>
        {pageTitle && (
          <div className='mb-3 flex items-start justify-between'>
            <Heading
              title={pageTitle}
              description={pageDescription}
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
      <div className={cn('flex flex-col p-3 md:p-5 w-full', className)}>
        {pageTitle && (
          <div className='mb-3 flex items-start justify-between'>
            <Heading
              title={pageTitle}
              description={pageDescription}
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
    <div className={cn('flex flex-col p-4 md:p-6 w-full', className)}>
      {pageTitle && (
        <div className='mb-4 flex items-start justify-between'>
          <Heading
            title={pageTitle}
            description={pageDescription}
          />
          {pageHeaderAction && <div>{pageHeaderAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
