'use client';

import Image from 'next/image';
import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

interface PageHeroProps {
  title: string;
  description?: string;
  imageUrl?: string;
  imageAlt?: string;
  gradient?: string;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  imagePosition?: 'left' | 'right';
}

const DEFAULT_GRADIENT = 'from-[#0a2e3e] via-[#0d3a4c] to-[#104a5e]';

export function PageHero({
  title,
  description,
  imageUrl,
  imageAlt = 'Hero image',
  gradient = DEFAULT_GRADIENT,
  actions,
  children,
  className,
  imagePosition = 'right',
}: PageHeroProps) {
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-2xl bg-gradient-to-r',
        gradient,
        'p-6 md:p-10 text-white',
        className
      )}
    >
      {imageUrl && (
        <div
          className={cn(
            'absolute top-0 h-full w-1/3 opacity-15',
            imagePosition === 'right' ? 'right-0' : 'left-0'
          )}
        >
          <Image
            src={imageUrl}
            alt={imageAlt}
            fill
            className='object-cover'
            priority
            unoptimized
          />
        </div>
      )}

      <div
        className={cn(
          'absolute -bottom-10 -left-10 h-32 w-32 rounded-full bg-[var(--brand-teal)]/10 blur-3xl',
          imagePosition === 'left' && '-left-20'
        )}
      />
      <div
        className={cn(
          'absolute top-10 h-24 w-24 rounded-full bg-[var(--brand-gold)]/10 blur-2xl',
          imagePosition === 'left' ? 'left-20' : 'right-20'
        )}
      />

      <div className='relative z-10 flex items-start justify-between gap-8'>
        <div className='space-y-2'>
          <h1 className='text-2xl md:text-3xl font-bold tracking-tight'>
            {title}
          </h1>
          {description && (
            <p className='text-base md:text-lg text-white/80 max-w-2xl'>
              {description}
            </p>
          )}
          {children}
        </div>

        {actions && (
          <div className='hidden md:flex items-center gap-3'>
            {actions}
          </div>
        )}
      </div>

      {actions && (
        <div className='flex md:hidden mt-4'>
          {actions}
        </div>
      )}
    </div>
  );
}
