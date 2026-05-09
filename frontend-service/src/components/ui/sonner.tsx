'use client';

import { useTheme } from 'next-themes';
import { Toaster as Sonner, ToasterProps } from 'sonner';

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = 'system' } = useTheme();

  return (
    <Sonner
      theme={theme as ToasterProps['theme']}
      className='toaster group'
      style={
        {
          '--normal-bg': 'var(--popover)',
          '--normal-text': 'var(--popover-foreground)',
          '--normal-border': 'var(--border)',
          '--success-bg': '#22c55e',
          '--success-text': '#ffffff',
          '--success-border': '#22c55e',
          '--error-bg': '#ef4444',
          '--error-text': '#ffffff',
          '--error-border': '#ef4444',
          '--warning-bg': '#f59e0b',
          '--warning-text': '#ffffff',
          '--warning-border': '#f59e0b',
          '--info-bg': '#3b82f6',
          '--info-text': '#ffffff',
          '--info-border': '#3b82f6',
        } as React.CSSProperties
      }
      {...props}
    />
  );
};

export { Toaster };
