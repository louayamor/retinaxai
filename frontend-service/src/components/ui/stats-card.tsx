'use client';

import { Card, CardContent } from '@/components/ui/card';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  color?: string;
  size?: 'compact' | 'default' | 'large';
  trendValue?: string;
}

export function StatsCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = 'var(--brand-teal)',
  size = 'default',
  trendValue,
}: StatsCardProps) {
  const sizeClasses = {
    compact: {
      container: 'p-3',
      icon: 'h-8 w-8',
      iconSize: 'h-4 w-4',
      value: 'text-xl',
      padding: 'gap-3',
    },
    default: {
      container: 'p-4',
      icon: 'h-12 w-12',
      iconSize: 'h-5 w-5',
      value: 'text-2xl',
      padding: 'gap-4',
    },
    large: {
      container: 'p-5',
      icon: 'h-14 w-14',
      iconSize: 'h-6 w-6',
      value: 'text-3xl',
      padding: 'gap-5',
    },
  };

  const classes = sizeClasses[size];

  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
  const trendColor = trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : 'text-muted-foreground';

  return (
    <Card className='hover:shadow-md transition-shadow duration-200'>
      <CardContent className={cn('flex items-center', classes.padding, classes.container)}>
        <div
          className={cn('flex items-center justify-center rounded-xl shrink-0', classes.icon)}
          style={{ backgroundColor: `${color}20` }}
        >
          <Icon className={classes.iconSize} style={{ color }} />
        </div>
        <div className='min-w-0 flex-1'>
          <p className='text-sm text-muted-foreground truncate'>{title}</p>
          <div className='flex items-baseline gap-2'>
            <p className={cn('font-bold truncate', classes.value)}>{value}</p>
            {trend && (
              <div className={cn('flex items-center gap-1', trendColor)}>
                <TrendIcon className='h-3 w-3' />
                {trendValue && <span className='text-xs'>{trendValue}</span>}
              </div>
            )}
          </div>
          {subtitle && (
            <p className='text-xs text-muted-foreground truncate'>{subtitle}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
