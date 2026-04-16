'use client';

import { useEffect, useState } from 'react';

interface PatientFiltersProps {
  total: number;
  maleCount: number;
  femaleCount: number;
  value: 'all' | 'M' | 'F';
  onChange: (value: 'all' | 'M' | 'F') => void;
}

export function PatientFilters({
  total,
  maleCount,
  femaleCount,
  value,
  onChange,
}: PatientFiltersProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="flex items-center gap-2">
        <div className="h-9 w-16 bg-muted animate-pulse rounded-full" />
        <div className="h-9 w-20 bg-muted animate-pulse rounded-full" />
        <div className="h-9 w-20 bg-muted animate-pulse rounded-full" />
      </div>
    );
  }

  const filters = [
    { key: 'all' as const, label: 'All', count: total },
    { key: 'M' as const, label: 'Male', count: maleCount },
    { key: 'F' as const, label: 'Female', count: femaleCount },
  ];

  return (
    <div className="flex items-center gap-2">
      {filters.map((filter) => (
        <button
          key={filter.key}
          onClick={() => onChange(filter.key)}
          className={`
            inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium
            transition-all duration-200
            ${
              value === filter.key
                ? 'bg-[var(--brand-teal)] text-white shadow-sm'
                : 'bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground'
            }
          `}
        >
          {filter.label}
          <span
            className={`
              text-xs px-1.5 py-0.5 rounded-full
              ${
                value === filter.key
                  ? 'bg-white/20 text-white'
                  : 'bg-muted text-muted-foreground'
              }
            `}
          >
            {filter.count}
          </span>
        </button>
      ))}
    </div>
  );
}
