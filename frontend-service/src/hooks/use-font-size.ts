'use client';

import { useCallback, useEffect, useState } from 'react';

type FontSizeLevel = 'normal' | 'large' | 'xlarge';

const LEVEL_MAP: Record<FontSizeLevel, number> = {
  normal: 100,
  large: 120,
  xlarge: 140,
};

const STORAGE_KEY = 'retinaxai-font-size';

export function useFontSize() {
  const [level, setLevelState] = useState<FontSizeLevel>('normal');

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as FontSizeLevel | null;
    if (stored && stored in LEVEL_MAP) {
      setLevelState(stored);
    }
  }, []);

  const setLevel = useCallback((next: FontSizeLevel) => {
    setLevelState(next);
    localStorage.setItem(STORAGE_KEY, next);
    const pct = LEVEL_MAP[next];
    document.documentElement.style.fontSize = `${pct}%`;
  }, []);

  const cycle = useCallback(() => {
    const order: FontSizeLevel[] = ['normal', 'large', 'xlarge'];
    const idx = order.indexOf(level);
    setLevel(order[(idx + 1) % order.length]);
  }, [level, setLevel]);

  return { level, setLevel, cycle };
}
