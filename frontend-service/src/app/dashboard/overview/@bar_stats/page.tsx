'use client';

import { useEffect, useState } from 'react';
import { BarGraph } from '@/features/overview/components/bar-graph';

interface TimelineData {
  date: string;
  predictions: number;
}

export default function BarStats() {
  const [data, setData] = useState<TimelineData[] | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' })
      .then(r => r.json())
      .then(json => {
        setData(json.predictions_timeline || undefined);
      })
      .catch(() => {
        setData(undefined);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return <BarGraph data={data} loading={loading} />;
}
