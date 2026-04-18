'use client';

import { useEffect, useState } from 'react';
import { PieGraph } from '@/features/overview/components/pie-graph';

export default function Stats() {
  const [data, setData] = useState<Record<string, number> | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' })
      .then(r => r.json())
      .then(json => {
        setData(json.gender_distribution || undefined);
      })
      .catch(() => {
        setData(undefined);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return <PieGraph data={data} loading={loading} />;
}
