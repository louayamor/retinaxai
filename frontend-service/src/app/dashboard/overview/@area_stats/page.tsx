'use client';

import { useEffect, useState } from 'react';
import { AreaGraph } from '@/features/overview/components/area-graph';

export default function AreaStats() {
  const [data, setData] = useState<Record<number, number> | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${BASE}/api/v1/dashboard/stats`, { credentials: 'include' })
      .then(r => r.json())
      .then(json => {
        setData(json.severity_distribution || undefined);
      })
      .catch(() => {
        setData(undefined);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return <AreaGraph data={data} loading={loading} />;
}
