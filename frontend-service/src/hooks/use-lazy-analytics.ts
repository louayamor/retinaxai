'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { getLLMOpsHealth, queryAnalytics } from '@/lib/api';
import type { AnalyticsQueryResponse, AnalyticsSection } from '@/lib/api';
import { ANALYTIC_QUERIES } from '@/lib/api';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000;
const STALE_TTL_MS = 4 * 60 * 1000;
const INTER_QUERY_DELAY_MS = 3000;
const PRE_FIRE_COUNT = 2;

export function useLazyAnalytics() {
  const [sections, setSections] = useState<AnalyticsSection[]>(() =>
    ANALYTIC_QUERIES.map((q) => ({
      ...q,
      response: null,
      loading: false,
      error: null,
    })),
  );
  const [llmopsOnline, setLlmopsOnline] = useState<boolean | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [paused, setPaused] = useState(false);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const observersRef = useRef<IntersectionObserver[]>([]);
  const timestampsRef = useRef<Map<string, number>>(new Map());
  const containerRef = useRef<HTMLDivElement | null>(null);

  const fetchOne = useCallback(async (q: typeof ANALYTIC_QUERIES[number]) => {
    setSections((prev) =>
      prev.map((s) => (s.key === q.key ? { ...s, loading: true, error: null } : s)),
    );
    try {
      const response = await queryAnalytics(q.question);
      timestampsRef.current.set(q.key, Date.now());
      if (response.error) {
        setSections((prev) =>
          prev.map((s) =>
            s.key === q.key
              ? { ...s, response, loading: false, error: response.error }
              : s,
          ),
        );
      } else {
        setSections((prev) =>
          prev.map((s) =>
            s.key === q.key ? { ...s, response, loading: false, error: null } : s,
          ),
        );
      }
    } catch (err) {
      setSections((prev) =>
        prev.map((s) =>
          s.key === q.key
            ? { ...s, response: null, loading: false, error: String(err).slice(0, 200) }
            : s,
        ),
      );
    }
  }, []);

  const loadAllSequential = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    const online = await getLLMOpsHealth().then((h) => h?.status === 'ok').catch(() => false);
    setLlmopsOnline(online);
    if (!online) {
      setSections((prev) =>
        prev.map((s) => ({
          ...s,
          loading: false,
          error: 'Analytics engine unavailable',
        })),
      );
      return;
    }

    for (const q of ANALYTIC_QUERIES) {
      if (abortRef.current?.signal.aborted) break;
      await fetchOne(q);
      await new Promise((r) => setTimeout(r, INTER_QUERY_DELAY_MS));
    }

    setLastUpdated(new Date());
  }, [fetchOne]);

  const loadVisible = useCallback(async (q: typeof ANALYTIC_QUERIES[number]) => {
    const ts = timestampsRef.current.get(q.key) || 0;
    if (Date.now() - ts < STALE_TTL_MS) return;
    await fetchOne(q);
    setLastUpdated(new Date());
  }, [fetchOne]);

  const setupObservers = useCallback(() => {
    observersRef.current.forEach((o) => o.disconnect());
    observersRef.current = [];

    const cards = containerRef.current?.querySelectorAll('[data-analytics-card]');
    if (!cards) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const key = (entry.target as HTMLElement).dataset.analyticsCard;
            if (!key) continue;
            observer.unobserve(entry.target);
            const q = ANALYTIC_QUERIES.find((q) => q.key === key);
            if (q) void loadVisible(q);
          }
        }
      },
      { rootMargin: '200px', threshold: 0.1 },
    );

    cards.forEach((card, index) => {
      if (index < PRE_FIRE_COUNT) return;
      observer.observe(card);
    });

    observersRef.current.push(observer);
  }, [loadVisible]);

  const refresh = useCallback(() => {
    void loadAllSequential();
  }, [loadAllSequential]);

  const retrySection = useCallback((key: string) => {
    const q = ANALYTIC_QUERIES.find((q) => q.key === key);
    if (q) void fetchOne(q);
  }, [fetchOne]);

  useEffect(() => {
    let visible = true;
    const handleVisibility = () => {
      visible = !document.hidden;
      setPaused(!visible);
    };
    document.addEventListener('visibilitychange', handleVisibility);

    const load = async () => {
      const online = await getLLMOpsHealth().then((h) => h?.status === 'ok').catch(() => false);
      setLlmopsOnline(online);
      if (!online) return;

      const allStale = ANALYTIC_QUERIES.every((q) => {
        const ts = timestampsRef.current.get(q.key) || 0;
        return Date.now() - ts > STALE_TTL_MS;
      });

      if (allStale) {
        for (const q of ANALYTIC_QUERIES.slice(0, PRE_FIRE_COUNT)) {
          await fetchOne(q);
          await new Promise((r) => setTimeout(r, INTER_QUERY_DELAY_MS));
        }
        setLastUpdated(new Date());
        requestAnimationFrame(() => setupObservers());
      }
    };

    void load();

    intervalRef.current = setInterval(() => {
      if (visible) {
        for (const q of ANALYTIC_QUERIES) {
          void loadVisible(q);
        }
        setLastUpdated(new Date());
      }
    }, REFRESH_INTERVAL_MS);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      if (intervalRef.current) clearInterval(intervalRef.current);
      abortRef.current?.abort();
      observersRef.current.forEach((o) => o.disconnect());
    };
  }, [fetchOne, loadVisible, setupObservers]);

  return {
    sections,
    llmopsOnline,
    lastUpdated,
    paused,
    refresh,
    retrySection,
    containerRef,
  };
}
