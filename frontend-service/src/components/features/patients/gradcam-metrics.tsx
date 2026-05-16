'use client';

import { Badge } from '@/components/ui/badge';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';

export type GradCAMRegion = {
  name?: string;
  intensity?: number;
  area?: number;
  center_x?: number;
  center_y?: number;
  saliency_score?: number;
};

export type GradCAMHotspot = {
  region?: string;
  intensity?: number;
  rank?: number;
};

export function GradCAMMetricsBlock({
  leftRegions,
  rightRegions,
  leftHotspots,
  rightHotspots,
}: {
  leftRegions?: GradCAMRegion[];
  rightRegions?: GradCAMRegion[];
  leftHotspots?: GradCAMHotspot[];
  rightHotspots?: GradCAMHotspot[];
}) {
  const hasAny =
    (leftRegions?.length ?? 0) > 0 ||
    (rightRegions?.length ?? 0) > 0 ||
    (leftHotspots?.length ?? 0) > 0 ||
    (rightHotspots?.length ?? 0) > 0;

  if (!hasAny) {
    return null;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      <GradCAMEyeMetrics
        title="Left Eye Metrics"
        accentClassName="border-[var(--brand-teal)]/30 bg-[var(--brand-teal)]/5"
        titleClassName="text-[var(--brand-teal)]"
        regions={leftRegions}
        hotspots={leftHotspots}
      />
      <GradCAMEyeMetrics
        title="Right Eye Metrics"
        accentClassName="border-[var(--brand-gold)]/30 bg-[var(--brand-gold)]/5"
        titleClassName="text-[var(--brand-gold)]"
        regions={rightRegions}
        hotspots={rightHotspots}
      />
    </div>
  );
}

function GradCAMEyeMetrics({
  title,
  accentClassName,
  titleClassName,
  regions,
  hotspots,
}: {
  title: string;
  accentClassName: string;
  titleClassName: string;
  regions?: GradCAMRegion[];
  hotspots?: GradCAMHotspot[];
}) {
  const regionList = (regions ?? []).slice(0, 4);
  const hotspotList = (hotspots ?? []).slice(0, 3);

  if (regionList.length === 0 && hotspotList.length === 0) {
    return null;
  }

  return (
    <div className={`rounded-lg border p-3 ${accentClassName}`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className={`font-semibold text-base ${titleClassName}`}>{title}</h4>
        {hotspotList.length > 0 && (
          <span className="text-xs uppercase tracking-widest text-muted-foreground">
            Top hotspots
          </span>
        )}
      </div>

      {hotspotList.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {hotspotList.map((hotspot, idx) => (
            <Badge
              key={`${hotspot.region ?? 'hotspot'}-${idx}`}
              variant="outline"
              className="text-xs bg-white/60 dark:bg-black/20"
            >
              #{hotspot.rank ?? idx + 1} {formatGradCAMLabel(hotspot.region)}
            </Badge>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {regionList.map((region, idx) => (
          <div key={`${region.name ?? 'region'}-${idx}`} className="rounded-md border border-border bg-card/80 p-3 shadow-sm">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-sm font-semibold capitalize leading-tight">
                  {formatGradCAMLabel(region.name)}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  ({region.center_x ?? '—'}, {region.center_y ?? '—'})
                </p>
              </div>
              <Badge variant="secondary" className="text-xs h-6 px-2">
                {(typeof region.intensity === 'number' ? (region.intensity * 100).toFixed(0) : '—')}%
              </Badge>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
              <MetricPill label="Area" value={region.area != null ? `${region.area}px` : '—'} />
              <MetricPill label="Saliency" value={formatDecimal(region.saliency_score)} />
            </div>
          </div>
        ))}
      </div>

      {regionList.length > 0 && (
        <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
          <div className="rounded-md border border-border bg-card/70 p-2.5">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs uppercase tracking-widest text-muted-foreground">Intensity Chart</p>
              <p className="text-xs text-muted-foreground">Top {regionList.length}</p>
            </div>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regionList.map((region) => ({
                  name: formatGradCAMLabel(region.name),
                  intensity: region.intensity ?? 0,
                }))} layout="vertical" margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" horizontal={false} />
                  <XAxis type="number" hide />
                  <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(0)}%`, 'Intensity']} />
                  <Bar dataKey="intensity" radius={[0, 6, 6, 0]}>
                    {regionList.map((region, idx) => (
                      <Cell
                        key={`${region.name ?? 'region'}-${idx}`}
                        fill={idx % 2 === 0 ? 'rgba(32, 189, 190, 0.85)' : 'rgba(200, 169, 81, 0.85)'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-md border border-border bg-card/70 p-2.5">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs uppercase tracking-widest text-muted-foreground">Spatial Map</p>
              <p className="text-xs text-muted-foreground">X/Y position</p>
            </div>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                  <XAxis
                    type="number"
                    dataKey="center_x"
                    name="x"
                    tick={{ fontSize: 10 }}
                    domain={[0, 224]}
                    allowDataOverflow
                  />
                  <YAxis
                    type="number"
                    dataKey="center_y"
                    name="y"
                    tick={{ fontSize: 10 }}
                    domain={[0, 224]}
                    allowDataOverflow
                  />
                  <ZAxis type="number" dataKey="saliency_score" range={[40, 140]} />
                  <Tooltip
                    formatter={(value: number, name: string) => {
                      if (name === 'saliency_score') return [value.toFixed(3), 'Saliency'];
                      return [value, name];
                    }}
                  />
<Scatter data={regionList} fill="rgba(32, 189, 190, 0.8)">
                    {regionList.map((region, idx) => (
                      <Cell
                        key={`${region.name ?? 'region'}-${idx}`}
                        fill={idx % 2 === 0 ? 'rgba(32, 189, 190, 0.85)' : 'rgba(200, 169, 81, 0.85)'}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-muted/40 px-2.5 py-1.5">
      <p className="uppercase tracking-widest text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-medium text-foreground truncate">{value}</p>
    </div>
  );
}

function formatGradCAMLabel(value?: string) {
  if (!value) return 'Unknown';
  return value.replace(/_/g, ' ');
}

function formatDecimal(value?: number) {
  return typeof value === 'number' ? value.toFixed(3) : '—';
}
