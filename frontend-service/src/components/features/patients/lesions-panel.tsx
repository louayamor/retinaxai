'use client';

import React from 'react';
import { Eye } from 'lucide-react';

type LesionCluster = {
  class?: string;
  class_name?: string;
  centroid_x?: number;
  centroid_y?: number;
  area?: number;
  [k: string]: any;
};

export default function LesionsPanel({ latestPrediction, onClusterClick }: { latestPrediction: any | null; onClusterClick?: (eye: 'left'|'right', cluster: any) => void }) {
  const leftLesions = (latestPrediction?.output_payload?.lesions_left as Record<string, number>) || {};
  const rightLesions = (latestPrediction?.output_payload?.lesions_right as Record<string, number>) || {};
  const leftClusters = (latestPrediction?.output_payload?.lesion_clusters_left as LesionCluster[]) || [];
  const rightClusters = (latestPrediction?.output_payload?.lesion_clusters_right as LesionCluster[]) || [];

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-md border border-border bg-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-foreground flex items-center gap-2">
              <Eye className="h-4 w-4 text-[var(--brand-teal)]" />
              Left Eye (OS)
            </h4>
            <span className="text-xs font-mono font-bold text-white px-1.5 py-0.5 rounded bg-[var(--brand-teal)]">
              {leftClusters.length} clusters
            </span>
          </div>

          <div className="mb-3">
            <p className="text-xs text-muted-foreground mb-2">Lesion pixel counts</p>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(leftLesions).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between border rounded px-2 py-1 bg-muted/5">
                  <span className="text-sm text-muted-foreground capitalize">{k}</span>
                  <span className="text-sm font-mono font-semibold">{v}</span>
                </div>
              ))}
              {Object.keys(leftLesions).length === 0 && <div className="text-xs text-muted-foreground">—</div>}
            </div>
          </div>

          <div>
            <p className="text-xs text-muted-foreground mb-2">Sample clusters</p>
            <div className="space-y-2 max-h-48 overflow-auto">
              {leftClusters.slice(0, 12).map((c, idx) => (
                <button key={idx} onClick={() => onClusterClick?.('left', c)} className="text-xs text-muted-foreground w-full text-left hover:bg-muted/10 px-1 py-0.5 rounded">
                  {c.class || c.class_name || 'lesion'} — area: {c.area ?? '-'} — ({c.centroid_x ?? c.center_x ?? '-'}, {c.centroid_y ?? c.center_y ?? '-'})
                </button>
              ))}
              {leftClusters.length === 0 && <div className="text-xs text-muted-foreground">No clusters</div>}
            </div>
          </div>
        </div>

        <div className="rounded-md border border-border bg-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-foreground flex items-center gap-2">
              <Eye className="h-4 w-4 text-[var(--brand-gold)]" />
              Right Eye (OD)
            </h4>
            <span className="text-xs font-mono font-bold text-white px-1.5 py-0.5 rounded bg-[var(--brand-gold)]">
              {rightClusters.length} clusters
            </span>
          </div>

          <div className="mb-3">
            <p className="text-xs text-muted-foreground mb-2">Lesion pixel counts</p>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(rightLesions).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between border rounded px-2 py-1 bg-muted/5">
                  <span className="text-sm text-muted-foreground capitalize">{k}</span>
                  <span className="text-sm font-mono font-semibold">{v}</span>
                </div>
              ))}
              {Object.keys(rightLesions).length === 0 && <div className="text-xs text-muted-foreground">—</div>}
            </div>
          </div>

          <div>
            <p className="text-xs text-muted-foreground mb-2">Sample clusters</p>
            <div className="space-y-2 max-h-48 overflow-auto">
              {rightClusters.slice(0, 12).map((c, idx) => (
                <button key={idx} onClick={() => onClusterClick?.('right', c)} className="text-xs text-muted-foreground w-full text-left hover:bg-muted/10 px-1 py-0.5 rounded">
                  {c.class || c.class_name || 'lesion'} — area: {c.area ?? '-'} — ({c.centroid_x ?? c.center_x ?? '-'}, {c.centroid_y ?? c.center_y ?? '-'})
                </button>
              ))}
              {rightClusters.length === 0 && <div className="text-xs text-muted-foreground">No clusters</div>}
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-md border border-border bg-card p-4">
        <p className="text-sm font-semibold mb-2">Cluster heatmap</p>
        <p className="text-xs text-muted-foreground">Click any cluster to jump to the GradCAM view and highlight it.</p>
      </div>
    </div>
  );
}
