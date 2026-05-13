'use client';

import * as React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import type { AnalyticsChartSpec } from '@/lib/api';

interface AIChartRendererProps {
  spec: AnalyticsChartSpec;
  height?: number;
}

const FALLBACK_COLORS = [
  '#3b82f6',
  '#22c55e',
  '#eab308',
  '#ef4444',
  '#8b5cf6',
  '#f97316',
  '#06b6d4',
  '#ec4899',
];

export function AIChartRenderer({ spec, height = 250 }: AIChartRendererProps) {
  const rawColors = spec.config?.colors;
  const colors: string[] =
    Array.isArray(rawColors) && rawColors.length > 0
      ? (rawColors as string[])
      : FALLBACK_COLORS;

  const rawDataKeys = spec.config?.dataKeys;
  const dataKeys: string[] =
    Array.isArray(rawDataKeys) && rawDataKeys.length > 0
      ? (rawDataKeys as string[])
      : spec.type === 'bar' && spec.data.length > 0
        ? Object.keys(spec.data[0]).filter((k) => k !== 'name')
        : spec.type === 'line' && spec.data.length > 0
          ? Object.keys(spec.data[0]).filter((k) => k !== 'name')
          : ['value'];

  const xKey: string = (spec.config?.xKey as string) || 'name';

  const chartConfig = colors.reduce<Record<string, { label: string; color: string }>>(
    (acc, color, i) => {
      const key = dataKeys?.[i] ?? dataKeys?.[0] ?? 'value';
      acc[key] = { label: key, color };
      return acc;
    },
    {} as Record<string, { label: string; color: string }>,
  );

  switch (spec.type) {
    case 'pie':
      return (
        <ChartContainer config={chartConfig} className="mx-auto" style={{ height }}>
          <PieChart>
            <Pie
              data={spec.data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) =>
                `${name}: ${(percent * 100).toFixed(0)}%`
              }
              outerRadius={height * 0.35}
              innerRadius={height * 0.18}
              dataKey="value"
              stroke="var(--background)"
              strokeWidth={2}
            >
              {spec.data.map((_entry: Record<string, unknown>, i: number) => (
                <Cell key={i} fill={colors[i % colors.length]} />
              ))}
            </Pie>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
          </PieChart>
        </ChartContainer>
      );
    case 'bar':
      return (
        <ChartContainer config={chartConfig} className="aspect-auto w-full" style={{ height }}>
          <BarChart data={spec.data} margin={{ left: 8, right: 8 }}>
            <CartesianGrid
              vertical={false}
              stroke="hsl(var(--border) / 0.55)"
            />
            <XAxis
              dataKey={xKey}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tick={{ fontSize: 11 }}
            />
            <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
            <ChartTooltip
              cursor={{ fill: 'var(--primary)', opacity: 0.1 }}
              content={<ChartTooltipContent indicator="dot" />}
            />
            {dataKeys.map((dk: string, i: number) => (
              <Bar
                key={dk}
                dataKey={dk}
                fill={colors[i % colors.length]}
                radius={[6, 6, 0, 0]}
              />
            ))}
          </BarChart>
        </ChartContainer>
      );
    case 'line':
      return (
        <ChartContainer config={chartConfig} className="aspect-auto w-full" style={{ height }}>
          <LineChart data={spec.data} margin={{ left: 8, right: 8 }}>
            <CartesianGrid
              vertical={false}
              stroke="hsl(var(--border) / 0.55)"
            />
            <XAxis
              dataKey={xKey}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tick={{ fontSize: 11 }}
            />
            <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
            <ChartTooltip
              cursor={{ stroke: 'var(--border)', strokeWidth: 1 }}
              content={<ChartTooltipContent indicator="dot" />}
            />
            {dataKeys.map((dk: string, i: number) => (
              <Line
                key={dk}
                type="monotone"
                dataKey={dk}
                stroke={colors[i % colors.length]}
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ChartContainer>
      );
    case 'area':
      return (
        <ChartContainer config={chartConfig} className="aspect-auto w-full" style={{ height }}>
          <AreaChart data={spec.data} margin={{ left: 8, right: 8 }}>
            <CartesianGrid
              vertical={false}
              stroke="hsl(var(--border) / 0.55)"
            />
            <XAxis
              dataKey={xKey}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tick={{ fontSize: 11 }}
            />
            <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
            <ChartTooltip
              cursor={{ stroke: 'var(--border)', strokeWidth: 1 }}
              content={<ChartTooltipContent indicator="dot" />}
            />
            {dataKeys.map((dk: string, i: number) => (
              <Area
                key={dk}
                type="monotone"
                dataKey={dk}
                stroke={colors[i % colors.length]}
                fill={colors[i % colors.length]}
                fillOpacity={0.15}
              />
            ))}
          </AreaChart>
        </ChartContainer>
      );
    case 'radar':
      return (
        <ChartContainer config={chartConfig} className="mx-auto" style={{ height }}>
          <RadarChart data={spec.data}>
            <PolarGrid stroke="hsl(var(--border) / 0.55)" />
            <PolarAngleAxis
              dataKey={xKey}
              tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
            />
            <PolarRadiusAxis
              tick={{ fill: 'var(--muted-foreground)', fontSize: 9 }}
            />
            {dataKeys.map((dk: string, i: number) => (
              <Radar
                key={dk}
                name={dk}
                dataKey={dk}
                stroke={colors[i % colors.length]}
                fill={colors[i % colors.length]}
                fillOpacity={0.3}
              />
            ))}
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
          </RadarChart>
        </ChartContainer>
      );
    case 'table':
      return (
        <div className="overflow-x-auto" style={{ maxHeight: height }}>
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Name</th>
                <th className="px-3 py-2 text-right font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {spec.data.map((row: Record<string, unknown>, i: number) => (
                <tr key={i} className="border-b last:border-0">
                <td className="px-3 py-2">{String(row[xKey] ?? row.name ?? row.label ?? '')}</td>
                <td className="px-3 py-2 text-right font-mono text-sm">
                  {String(row.value ?? row[dataKeys?.[0] ?? 'value'] ?? '')}
                </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    default:
      return (
        <div className="flex items-center justify-center text-sm text-muted-foreground" style={{ height }}>
          Unsupported chart type: {spec.type}
        </div>
      );
  }
}
