import type { DriftFile } from "./data";
import { formatMonth } from "./format";

/**
 * Build an SVG line chart of monthly MAE plus optional signal overlays.
 * Pure function: returns SVG markup; main.ts injects it.
 */

export interface ChartSeries {
  name: string;
  values: number[];
  color: string;
}

export interface ChartModel {
  width: number;
  height: number;
  months: string[];
  series: ChartSeries[];
  /** polyline point strings per series, in SVG coords */
  paths: { name: string; color: string; points: string }[];
  yMin: number;
  yMax: number;
  /** indices of degraded months for tick markers */
  degradedIdx: number[];
}

const W = 260;
const H = 110;
const PAD_L = 34;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 20;

export function buildDriftChart(drift: DriftFile): ChartModel {
  const months = drift.months;
  const maeVals = months.map((m) => drift.mae[m] ?? NaN);
  const finite = maeVals.filter((v) => Number.isFinite(v));
  const yMin = finite.length ? Math.min(...finite) * 0.9 : 0;
  const yMax = finite.length ? Math.max(...finite) * 1.1 : 1;

  const x = (i: number) =>
    PAD_L + (months.length <= 1 ? 0 : (i / (months.length - 1)) * (W - PAD_L - PAD_R));
  const y = (v: number) =>
    PAD_T + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * (H - PAD_T - PAD_B);

  const series: ChartSeries[] = [
    { name: "MAE (min)", values: maeVals, color: "#ffd166" },
  ];
  // Overlay the first drift signal normalized to its own max, faint.
  const sigNames = Object.keys(drift.signals);
  if (sigNames.length > 0) {
    const s0 = sigNames[0];
    const vals = months.map((m) => drift.signals[s0]?.[m] ?? NaN);
    const f = vals.filter((v) => Number.isFinite(v));
    const sMax = f.length ? Math.max(...f) : 1;
    series.push({
      name: `${s0} (scaled)`,
      values: vals.map((v) => (Number.isFinite(v) ? (v / sMax) * yMax : NaN)),
      color: "#4cc2ff",
    });
  }

  const paths = series.map((s) => {
    const pts: string[] = [];
    s.values.forEach((v, i) => {
      if (Number.isFinite(v)) pts.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    });
    return { name: s.name, color: s.color, points: pts.join(" ") };
  });

  const degradedIdx = months
    .map((m, i) => (drift.degraded[m] ? i : -1))
    .filter((i) => i >= 0);

  return {
    width: W,
    height: H,
    months,
    series,
    paths,
    yMin,
    yMax,
    degradedIdx,
  };
}

export function renderDriftChartSvg(model: ChartModel): string {
  const { width: w, height: h, months, paths, degradedIdx } = model;
  const x = (i: number) =>
    PAD_L + (months.length <= 1 ? 0 : (i / (months.length - 1)) * (w - PAD_L - PAD_R));
  const yMin = model.yMin;
  const yMax = model.yMax;
  const y = (v: number) =>
    PAD_T + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * (h - PAD_T - PAD_B);

  const gridLines = [0, 0.5, 1].map((f) => {
    const v = yMin + f * (yMax - yMin);
    const yy = y(v).toFixed(1);
    return `<line x1="${PAD_L}" y1="${yy}" x2="${w - PAD_R}" y2="${yy}" stroke="#232a3a" stroke-width="1"/>` +
      `<text x="${PAD_L - 4}" y="${Number(yy) + 3}" text-anchor="end" font-size="8" fill="#8b96ad">${v.toFixed(0)}</text>`;
  }).join("");

  const monthTicks = months
    .map((m, i) =>
      i % Math.ceil(months.length / 6) === 0
        ? `<text x="${x(i).toFixed(1)}" y="${h - 6}" text-anchor="middle" font-size="8" fill="#8b96ad">${formatMonth(m)}</text>`
        : "",
    )
    .join("");

  const degradedMarks = degradedIdx
    .map(
      (i) =>
        `<circle cx="${x(i).toFixed(1)}" cy="${PAD_T + 2}" r="2.5" fill="#ff6b6b"/>`,
    )
    .join("");

  const lines = paths
    .map(
      (p) =>
        `<polyline points="${p.points}" fill="none" stroke="${p.color}" stroke-width="1.6"/>`,
    )
    .join("");

  const legend = paths
    .map(
      (p, i) =>
        `<text x="${PAD_L + i * 90}" y="${h - 8}" font-size="8" fill="${p.color}">${p.name}</text>`,
    )
    .join("");

  return (
    `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">` +
    gridLines +
    `<g>${degradedMarks}</g>` +
    lines +
    monthTicks +
    legend +
    `</svg>`
  );
}
