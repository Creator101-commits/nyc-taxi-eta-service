import type { DriftFile } from "./data";
import { formatMonth } from "./format";

/**
 * Build an SVG line chart of per-policy monthly MAE with drift overlays:
 * degraded-cut line, any-alarm month markers, drift-policy retrain ticks
 * and a COVID annotation at 2020-03. Pure function: returns SVG markup;
 * main.ts injects it.
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
  /** indices of months where any drift signal alarmed */
  alarmIdx: number[];
  /** x positions (month indices) where the drift policy retrained */
  retrainIdx: number[];
  /** index of the 2020-03 COVID annotation, or -1 */
  covidIdx: number;
  /** degraded cut value (1.1 x validation MAE) */
  degradedCut: number;
}

const W = 260;
const H = 110;
const PAD_L = 34;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 20;

const POLICY_ORDER = ["static", "monthly", "drift"];
const POLICY_COLORS: Record<string, string> = {
  static: "#ffd166",
  monthly: "#4cc2ff",
  drift: "#ff9e64",
};
const COVID_MONTH = "2020-03";

export function buildDriftChart(drift: DriftFile): ChartModel {
  const months = drift.months;
  const n = months.length;

  // One MAE series per retraining policy, in a fixed order.
  const series: ChartSeries[] = [];
  for (const name of POLICY_ORDER) {
    const pol = drift.policies?.[name];
    if (!pol) continue;
    series.push({
      name: `${name} MAE`,
      values: months.map((m) => pol.monthly_mae[m] ?? NaN),
      color: POLICY_COLORS[name] ?? "#8b96ad",
    });
  }
  // Fall back to the static mae map when policies are absent.
  if (series.length === 0) {
    series.push({
      name: "static MAE",
      values: months.map((m) => drift.mae[m] ?? NaN),
      color: POLICY_COLORS.static,
    });
  }

  const allVals = series.flatMap((s) => s.values).filter(Number.isFinite);
  const cut = Number.isFinite(drift.degraded_cut) ? drift.degraded_cut : NaN;
  const fitVals = cut !== undefined && Number.isFinite(cut)
    ? [...allVals, cut]
    : allVals;
  const yMin = fitVals.length ? Math.min(...fitVals) * 0.9 : 0;
  const yMax = fitVals.length ? Math.max(...fitVals) * 1.08 : 1;

  const x = (i: number) =>
    PAD_L + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD_L - PAD_R));
  const y = (v: number) =>
    PAD_T + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * (H - PAD_T - PAD_B);

  const paths = series.map((s) => {
    const pts: string[] = [];
    s.values.forEach((v, i) => {
      if (Number.isFinite(v)) pts.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    });
    return { name: s.name, color: s.color, points: pts.join(" ") };
  });

  const degradedIdx = months
    .map((m, i) => (drift.degraded?.[m] ? i : -1))
    .filter((i) => i >= 0);
  const alarmIdx = months
    .map((m, i) => (drift.alarms?.[m]?.any ? i : -1))
    .filter((i) => i >= 0);
  const retrainSet = new Set(drift.policies?.drift?.retrain_months ?? []);
  const retrainIdx = months
    .map((m, i) => (retrainSet.has(m) ? i : -1))
    .filter((i) => i >= 0);
  const covidIdx = months.indexOf(COVID_MONTH);

  return {
    width: W,
    height: H,
    months,
    series,
    paths,
    yMin,
    yMax,
    degradedIdx,
    alarmIdx,
    retrainIdx,
    covidIdx,
    degradedCut: Number.isFinite(cut) ? cut : NaN,
  };
}

export function renderDriftChartSvg(model: ChartModel): string {
  const { width: w, height: h, months, paths } = model;
  const n = months.length;
  const x = (i: number) =>
    PAD_L + (n <= 1 ? 0 : (i / (n - 1)) * (w - PAD_L - PAD_R));
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

  // Degraded cut: dashed horizontal line at 1.1 x validation MAE.
  const cutLine = Number.isFinite(model.degradedCut)
    ? `<line x1="${PAD_L}" y1="${y(model.degradedCut).toFixed(1)}" x2="${w - PAD_R}" y2="${y(model.degradedCut).toFixed(1)}" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3"/>`
    : "";

  // COVID annotation: vertical dashed line at 2020-03.
  const covidLine = model.covidIdx >= 0
    ? `<line x1="${x(model.covidIdx).toFixed(1)}" y1="${PAD_T}" x2="${x(model.covidIdx).toFixed(1)}" y2="${h - PAD_B}" stroke="#8b96ad" stroke-width="1" stroke-dasharray="2 3"/>` +
      `<text x="${(x(model.covidIdx) + 2).toFixed(1)}" y="${PAD_T + 8}" font-size="8" fill="#8b96ad">COVID</text>`
    : "";

  const monthTicks = months
    .map((m, i) =>
      i % Math.ceil(n / 6) === 0
        ? `<text x="${x(i).toFixed(1)}" y="${h - 6}" text-anchor="middle" font-size="8" fill="#8b96ad">${formatMonth(m)}</text>`
        : "",
    )
    .join("");

  const degradedMarks = model.degradedIdx
    .map(
      (i) =>
        `<circle cx="${x(i).toFixed(1)}" cy="${PAD_T + 2}" r="2.5" fill="#ff6b6b"/>`,
    )
    .join("");

  // Any-alarm months: amber diamonds along the top edge.
  const alarmMarks = model.alarmIdx
    .map(
      (i) => {
        const cx = x(i).toFixed(1);
        const cy = (PAD_T + 8).toFixed(1);
        return `<path d="M ${cx} ${Number(cy) - 3} L ${Number(cx) + 3} ${cy} L ${cx} ${Number(cy) + 3} L ${Number(cx) - 3} ${cy} Z" fill="#ffd166"/>`;
      },
    )
    .join("");

  // Drift-policy retrain months: short vertical ticks on the x axis.
  const retrainTicks = model.retrainIdx
    .map(
      (i) =>
        `<line x1="${x(i).toFixed(1)}" y1="${h - PAD_B}" x2="${x(i).toFixed(1)}" y2="${h - PAD_B + 4}" stroke="#ff9e64" stroke-width="1.4"/>`,
    )
    .join("");

  const lines = paths
    .map(
      (p) =>
        `<polyline points="${p.points}" fill="none" stroke="${p.color}" stroke-width="1.6"/>`,
    )
    .join("");

  return (
    `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">` +
    gridLines +
    cutLine +
    covidLine +
    `<g>${degradedMarks}${alarmMarks}</g>` +
    lines +
    retrainTicks +
    monthTicks +
    `</svg>`
  );
}
