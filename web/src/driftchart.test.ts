import { describe, expect, it } from "vitest";
import { buildDriftChart, renderDriftChartSvg } from "./driftchart";
import type { DriftFile } from "./data";

const MONTHS = [
  "2019-10", "2019-11", "2019-12",
  "2020-01", "2020-02", "2020-03", "2020-04",
];

function maeFor(scale: number): Record<string, number> {
  return Object.fromEntries(MONTHS.map((m, i) => [m, 10 + i * scale]));
}

const drift: DriftFile = {
  months: MONTHS,
  mae: maeFor(1),
  signals: {
    drift_share: Object.fromEntries(MONTHS.map((m) => [m, 0.1])),
  },
  degraded: Object.fromEntries(MONTHS.map((m, i) => [m, i === 6])),
  thresholds: { drift_share: 0.3, pred_wasserstein: 0.45, volume_ratio: 1.25 },
  alarms: Object.fromEntries(
    MONTHS.map((m, i) => [
      m,
      i === 5
        ? { drift_share: false, pred_wasserstein: false, volume_ratio: true, any: true }
        : { drift_share: false, pred_wasserstein: false, volume_ratio: false, any: false },
    ]),
  ),
  static_coverage: Object.fromEntries(MONTHS.map((m) => [m, 0.8])),
  val_mae: 10,
  degraded_cut: 11,
  policies: {
    static: { monthly_mae: maeFor(1), retrain_months: [] },
    monthly: { monthly_mae: maeFor(0.5), retrain_months: ["2020-03"] },
    drift: { monthly_mae: maeFor(0.8), retrain_months: ["2020-03", "2020-04"] },
  },
};

describe("buildDriftChart", () => {
  it("produces one path per policy plus markers", () => {
    const model = buildDriftChart(drift);
    expect(model.paths).toHaveLength(3);
    expect(model.paths.map((p) => p.name)).toEqual([
      "static MAE",
      "monthly MAE",
      "drift MAE",
    ]);
    for (const p of model.paths) {
      expect(p.points.split(" ")).toHaveLength(MONTHS.length);
    }
    expect(model.degradedIdx).toEqual([6]);
    expect(model.alarmIdx).toEqual([5]); // 2020-03 any-alarm
    expect(model.retrainIdx).toEqual([5, 6]); // drift retrains at 03, 04
    expect(model.covidIdx).toBe(5);
    expect(model.degradedCut).toBe(11);
  });

  it("falls back to the static mae map without policies", () => {
    const legacy: DriftFile = {
      months: ["2019-10"],
      mae: { "2019-10": 10 },
      signals: {},
      degraded: { "2019-10": false },
      thresholds: {},
      alarms: {},
      static_coverage: {},
      val_mae: 10,
      degraded_cut: 11,
      policies: {},
    };
    const model = buildDriftChart(legacy);
    expect(model.paths).toHaveLength(1);
    expect(model.paths[0].name).toBe("static MAE");
    expect(model.covidIdx).toBe(-1);
  });
});

describe("renderDriftChartSvg", () => {
  it("returns svg markup with lines, cut, alarms and covid mark", () => {
    const svg = renderDriftChartSvg(buildDriftChart(drift));
    expect(svg).toContain("<svg");
    expect(svg).toContain("<polyline");
    expect(svg).toContain("degraded cut");
    expect(svg).toContain("COVID");
    expect(svg).toContain("#ff6b6b"); // degraded marker + cut line
    expect(svg).toContain("#ffd166"); // static MAE + alarm diamonds
  });
});
