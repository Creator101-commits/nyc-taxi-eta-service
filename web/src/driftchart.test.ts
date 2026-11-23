import { describe, expect, it } from "vitest";
import { buildDriftChart, renderDriftChartSvg } from "./driftchart";
import type { DriftFile } from "./data";

const drift: DriftFile = {
  months: ["2019-07", "2019-08", "2019-09"],
  mae: { "2019-07": 10, "2019-08": 12, "2019-09": 15 },
  signals: {
    drift_share: { "2019-07": 0.1, "2019-08": 0.2, "2019-09": 0.4 },
  },
  degraded: { "2019-07": false, "2019-08": false, "2019-09": true },
};

describe("buildDriftChart", () => {
  it("produces one path per series with finite points", () => {
    const model = buildDriftChart(drift);
    expect(model.paths).toHaveLength(2); // MAE + scaled signal
    for (const p of model.paths) {
      expect(p.points.split(" ")).toHaveLength(3);
    }
    expect(model.degradedIdx).toEqual([2]);
  });

  it("handles a single month", () => {
    const one: DriftFile = {
      months: ["2019-07"],
      mae: { "2019-07": 10 },
      signals: {},
      degraded: { "2019-07": false },
    };
    const model = buildDriftChart(one);
    expect(model.paths).toHaveLength(1);
    expect(model.paths[0].points).not.toBe("");
  });
});

describe("renderDriftChartSvg", () => {
  it("returns svg markup with polylines", () => {
    const svg = renderDriftChartSvg(buildDriftChart(drift));
    expect(svg).toContain("<svg");
    expect(svg).toContain("<polyline");
    expect(svg).toContain("#ff6b6b"); // degraded marker
  });
});
