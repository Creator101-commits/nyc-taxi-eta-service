import { describe, expect, it } from "vitest";
import {
  LOOP_MS,
  MIN_TRIP_MS,
  buildTripRecords,
  pointAlongRoute,
  routeKey,
  tripToRecord,
} from "./trips";
import type { RoutesFile, TripRow } from "./data";

const routes: RoutesFile = {
  "1>2": [
    [-74.0, 40.7],
    [-73.99, 40.71],
    [-73.98, 40.72],
  ],
};

describe("routeKey", () => {
  it("formats pu>do", () => {
    expect(routeKey(1, 2)).toBe("1>2");
  });
});

describe("tripToRecord", () => {
  const trip: TripRow = { pu: 1, do: 2, minute: 600, pred_minutes: 30 };

  it("builds path and increasing timestamps", () => {
    const rec = tripToRecord(trip, routes);
    expect(rec).not.toBeNull();
    expect(rec!.path).toHaveLength(3);
    expect(rec!.timestamps).toHaveLength(3);
    expect(rec!.timestamps[0]).toBeLessThan(rec!.timestamps[1]);
    expect(rec!.timestamps[1]).toBeLessThan(rec!.timestamps[2]);
    // start time = minute/1440 * LOOP_MS
    expect(rec!.timestamps[0]).toBeCloseTo((600 / 1440) * LOOP_MS, 5);
  });

  it("returns null when the route is missing", () => {
    expect(tripToRecord({ pu: 9, do: 9, minute: 0, pred_minutes: 5 }, routes)).toBeNull();
  });

  it("enforces a minimum trip duration", () => {
    const rec = tripToRecord(
      { pu: 1, do: 2, minute: 0, pred_minutes: 0.001 },
      routes,
    );
    expect(rec!.timestamps[2] - rec!.timestamps[0]).toBeGreaterThanOrEqual(
      MIN_TRIP_MS - 1e-6,
    );
  });
});

describe("buildTripRecords", () => {
  it("skips trips without routes and caps the count", () => {
    const trips: TripRow[] = [
      { pu: 1, do: 2, minute: 100, pred_minutes: 10 },
      { pu: 7, do: 8, minute: 100, pred_minutes: 10 }, // no route
      { pu: 1, do: 2, minute: 200, pred_minutes: 12 },
    ];
    const recs = buildTripRecords(trips, routes, 1);
    expect(recs).toHaveLength(1);
    expect(recs[0].pu).toBe(1);
  });
});

describe("pointAlongRoute", () => {
  const line: [number, number][] = [
    [0, 0],
    [10, 0],
    [10, 10],
  ];

  it("interpolates endpoints and midpoint", () => {
    expect(pointAlongRoute(line, 0)).toEqual([0, 0]);
    expect(pointAlongRoute(line, 1)).toEqual([10, 10]);
    expect(pointAlongRoute(line, 0.5)[0]).toBeCloseTo(10, 5);
  });

  it("handles degenerate input", () => {
    expect(pointAlongRoute([], 0.5)).toEqual([0, 0]);
    expect(pointAlongRoute([[5, 5]], 0.5)).toEqual([5, 5]);
  });
});
