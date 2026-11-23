import { describe, expect, it } from "vitest";
import {
  LOOP_MS,
  MIN_TRIP_MS,
  buildTripRecords,
  dayRouteKeys,
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
  const trip: TripRow = {
    pu: 1, do: 2, minute: 600, minutes: 30, pred_minutes: 28,
  };

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
    expect(
      tripToRecord({ pu: 9, do: 9, minute: 0, minutes: 5, pred_minutes: 5 }, routes),
    ).toBeNull();
  });

  it("animates with actual minutes, not predicted", () => {
    const rec = tripToRecord(
      { pu: 1, do: 2, minute: 0, minutes: 60, pred_minutes: 5 },
      routes,
    );
    expect(rec!.minutes).toBe(60);
    expect(rec!.pred_minutes).toBe(5);
    const dur = rec!.timestamps[2] - rec!.timestamps[0];
    expect(dur).toBeCloseTo((60 / 1440) * LOOP_MS, 5);
  });

  it("enforces a minimum trip duration", () => {
    const rec = tripToRecord(
      { pu: 1, do: 2, minute: 0, minutes: 0.001, pred_minutes: 30 },
      routes,
    );
    expect(rec!.timestamps[2] - rec!.timestamps[0]).toBeGreaterThanOrEqual(
      MIN_TRIP_MS - 1e-6,
    );
  });

  it("spaces timestamps by segment length for constant speed", () => {
    // Uneven vertex spacing: 1, 1, 2 units -> time shares 1/4, 1/4, 1/2.
    const uneven: RoutesFile = {
      "1>2": [
        [0, 0],
        [0.001, 0],
        [0.002, 0],
        [0.004, 0],
      ],
    };
    const rec = tripToRecord(
      { pu: 1, do: 2, minute: 0, minutes: 40, pred_minutes: 40 },
      uneven,
    )!;
    const d01 = rec.timestamps[1] - rec.timestamps[0];
    const d12 = rec.timestamps[2] - rec.timestamps[1];
    const d23 = rec.timestamps[3] - rec.timestamps[2];
    expect(d01).toBeCloseTo(d12, 6);
    expect(d23).toBeCloseTo(2 * d01, 6);
  });
});

describe("buildTripRecords", () => {
  it("skips trips without routes and caps the count", () => {
    const trips: TripRow[] = [
      { pu: 1, do: 2, minute: 100, minutes: 10, pred_minutes: 10 },
      { pu: 7, do: 8, minute: 100, minutes: 10, pred_minutes: 10 }, // no route
      { pu: 1, do: 2, minute: 200, minutes: 12, pred_minutes: 12 },
    ];
    const recs = buildTripRecords(trips, routes, 1);
    expect(recs).toHaveLength(1);
    expect(recs[0].pu).toBe(1);
  });

  it("keeps every trip that has a route (no cap by default)", () => {
    const trips: TripRow[] = Array.from({ length: 450 }, (_, i) => ({
      pu: 1, do: 2, minute: i % 1440, minutes: 10, pred_minutes: 10,
    }));
    const recs = buildTripRecords(trips, routes);
    expect(recs).toHaveLength(450);
  });
});

describe("dayRouteKeys", () => {
  it("returns only keys used by the day's trips with valid routes", () => {
    const trips: TripRow[] = [
      { pu: 1, do: 2, minute: 0, minutes: 5, pred_minutes: 5 },
      { pu: 1, do: 2, minute: 60, minutes: 5, pred_minutes: 5 },
      { pu: 3, do: 4, minute: 0, minutes: 5, pred_minutes: 5 }, // no route
    ];
    expect(dayRouteKeys(trips, routes)).toEqual(["1>2"]);
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
