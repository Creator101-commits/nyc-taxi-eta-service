import type { TripRow, RoutesFile } from "./data";

/**
 * Build deck.gl TripsLayer records from trips.json + routes.json.
 *
 * TripsLayer animates each trip along `path` between `timestamps`.
 * We compress a demo day into a LOOP_MS loop: a trip picking up at
 * minute m of the day starts at (m / 1440) * LOOP_MS and lasts
 * max(MIN_TRIP_MS, minutes / 1440 * LOOP_MS), where minutes is the
 * trip's ACTUAL duration.
 */

export const LOOP_MS = 120_000;
export const MIN_TRIP_MS = 4_000;

export interface TripRecord {
  /** [lon, lat] positions along the route */
  path: [number, number][];
  /** epoch-ms timestamps, one per path vertex, strictly increasing */
  timestamps: number[];
  /** display metadata kept alongside (not used by the layer) */
  pu: number;
  do: number;
  minutes: number;
  pred_minutes: number;
}

export function routeKey(pu: number, doZone: number): string {
  return `${pu}>${doZone}`;
}

/**
 * Convert one TripRow into a TripsLayer record, or null when the route
 * for its zone pair is missing.
 */
export function tripToRecord(
  trip: TripRow,
  routes: RoutesFile,
): TripRecord | null {
  const path = routes[routeKey(trip.pu, trip.do)];
  if (!path || path.length < 2) return null;

  const start = (trip.minute / 1440) * LOOP_MS;
  const rawDur = (trip.minutes / 1440) * LOOP_MS;
  const dur = Math.max(MIN_TRIP_MS, rawDur);

  // Timestamps follow cumulative segment length so each taxi moves at
  // constant speed even after route simplification thins the vertices.
  const n = path.length;
  const cum: number[] = [0];
  for (let i = 1; i < n; i++) {
    const dx = path[i][0] - path[i - 1][0];
    const dy = path[i][1] - path[i - 1][1];
    cum.push(cum[i - 1] + Math.hypot(dx, dy));
  }
  const total = cum[n - 1];
  const timestamps = Array.from({ length: n }, (_, i) =>
    total > 0 ? start + (dur * cum[i]) / total : start + (dur * i) / (n - 1),
  );

  return {
    path: path.map((p) => [p[0], p[1]] as [number, number]),
    timestamps,
    pu: trip.pu,
    do: trip.do,
    minutes: trip.minutes,
    pred_minutes: trip.pred_minutes,
  };
}

/** All trips for one demo day that have routes, capped at maxTrips. */
export function buildTripRecords(
  trips: TripRow[],
  routes: RoutesFile,
  maxTrips = 400,
): TripRecord[] {
  const out: TripRecord[] = [];
  for (const t of trips) {
    const rec = tripToRecord(t, routes);
    if (rec) out.push(rec);
    if (out.length >= maxTrips) break;
  }
  return out;
}

/** Position along a polyline at fraction t in [0,1]. */
export function pointAlongRoute(
  coords: [number, number][],
  t: number,
): [number, number] {
  if (coords.length === 0) return [0, 0];
  if (coords.length === 1 || t <= 0) return coords[0];
  if (t >= 1) return coords[coords.length - 1];

  // Walk segments by count (routes are stride-sampled evenly enough for
  // display purposes).
  const seg = t * (coords.length - 1);
  const i = Math.min(Math.floor(seg), coords.length - 2);
  const f = seg - i;
  const a = coords[i];
  const b = coords[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}
