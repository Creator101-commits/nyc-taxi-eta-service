/** Data types for the generated files under public/data/. */

export interface ZoneFeature {
  type: "Feature";
  properties: { location_id: number; zone: string; borough: string };
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
}

export interface ZoneCollection {
  type: "FeatureCollection";
  features: ZoneFeature[];
}

export interface TripRow {
  /** pickup zone LocationID */
  pu: number;
  /** dropoff zone LocationID */
  do: number;
  /** minute of day of pickup, 0..1439 */
  minute: number;
  /** actual trip duration in minutes */
  minutes: number;
  /** model-predicted trip minutes */
  pred_minutes: number;
}

export interface DayMeta {
  date: string;
  /** sampled rows on that day */
  n_sampled: number;
  /** estimated real trips that day (scaled by month cleaned count) */
  est_real_trips: number;
  /** trips actually exported to trips.json */
  n_exported: number;
  /** est_real_trips / n_exported; how many real trips each taxi stands for */
  real_trips_per_taxi: number | null;
}

/** trips.json: { days: {tag: DayMeta}, trips: {tag: TripRow[]} } */
export interface TripsFile {
  days: Record<string, DayMeta>;
  trips: Record<string, TripRow[]>;
}

/** routes.json: { "pu>do": [[lon, lat], ...] } */
export type RoutesFile = Record<string, [number, number][]>;

export interface PolicySeries {
  monthly_mae: Record<string, number>;
  retrain_months: string[];
}

export interface DriftFile {
  months: string[];
  /** static-model MAE per month (kept for backward readability) */
  mae: Record<string, number>;
  signals: Record<string, Record<string, number>>;
  degraded: Record<string, boolean>;
  thresholds: Record<string, number>;
  /** per month: per-signal alarm flags plus "any" */
  alarms: Record<string, Record<string, boolean>>;
  /** static-model interval coverage per month */
  static_coverage: Record<string, number>;
  val_mae: number;
  /** MAE above this counts as degraded (1.1 x val_mae) */
  degraded_cut: number;
  /** per-policy monthly MAE and retrain months */
  policies: Record<string, PolicySeries>;
}

export interface SiteData {
  zones: ZoneCollection;
  trips: TripsFile;
  routes: RoutesFile;
  drift: DriftFile | null;
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function loadSiteData(base = "data/"): Promise<SiteData> {
  const [zones, trips, routes, drift] = await Promise.all([
    fetchJson<ZoneCollection>(`${base}zones.geojson`),
    fetchJson<TripsFile>(`${base}trips.json`),
    fetchJson<RoutesFile>(`${base}routes.json`),
    fetchJson<DriftFile>(`${base}drift.json`),
  ]);
  return {
    zones: zones ?? { type: "FeatureCollection", features: [] },
    trips: trips ?? { days: {}, trips: {} },
    routes: routes ?? {},
    drift,
  };
}
