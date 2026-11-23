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
  /** model-predicted trip minutes */
  pred_minutes: number;
}

/** trips.json: { "pre": TripRow[], "covid": TripRow[] } */
export type TripsFile = Record<string, TripRow[]>;

/** routes.json: { "pu>do": [[lon, lat], ...] } */
export type RoutesFile = Record<string, [number, number][]>;

export interface DriftFile {
  months: string[];
  mae: Record<string, number>;
  signals: Record<string, Record<string, number>>;
  degraded: Record<string, boolean>;
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
    trips: trips ?? {},
    routes: routes ?? {},
    drift,
  };
}
