import type { ZoneCollection, ZoneFeature } from "./data";

/**
 * Map overlay for a live /predict result.
 *
 * The service returns a duration estimate only; it does not return a
 * road route. The map therefore highlights the two zones and draws a
 * dashed straight connector between their centroids, explicitly
 * labeled as not a route. Road geometry appears only in the animated
 * replay trips, which carry the "illustrative" label.
 */

export interface PredictionOverlay {
  /** the pickup and dropoff zone polygons to highlight */
  zones: GeoJSON.FeatureCollection;
  /** dashed straight connector between zone centroids */
  link: GeoJSON.FeatureCollection;
  /** midpoint label point for the connector */
  label: GeoJSON.FeatureCollection;
}

export const CONNECTOR_LABEL = "straight-line connector, not a route";

function centroidOf(feature: ZoneFeature): [number, number] {
  const ring = feature.geometry.coordinates[0];
  let sx = 0;
  let sy = 0;
  for (const [x, y] of ring) {
    sx += x;
    sy += y;
  }
  return [sx / ring.length, sy / ring.length];
}

function findZone(
  zones: ZoneCollection,
  locationId: number,
): ZoneFeature | undefined {
  return zones.features.find(
    (f) => f.properties.location_id === locationId,
  );
}

export function predictionOverlay(
  zones: ZoneCollection,
  pickupZoneId: number,
  dropoffZoneId: number,
): PredictionOverlay | null {
  const pu = findZone(zones, pickupZoneId);
  const doZone = findZone(zones, dropoffZoneId);
  if (!pu || !doZone) return null;
  const a = centroidOf(pu);
  const b = centroidOf(doZone);
  const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  return {
    zones: {
      type: "FeatureCollection",
      features: [pu, doZone],
    },
    link: {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { kind: "connector" },
          geometry: { type: "LineString", coordinates: [a, b] },
        },
      ],
    },
    label: {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { label: CONNECTOR_LABEL },
          geometry: { type: "Point", coordinates: mid },
        },
      ],
    },
  };
}
