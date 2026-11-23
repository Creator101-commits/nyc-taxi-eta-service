import { describe, expect, it } from "vitest";

import type { ZoneCollection } from "./data";
import { CONNECTOR_LABEL, predictionOverlay } from "./prediction";

function zone(id: number, coords: number[][]): ZoneCollection["features"][number] {
  return {
    type: "Feature",
    properties: { location_id: id, zone: `Z${id}`, borough: "Test" },
    geometry: { type: "Polygon", coordinates: [coords] },
  };
}

const zones: ZoneCollection = {
  type: "FeatureCollection",
  features: [
    zone(1, [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]),
    zone(2, [[10, 10], [12, 10], [12, 12], [10, 12], [10, 10]]),
  ],
};

describe("predictionOverlay", () => {
  it("builds highlight zones, a centroid connector and a midpoint label", () => {
    const o = predictionOverlay(zones, 1, 2);
    expect(o).not.toBeNull();
    expect(o!.zones.features).toHaveLength(2);
    const link = o!.link.features[0];
    expect(link.geometry.type).toBe("LineString");
    if (link.geometry.type === "LineString") {
      // vertex mean of the closed 5-point rings: (0.8,0.8) and (10.8,10.8)
      expect(link.geometry.coordinates).toEqual([
        [0.8, 0.8],
        [10.8, 10.8],
      ]);
    }
    const label = o!.label.features[0];
    expect(label.geometry.type).toBe("Point");
    if (label.geometry.type === "Point") {
      expect(label.geometry.coordinates[0]).toBeCloseTo(5.8);
      expect(label.geometry.coordinates[1]).toBeCloseTo(5.8);
      expect(label.properties?.label).toBe(CONNECTOR_LABEL);
    }
  });

  it("returns null when a zone is missing", () => {
    expect(predictionOverlay(zones, 1, 999)).toBeNull();
    expect(predictionOverlay(zones, 999, 1)).toBeNull();
  });
});
