import * as maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { TripsLayer } from "@deck.gl/geo-layers";
import "maplibre-gl/dist/maplibre-gl.css";
import "./style.css";

import { loadSiteData, type SiteData } from "./data";
import { LOOP_MS, buildTripRecords, type TripRecord } from "./trips";
import { buildDriftChart, renderDriftChartSvg } from "./driftchart";
import { formatMinutes } from "./format";

const DARK_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    {
      id: "background",
      type: "background",
      paint: { "background-color": "#0b0e14" },
    },
  ],
};

function routesToGeoJSON(routes: SiteData["routes"]) {
  return {
    type: "FeatureCollection" as const,
    features: Object.entries(routes).map(([key, coords]) => ({
      type: "Feature" as const,
      properties: { pair: key },
      geometry: {
        type: "LineString" as const,
        coordinates: coords,
      },
    })),
  };
}

async function tryLivePrediction(): Promise<void> {
  const card = document.getElementById("prediction-card");
  const body = document.getElementById("prediction-body");
  if (!card || !body) return;
  try {
    const res = await fetch("predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pu_zone: 161, // Midtown
        do_zone: 132, // JFK
        pickup_datetime: new Date().toISOString(),
      }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as {
      predicted_minutes: number;
      interval_low_minutes: number;
      interval_high_minutes: number;
      model_note: string;
    };
    body.innerHTML =
      `<div class="big">${formatMinutes(data.predicted_minutes)}</div>` +
      `<div>80% interval: ${formatMinutes(data.interval_low_minutes)} - ` +
      `${formatMinutes(data.interval_high_minutes)}</div>` +
      `<div style="color:var(--muted)">${data.model_note}</div>`;
    card.hidden = false;
  } catch {
    // Static site without the API: leave the card hidden.
  }
}

function renderDrift(drift: SiteData["drift"]): void {
  const el = document.getElementById("drift-chart");
  const cap = document.getElementById("drift-caption");
  if (!el || !cap || !drift) return;
  const model = buildDriftChart(drift);
  el.innerHTML = renderDriftChartSvg(model);
  const months = drift.months;
  const degraded = months.filter((m) => drift.degraded[m]);
  cap.textContent = degraded.length
    ? `Degraded months (MAE > 10% above validation): ${degraded.join(", ")}`
    : "No degraded months in this window.";
}

function tripLayerProps(records: TripRecord[], currentTime: number) {
  return {
    id: "taxis",
    data: records,
    getPath: (d: TripRecord) => d.path,
    getTimestamps: (d: TripRecord) => d.timestamps,
    getColor: [255, 209, 102] as [number, number, number],
    opacity: 0.95,
    trailLength: 0.15,
    currentTime,
  };
}

async function main(): Promise<void> {
  const mapEl = document.getElementById("map");
  if (!mapEl) return;

  const data = await loadSiteData();

  const map = new maplibregl.Map({
    container: mapEl,
    style: DARK_STYLE,
    center: [-73.98, 40.75],
    zoom: 11,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");

  map.on("load", () => {
    // Taxi zones: subtle fill + border.
    map.addSource("zones", { type: "geojson", data: data.zones });
    map.addLayer({
      id: "zones-fill",
      type: "fill",
      source: "zones",
      paint: { "fill-color": "#1d2b45", "fill-opacity": 0.55 },
    });
    map.addLayer({
      id: "zones-line",
      type: "line",
      source: "zones",
      paint: { "line-color": "#33507e", "line-width": 0.6 },
    });

    // Illustrative routes between zone centroids.
    const routeData = routesToGeoJSON(data.routes);
    map.addSource("routes", { type: "geojson", data: routeData });
    map.addLayer({
      id: "routes-line",
      type: "line",
      source: "routes",
      paint: {
        "line-color": "#4cc2ff",
        "line-width": 1.1,
        "line-opacity": 0.55,
      },
    });

    // Animated taxis via deck.gl TripsLayer; currentTime advances on a
    // LOOP_MS clock and the layer props are re-set each frame (deck.gl
    // diffs by layer id, so the data buffers are reused).
    const tripsKey = Object.keys(data.trips)[0];
    const records: TripRecord[] = tripsKey
      ? buildTripRecords(data.trips[tripsKey], data.routes)
      : [];

    const overlay = new MapboxOverlay({
      interleaved: true,
      layers: [new TripsLayer(tripLayerProps(records, 0))],
    });
    map.addControl(overlay as unknown as maplibregl.IControl);

    const start = performance.now();
    const tick = () => {
      const t = (performance.now() - start) % LOOP_MS;
      overlay.setProps({
        layers: [new TripsLayer(tripLayerProps(records, t))],
      });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  renderDrift(data.drift);
  void tryLivePrediction();
}

void main();
