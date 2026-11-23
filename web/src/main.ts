import * as maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { TripsLayer } from "@deck.gl/geo-layers";
import "maplibre-gl/dist/maplibre-gl.css";
// The ESM build loads its worker by URL; let Vite bundle and emit it.
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "./style.css";

import {
  loadSiteData,
  type SiteData,
  type TripsFile,
  type ZoneCollection,
} from "./data";
import { LOOP_MS, buildTripRecords, type TripRecord } from "./trips";
import { buildDriftChart, renderDriftChartSvg } from "./driftchart";
import { predictionOverlay } from "./prediction";
import { formatClock, formatDayDate, formatMinutes } from "./format";

maplibregl.setWorkerUrl(maplibreWorkerUrl);

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

const PRED_PU_ZONE = 161; // Midtown
const PRED_DO_ZONE = 132; // JFK

function addPredictionOverlay(
  map: maplibregl.Map,
  zones: ZoneCollection,
): void {
  const overlay = predictionOverlay(zones, PRED_PU_ZONE, PRED_DO_ZONE);
  if (!overlay) return;
  map.addSource("pred-zones", { type: "geojson", data: overlay.zones });
  map.addSource("pred-link", { type: "geojson", data: overlay.link });
  map.addSource("pred-label", { type: "geojson", data: overlay.label });
  map.addLayer({
    id: "pred-zones-fill",
    type: "fill",
    source: "pred-zones",
    paint: { "fill-color": "#f5b942", "fill-opacity": 0.4 },
  });
  map.addLayer({
    id: "pred-zones-line",
    type: "line",
    source: "pred-zones",
    paint: { "line-color": "#f5b942", "line-width": 1.4 },
  });
  map.addLayer({
    id: "pred-link-line",
    type: "line",
    source: "pred-link",
    paint: {
      "line-color": "#ffd166",
      "line-width": 2,
      "line-opacity": 0.9,
      "line-dasharray": [2, 1.6],
    },
  });
  map.addLayer({
    id: "pred-label",
    type: "symbol",
    source: "pred-label",
    layout: {
      "text-field": ["get", "label"],
      "text-size": 11,
      "text-offset": [0, 1.4],
    },
    paint: {
      "text-color": "#c8d3e8",
      "text-halo-color": "#0b0e14",
      "text-halo-width": 1.2,
    },
  });
}

async function tryLivePrediction(
  map: maplibregl.Map,
  zones: ZoneCollection,
): Promise<void> {
  const card = document.getElementById("prediction-card");
  const body = document.getElementById("prediction-body");
  if (!card || !body) return;
  try {
    const res = await fetch("predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pickup_zone_id: PRED_PU_ZONE,
        dropoff_zone_id: PRED_DO_ZONE,
        pickup_datetime: new Date().toISOString(),
      }),
    });
    if (!res.ok) {
      showPredictionFallback(card, body);
      return;
    }
    const data = (await res.json()) as {
      minutes: number;
      interval_80: { low: number; high: number };
      model_version: string;
    };
    body.innerHTML =
      `<div class="big">${formatMinutes(data.minutes)}</div>` +
      `<div>80% interval: ${formatMinutes(data.interval_80.low)} - ` +
      `${formatMinutes(data.interval_80.high)}</div>` +
      `<div style="color:var(--muted)">model ${data.model_version.slice(0, 7)}</div>`;
    card.hidden = false;
    addPredictionOverlay(map, zones);
  } catch {
    showPredictionFallback(card, body);
  }
}

function showPredictionFallback(card: HTMLElement, body: HTMLElement): void {
  body.innerHTML =
    '<div class="note">Live predictions need the API: run the Docker ' +
    "image or uvicorn; see README.</div>";
  card.hidden = false;
}

function renderDrift(drift: SiteData["drift"]): void {
  const el = document.getElementById("drift-chart");
  const cap = document.getElementById("drift-caption");
  if (!el || !cap || !drift) return;
  const model = buildDriftChart(drift);
  el.innerHTML = renderDriftChartSvg(model);
  const months = drift.months;
  const degraded = months.filter((m) => drift.degraded[m]);
  const cut = Number.isFinite(drift.degraded_cut)
    ? formatMinutes(drift.degraded_cut)
    : "the cut";
  cap.textContent = degraded.length
    ? `Degraded months (MAE above ${cut}): ${degraded.join(", ")}`
    : `No degraded months in this window (cut ${cut}).`;
}

function renderDays(trips: TripsFile, selected: string): void {
  const el = document.getElementById("days-body");
  if (!el) return;
  const tags = Object.keys(trips.days);
  if (tags.length === 0) {
    el.textContent = "No demo days in this export.";
    return;
  }
  el.innerHTML = tags
    .map((tag) => {
      const d = trips.days[tag];
      const mark = tag === selected ? " (selected)" : "";
      const per = d.real_trips_per_taxi;
      const perTxt =
        per != null
          ? `; 1 taxi = about ${Math.round(per).toLocaleString()} real trips`
          : "";
      return (
        `<div>${d.date}${mark}: about ` +
        `${d.est_real_trips.toLocaleString()} real trips, ` +
        `${d.n_exported.toLocaleString()} shown${perTxt}</div>`
      );
    })
    .join("");
}

function renderDaySelector(
  trips: TripsFile,
  selected: string,
  onPick: (tag: string) => void,
): void {
  const el = document.getElementById("day-selector");
  if (!el) return;
  el.innerHTML = "";
  for (const tag of Object.keys(trips.days)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "day-btn" + (tag === selected ? " active" : "");
    btn.textContent = formatDayDate(trips.days[tag].date);
    btn.addEventListener("click", () => onPick(tag));
    el.appendChild(btn);
  }
}

function tripLayerProps(records: TripRecord[], currentTime: number) {
  return {
    id: "taxis",
    data: records,
    getPath: (d: TripRecord) => d.path,
    getTimestamps: (d: TripRecord) => d.timestamps,
    // Route coords are [lon, lat] pairs; the default "XYZ" format reads
    // three floats per position and corrupts every path.
    positionFormat: "XY" as const,
    getColor: [255, 209, 102] as [number, number, number],
    opacity: 0.95,
    trailLength: 1500,
    currentTime,
    pickable: true,
    widthMinPixels: 2,
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
    // diffs by layer id, so the data buffers are reused). The day
    // selector swaps the record set and restarts the loop.
    const dayTags = Object.keys(data.trips.trips);
    let activeTag = dayTags[0];
    let records: TripRecord[] = activeTag
      ? buildTripRecords(data.trips.trips[activeTag], data.routes)
      : [];

    // Hover tooltip: actual vs predicted minutes for the picked trip.
    const tip = document.createElement("div");
    tip.id = "trip-tooltip";
    tip.hidden = true;
    mapEl.appendChild(tip);

    // Overlaid mode: deck.gl 9.4's interleaved path reads map.transform,
    // which maplibre-gl v6 removed, so the custom layer threw every frame
    // and nothing rendered. The overlaid canvas owns its own animation
    // loop and needs only public map APIs.
    const overlay = new MapboxOverlay({
      interleaved: false,
      layers: [new TripsLayer(tripLayerProps(records, 0))],
      onHover: (info) => {
        const obj = info.object as TripRecord | undefined;
        if (obj && typeof obj.minutes === "number") {
          tip.textContent =
            `${formatMinutes(obj.minutes)} actual / ` +
            `${formatMinutes(obj.pred_minutes)} predicted`;
          tip.style.left = `${info.x + 14}px`;
          tip.style.top = `${info.y + 14}px`;
          tip.hidden = false;
        } else {
          tip.hidden = true;
        }
      },
    });
    map.addControl(overlay as unknown as maplibregl.IControl);

    // Play/pause + clock. prefers-reduced-motion starts paused on one
    // static frame; the button is always available.
    const clockEl = document.getElementById("clock");
    const playBtn = document.getElementById("play-pause");
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    let playing = !reduced;
    let frozenT = 0;

    const syncPlayBtn = () => {
      if (playBtn) playBtn.textContent = playing ? "Pause" : "Play";
    };
    syncPlayBtn();
    playBtn?.addEventListener("click", () => {
      playing = !playing;
      if (playing) start = performance.now() - frozenT;
      syncPlayBtn();
    });

    let start = performance.now() - frozenT;
    const tick = () => {
      const now = performance.now();
      const t = playing ? (now - start) % LOOP_MS : frozenT;
      if (playing) frozenT = t;
      if (clockEl) clockEl.textContent = formatClock(t);
      overlay.setProps({
        layers: [new TripsLayer(tripLayerProps(records, t))],
      });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    const setDay = (tag: string) => {
      if (!(tag in data.trips.trips)) return;
      activeTag = tag;
      records = buildTripRecords(data.trips.trips[tag], data.routes);
      start = performance.now();
      frozenT = 0;
      overlay.setProps({
        layers: [new TripsLayer(tripLayerProps(records, 0))],
      });
      renderDaySelector(data.trips, activeTag, setDay);
      renderDays(data.trips, activeTag);
    };
    renderDaySelector(data.trips, activeTag, setDay);

    void tryLivePrediction(map, data.zones);
  });

  renderDrift(data.drift);
  renderDays(data.trips, Object.keys(data.trips.trips)[0] ?? "");
}

void main();
