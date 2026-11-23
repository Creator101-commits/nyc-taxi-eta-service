"""One-off kernel: build the NYC drive graph dataset.

Downloads the OSMnx drive graph for New York City once, computes the
263x263 directed zone-centroid road distances, and writes the dataset
payload to /kaggle/working/graph_dataset/:

  - nyc_drive_graph.graphml  (the graph itself)
  - distances.npz            (zone_ids + directed km matrix)
  - osm_base.txt             (Overpass replication timestamp)
  - provenance.json          (sha256 of graphml, node/edge counts,
                              zone-file hashes, timings)

scripts/build_graph_dataset.sh pushes this kernel, downloads the output
and creates the private Kaggle dataset sreeharshak101/nyc-taxi-eta-osm-graph.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "graph_dataset"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "nyc_eta").is_dir():
            return candidate.parent
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    proj_root = _find_project_root()
    if proj_root is None:
        raise FileNotFoundError("source dataset not found")
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    _run([sys.executable, "-m", "pip", "install", "--quiet",
          "osmnx==2.1.1"])

    from nyc_eta import config, data, routing  # noqa: PLC0415
    from nyc_eta.zones import build_zone_table  # noqa: PLC0415

    raw = WORK / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lookup = raw / "taxi_zone_lookup.csv"
    zones_zip = raw / "taxi_zones.zip"
    zone_records = []
    for url, dest in (
        (config.TLC_ZONE_LOOKUP_URL, lookup),
        (config.TLC_ZONES_URL, zones_zip),
    ):
        rec = data.download_file(url, dest)
        zone_records.append({"url": rec.url, "sha256": rec.sha256,
                             "size_bytes": rec.size_bytes,
                             "last_modified": rec.last_modified})

    _, zone_infos, _ = build_zone_table(lookup, zones_zip)

    graph, grec = routing.download_drive_graph(WORK)
    nodes = routing.zone_nodes(graph, zone_infos)
    print(f"graph: {grec.n_nodes} nodes, {grec.n_edges} edges; "
          f"{len(nodes)} zone nodes", flush=True)

    t_dist = time.time()
    dists = routing.all_pairs_road_km(graph, nodes)
    zone_ids = sorted(zone_infos)
    OUT.mkdir(parents=True, exist_ok=True)
    dist_path = OUT / "distances.npz"
    routing.save_distance_table(dists, zone_ids, dist_path)
    print(f"distances: {len(dists)} pairs in "
          f"{time.time() - t_dist:.1f}s", flush=True)

    graphml = Path(grec.graphml_path)
    shutil.copyfile(graphml, OUT / "nyc_drive_graph.graphml")
    (OUT / "osm_base.txt").write_text(grec.osm_base + "\n")
    provenance = {
        "graphml_sha256": _sha256(OUT / "nyc_drive_graph.graphml"),
        "distances_sha256": _sha256(dist_path),
        "osm_base": grec.osm_base,
        "n_nodes": grec.n_nodes,
        "n_edges": grec.n_edges,
        "n_zone_pairs": len(dists),
        "zone_files": zone_records,
        "place": config.OSM_PLACE,
        "network_type": config.OSM_NETWORK_TYPE,
        "total_s": round(time.time() - t0, 2),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)
    print(f"graph dataset payload -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
