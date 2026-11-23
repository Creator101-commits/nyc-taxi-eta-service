"""OSM drive-graph routing for zone-centroid road distances.

Heavy dependencies (osmnx, networkx, geopandas) are imported lazily inside
the functions that need them; they are installed only in the Kaggle kernel,
not in the base package environment.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nyc_eta import config
from nyc_eta.zones import ZoneInfo

OVERPASS_STATUS_URL = "https://overpass-api.de/api/status"


@dataclass(frozen=True)
class GraphRecord:
    graphml_path: str
    sha256: str
    osm_base: str
    n_nodes: int
    n_edges: int


def fetch_osm_base() -> str:
    """Overpass replication timestamp (osm_base) at query time."""
    try:
        req = urllib.request.Request(
            OVERPASS_STATUS_URL, headers={"User-Agent": "nyc-eta-research/0.1"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = json.loads(resp.read().decode())
        return str(status.get("osm3s", {}).get("timestamp_osm_base", ""))
    except Exception:
        return ""


def download_drive_graph(work_dir: Path) -> tuple[Any, GraphRecord]:
    """Download the NYC drive graph via OSMnx and save GraphML."""
    import osmnx as ox  # noqa: PLC0415 - kernel-only dependency

    osm_base = fetch_osm_base()
    graph = ox.graph_from_place(
        config.OSM_PLACE, network_type=config.OSM_NETWORK_TYPE
    )
    graphml = work_dir / "nyc_drive_graph.graphml"
    ox.save_graphml(graph, graphml)
    sha = hashlib.sha256(graphml.read_bytes()).hexdigest()
    record = GraphRecord(
        graphml_path=str(graphml),
        sha256=sha,
        osm_base=osm_base,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
    )
    return graph, record


def load_graph(graphml_path: Path) -> Any:
    import osmnx as ox  # noqa: PLC0415

    return ox.load_graphml(graphml_path)


def zone_nodes(graph: Any, infos: dict[int, ZoneInfo]) -> dict[int, int]:
    """Nearest drive-graph node to each zone centroid."""
    import osmnx as ox  # noqa: PLC0415

    nearest = ox.distance.nearest_nodes
    nodes: dict[int, int] = {}
    for loc_id, info in infos.items():
        if np.isnan(info.lat) or np.isnan(info.lon):
            continue
        nodes[loc_id] = int(nearest(graph, info.lon, info.lat))
    return nodes


def all_pairs_road_km(
    graph: Any, nodes: dict[int, int]
) -> dict[tuple[int, int], float]:
    """Directed shortest-path distances between every zone pair, in km.

    One Dijkstra per source zone (263 runs) covers all targets.
    """
    import networkx as nx  # noqa: PLC0415

    distances: dict[tuple[int, int], float] = {}
    for pu_id, src in nodes.items():
        lengths = nx.single_source_dijkstra_path_length(
            graph, src, weight="length"
        )
        for do_id, dst in nodes.items():
            if pu_id == do_id:
                distances[(pu_id, do_id)] = 0.0
                continue
            meters = lengths.get(dst)
            if meters is not None:
                distances[(pu_id, do_id)] = meters / 1000.0
    return distances


def save_distance_table(
    distances: dict[tuple[int, int], float], zone_ids: list[int], path: Path
) -> None:
    """Persist the directed distance matrix as npz."""
    n = len(zone_ids)
    index = {z: i for i, z in enumerate(zone_ids)}
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    for (pu, do), km in distances.items():
        if pu in index and do in index:
            matrix[index[pu], index[do]] = km
    np.savez_compressed(path, zone_ids=np.array(zone_ids), matrix=matrix)


def load_distance_table(path: Path) -> tuple[list[int], np.ndarray]:
    data = np.load(path)
    return [int(z) for z in data["zone_ids"]], data["matrix"]


def distance_dict(zone_ids: list[int], matrix: np.ndarray) -> dict[tuple[int, int], float]:
    """Matrix -> {(pu, do): km} for FeatureBuilder."""
    out: dict[tuple[int, int], float] = {}
    for i, pu in enumerate(zone_ids):
        for j, do in enumerate(zone_ids):
            v = matrix[i, j]
            if np.isfinite(v):
                out[(int(pu), int(do))] = float(v)
    return out


def route_geometries(
    graph: Any,
    nodes: dict[int, int],
    infos: dict[int, ZoneInfo],
    pairs: set[tuple[int, int]],
    max_points: int = 120,
) -> dict[str, list[list[float]]]:
    """Simplified lon/lat polylines for the given directed zone pairs.

    Each unique pair is routed once; coordinates are stride-sampled to at
    most ``max_points``. These are drive-graph paths between centroids and
    are illustrative, not GPS traces.
    """
    import networkx as nx  # noqa: PLC0415

    routes: dict[str, list[list[float]]] = {}
    for pu, do in pairs:
        if pu not in nodes or do not in nodes:
            continue
        try:
            path = nx.shortest_path(graph, nodes[pu], nodes[do], weight="length")
        except nx.NetworkXNoPath:
            continue
        coords = [
            [float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])]
            for n in path
        ]
        if len(coords) > max_points:
            stride = int(np.ceil(len(coords) / max_points))
            coords = coords[::stride]
            if coords[-1] != [float(graph.nodes[path[-1]]["x"]),
                              float(graph.nodes[path[-1]]["y"])]:
                coords.append(
                    [float(graph.nodes[path[-1]]["x"]),
                     float(graph.nodes[path[-1]]["y"])]
                )
        routes[f"{pu}>{do}"] = coords
    return routes
