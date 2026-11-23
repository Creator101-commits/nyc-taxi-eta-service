"""Kaggle entrypoint for the NYC taxi ETA benchmark (CPU kernel).

Installs ``nyc_eta`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, locates the OSM graph dataset explicitly under
``/kaggle/input/datasets/``, runs the frozen protocol via
``nyc_eta.experiments.run_experiment`` and writes ``benchmark/`` artifacts
including ``manifest.json`` with the git SHA.

``NYC_ETA_SMOKE=1`` (set by the ``kaggle_smoke`` kernel wrapper) runs the
smoke protocol: three months, one per split, small samples.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"
WORK_DIR = WORK / "nyc_eta_work"

# Kernel-only dependencies, pinned to releases >= 7 days old.
KERNEL_DEPS = [
    "osmnx==2.1.1",
    "evidently==0.7.23",
    "mlflow==3.18.0",
    "shap==0.54.1",
]

GRAPH_DATASET_ID = "sreeharshak101/nyc-taxi-eta-osm-graph"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "nyc_eta").is_dir():
            return candidate.parent
    return None


def _install_package() -> Path:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/nyc_eta)."
        )
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    if KERNEL_DEPS:
        _run([sys.executable, "-m", "pip", "install", "--quiet",
              *KERNEL_DEPS])
    return proj


def _git_sha(proj: Path) -> str:
    info = proj / "run_info.json"
    if info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _find_graph_dataset() -> Path | None:
    """Explicit dataset mount: /kaggle/input/datasets/<owner>/<slug>."""
    explicit = KAGGLE_INPUT / "datasets" / GRAPH_DATASET_ID
    if (explicit / "nyc_drive_graph.graphml").exists():
        return explicit
    # Legacy flat mount fallback.
    flat = KAGGLE_INPUT / GRAPH_DATASET_ID.split("/")[-1]
    if (flat / "nyc_drive_graph.graphml").exists():
        return flat
    matches = [
        p.parent
        for p in KAGGLE_INPUT.rglob("nyc_drive_graph.graphml")
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"OSM graph dataset {GRAPH_DATASET_ID} not found under "
        f"{KAGGLE_INPUT}; matches={matches}"
    )


def main() -> None:
    proj = _install_package()
    sha = _git_sha(proj)
    smoke = bool(os.environ.get("NYC_ETA_SMOKE"))
    print(f"git_sha={sha} smoke={smoke}", flush=True)

    from nyc_eta.experiments import run_experiment  # noqa: PLC0415

    graph_dir = _find_graph_dataset()
    print(f"osm graph dataset: {graph_dir}", flush=True)
    # experiments._find_kaggle_input resolves the same explicit path.
    BENCH.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    manifest = run_experiment(WORK_DIR, smoke=smoke, git_sha=sha)

    # Publish artifacts: plan, manifest, web data, mlflow db, model bundle.
    out = WORK_DIR / "out"
    for name in ("analysis_plan.json", "manifest.json"):
        src = out / name
        if src.exists():
            shutil.copyfile(src, BENCH / name)
    web_src = out / "web"
    if web_src.exists():
        shutil.copytree(web_src, BENCH / "web", dirs_exist_ok=True)
    for extra in ("shap_importance.json",):
        src = out / extra
        if src.exists():
            shutil.copyfile(src, BENCH / extra)
    mlflow_db = WORK_DIR / "mlflow.db"
    if mlflow_db.exists():
        shutil.copyfile(mlflow_db, BENCH / "mlflow.db")
    bundle = WORK_DIR / "model_bundle"
    if bundle.exists():
        shutil.copytree(bundle, BENCH / "model_bundle", dirs_exist_ok=True)
    dist_table = WORK_DIR / "distances.npz"
    if dist_table.exists():
        shutil.copyfile(dist_table, BENCH / "distances.npz")

    print("=== GATES ===", flush=True)
    print(json.dumps({
        "ordering_ok": manifest["ordering_ok"],
        "git_sha": manifest["git_sha"],
        "smoke": manifest["smoke"],
        "total_s": manifest["total_s"],
        "timings_s": manifest["timings_s"],
    }, indent=2), flush=True)
    print(f"artifacts -> {BENCH}", flush=True)


if __name__ == "__main__":
    main()
