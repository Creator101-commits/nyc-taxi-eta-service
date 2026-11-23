"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from nyc_eta import config
from nyc_eta.experiments import run_experiment

app = typer.Typer(help="NYC taxi ETA study and service.")


@app.command()
def run(
    work_dir: Path = typer.Option(Path("work"), help="Working directory."),
    smoke: bool = typer.Option(False, help="Smoke mode: 3 months, small samples."),
) -> None:
    """Run the frozen protocol (train/val/deploy months)."""
    manifest = run_experiment(work_dir, smoke=smoke)
    typer.echo(json.dumps({
        "ordering_ok": manifest["ordering_ok"],
        "total_s": manifest["total_s"],
        "git_sha": manifest["git_sha"],
    }, indent=2))


@app.command()
def serve(
    bundle: Path = typer.Option(Path("results/model_bundle")),
    web_dist: Path = typer.Option(Path("web/dist")),
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Serve the prediction API and the built website."""
    import uvicorn

    from nyc_eta.service import create_app

    uvicorn.run(create_app(bundle, web_dist), host=host, port=port)


@app.command()
def months() -> None:
    """Print the frozen month split."""
    typer.echo(json.dumps({
        "train": config.TRAIN_MONTHS,
        "validation": config.VALIDATION_MONTHS,
        "deploy_first": config.DEPLOY_MONTHS[0],
        "deploy_last": config.DEPLOY_MONTHS[-1],
        "served_train": config.SERVED_TRAIN_MONTHS,
        "smoke": config.SMOKE_MONTHS,
    }, indent=2))
