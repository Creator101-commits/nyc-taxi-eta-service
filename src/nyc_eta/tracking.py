"""MLflow tracking with an explicit sqlite backend in the kernel output."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


def init_tracking(mlruns_db: Path) -> str:
    """Point MLflow at a sqlite file inside the run output directory."""
    import mlflow  # noqa: PLC0415 - kernel-only dependency

    mlruns_db.parent.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{mlruns_db}"
    mlflow.set_tracking_uri(uri)
    return uri


@contextmanager
def run_context(
    name: str, params: dict[str, Any], tags: dict[str, str] | None = None
) -> Iterator[object]:
    """One MLflow run; yields the run object for metric logging."""
    import mlflow  # noqa: PLC0415

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_metrics(run: object, metrics: dict[str, float]) -> None:
    import mlflow  # noqa: PLC0415

    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def log_training_frame(run: object, df: pd.DataFrame, name: str) -> None:
    import mlflow  # noqa: PLC0415

    mlflow.log_text(df.head(200).to_csv(index=False), f"{name}_sample.csv")
