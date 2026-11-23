"""TLC yellow-taxi monthly downloads, frozen cleaning and sampling.

Cleaning is identical for every month and every run; the count after each
filter is recorded so the manifest can show exactly what was dropped.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nyc_eta import config

TRIP_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
]


@dataclass(frozen=True)
class FileRecord:
    """Provenance for one downloaded file."""

    url: str
    path: str
    sha256: str
    size_bytes: int
    last_modified: str


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> FileRecord:
    """Download ``url`` to ``dest`` while hashing; returns provenance."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    size = 0
    last_modified = ""
    req = urllib.request.Request(url, headers={"User-Agent": "nyc-eta-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        last_modified = resp.headers.get("Last-Modified", "")
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            sha.update(chunk)
            size += len(chunk)
    return FileRecord(
        url=url,
        path=str(dest),
        sha256=sha.hexdigest(),
        size_bytes=size,
        last_modified=last_modified,
    )


def month_url(month: str) -> str:
    return f"{config.TLC_BASE_URL}/yellow_tripdata_{month}.parquet"


def month_seed(month: str) -> int:
    """Deterministic per-month sampling seed."""
    return config.SAMPLE_SEED_BASE + int(month.replace("-", ""))


def load_month_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=TRIP_COLUMNS)


def clean_trips(df: pl.DataFrame, month: str) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply the frozen cleaning chain; returns filtered frame + counts.

    Filter order (identical every month):
      1. pickup inside the file's month
      2. duration in [1, 180] minutes
      3. pickup and dropoff zone ids in 1..263
      4. dropoff not before pickup
    """
    year, mon = (int(p) for p in month.split("-"))
    counts: dict[str, int] = {"raw": df.height}

    df = df.filter(
        (pl.col("tpep_pickup_datetime").dt.year() == year)
        & (pl.col("tpep_pickup_datetime").dt.month() == mon)
    )
    counts["pickup_in_month"] = df.height

    minutes = (
        pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
    ).dt.total_minutes()
    df = df.with_columns(minutes.alias("_minutes_raw")).filter(
        pl.col("_minutes_raw").is_between(
            config.MIN_TRIP_MINUTES,
            config.MAX_TRIP_MINUTES,
            closed="both",
        )
    )
    counts["duration_ok"] = df.height

    df = df.filter(
        pl.col("PULocationID").is_between(
            config.VALID_ZONE_MIN, config.VALID_ZONE_MAX
        )
        & pl.col("DOLocationID").is_between(
            config.VALID_ZONE_MIN, config.VALID_ZONE_MAX
        )
    )
    counts["zones_ok"] = df.height

    df = df.filter(
        pl.col("tpep_dropoff_datetime") >= pl.col("tpep_pickup_datetime")
    )
    counts["dropoff_after_pickup"] = df.height

    return df, counts


def sample_trips(
    df: pl.DataFrame, month: str, n: int, seed_base: int | None = None
) -> pl.DataFrame:
    """Uniform sample of up to ``n`` rows with a fixed per-month seed."""
    base = config.SAMPLE_SEED_BASE if seed_base is None else seed_base
    seed = base + int(month.replace("-", ""))
    take = min(n, df.height)
    return df.sample(n=take, seed=seed, shuffle=True)


def month_frame(
    path: Path, month: str, sample_n: int, seed_base: int | None = None
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Load, clean and sample one month file."""
    df = load_month_parquet(path)
    cleaned, counts = clean_trips(df, month)
    sampled = sample_trips(cleaned, month, sample_n, seed_base=seed_base)
    return sampled, counts
