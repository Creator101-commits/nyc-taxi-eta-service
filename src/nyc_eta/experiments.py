"""Frozen-protocol orchestration for the NYC taxi ETA study.

Phase order is a leakage contract:

  1. download + clean + sample TRAIN and VALIDATION months
  2. build features, run the model grid, select on validation MAE
  3. fit conformal interval on validation residuals
  4. compute label-free drift signals on validation months -> thresholds
  5. persist analysis_plan.json (the freeze point)
  6. only then read DEPLOYMENT month files; compute signals, run the
     static / monthly / drift policies, monthly metrics, coverage
  7. write manifest.json with ordering_ok and all provenance

Smoke mode uses SMOKE_MONTHS and SMOKE_SAMPLE_PER_MONTH but keeps the
same phase order and records every reduction in the manifest.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nyc_eta import config, data, drift, metrics, models, policies
from nyc_eta.conformal import empirical_coverage, fit_conformal
from nyc_eta.features import FeatureBuilder
from nyc_eta.routing import (
    all_pairs_road_km,
    distance_dict,
    download_drive_graph,
    load_distance_table,
    load_graph,
    route_geometries,
    save_distance_table,
    zone_nodes,
)
from nyc_eta.zones import ZoneInfo, build_zone_table


@dataclass
class PhaseLog:
    timings: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, t0: float) -> None:
        self.timings[name] = round(time.time() - t0, 2)


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return os.environ.get("NYC_ETA_GIT_SHA", "unknown")



def _download_months(
    months: list[str], raw_dir: Path,
    records: list[data.FileRecord], sample_n: int
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Download, clean and sample each month; returns frames + counts."""
    frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, int]] = {}
    for month in months:
        url = data.month_url(month)
        dest = raw_dir / f"yellow_tripdata_{month}.parquet"
        rec = data.download_file(url, dest)
        records.append(rec)
        sampled, cnt = data.month_frame(dest, month, sample_n)
        counts[month] = cnt
        frames[month] = sampled.to_pandas()
        # sha256 is already recorded in ``rec``; the raw file is not
        # needed again, so delete it to keep the work dir small.
        dest.unlink(missing_ok=True)
    return frames, counts


def _signals_for_month(
    builder_ref: pd.DataFrame,
    ref_pred: np.ndarray,
    ref_monthly_mean: float,
    month_df: pd.DataFrame,
    pred: np.ndarray,
    cur_n: int,
) -> drift.DriftSignals:
    return drift.compute_signals(
        ref_features=builder_ref[config.FEATURE_COLUMNS],
        ref_pred=ref_pred,
        ref_monthly_mean=ref_monthly_mean,
        cur_features=month_df[config.FEATURE_COLUMNS],
        cur_pred=pred,
        cur_n=cur_n,
    )


def run_experiment(
    work_dir: Path,
    smoke: bool = False,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Run the frozen protocol; returns the manifest dict."""
    t_start = time.time()
    log = PhaseLog()
    git_sha = git_sha or _git_sha()
    sample_n = config.SMOKE_SAMPLE_PER_MONTH if smoke else config.SAMPLE_PER_MONTH
    train_months = config.SMOKE_MONTHS["train"] if smoke else config.TRAIN_MONTHS
    val_months = (
        config.SMOKE_MONTHS["validation"] if smoke else config.VALIDATION_MONTHS
    )
    deploy_months = (
        config.SMOKE_MONTHS["deploy"] if smoke else config.DEPLOY_MONTHS
    )

    raw_dir = work_dir / "raw"
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_records: list[data.FileRecord] = []

    from nyc_eta import tracking  # noqa: PLC0415 - kernel-only dep chain

    tracking_uri = tracking.init_tracking(work_dir / "mlflow.db")
    print(f"mlflow tracking uri: {tracking_uri}", flush=True)

    # --- zones + lookup -------------------------------------------------
    t0 = time.time()
    zone_lookup_csv = raw_dir / "taxi_zone_lookup.csv"
    zones_zip = raw_dir / "taxi_zones.zip"
    if not zone_lookup_csv.exists():
        file_records.append(
            data.download_file(config.TLC_ZONE_LOOKUP_URL, zone_lookup_csv)
        )
        file_records.append(
            data.download_file(config.TLC_ZONES_URL, zones_zip)
        )
    _, zone_infos, zones_geojson = build_zone_table(zone_lookup_csv, zones_zip)
    log.mark("zones", t0)

    # --- OSM graph: reuse the pinned private dataset when mounted -------
    # The GraphML is only parsed lazily (routes / fallback distances); the
    # dataset's provenance.json supplies the recorded hashes and counts.
    t0 = time.time()
    graph_dataset = _find_kaggle_input(config.OSM_GRAPH_DATASET_ID)
    dist_table_path = work_dir / "distances.npz"
    graphml_path: Path | None = None
    graph_record: dict[str, Any] = {}
    if graph_dataset is not None:
        prov_path = graph_dataset / "provenance.json"
        prov: dict[str, Any] = (
            json.loads(prov_path.read_text()) if prov_path.exists() else {}
        )
        graphml_path = graph_dataset / "nyc_drive_graph.graphml"
        ds_dist = graph_dataset / "distances.npz"
        if ds_dist.exists():
            import shutil

            shutil.copyfile(ds_dist, dist_table_path)
        graph_record = {
            "source": "kaggle_dataset",
            "dataset_id": config.OSM_GRAPH_DATASET_ID,
            "sha256": prov.get("graphml_sha256")
            or (_sha256_file(graphml_path) if graphml_path.exists() else ""),
            "osm_base": prov.get("osm_base")
            or _read_text(graph_dataset / "osm_base.txt"),
            "n_nodes": prov.get("n_nodes"),
            "n_edges": prov.get("n_edges"),
            "distances_sha256": prov.get("distances_sha256", ""),
        }
    else:
        # Local/dev fallback: build the graph once (kernel path uses dataset).
        _graph, grec = download_drive_graph(work_dir)
        graphml_path = Path(grec.graphml_path)
        graph_record = {
            "source": "downloaded",
            "sha256": grec.sha256,
            "osm_base": grec.osm_base,
            "n_nodes": grec.n_nodes,
            "n_edges": grec.n_edges,
        }
    log.mark("osm_graph", t0)

    # --- distances between zone centroids -------------------------------
    t0 = time.time()
    if dist_table_path.exists():
        zone_ids, matrix = load_distance_table(dist_table_path)
    else:
        if graphml_path is None or not graphml_path.exists():
            raise FileNotFoundError("no graphml available for distances")
        graph = load_graph(graphml_path)
        nodes = zone_nodes(graph, zone_infos)
        dists = all_pairs_road_km(graph, nodes)
        zone_ids = sorted(zone_infos)
        save_distance_table(dists, zone_ids, dist_table_path)
        zone_ids, matrix = load_distance_table(dist_table_path)
    road_km = distance_dict(zone_ids, matrix)
    log.mark("distances", t0)

    builder = FeatureBuilder(zone_infos, road_km)

    # --- train + validation months (pre-freeze reads) --------------------
    t0 = time.time()
    pre_months = train_months + val_months
    pre_frames, clean_counts = _download_months(
        pre_months, raw_dir, file_records, sample_n
    )
    log.mark("download_pre", t0)

    t0 = time.time()
    feats: dict[str, pd.DataFrame] = {}
    for m, df in pre_frames.items():
        fdf = builder.build_frame(df)
        feats[m] = fdf
    log.mark("features_pre", t0)

    train_df = pd.concat([feats[m] for m in train_months], ignore_index=True)
    val_df = pd.concat([feats[m] for m in val_months], ignore_index=True)

    # --- model grid + validation-only selection -------------------------
    t0 = time.time()
    with tracking.run_context(
        "model_selection",
        params={"grid": config.MODEL_GRID, "git_sha": git_sha,
                "smoke": smoke},
        tags={"git_sha": git_sha},
    ) as sel_run:
        best, grid_records = models.select_model(train_df, val_df)
        for rec in grid_records:
            tracking.log_metrics(sel_run, {
                f"val_mae_leaves{rec['num_leaves']}_leaf{rec['min_data_in_leaf']}":
                    float(rec["val_mae"]),
                f"best_iter_leaves{rec['num_leaves']}_leaf{rec['min_data_in_leaf']}":
                    float(rec["best_iteration"]),
            })
    log.mark("model_grid", t0)

    # --- zone-pair baseline reference (train-only fit, val-scored) -----
    base = models.ZonePairBaseline().fit(train_df)
    baseline_val_mae = metrics.mae(
        val_df["minutes"].to_numpy(), base.predict(val_df)
    )

    # --- conformal on validation residuals ------------------------------
    val_pred_min = models.predict_minutes(best, val_df)
    interval = fit_conformal(
        val_df["minutes"].to_numpy(), val_pred_min, config.CONFORMAL_ALPHA
    )
    val_coverage = empirical_coverage(
        val_df["minutes"].to_numpy(), val_pred_min, interval
    )
    val_borough_slices = metrics.borough_slices(val_df, val_pred_min)

    # --- label-free drift signals on validation months ------------------
    t0 = time.time()
    ref_features = feats[train_months[0]][config.FEATURE_COLUMNS]
    ref_pred = models.predict_minutes(best, feats[train_months[0]])
    # Volume uses cleaned trip counts, not the capped sample size, so a
    # real demand drop registers instead of always reading ~1.0.
    pre_volume_n = drift.cleaned_counts_for_months(clean_counts, pre_months)
    ref_monthly_mean = float(
        np.mean([pre_volume_n[m] for m in train_months])
    )
    val_signals = [
        _signals_for_month(
            ref_features, ref_pred, ref_monthly_mean,
            feats[m], models.predict_minutes(best, feats[m]),
            cur_n=pre_volume_n[m],
        )
        for m in val_months
    ]
    thresholds = drift.alarm_thresholds(val_signals)
    log.mark("drift_val", t0)

    # --- FREEZE: analysis_plan.json before any deploy-month read ---------
    card_write_epoch = time.time()
    plan: dict[str, Any] = {
        "protocol": config.PROTOCOL,
        "git_sha": git_sha,
        "smoke": smoke,
        "sample_per_month": sample_n,
        "train_months": train_months,
        "validation_months": val_months,
        "deploy_months": deploy_months,
        "model_grid": grid_records,
        "selected": {
            "num_leaves": best.config.num_leaves,
            "min_data_in_leaf": best.config.min_data_in_leaf,
            "best_iteration": best.best_iteration,
            "cap_bound": best.cap_bound,
            "val_mae": best.val_mae,
        },
        "conformal": {"alpha": interval.alpha, "q": interval.q,
                      "val_coverage": val_coverage},
        "baseline_val_mae": baseline_val_mae,
        "val_borough_slices": val_borough_slices,
        "drift_thresholds": thresholds,
        "volume_ref_monthly_mean": ref_monthly_mean,
        "val_signals": [
            {"month": m, **{k: getattr(s, k) for k in config.DRIFT_SIGNALS},
             "cleaned_n": int(pre_volume_n[m]),
             "sample_n": int(len(feats[m]))}
            for m, s in zip(val_months, val_signals, strict=False)
        ],
        "card_write_epoch": card_write_epoch,
    }
    (out_dir / "analysis_plan.json").write_text(json.dumps(plan, indent=2))

    # --- deployment months: first deploy read happens now ---------------
    first_deploy_read_epoch = time.time()
    deploy_frames, deploy_counts = _download_months(
        deploy_months, raw_dir, file_records, sample_n
    )
    clean_counts.update(deploy_counts)
    log.mark("download_deploy", first_deploy_read_epoch)

    t0 = time.time()
    for m, df in deploy_frames.items():
        feats[m] = builder.build_frame(df)
    log.mark("features_deploy", t0)

    # --- label-free signals + degraded flags + policies ------------------
    t0 = time.time()
    deploy_volume_n = drift.cleaned_counts_for_months(
        clean_counts, deploy_months
    )
    deploy_signals: dict[str, drift.DriftSignals] = {}
    monthly_records: dict[str, dict[str, Any]] = {}
    degraded: dict[str, bool] = {}
    val_mae = best.val_mae

    month_data: dict[str, policies.MonthData] = {}
    deploy_dates: dict[str, np.ndarray] = {}
    for m in deploy_months:
        fdf = feats[m]
        pred = models.predict_minutes(best, fdf)
        sig = _signals_for_month(
            ref_features, ref_pred, ref_monthly_mean, fdf, pred,
            cur_n=deploy_volume_n[m],
        )
        deploy_signals[m] = sig
        y_m = fdf["minutes"].to_numpy()
        mae_m = metrics.mae(y_m, pred)
        degraded[m] = bool(mae_m > config.DEGRADED_MAE_RATIO * val_mae)
        monthly_records[m] = {
            "mae": mae_m,
            "rmse": metrics.rmse(y_m, pred),
            "median_ape": metrics.median_ape(y_m, pred),
            "n": len(fdf),
            "cleaned_n": int(deploy_volume_n[m]),
            "baseline_mae": metrics.mae(y_m, base.predict(fdf)),
            "borough_mae": metrics.borough_slices(fdf, pred),
            "coverage": empirical_coverage(y_m, pred, interval),
            "signals": {k: getattr(sig, k) for k in config.DRIFT_SIGNALS},
            "signal_alarms": drift.signal_alarms(sig, thresholds),
            "degraded": degraded[m],
        }
        month_data[m] = policies.MonthData(
            month=m,
            features=fdf[["minutes"] + config.FEATURE_COLUMNS],
            minutes=y_m,
            signals=sig,
        )
        deploy_dates[m] = fdf["pickup"].dt.floor("D").to_numpy()
    log.mark("deploy_eval", t0)

    # --- retraining policies ---------------------------------------------
    t0 = time.time()
    labeled_pool = {
        m: feats[m][["minutes"] + config.FEATURE_COLUMNS]
        for m in train_months + val_months
    }

    def fit_fn(train_pdf: pd.DataFrame) -> models.TrainedModel:
        # Frozen selected config AND frozen iteration count: retrain for
        # exactly best_iteration rounds with no early stopping.
        cfg = models.LGBMConfig(
            num_leaves=best.config.num_leaves,
            min_data_in_leaf=best.config.min_data_in_leaf,
        )
        with tracking.run_context(
            "policy_retrain",
            params={
                "num_leaves": cfg.num_leaves,
                "min_data_in_leaf": cfg.min_data_in_leaf,
                "iterations": best.best_iteration,
                "n_rows": int(len(train_pdf)),
            },
            tags={"git_sha": git_sha},
        ) as rrun:
            model = models.train_lgbm_fixed(
                train_pdf, cfg, iterations=best.best_iteration
            )
            tracking.log_metrics(rrun, {"n_rows": float(len(train_pdf))})
            return model

    def predict_fn(model: object, fdf: pd.DataFrame) -> np.ndarray:
        if isinstance(model, models.TrainedModel):
            return models.predict_minutes(model, fdf)
        if isinstance(model, models.ZonePairBaseline):
            return model.predict(fdf)
        raise TypeError(f"unknown model type: {type(model)!r}")

    pol_results = policies.simulate_policies(
        deploy_months, month_data, labeled_pool,
        fit_fn=fit_fn, predict_fn=predict_fn,
        static_model=best, signal_thresholds=thresholds,
    )
    log.mark("policies", t0)

    # --- drift <-> MAE correlations + alarm scores -----------------------
    deploy_list = list(deploy_months)
    maes = [monthly_records[m]["mae"] for m in deploy_list]
    val_mae_ratio = [m / val_mae for m in maes]

    def _spearman_lagged(xs: list[float], lag: int) -> float:
        # lag +1: signal at t vs MAE ratio at t+1 (signal leads error);
        # lag -1: signal at t vs MAE ratio at t-1; lag 0 is contemporaneous.
        if lag == 0:
            return metrics.spearman(xs, val_mae_ratio)
        if lag == 1:
            return metrics.spearman(xs[:-1], val_mae_ratio[1:])
        return metrics.spearman(xs[1:], val_mae_ratio[:-1])

    drift_report: dict[str, Any] = {}
    for sig_name in config.DRIFT_SIGNALS:
        xs = [getattr(deploy_signals[m], sig_name) for m in deploy_list]
        drift_report[sig_name] = {
            "spearman_vs_mae_ratio": metrics.spearman(xs, val_mae_ratio),
            "spearman_lagged": {
                str(lag): _spearman_lagged(xs, lag) for lag in (-1, 0, 1)
            },
            "values": {m: float(getattr(deploy_signals[m], sig_name))
                       for m in deploy_list},
        }
    all_alarms = [
        any(drift.signal_alarms(deploy_signals[m], thresholds).values())
        for m in deploy_list
    ]
    drift_report["alarm_scores"] = metrics.alarm_scores(
        [degraded[m] for m in deploy_list], all_alarms
    )

    # Post-hoc sensitivity (never used by any policy): two-sided volume
    # alarm outside the validation months' [min, max] range, combined
    # with the unchanged one-sided drift_share / pred_wasserstein rules.
    val_vol = [s.volume_ratio for s in val_signals]
    vol_low, vol_high = min(val_vol), max(val_vol)
    two_sided: dict[str, bool] = {}
    posthoc_alarms: list[bool] = []
    for m in deploy_list:
        sig = deploy_signals[m]
        vol_out = not (vol_low <= sig.volume_ratio <= vol_high)
        alarms = drift.signal_alarms(sig, thresholds)
        two_sided[m] = bool(vol_out)
        posthoc_alarms.append(
            bool(vol_out or alarms["drift_share"]
                 or alarms["pred_wasserstein"])
        )
    drift_report["posthoc_two_sided_volume"] = {
        "label": "post-hoc sensitivity; not used by any policy",
        "rule": "volume alarm when outside [min, max] of the validation months",
        "low": float(vol_low),
        "high": float(vol_high),
        "alarm_months": [m for m in deploy_list if two_sided[m]],
        "alarm_scores": metrics.alarm_scores(
            [degraded[m] for m in deploy_list], posthoc_alarms
        ),
    }

    # --- policy summaries --------------------------------------------------
    policy_summary: dict[str, Any] = {}
    for name, res in pol_results.items():
        vals = [res.monthly_mae[m] for m in deploy_list if m in res.monthly_mae]
        policy_summary[name] = {
            "monthly_mae": res.monthly_mae,
            "mean_mae": float(np.mean(vals)) if vals else float("nan"),
            "n_retrains": res.n_retrains,
            "retrain_months": res.retrain_months,
        }

    # --- paired day-cluster bootstrap over policy monthly MAE -----------
    t0 = time.time()
    ae_by_policy = {
        name: {m: res.monthly_ae[m] for m in deploy_list
               if m in res.monthly_ae}
        for name, res in pol_results.items()
    }
    bootstrap = metrics.paired_day_cluster_bootstrap(
        ae_by_policy, deploy_dates, n_boot=1000, seed=0
    )
    log.mark("bootstrap", t0)

    # --- SHAP global importance: pre-COVID vs April 2020 (full run) ------
    shap_report: dict[str, Any] = {"skipped": smoke}
    if not smoke and "2020-04" in feats:
        t0 = time.time()
        shap_report = _compute_shap(best, train_df, feats["2020-04"],
                                    out_dir)
        log.mark("shap", t0)

    # --- served model: final train on SERVED_TRAIN_MONTHS -----------------
    # Only in the full run; smoke records the plan but skips heavy work.
    served: dict[str, Any] = {"skipped": smoke}
    if not smoke:
        t0 = time.time()
        served = _train_served_model(
            work_dir, raw_dir, feats, builder, zone_infos,
            best, git_sha,
        )
        log.mark("served_model", t0)

    # --- web data export (always, incl. smoke) -----------------------------
    t0 = time.time()
    web_export = _export_web_data(
        out_dir, zones_geojson, feats, deploy_signals,
        monthly_records, deploy_months, graphml_path, zone_infos,
        static_model=best, pol_results=pol_results,
        thresholds=thresholds, val_mae=val_mae,
        clean_counts=clean_counts, sample_n=sample_n,
    )
    log.mark("web_export", t0)

    # --- manifest ----------------------------------------------------------
    ordering_ok = bool(card_write_epoch < first_deploy_read_epoch)
    manifest: dict[str, Any] = {
        "protocol": config.PROTOCOL,
        "git_sha": git_sha,
        "smoke": smoke,
        "ordering_ok": ordering_ok,
        "card_write_epoch": plan["card_write_epoch"],
        "first_deploy_read_epoch": first_deploy_read_epoch,
        "timings_s": log.timings,
        "total_s": round(time.time() - t_start, 2),
        "files": [
            {"url": r.url, "sha256": r.sha256, "size_bytes": r.size_bytes,
             "last_modified": r.last_modified}
            for r in file_records
        ],
        "clean_counts": clean_counts,
        "osm_graph": graph_record,
        "distances_sha256": _sha256_file(dist_table_path)
        if dist_table_path.exists() else "",
        "model_grid": grid_records,
        "selected": plan["selected"],
        "conformal": plan["conformal"],
        "baseline_val_mae": baseline_val_mae,
        "drift_thresholds": thresholds,
        "drift_report": drift_report,
        "monthly_metrics": monthly_records,
        "policy_summary": policy_summary,
        "bootstrap": bootstrap,
        "shap": shap_report,
        "served_model": served,
        "sample_per_month": sample_n,
        "web_export": web_export,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _find_kaggle_input(dataset_id: str) -> Path | None:
    """Locate a mounted Kaggle dataset under /kaggle/input/datasets/owner/slug."""
    base = Path("/kaggle/input/datasets")
    if not base.exists():
        # Legacy flat layout
        flat = Path("/kaggle/input") / dataset_id.split("/")[-1]
        return flat if flat.exists() else None
    candidate = base / dataset_id
    return candidate if candidate.exists() else None


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text().strip() if path.exists() else ""


def _compute_shap(
    best: models.TrainedModel,
    pre_df: pd.DataFrame,
    covid_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    """SHAP TreeExplainer mean |SHAP| on fixed samples: pre-COVID vs 2020-04."""
    import shap  # noqa: PLC0415 - kernel-only dep

    feats = config.FEATURE_COLUMNS
    rng = np.random.default_rng(0)
    n = min(20_000, len(pre_df), len(covid_df))
    pre_s = pre_df.iloc[rng.choice(len(pre_df), n, replace=False)]
    cov_s = covid_df.iloc[rng.choice(len(covid_df), n, replace=False)]
    explainer = shap.TreeExplainer(best.booster)
    out: dict[str, Any] = {"n_per_group": int(n)}
    for tag, frame in (("pre_covid", pre_s), ("covid_2020_04", cov_s)):
        vals = explainer.shap_values(frame[feats])
        arr = np.asarray(vals)
        if arr.ndim == 3:  # per-class layout; regression gives (n, f)
            arr = arr[:, :, 0]
        out[tag] = {
            "mean_abs_shap": {
                f: float(np.mean(np.abs(arr[:, i])))
                for i, f in enumerate(feats)
            }
        }
    (out_dir / "shap_importance.json").write_text(json.dumps(out, indent=2))
    return out


def _train_served_model(
    work_dir: Path, raw_dir: Path,
    feats: dict[str, pd.DataFrame],
    builder: FeatureBuilder, zone_infos: dict[int, ZoneInfo],
    best: models.TrainedModel,
    git_sha: str,
) -> dict[str, Any]:
    """Train the final served model on SERVED_TRAIN_MONTHS.

    Those months are deployment months, so their cleaned, sampled feature
    frames already exist in ``feats``; no re-download is needed.
    """
    missing = [m for m in config.SERVED_TRAIN_MONTHS if m not in feats]
    if missing:
        raise ValueError(f"served months missing from feats: {missing}")
    train_df = pd.concat(
        [feats[m] for m in config.SERVED_TRAIN_MONTHS], ignore_index=True
    )
    counts = {"reused_deploy_frames": True}

    # 80/20 conformal holdout: fixed random split on those months.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_df))
    n_hold = int(len(train_df) * 0.2)
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
    hold_df = train_df.iloc[hold_idx]
    fit_df = train_df.iloc[fit_idx]

    cfg = models.LGBMConfig(
        num_leaves=best.config.num_leaves,
        min_data_in_leaf=best.config.min_data_in_leaf,
    )
    # Frozen iteration count on the 80% fit split.
    from nyc_eta import tracking as _tracking  # noqa: PLC0415

    with _tracking.run_context(
        "served_model",
        params={"trained_months": config.SERVED_TRAIN_MONTHS,
                "iterations": best.best_iteration, "git_sha": git_sha},
        tags={"git_sha": git_sha},
    ) as srun:
        model = models.train_lgbm_fixed(
            fit_df, cfg, iterations=best.best_iteration
        )
        _tracking.log_metrics(srun, {"n_fit": float(len(fit_df))})

    # Conformal quantile from the 20% holdout residuals.
    hold_pred = models.predict_minutes(model, hold_df)
    served_interval = fit_conformal(
        hold_df["minutes"].to_numpy(), hold_pred, config.CONFORMAL_ALPHA
    )

    bundle = work_dir / "model_bundle"
    bundle.mkdir(exist_ok=True)
    models.save_booster(model, str(bundle / "model.txt"))
    (bundle / "taxi_zone_lookup.csv").write_bytes(
        (raw_dir / "taxi_zone_lookup.csv").read_bytes()
    )
    (bundle / "taxi_zones.zip").write_bytes(
        (raw_dir / "taxi_zones.zip").read_bytes()
    )
    dist_src = work_dir / "distances.npz"
    if dist_src.exists():
        (bundle / "distances.npz").write_bytes(dist_src.read_bytes())
    meta = {
        "booster_file": "model.txt",
        "best_iteration": best.best_iteration,
        "conformal_q": served_interval.q,
        "alpha": served_interval.alpha,
        "zone_lookup_file": "taxi_zone_lookup.csv",
        "zone_shapefile_zip": "taxi_zones.zip",
        "distance_table": "distances.npz",
        "trained_through": "2021-12",
        "trained_months": config.SERVED_TRAIN_MONTHS,
        "validation_months": config.VALIDATION_MONTHS,
        "val_mae": float(best.val_mae),
        "model_note": (
            "Served model trained on 2021-10..2021-12; predictions "
            "reflect late-2021 traffic patterns."
        ),
        "confidence": config.CONFORMAL_ALPHA,
        "git_sha": git_sha,
        "clean_counts": counts,
    }
    (bundle / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return {
        "bundle": str(bundle),
        "trained_months": config.SERVED_TRAIN_MONTHS,
        "n_train": int(len(fit_df)),
        "n_holdout": int(len(hold_df)),
        "best_iteration": best.best_iteration,
        "conformal_q": served_interval.q,
        "holdout_coverage": empirical_coverage(
            hold_df["minutes"].to_numpy(), hold_pred, served_interval
        ),
        "clean_counts": counts,
    }


def _round_coords(coords: Any, decimals: int) -> list[list[float]]:
    return [
        [round(float(x), decimals), round(float(y), decimals)]
        for x, y in coords
    ]


def _simplify_route_coords(
    coords: list[list[float]],
) -> list[list[float]]:
    """shapely simplify for one route polyline; keeps endpoints."""
    from shapely.geometry import LineString  # noqa: PLC0415

    # A one-point path (e.g. same-node pair) is not a valid LineString.
    if len(coords) < 2:
        return _round_coords(coords, config.WEB_COORD_DECIMALS)
    line = LineString(coords)
    simp = line.simplify(
        config.WEB_ROUTE_SIMPLIFY_TOL, preserve_topology=True
    )
    if simp.is_empty:
        return _round_coords(coords, config.WEB_COORD_DECIMALS)
    return _round_coords(simp.coords, config.WEB_COORD_DECIMALS)


def _simplify_zones_geojson(zones_geojson: dict[str, Any]) -> None:
    """Simplify zone polygons in place; rounds coordinates.

    Handles Polygon and MultiPolygon (zones can be multi-part); every
    geometry's coordinates are rounded to WEB_COORD_DECIMALS whether or
    not simplification produced a usable result.
    """
    from shapely.geometry import Polygon  # noqa: PLC0415

    d = config.WEB_COORD_DECIMALS
    tol = config.WEB_ZONE_SIMPLIFY_TOL

    def _poly_coords(poly: Any) -> list[Any]:
        rings = [poly.exterior, *poly.interiors]
        return [_round_coords(list(r.coords), d) for r in rings]

    def _one_polygon(rings: list[Any]) -> list[Any]:
        if not rings:
            return rings
        poly = Polygon(shell=rings[0], holes=rings[1:])
        simp = poly.simplify(tol, preserve_topology=True)
        if simp.is_empty or simp.geom_type != "Polygon":
            return [_round_coords(r, d) for r in rings]
        return _poly_coords(simp)

    for feature in zones_geojson.get("features", []):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            geom["coordinates"] = _one_polygon(coords)
        elif gtype == "MultiPolygon":
            geom["coordinates"] = [_one_polygon(p) for p in coords]


def _demo_day_frame(
    feats: dict[str, pd.DataFrame], day: str
) -> tuple[str, pd.DataFrame] | None:
    """Feature rows for one demo day, or None when the month is absent."""
    day_ts = pd.Timestamp(day)
    month = f"{day_ts.year}-{day_ts.month:02d}"
    if month not in feats:
        return None
    fdf = feats[month]
    day_df = fdf.loc[fdf["pickup"].dt.date == day_ts.date()]
    if len(day_df) == 0:
        return None
    return month, day_df


def _export_web_data(
    out_dir: Path,
    zones_geojson: dict[str, Any],
    feats: dict[str, pd.DataFrame],
    deploy_signals: dict[str, drift.DriftSignals],
    monthly_records: dict[str, dict[str, Any]],
    deploy_months: list[str],
    graphml_path: Path | None,
    zone_infos: dict[int, ZoneInfo],
    static_model: models.TrainedModel | models.ZonePairBaseline,
    pol_results: dict[str, policies.PolicyResult],
    thresholds: dict[str, float],
    val_mae: float,
    clean_counts: dict[str, dict[str, int]],
    sample_n: int,
    route_fn: Any = None,
) -> dict[str, Any]:
    """Write the static data files the website loads.

    ``route_fn(pairs) -> {"pu>do": [[lon, lat], ...]}`` is injectable so
    tests can stub routing; the kernel path routes on the OSM graph.
    Returns the web_export provenance block for the manifest.
    """
    web = out_dir / "web"
    web.mkdir(exist_ok=True)

    zones_simplified = json.loads(json.dumps(zones_geojson))
    _simplify_zones_geojson(zones_simplified)
    (web / "zones.geojson").write_text(json.dumps(zones_simplified))

    # Drift chart data: static MAE, signals, alarms, policy series.
    drift_json: dict[str, Any] = {
        "months": list(deploy_months),
        "mae": {m: monthly_records[m]["mae"] for m in deploy_months},
        "signals": {
            name: {
                m: float(getattr(deploy_signals[m], name))
                for m in deploy_months
            }
            for name in config.DRIFT_SIGNALS
        },
        "degraded": {
            m: monthly_records[m]["degraded"] for m in deploy_months
        },
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "alarms": {
            m: {
                **{
                    k: bool(v)
                    for k, v in drift.signal_alarms(
                        deploy_signals[m], thresholds
                    ).items()
                },
                "any": drift.any_alarm(
                    drift.signal_alarms(deploy_signals[m], thresholds)
                ),
            }
            for m in deploy_months
        },
        "static_coverage": {
            m: monthly_records[m]["coverage"] for m in deploy_months
        },
        "val_mae": float(val_mae),
        "degraded_cut": float(config.DEGRADED_MAE_RATIO * val_mae),
        "policies": {
            name: {
                "monthly_mae": res.monthly_mae,
                "retrain_months": res.retrain_months,
            }
            for name, res in pol_results.items()
        },
    }
    (web / "drift.json").write_text(json.dumps(drift_json))

    # Demo-day trips: seed-0 subsets scaled to estimated real volume.
    rng = np.random.default_rng(config.WEB_SUBSET_SEED)
    days_meta: dict[str, Any] = {}
    trips_out: dict[str, list[dict[str, Any]]] = {}
    exported_pairs: set[tuple[int, int]] = set()
    est_real: dict[str, int] = {}
    n_exported: dict[str, int] = {}

    day_frames: dict[str, tuple[str, pd.DataFrame]] = {}
    for tag, day in config.DEMO_DAYS.items():
        found = _demo_day_frame(feats, day)
        if found is not None:
            day_frames[tag] = found

    # Pass 1: sample sizes and estimated real trips per demo day.
    for tag, (month, day_df) in day_frames.items():
        n_sampled = len(day_df)
        cleaned_n = int(clean_counts[month]["dropoff_after_pickup"])
        est = int(round(n_sampled * cleaned_n / sample_n))
        est_real[tag] = est
        days_meta[tag] = {
            "date": config.DEMO_DAYS[tag],
            "n_sampled": int(n_sampled),
            "est_real_trips": est,
            "n_exported": 0,
            "real_trips_per_taxi": None,
        }

    # Pass 2: export subsets. Pre day capped; covid day scaled by the
    # estimated real-trip ratio so taxi density matches real volume.
    for tag, (_month, day_df) in day_frames.items():
        n_sampled = len(day_df)
        if tag == "pre":
            k = min(config.WEB_TRIPS_PRE_CAP, n_sampled)
        else:
            pre_est = est_real.get("pre", 0)
            pre_n = n_exported.get("pre", 0)
            if pre_est <= 0 or pre_n <= 0:
                continue
            k = int(round(
                pre_n * est_real[tag] / pre_est
            ))
            k = min(k, n_sampled)
        if k <= 0:
            continue
        idx = np.sort(rng.choice(n_sampled, size=k, replace=False))
        subset = day_df.iloc[idx].sort_values("pickup")
        pred = models.predict_minutes(static_model, subset)
        rows = []
        for (_, r), p in zip(subset.iterrows(), pred, strict=False):
            pu, do = int(r["pu_zone"]), int(r["do_zone"])
            exported_pairs.add((pu, do))
            rows.append({
                "pu": pu,
                "do": do,
                "minute": int(r["pickup"].hour * 60 + r["pickup"].minute),
                "minutes": round(float(r["minutes"]), 1),
                "pred_minutes": round(float(p), 1),
            })
        trips_out[tag] = rows
        n_exported[tag] = len(rows)
        days_meta[tag]["n_exported"] = len(rows)
        days_meta[tag]["real_trips_per_taxi"] = round(
            est_real[tag] / len(rows), 1
        ) if rows else None

    trips_json = {"days": days_meta, "trips": trips_out}
    (web / "trips.json").write_text(json.dumps(trips_json))

    # Routes only for pairs used by exported trips; GraphML loads lazily.
    routes: dict[str, Any] = {}
    if exported_pairs:
        if route_fn is not None:
            raw_routes = route_fn(exported_pairs)
        elif graphml_path is not None and graphml_path.exists():
            graph = load_graph(graphml_path)
            nodes = zone_nodes(graph, zone_infos)
            raw_routes = route_geometries(graph, nodes, exported_pairs)
        else:
            raw_routes = {}
        routes = {
            key: _simplify_route_coords(coords)
            for key, coords in raw_routes.items()
            if tuple(int(v) for v in key.split(">")) in exported_pairs
        }
    (web / "routes.json").write_text(json.dumps(routes))

    file_sizes = {
        name: (web / name).stat().st_size
        for name in ("zones.geojson", "drift.json", "trips.json",
                     "routes.json")
    }
    return {
        "caps": {"pre_day_trips": config.WEB_TRIPS_PRE_CAP},
        "seeds": {"subset": config.WEB_SUBSET_SEED},
        "tolerances": {
            "route_simplify": config.WEB_ROUTE_SIMPLIFY_TOL,
            "zone_simplify": config.WEB_ZONE_SIMPLIFY_TOL,
        },
        "coordinate_decimals": config.WEB_COORD_DECIMALS,
        "demo_days": config.DEMO_DAYS,
        "files": file_sizes,
    }
