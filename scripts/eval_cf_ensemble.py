"""Prediction-level CF ensemble comparison.

Compares: simple mean, softmax CF (additive/multiplicative), Bayesian+CF.
Uses same train/val/test split as train_multi_probe.py.

Usage:
    python scripts/eval_cf_ensemble.py
    python scripts/eval_cf_ensemble.py --probes 1 2 3 5 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DATA_DIR = Path("data/features")
RESULTS_DIR = Path("results/multi_probe")

TOP_FEATURES = [
    "speed_mean", "speed_std", "speed_min", "speed_cv",
    "ax_std", "ax_mean", "stop_count", "brake_time_ratio",
]
SINGLE_FEATURES = TOP_FEATURES + ["num_lanes", "speed_limit"]
TARGET = "density_per_lane"


# ---- CF score functions ----

def cf_additive(row: pd.Series) -> float:
    return row["ax_std"] + row["brake_time_ratio"] + row["speed_cv"]


def cf_multiplicative(row: pd.Series) -> float:
    return row["ax_std"] * row["speed_cv"] * (1.0 + row["brake_time_ratio"])


# ---- Ensemble methods ----

def softmax_weights(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    s = scores / temperature
    mx = s.max()
    exps = np.exp(s - mx)
    return exps / exps.sum()


def ensemble_softmax_cf(
    preds: np.ndarray, cf_scores: np.ndarray, temperature: float = 1.0,
) -> float:
    w = softmax_weights(cf_scores, temperature)
    return float((w * preds).sum())


def ensemble_bayesian_cf(
    preds: np.ndarray,
    cf_scores: np.ndarray,
    prior_mu: float = 9.4,
    prior_sigma: float = 36.25,
    cf_scale: float = 2.59,
    base_obs_sigma: float = 0.001,
) -> float:
    """Bayesian sequential update with CF-based observation noise.

    obs_sigma = base_obs_sigma * exp(-cf_score * cf_scale)
    High CF → low obs_sigma → more trusted observation.
    """
    mu = prior_mu
    var = prior_sigma**2

    for pred, cf in zip(preds, cf_scores):
        obs_sigma = base_obs_sigma * np.exp(-cf * cf_scale)
        obs_var = obs_sigma**2
        # Bayesian update
        new_var = 1.0 / (1.0 / var + 1.0 / obs_var)
        mu = new_var * (mu / var + pred / obs_var)
        var = new_var

    return float(mu)


def ensemble_simple_mean(preds: np.ndarray) -> float:
    return float(preds.mean())


# ---- Evaluation ----

def evaluate_ensemble(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    n_probes: int,
    test_sids: np.ndarray,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    rng = np.random.RandomState(seed)
    sub = df[df["scenario_id"].isin(set(test_sids))]

    results: dict[str, list[float]] = {
        "simple_mean": [], "cf_add_softmax": [], "cf_mul_softmax": [],
        "bayesian_cf_add": [], "bayesian_cf_mul": [],
    }
    y_trues: list[float] = []

    for _sid, g in sub.groupby("scenario_id"):
        if len(g) < n_probes:
            continue
        s = g.iloc[rng.choice(len(g), size=n_probes, replace=False)]
        preds = model.predict(s[SINGLE_FEATURES].values)
        cf_add = np.array([cf_additive(s.iloc[i]) for i in range(len(s))])
        cf_mul = np.array([cf_multiplicative(s.iloc[i]) for i in range(len(s))])

        y_trues.append(float(g[TARGET].iloc[0]))
        results["simple_mean"].append(ensemble_simple_mean(preds))
        results["cf_add_softmax"].append(ensemble_softmax_cf(preds, cf_add))
        results["cf_mul_softmax"].append(ensemble_softmax_cf(preds, cf_mul))
        results["bayesian_cf_add"].append(ensemble_bayesian_cf(preds, cf_add))
        results["bayesian_cf_mul"].append(ensemble_bayesian_cf(preds, cf_mul))

    yt = np.array(y_trues)
    metrics = {}
    for method, yp_list in results.items():
        yp = np.array(yp_list)
        metrics[method] = {
            "r2": round(r2_score(yt, yp), 4),
            "rmse": round(np.sqrt(mean_squared_error(yt, yp)), 4),
            "mae": round(mean_absolute_error(yt, yp), 4),
        }
    return metrics


# ---- Feature-level ensemble (baseline) ----

def feature_ensemble_r2(
    df: pd.DataFrame, n_probes: int,
    train_sids: np.ndarray, val_sids: np.ndarray, test_sids: np.ndarray,
) -> float:
    def build(sids: np.ndarray, seed: int) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        sub = df[df["scenario_id"].isin(set(sids))]
        rows = []
        for _sid, g in sub.groupby("scenario_id"):
            if len(g) < n_probes:
                continue
            s = g.iloc[rng.choice(len(g), size=n_probes, replace=False)]
            row: dict[str, float] = {}
            for c in TOP_FEATURES:
                row[f"{c}_mean"] = s[c].mean()
                row[f"{c}_std"] = s[c].std() if n_probes > 1 else 0.0
            row["num_lanes"] = g["num_lanes"].iloc[0]
            row["speed_limit"] = g["speed_limit"].iloc[0]
            row["target"] = g[TARGET].iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    tr = build(train_sids, SEED)
    va = build(val_sids, SEED + 1)
    te = build(test_sids, SEED + 2)

    acols = (
        [f"{c}_mean" for c in TOP_FEATURES]
        + [f"{c}_std" for c in TOP_FEATURES]
        + ["num_lanes", "speed_limit"]
    )
    fm = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20,
        random_state=SEED, tree_method="hist",
    )
    fm.fit(tr[acols].values, tr["target"].values,
           eval_set=[(va[acols].values, va["target"].values)], verbose=False)
    fp = fm.predict(te[acols].values)
    return round(r2_score(te["target"].values, fp), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", nargs="+", type=int, default=[1, 2, 3, 5])
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / "dataset_1km.parquet")
    sid_counts = df.groupby("scenario_id").size()
    valid_sids = sid_counts[sid_counts >= 5].index.values

    train_sids, temp = train_test_split(valid_sids, test_size=0.3, random_state=SEED)
    val_sids, test_sids = train_test_split(temp, test_size=0.5, random_state=SEED)
    logger.info("Split: train=%d, val=%d, test=%d", len(train_sids), len(val_sids), len(test_sids))

    # Train single-probe model
    train_sub = df[df["scenario_id"].isin(set(train_sids))]
    val_sub = df[df["scenario_id"].isin(set(val_sids))]
    raw_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20,
        random_state=SEED, tree_method="hist",
    )
    raw_model.fit(
        train_sub[SINGLE_FEATURES].values, train_sub[TARGET].values,
        eval_set=[(val_sub[SINGLE_FEATURES].values, val_sub[TARGET].values)],
        verbose=False,
    )
    logger.info("Single-probe model trained")

    all_results: dict = {}

    header = f"{'N':>3} | {'Feature앙상블':>12} | {'Bayes+CF덧':>11} | {'Bayes+CF곱':>11} | {'Softmax덧':>10} | {'Softmax곱':>10} | {'단순평균':>8}"
    print("\n" + header)
    print("-" * len(header))

    for n in args.probes:
        feat_r2 = feature_ensemble_r2(df, n, train_sids, val_sids, test_sids)

        if n == 1:
            m = {"r2": feat_r2, "rmse": 0.0, "mae": 0.0}
            metrics = {k: m for k in ["simple_mean", "cf_add_softmax", "cf_mul_softmax", "bayesian_cf_add", "bayesian_cf_mul"]}
        else:
            metrics = evaluate_ensemble(df, raw_model, n, test_sids, seed=SEED + 2)

        all_results[f"N={n}"] = {"feature_ensemble_r2": feat_r2, **metrics}

        ba = metrics["bayesian_cf_add"]["r2"]
        bm = metrics["bayesian_cf_mul"]["r2"]
        sa = metrics["cf_add_softmax"]["r2"]
        sm = metrics["cf_mul_softmax"]["r2"]
        mn = metrics["simple_mean"]["r2"]
        print(f"{n:3d} | {feat_r2:12.4f} | {ba:11.4f} | {bm:11.4f} | {sa:10.4f} | {sm:10.4f} | {mn:8.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cf_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
