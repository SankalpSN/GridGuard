"""Inference helpers: load models, score feature rows, run what-if scenarios."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, TEMP_THRESHOLD_C

MODELS = Path("models")
MODEL_FILES = {
    "logistic_regression": "logistic_model.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost_model.pkl",
}

# Project-defined communication bands (NOT utility operating standards)
RISK_BANDS = [(0.30, "Low", "#2e9e5b"), (0.60, "Moderate", "#e0a800"),
              (0.80, "High", "#e8590c"), (1.01, "Critical", "#c92a2a")]


def risk_band(p: float) -> tuple[str, str]:
    for upper, name, color in RISK_BANDS:
        if p < upper:
            return name, color
    return "Critical", RISK_BANDS[-1][2]


def load_artifacts(models_dir: str | Path = MODELS) -> dict:
    models_dir = Path(models_dir)
    meta = json.loads((models_dir / "metadata.json").read_text())
    models = {n: joblib.load(models_dir / f) for n, f in MODEL_FILES.items() if (models_dir / f).exists()}
    return {"meta": meta, "models": models, "scaler": joblib.load(models_dir / "scaler.pkl")}


def predict_proba(art: dict, X: pd.DataFrame, model_name: str | None = None) -> np.ndarray:
    name = model_name or art["meta"]["best_model"]
    X = X[FEATURES]
    data = art["scaler"].transform(X) if name == "logistic_regression" else X.values
    return art["models"][name].predict_proba(data)[:, 1]


def apply_what_if(baseline: pd.Series, art: dict, **overrides) -> pd.DataFrame:
    """Return a one-row feature frame = baseline hour with overrides applied.

    Derived features (renewables, net load, penetration, ramp, demand change,
    peak flag, temperature stress, rolling demand averages) are recomputed so
    the scenario stays internally consistent.
    """
    r = baseline.copy()
    base_demand = float(r["demand_mw"])
    prev_demand = float(r.get("demand_prev", base_demand))
    prev_ren = float(r["renewable_mw"]) - float(r["renewable_ramp"])
    base_avgs = {h: float(baseline[f"demand_{h}h_avg"]) for h in (3, 6, 24)}

    for k, v in overrides.items():
        r[k] = v

    r["renewable_mw"] = r["solar_mw"] + r["wind_mw"]
    r["net_load"] = r["demand_mw"] - r["renewable_mw"]
    r["renewable_penetration"] = min(r["renewable_mw"] / max(r["demand_mw"], 1e-6), 1.5)
    r["renewable_ramp"] = r["renewable_mw"] - prev_ren
    r["demand_change"] = (r["demand_mw"] - prev_demand) / max(prev_demand, 1e-6)
    r["peak_load"] = int(r["demand_mw"] >= art["meta"]["peak_threshold_mw"])
    r["temp_stress"] = max(r["temperature"] - TEMP_THRESHOLD_C, 0)
    delta = float(r["demand_mw"]) - base_demand
    for h in (3, 6, 24):
        r[f"demand_{h}h_avg"] = base_avgs[h] + delta / h
    return pd.DataFrame([r])[FEATURES].astype(float)
