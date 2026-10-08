"""Train, evaluate and save GridGuard models.

Usage:
    python -m src.train                       # uses data/raw/grid_data.csv
    python -m src.train --data path/to.csv

Time-aware split (no shuffling): train = past, validation = middle,
test = most recent period (default: last 2 months = test, the 2 months before = validation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, TARGET, build_features
from .preprocessing import clean, load_raw

MODELS = Path("models")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
SEED = 42


def time_split(df: pd.DataFrame, val_months: int = 2, test_months: int = 2) -> pd.DataFrame:
    end = df["timestamp"].max()
    test_start = (end - pd.DateOffset(months=test_months)).normalize() + pd.Timedelta(days=1)
    val_start = test_start - pd.DateOffset(months=val_months)
    out = df.copy()
    out["split"] = np.where(out["timestamp"] >= test_start, "test",
                            np.where(out["timestamp"] >= val_start, "val", "train"))
    return out


def evaluate(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    two = y.nunique() > 1
    return {
        "roc_auc": float(roc_auc_score(y, proba)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if two else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "event_rate": float(y.mean()),
    }


def best_f1_threshold(y, proba) -> float:
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.nanargmax(f1))]) if len(t) else 0.5


def main(data_path: str = "data/raw/grid_data.csv") -> dict:
    for p in (MODELS, PROCESSED, REPORTS):
        p.mkdir(parents=True, exist_ok=True)

    raw = time_split(clean(load_raw(data_path)))
    peak_thr = float(raw.loc[raw["split"] == "train", "demand_mw"].quantile(0.90))  # train-only

    df = build_features(raw, peak_threshold=peak_thr)
    df.to_csv(PROCESSED / "features.csv", index=False)

    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, ytr, Xva, yva, Xte, yte = tr[FEATURES], tr[TARGET], va[FEATURES], va[TARGET], te[FEATURES], te[TARGET]
    print(f"Rows  train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    print(f"Event rate  train={ytr.mean():.2%}  val={yva.mean():.2%}  test={yte.mean():.2%}")

    scaler = StandardScaler().fit(Xtr)
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, max_depth=14, max_features="sqrt", n_jobs=-1, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)
    except ImportError:
        print("xgboost not installed - skipping (optional).")

    def feats(name, X):
        return scaler.transform(X) if name == "logistic_regression" else X.values

    results, thresholds, fitted = {}, {}, {}
    for name, m in models.items():
        m.fit(feats(name, Xtr), ytr)
        p_val = m.predict_proba(feats(name, Xva))[:, 1]
        thr = best_f1_threshold(yva, p_val)          # threshold chosen on VALIDATION only
        p_te = m.predict_proba(feats(name, Xte))[:, 1]
        results[name] = {"validation": evaluate(yva, p_val, thr), "test": evaluate(yte, p_te, thr)}
        thresholds[name], fitted[name] = thr, m
        t = results[name]["test"]
        print(f"{name:20s} TEST  ROC-AUC={t['roc_auc']:.3f}  PR-AUC={t['pr_auc']:.3f}  "
              f"P={t['precision']:.3f}  R={t['recall']:.3f}  F1={t['f1']:.3f}")

    best = max(results, key=lambda k: results[k]["validation"]["pr_auc"])
    print(f"Best model (validation PR-AUC): {best}")

    # Permutation importance on the held-out test set (PR-AUC based)
    rows = []
    for name, m in fitted.items():
        r = permutation_importance(m, feats(name, Xte), yte, scoring="average_precision",
                                   n_repeats=5, random_state=SEED, n_jobs=-1)
        rows += [{"model": name, "feature": f, "importance": float(v)}
                 for f, v in zip(FEATURES, r.importances_mean)]
    pd.DataFrame(rows).to_csv(REPORTS / "permutation_importance.csv", index=False)

    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(fitted["logistic_regression"], MODELS / "logistic_model.pkl")
    joblib.dump(fitted["random_forest"], MODELS / "random_forest.pkl")
    if "xgboost" in fitted:
        joblib.dump(fitted["xgboost"], MODELS / "xgboost_model.pkl")

    meta = {
        "features": FEATURES,
        "best_model": best,
        "peak_threshold_mw": peak_thr,
        "thresholds": thresholds,
        "metrics": results,
        "split_dates": {s: [str(df[df.split == s].timestamp.min()), str(df[df.split == s].timestamp.max())]
                        for s in ("train", "val", "test")},
        "note": "Metrics refer to the dataset used at training time. If that was the bundled "
                "SYNTHETIC dataset they only demonstrate that the pipeline works.",
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({m: r["test"] for m, r in results.items()}).T.to_csv(REPORTS / "test_metrics.csv")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/grid_data.csv")
    main(ap.parse_args().data)
