"""Loading, cleaning and timestamp alignment for GridGuard."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "demand_mw", "solar_mw", "wind_mw", "temperature"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def define_proxy_event(df: pd.DataFrame, quantile: float = 0.97) -> pd.Series:
    """Reliability-event PROXY for datasets without real outage records.

    Flags hours where net load (demand - solar - wind) is in the top
    (1 - quantile) of the distribution. This is a stress indicator, NOT an
    actual outage - label it as a 'reliability-risk event' everywhere.
    """
    net = df["demand_mw"] - df["solar_mw"] - df["wind_mw"]
    return (net >= net.quantile(quantile)).astype(int)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop bad timestamps/duplicates, enforce hourly grid, fix gaps/outliers."""
    report = {"rows_in": len(df)}
    df = df.dropna(subset=["timestamp"]).copy()
    report["duplicate_timestamps"] = int(df.duplicated("timestamp").sum())
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")

    # Regular hourly index (timestamp alignment)
    df = df.set_index("timestamp")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    report["missing_hours"] = int(len(full) - len(df))
    df = df.reindex(full)
    df.index.name = "timestamp"

    # Unit sanity: negative generation/demand is physically invalid here
    for c in ["demand_mw", "solar_mw", "wind_mw"]:
        df.loc[df[c] < 0, c] = np.nan
    if "humidity" in df:
        df.loc[(df["humidity"] < 0) | (df["humidity"] > 100), "humidity"] = np.nan

    # Winsorise extreme outliers (0.1% / 99.9%) on continuous columns
    for c in [c for c in ["demand_mw", "solar_mw", "wind_mw", "temperature", "humidity"] if c in df]:
        lo, hi = df[c].quantile([0.001, 0.999])
        df[c] = df[c].clip(lo, hi)

    report["missing_values_before_fill"] = int(df.isna().sum().sum())
    skip = [c for c in ("outage_event", "outage_duration_min") if c in df]
    num = [c for c in df.select_dtypes("number").columns if c not in skip]
    df[num] = df[num].interpolate(limit=6, limit_direction="both").ffill().bfill()

    if "outage_event" not in df:
        df["outage_event"] = define_proxy_event(df).values
        report["target"] = "PROXY (net-load stress) - not real outages"
    df["outage_event"] = df["outage_event"].fillna(0).astype(int)
    if "outage_duration_min" in df:
        df["outage_duration_min"] = df["outage_duration_min"].fillna(0)

    report["rows_out"] = len(df)
    if verbose:
        print("Cleaning report:", report)
    return df.reset_index()
