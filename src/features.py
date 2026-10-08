"""Grid-specific feature engineering for GridGuard.

All rolling features look only at the PAST (no look-ahead), and the target is
the reliability event in the NEXT interval (t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "outage_next_interval"
TEMP_THRESHOLD_C = 25.0       # temperature above which cooling stress starts
PEAK_QUANTILE = 0.90          # peak-load percentile

FEATURES = [
    "demand_mw", "solar_mw", "wind_mw", "temperature", "humidity",
    "renewable_mw", "renewable_penetration", "net_load",
    "renewable_ramp", "demand_change", "peak_load", "temp_stress",
    "demand_3h_avg", "demand_6h_avg", "demand_24h_avg", "renewable_volatility",
    "outages_24h", "outages_7d", "previous_outage",
    "hour", "is_weekend",
]

LABELS = {
    "demand_mw": "Demand (MW)", "solar_mw": "Solar (MW)", "wind_mw": "Wind (MW)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "renewable_mw": "Renewable generation (MW)",
    "renewable_penetration": "Renewable penetration", "net_load": "Net load (MW)",
    "renewable_ramp": "Renewable ramp (MW/h)", "demand_change": "Demand change (1h)",
    "peak_load": "Peak-load flag", "temp_stress": "Temperature stress (°C)",
    "demand_3h_avg": "Demand 3h avg", "demand_6h_avg": "Demand 6h avg",
    "demand_24h_avg": "Demand 24h avg", "renewable_volatility": "Renewable volatility (6h std)",
    "outages_24h": "Events in last 24h", "outages_7d": "Events in last 7d",
    "previous_outage": "Event in current hour", "hour": "Hour of day", "is_weekend": "Weekend",
}


def build_features(df: pd.DataFrame, peak_threshold: float | None = None) -> pd.DataFrame:
    """Return a copy of `df` with engineered features and the target column.

    peak_threshold: demand (MW) above which the peak-load flag is 1. Pass the
    value computed on the TRAINING period to avoid leaking test information;
    if None it is computed on the whole frame (fine for exploration only).
    """
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    if "humidity" not in d:
        d["humidity"] = np.nan
    d["humidity"] = d["humidity"].fillna(d["humidity"].median() if d["humidity"].notna().any() else 50.0)

    d["renewable_mw"] = d["solar_mw"] + d["wind_mw"]
    # Denominator = demand (total generation is rarely available in public data)
    d["renewable_penetration"] = (d["renewable_mw"] / d["demand_mw"]).clip(0, 1.5)
    d["net_load"] = d["demand_mw"] - d["renewable_mw"]
    d["renewable_ramp"] = d["renewable_mw"].diff().fillna(0)
    d["demand_prev"] = d["demand_mw"].shift(1).bfill()          # helper (not a model input)
    d["demand_change"] = ((d["demand_mw"] - d["demand_prev"]) / d["demand_prev"]).fillna(0)

    if peak_threshold is None:
        peak_threshold = float(d["demand_mw"].quantile(PEAK_QUANTILE))
    d["peak_load"] = (d["demand_mw"] >= peak_threshold).astype(int)
    d["temp_stress"] = (d["temperature"] - TEMP_THRESHOLD_C).clip(lower=0)

    for h in (3, 6, 24):
        d[f"demand_{h}h_avg"] = d["demand_mw"].rolling(h, min_periods=1).mean()
    d["renewable_volatility"] = d["renewable_mw"].rolling(6, min_periods=2).std().fillna(0)

    ev = d["outage_event"].astype(int)
    d["previous_outage"] = ev
    # events in the previous 24h / 7d, excluding the current hour (that is `previous_outage`)
    d["outages_24h"] = ev.shift(1).rolling(24, min_periods=1).sum().fillna(0)
    d["outages_7d"] = ev.shift(1).rolling(24 * 7, min_periods=1).sum().fillna(0)

    d["hour"] = d["timestamp"].dt.hour
    d["is_weekend"] = (d["timestamp"].dt.dayofweek >= 5).astype(int)

    d[TARGET] = ev.shift(-1)
    d = d.iloc[:-1].copy()            # last row has no known next interval
    d[TARGET] = d[TARGET].astype(int)
    return d
