"""
Generate a SYNTHETIC hourly grid dataset so GridGuard runs end-to-end without
external downloads.

!! IMPORTANT !!
Everything produced here is SYNTHETIC. The `outage_event` column is a
*simulated reliability-risk event*, NOT a real utility outage. Metrics obtained
on this data demonstrate that the pipeline works; they say nothing about
real-world grid performance. Replace data/raw/grid_data.csv with real data
(see README -> "Using real data") before drawing any conclusions.

Usage:
    python -m src.generate_data --start 2022-01-01 --days 1095 --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/grid_data.csv")


def generate(start="2022-01-01", days=1095, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    n = len(idx)
    hour = idx.hour.values
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values

    # --- Weather -----------------------------------------------------------
    seasonal_t = 15 + 12 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    daily_t = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = seasonal_t + daily_t + rng.normal(0, 1.5, n)
    heat = np.zeros(n)  # occasional heat-waves
    for s in rng.choice(n - 120, size=max(3, days // 90), replace=False):
        heat[s:s + rng.integers(48, 120)] = rng.uniform(4, 9)
    temp = temp + heat
    humidity = np.clip(65 - 1.2 * (temp - 15) + rng.normal(0, 8, n), 15, 100)
    cloud = np.clip(pd.Series(rng.normal(0.4, 0.3, n)).rolling(6, min_periods=1).mean().values, 0, 1)

    # --- Demand (MW) -------------------------------------------------------
    daily_load = 0.18 * np.sin(2 * np.pi * (hour - 14) / 24) + 0.08 * np.sin(4 * np.pi * (hour - 8) / 24)
    weekend = np.where(dow >= 5, -0.07, 0.0)
    cooling = 0.018 * np.maximum(temp - 22, 0)
    heating = 0.010 * np.maximum(10 - temp, 0)
    demand = 1000 * (1 + daily_load + weekend + cooling + heating) + rng.normal(0, 25, n)

    # --- Renewables (MW) ---------------------------------------------------
    sun = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0) * (0.7 + 0.3 * np.sin(2 * np.pi * (doy - 80) / 365.25))
    solar = 450 * sun * (1 - 0.75 * cloud) + rng.normal(0, 5, n) * (sun > 0)
    solar = np.clip(solar, 0, None)
    wind = np.zeros(n)
    wind[0] = 150
    for t in range(1, n):  # AR(1) wind process -> intermittency
        wind[t] = 0.93 * wind[t - 1] + 0.07 * 170 + rng.normal(0, 22)
    wind = np.clip(wind, 0, 400)
    wind_speed = np.clip(wind / 40 + rng.normal(0, 0.5, n), 0, None)
    renewable = solar + wind

    # --- Latent risk -> simulated reliability event ------------------------
    net_load = demand - renewable
    z_net = (net_load - net_load.mean()) / net_load.std()
    ren_ramp = np.r_[0, np.diff(renewable)]
    z_ramp = np.clip(-ren_ramp / (ren_ramp.std() + 1e-9), 0, None)  # sudden drops hurt
    t_stress = np.maximum(temp - 28, 0)

    events = np.zeros(n, dtype=int)
    duration = np.zeros(n)
    base = -4.6 + 1.6 * np.maximum(z_net - 0.4, 0) + 0.7 * z_ramp + 0.25 * t_stress + 0.5 * (cloud > 0.8) * sun
    for t in range(1, n):
        recent = events[max(0, t - 24):t].sum()
        logit = base[t] + 1.2 * events[t - 1] + 0.30 * min(recent, 4)
        p = 1 / (1 + np.exp(-logit))
        if rng.random() < p:
            events[t] = 1
            duration[t] = rng.gamma(2.0, 25)

    return pd.DataFrame({
        "timestamp": idx,
        "demand_mw": demand.round(1),
        "solar_mw": solar.round(1),
        "wind_mw": wind.round(1),
        "temperature": temp.round(2),
        "humidity": humidity.round(1),
        "wind_speed": wind_speed.round(2),
        "cloud_cover": cloud.round(3),
        "outage_event": events,
        "outage_duration_min": duration.round(1),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RAW_PATH))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = generate(a.start, a.days, a.seed)
    d.to_csv(out, index=False)
    print(f"Wrote {len(d):,} rows to {out}  (event rate: {d.outage_event.mean():.2%})  [SYNTHETIC DATA]")
