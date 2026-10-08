# ⚡ GridGuard – Smart Grid Reliability Risk Predictor

An ML-based **decision-support** prototype that estimates the probability of a
grid reliability-risk event in the **next hour** from electricity demand,
renewable generation, weather and recent reliability history. Results are served
through an interactive Streamlit dashboard with a what-if simulator and
model explanations.

> **Read this first – data disclosure.** The repository ships with a **synthetic
> dataset** (`src/generate_data.py`) so that everything runs out of the box. The
> `outage_event` target in that data is a *simulated reliability-risk event*, **not**
> a real utility outage, and the reported metrics only demonstrate that the
> pipeline works. To obtain meaningful results, train on real data (see
> [Using real data](#using-real-data)).
>
> GridGuard does **not** prevent outages and must not be used for operational or
> autonomous grid control.

![dashboard](docs/screenshots/dashboard.png)
<!-- Add screenshots to docs/screenshots/ and update the path above -->

## Features

- Time-aware pipeline (train on the past, test on the future – no shuffling)
- Grid-specific feature engineering: renewable penetration, net load, renewable ramp,
  demand change, peak-load flag, temperature stress, rolling demand (3/6/24 h),
  renewable volatility, recent-event counts (24 h / 7 d)
- Models: Logistic Regression (baseline), Random Forest, optional XGBoost
- Metrics for imbalanced data: precision, recall, F1, ROC-AUC, PR-AUC
- Decision threshold tuned on **validation** data only; peak-load threshold computed on **training** data only
- Streamlit dashboard: **Overview · Risk Prediction (what-if) · Grid Analytics · Model Explainability**
- Explainability: permutation importance (global), SHAP for tree models / coefficient contributions for logistic regression (local)

## Quick start

```bash
git clone https://github.com/<your-username>/GridGuard.git
cd GridGuard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional) regenerate data and retrain – trained models are already included
python -m src.generate_data
python -m src.train

streamlit run app.py
```

## Results on the bundled synthetic data (test = last 2 months)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.760 | 0.165 | 0.212 | 0.151 | 0.176 |
| Random Forest | 0.764 | 0.176 | 0.202 | 0.315 | 0.246 |

Test event rate ≈ 5 %, so a random PR-AUC would be ≈ 0.05. The threshold is the
max-F1 point on the validation set; lower it for higher recall at the cost of
more false alarms. **These numbers are for synthetic data – replace them with
your own when you train on real data** (`reports/test_metrics.csv`).

## Project structure

```
GridGuard/
├── app.py                      # Streamlit dashboard
├── data/
│   ├── raw/grid_data.csv       # input data (synthetic by default)
│   └── processed/features.csv  # engineered features + split labels
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── generate_data.py        # synthetic data generator (clearly labelled)
│   ├── preprocessing.py        # loading, timestamp alignment, cleaning, proxy-event helper
│   ├── features.py             # feature engineering + target
│   ├── train.py                # time-aware split, training, evaluation, saving
│   └── predict.py              # inference, risk bands, what-if logic
├── models/                     # logistic_model.pkl, random_forest.pkl, scaler.pkl, metadata.json
├── reports/                    # test_metrics.csv, permutation_importance.csv
├── tests/test_pipeline.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Using real data

Put a CSV at `data/raw/grid_data.csv` with an hourly timestamp column and:

| Column | Required | Notes |
|---|---|---|
| `timestamp` | yes | parseable datetime, hourly |
| `demand_mw` | yes | MW |
| `solar_mw`, `wind_mw` | yes | MW |
| `temperature` | yes | °C |
| `humidity` | no | % |
| `outage_event` | no | 1 if a reliability/outage event occurred in that hour |

Then run `python -m src.train`. If `outage_event` is absent, a **proxy** is
derived automatically (top 3 % net-load hours, `define_proxy_event` in
`src/preprocessing.py`). If you use a proxy, state it prominently – it is a
stress indicator, not an actual outage.

Possible public sources: grid demand/generation from regional operators or the
US EIA Open Data API, weather from NOAA/Meteostat/Open-Meteo, outage records from
EIA-OE-417 disturbance reports or a utility's published reliability data.
Check each source's licence and document it here.

## Methodology notes

- **Target:** `outage_next_interval` = reliability event in hour *t+1*.
- **Split:** by time – last 2 months test, previous 2 months validation, rest train.
- **Leakage controls:** rolling features use only past values; peak-load threshold from train period only; threshold tuned on validation only.
- **Risk bands** (0–30 % Low, 30–60 % Moderate, 60–80 % High, 80–100 % Critical) are project-defined communication bands, not utility standards. Probabilities are not calibrated; the dashboard also shows risk relative to the base event rate.
- `renewable_penetration` is renewables ÷ demand (total generation is rarely available publicly).

## Limitations & ethics

- Public/synthetic data does not represent a specific utility, feeder or geography.
- Historical relationships do not guarantee future grid behaviour.
- Feature importance describes what the model uses, not causes.
- Not for autonomous control or operational decisions.
- Future work: SCADA/AMI data, feeder-level models, storage optimisation, real-time ingestion, probability calibration.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. On <https://share.streamlit.io> choose *New app* → select the repo, branch `main`, main file `app.py`.
3. Add the live URL to this README and your CV.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Resume description

**GridGuard – Smart Grid Reliability Predictor:** Built an ML-based decision-support system that predicts
short-term grid reliability risk from demand, renewable generation, weather and reliability indicators.
Engineered grid features (renewable penetration, net load, ramps, volatility), compared Logistic Regression and
Random Forest (XGBoost optional) with time-aware validation and imbalance-aware metrics, and deployed the
model in a Streamlit dashboard with a what-if simulator and explainability.
*Add your real metrics, GitHub link and live-demo link once you train on real data.*

## License

MIT – see [LICENSE](LICENSE). Replace `<Your Name>` in the license file.
