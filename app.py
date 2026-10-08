"""GridGuard - Smart Grid Reliability Risk Predictor (Streamlit dashboard).

Decision-support prototype. NOT an autonomous grid-control system.
Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.features import FEATURES, LABELS, TARGET
from src.predict import apply_what_if, load_artifacts, predict_proba, risk_band

st.set_page_config(page_title="GridGuard", page_icon="⚡", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_data():
    df = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])
    return df


if not (Path("models/metadata.json").exists() and Path("data/processed/features.csv").exists()):
    st.error("Models or processed data not found. Run `python -m src.generate_data && python -m src.train` first.")
    st.stop()

art, df = get_artifacts(), get_data()
meta = art["meta"]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ GridGuard")
model_name = st.sidebar.selectbox(
    "Model", list(art["models"].keys()),
    index=list(art["models"].keys()).index(meta["best_model"]),
    help="Default = best model by validation PR-AUC.")
st.sidebar.caption("Risk bands (project-defined, not utility standards): "
                   "0–30% Low · 30–60% Moderate · 60–80% High · 80–100% Critical")
st.sidebar.warning("Decision-support prototype only. Not for operational or autonomous grid control. "
                   "Bundled data is SYNTHETIC unless you replaced it with real data.")

df["risk"] = predict_proba(art, df, model_name)
base_rate = float(df.loc[df.split == "train", TARGET].mean())

st.title("GridGuard – Smart Grid Reliability Risk Predictor")
st.caption("ML-based decision-support system that estimates the probability of a reliability-risk event "
           "in the next hour.")

tab_over, tab_pred, tab_an, tab_exp = st.tabs(
    ["Overview", "Risk Prediction", "Grid Analytics", "Model Explainability"])

# ------------------------------------------------------------------ overview
with tab_over:
    test = df[df.split == "test"].reset_index(drop=True)
    idx = st.slider("Select hour in the test period", 0, len(test) - 1, len(test) - 1)
    row = test.iloc[idx]
    p = float(row["risk"])
    band, color = risk_band(p)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk probability", f"{p:.1%}", f"{p / base_rate:.1f}× baseline rate")
    c2.markdown(f"**Risk level**<br><span style='font-size:1.8rem;color:{color}'>{band}</span>",
                unsafe_allow_html=True)
    c3.metric("Demand", f"{row.demand_mw:,.0f} MW")
    c4.metric("Renewables", f"{row.renewable_mw:,.0f} MW", f"{row.renewable_penetration:.0%} of demand")
    c5.metric("Temperature", f"{row.temperature:.1f} °C")
    st.caption(f"Timestamp: {row.timestamp}  ·  Events in last 24h: {int(row.outages_24h)}  ·  "
               f"Peak-load flag: {'yes' if row.peak_load else 'no'}  ·  Net load: {row.net_load:,.0f} MW")

    window = test.iloc[max(0, idx - 168): idx + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window.timestamp, y=window.risk, name="Predicted risk", line=dict(color="#c92a2a")))
    ev = window[window[TARGET] == 1]
    fig.add_trace(go.Scatter(x=ev.timestamp, y=ev.risk, mode="markers", name="Event occurred next hour",
                             marker=dict(color="black", size=9, symbol="x")))
    for lim in (0.3, 0.6, 0.8):
        fig.add_hline(y=lim, line_dash="dot", line_color="grey")
    fig.update_layout(title="Last 7 days: predicted risk vs. observed events", yaxis_tickformat=".0%",
                      height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ prediction + what-if
with tab_pred:
    st.subheader("Risk prediction & what-if simulator")
    st.write("Start from a real hour in the data, then change the sliders to see how the estimated risk reacts.")
    c0, c00 = st.columns([2, 1])
    pool = df[df.split == "test"].reset_index(drop=True)
    sel = c0.selectbox("Baseline hour", range(len(pool)), index=len(pool) - 1,
                       format_func=lambda i: str(pool.timestamp[i]))
    base = pool.iloc[sel]

    a, b = st.columns(2)
    demand = a.slider("Demand (MW)", float(df.demand_mw.min()), float(df.demand_mw.max() * 1.1), float(base.demand_mw))
    solar = a.slider("Solar (MW)", 0.0, float(df.solar_mw.max() * 1.2), float(base.solar_mw))
    wind = a.slider("Wind (MW)", 0.0, float(df.wind_mw.max() * 1.2), float(base.wind_mw))
    temp = b.slider("Temperature (°C)", float(df.temperature.min() - 5), float(df.temperature.max() + 10),
                    float(base.temperature))
    humidity = b.slider("Humidity (%)", 0.0, 100.0, float(base.humidity))
    o24 = b.slider("Events in last 24h", 0, 12, int(base.outages_24h))
    o7 = b.slider("Events in last 7 days", 0, 60, int(base.outages_7d))
    prev = b.checkbox("Reliability event in the current hour", value=bool(base.previous_outage))

    X0 = apply_what_if(base, art)
    X1 = apply_what_if(base, art, demand_mw=demand, solar_mw=solar, wind_mw=wind, temperature=temp,
                       humidity=humidity, outages_24h=o24, outages_7d=o7, previous_outage=int(prev))
    p0, p1 = float(predict_proba(art, X0, model_name)[0]), float(predict_proba(art, X1, model_name)[0])
    band1, col1 = risk_band(p1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline risk", f"{p0:.1%}")
    m2.metric("Scenario risk", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pp")
    m3.markdown(f"**Scenario level**<br><span style='font-size:1.8rem;color:{col1}'>{band1}</span>",
                unsafe_allow_html=True)
    st.progress(min(max(p1, 0.0), 1.0))
    st.caption(f"Scenario net load: {float(X1.net_load.iloc[0]):,.0f} MW · renewable penetration "
               f"{float(X1.renewable_penetration.iloc[0]):.0%} · baseline event rate in training data: {base_rate:.1%}. "
               "Probabilities are model estimates and are not guaranteed to be calibrated.")

# ------------------------------------------------------------------ analytics
with tab_an:
    st.subheader("Grid analytics")
    lo, hi = df.timestamp.min().date(), df.timestamp.max().date()
    rng = st.date_input("Date range", (hi - pd.Timedelta(days=14), hi), min_value=lo, max_value=hi)
    if isinstance(rng, tuple) and len(rng) == 2:
        sub = df[(df.timestamp.dt.date >= rng[0]) & (df.timestamp.dt.date <= rng[1])]
    else:
        sub = df.tail(24 * 14)

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=sub.timestamp, y=sub.demand_mw, name="Demand"))
    f1.add_trace(go.Scatter(x=sub.timestamp, y=sub.net_load, name="Net load"))
    f1.add_trace(go.Scatter(x=sub.timestamp, y=sub.renewable_mw, name="Renewables"))
    ev = sub[sub.previous_outage == 1]
    f1.add_trace(go.Scatter(x=ev.timestamp, y=ev.demand_mw, mode="markers", name="Reliability event",
                            marker=dict(color="red", size=8)))
    f1.update_layout(title="Demand, net load and renewables (MW)", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(f1, use_container_width=True)

    c1, c2 = st.columns(2)
    stack = go.Figure()
    stack.add_trace(go.Scatter(x=sub.timestamp, y=sub.solar_mw, stackgroup="g", name="Solar"))
    stack.add_trace(go.Scatter(x=sub.timestamp, y=sub.wind_mw, stackgroup="g", name="Wind"))
    stack.update_layout(title="Solar + wind (intermittency)", height=320, margin=dict(t=50, b=10))
    c1.plotly_chart(stack, use_container_width=True)
    c2.plotly_chart(px.line(sub, x="timestamp", y="renewable_penetration", title="Renewable penetration")
                    .update_layout(height=320, margin=dict(t=50, b=10), yaxis_tickformat=".0%"),
                    use_container_width=True)

    c3, c4 = st.columns(2)
    sample = df.sample(min(4000, len(df)), random_state=0)
    sample["event_next_hour"] = sample[TARGET].map({0: "No", 1: "Yes"})
    c3.plotly_chart(px.scatter(sample, x="temperature", y="demand_mw", color="event_next_hour", opacity=0.5,
                               title="Demand vs. temperature"), use_container_width=True)
    by_hour = df.groupby("hour")[TARGET].mean().reset_index()
    c4.plotly_chart(px.bar(by_hour, x="hour", y=TARGET, title="Observed event rate by hour of day")
                    .update_layout(yaxis_tickformat=".1%"), use_container_width=True)

    dfr = df.assign(risk_decile=pd.qcut(df.risk, 10, duplicates="drop", labels=False))
    obs = dfr.groupby("risk_decile")[TARGET].mean().reset_index()
    st.plotly_chart(px.bar(obs, x="risk_decile", y=TARGET,
                           title="Observed event rate by predicted-risk decile (higher decile = higher predicted risk)")
                    .update_layout(yaxis_tickformat=".1%"), use_container_width=True)

# ------------------------------------------------------------------ explainability
with tab_exp:
    st.subheader("Model performance")
    st.info("Time-aware split — train: past · validation: middle · test: most recent. "
            f"Test period: {meta['split_dates']['test'][0][:10]} → {meta['split_dates']['test'][1][:10]}. "
            "Decision threshold was tuned on validation data (max F1).")
    rows = []
    for m, r in meta["metrics"].items():
        t = r["test"]
        rows.append({"Model": m, "ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"], "Precision": t["precision"],
                     "Recall": t["recall"], "F1": t["f1"], "Threshold": t["threshold"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
    st.caption(f"Test-set event rate: {meta['metrics'][model_name]['test']['event_rate']:.1%}. "
               "With rare events, accuracy is misleading — focus on recall, F1 and PR-AUC.")

    te = df[df.split == "test"]
    y, s = te[TARGET], te["risk"]
    c1, c2 = st.columns(2)
    fpr, tpr, _ = roc_curve(y, s)
    c1.plotly_chart(go.Figure([go.Scatter(x=fpr, y=tpr, name="ROC"),
                               go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot"), name="Chance")])
                    .update_layout(title="ROC curve (test)", xaxis_title="FPR", yaxis_title="TPR", height=330),
                    use_container_width=True)
    pr, rc, _ = precision_recall_curve(y, s)
    c2.plotly_chart(go.Figure([go.Scatter(x=rc, y=pr, name="PR")])
                    .update_layout(title="Precision–recall curve (test)", xaxis_title="Recall",
                                   yaxis_title="Precision", height=330), use_container_width=True)

    st.subheader("Global feature importance")
    st.caption("Permutation importance on the test set (drop in PR-AUC when a feature is shuffled). "
               "This describes what the model relies on — it is NOT a causal explanation.")
    imp_path = Path("reports/permutation_importance.csv")
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        imp = imp[imp.model == model_name].sort_values("importance").tail(15)
        imp["feature"] = imp.feature.map(LABELS)
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h").update_layout(height=450),
                        use_container_width=True)

    st.subheader("Why is this hour risky? (local explanation)")
    pool = df[df.split == "test"].reset_index(drop=True)
    k = st.selectbox("Hour to explain", range(len(pool)), index=int(pool.risk.values.argmax()),
                     format_func=lambda i: f"{pool.timestamp[i]}  (risk {pool.risk[i]:.1%})")
    Xk = pool.loc[[k], FEATURES]
    contrib = None
    try:
        if model_name in ("random_forest", "xgboost"):
            import shap  # optional dependency
            sv = shap.TreeExplainer(art["models"][model_name]).shap_values(Xk)
            sv = sv[1] if isinstance(sv, list) else sv
            sv = np.asarray(sv)
            contrib = sv[0, :, 1] if sv.ndim == 3 else sv[0]
            note = "SHAP values (log-odds / probability contribution depending on model)."
    except Exception:
        contrib = None
    if contrib is None and model_name == "logistic_regression":
        xs = art["scaler"].transform(Xk)[0]
        contrib = art["models"]["logistic_regression"].coef_[0] * xs
        note = "Logistic-regression contributions: coefficient × standardised value (log-odds)."
    if contrib is None:
        st.info("Install `shap` (`pip install shap`) to see local explanations for tree models, "
                "or choose Logistic Regression in the sidebar.")
    else:
        cdf = pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "contribution": contrib})
        cdf = cdf.reindex(cdf.contribution.abs().sort_values().tail(10).index)
        cdf["direction"] = np.where(cdf.contribution >= 0, "increases risk", "decreases risk")
        st.plotly_chart(px.bar(cdf, x="contribution", y="feature", orientation="h", color="direction",
                               color_discrete_map={"increases risk": "#c92a2a", "decreases risk": "#2e9e5b"})
                        .update_layout(height=420), use_container_width=True)
        st.caption(note)
