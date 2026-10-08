import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET, build_features
from src.generate_data import generate
from src.preprocessing import clean


def _df():
    return clean(generate(days=40, seed=1), verbose=False)


def test_features_complete_and_no_nans():
    f = build_features(_df())
    assert all(c in f for c in FEATURES + [TARGET])
    assert not f[FEATURES + [TARGET]].isna().any().any()


def test_target_is_next_interval():
    d = _df()
    f = build_features(d)
    assert (f[TARGET].values == d["outage_event"].shift(-1).iloc[:-1].astype(int).values).all()


def test_rolling_features_do_not_look_ahead():
    d = _df()
    f1 = build_features(d)
    d2 = d.copy()
    d2.loc[d2.index[-100:], "demand_mw"] *= 5          # change the FUTURE
    f2 = build_features(d2, peak_threshold=f1["demand_mw"].quantile(0.9))
    cut = len(f1) - 101
    assert np.allclose(f1.loc[:cut, "demand_24h_avg"], f2.loc[:cut, "demand_24h_avg"])


def test_cleaning_fills_gaps_and_dupes():
    d = generate(days=10, seed=2)
    d = pd.concat([d.drop(index=[5, 6]), d.iloc[[10]]])
    c = clean(d, verbose=False)
    assert c.timestamp.is_unique and c.timestamp.diff().dropna().eq(pd.Timedelta("1h")).all()
