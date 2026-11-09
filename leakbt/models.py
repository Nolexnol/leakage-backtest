"""Model fitting under different validation schemes.

The models are deliberately simple (ridge regression and a small gradient-boosted
tree ensemble). The point of the project is not the model; it is how much the
*evaluation protocol* changes the answer for the same model.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .validation import Split, shuffled_kfold_rows

ModelFactory = Callable[[], object]

MODELS: dict[str, ModelFactory] = {
    "ridge": lambda: Ridge(alpha=1.0),
    "gbm": lambda: HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500,
        l2_regularization=1.0, random_state=0,
    ),
}


def _xy(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[features].to_numpy(), df["target"].to_numpy()


def predict_by_date_splits(
    data: pd.DataFrame,
    features: list[str],
    splits: Iterable[Split],
    model: str = "ridge",
    winsor: float = 0.05,
) -> pd.Series:
    """Fit on each split's training dates, predict its test dates.

    Targets are clipped to +/- ``winsor`` before fitting so a handful of
    earnings-day jumps do not dominate the loss (clipping thresholds are fixed
    constants, so they carry no information from the test period).
    """
    dates = data.index.get_level_values("date")
    preds = []
    for train_dates, test_dates in splits:
        tr = data[dates.isin(train_dates)]
        te = data[dates.isin(test_dates)]
        if len(tr) == 0 or len(te) == 0:
            continue
        X, y = _xy(tr, features)
        m = MODELS[model]()
        m.fit(X, np.clip(y, -winsor, winsor))
        preds.append(pd.Series(m.predict(te[features].to_numpy()), index=te.index))
    return pd.concat(preds).sort_index()


def predict_shuffled_kfold(
    data: pd.DataFrame, features: list[str], k: int = 5, model: str = "ridge", winsor: float = 0.05, seed: int = 0
) -> pd.Series:
    """Row-level shuffled K-fold: the common, leaky default."""
    out = pd.Series(np.nan, index=data.index)
    X_all, y_all = _xy(data, features)
    for tr, te in shuffled_kfold_rows(len(data), k=k, seed=seed):
        m = MODELS[model]()
        m.fit(X_all[tr], np.clip(y_all[tr], -winsor, winsor))
        out.iloc[te] = m.predict(X_all[te])
    return out


def to_wide(pred: pd.Series) -> pd.DataFrame:
    return pred.unstack("ticker")
