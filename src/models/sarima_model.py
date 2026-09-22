import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time

PERIOD = 144          # 10-min intervals in a day
N_HARMONICS = 3        # captures the daily shape (peak/trough + finer structure)


def make_fourier_features(index: pd.DatetimeIndex, t_start: int,
                           period: int = PERIOD, n_harmonics: int = N_HARMONICS) -> pd.DataFrame:
    """Fourier terms for the given index, with t continuing from t_start
    (so train and test features are on the same continuous time axis)."""
    t = np.arange(t_start, t_start + len(index))
    cols = {}
    for k in range(1, n_harmonics + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def fit_predict_sarima(train_series: pd.Series, test_series: pd.Series,
                        order=(2, 0, 2), refit_every=144):
    """
    One-step-ahead rolling forecast. `refit_every`: periodically refit
    (default once per simulated day) so parameters don't go stale over
    the full 1008-step (one week) rolling horizon, while most steps stay
    cheap (refit=False).
    """
    exog_train = make_fourier_features(train_series.index, t_start=0)
    exog_test = make_fourier_features(test_series.index, t_start=len(train_series))

    t0 = time.time()
    model = SARIMAX(train_series, exog=exog_train, order=order, trend="c",
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=100, method="lbfgs")
    train_time = time.time() - t0

    t0 = time.time()
    preds = []
    history_model = fit
    for i in range(len(test_series)):
        exog_step = exog_test.iloc[[i]]
        pred = history_model.forecast(steps=1, exog=exog_step)
        pred_val = pred.iloc[0]
        if not np.isfinite(pred_val):
            pred_val = train_series.iloc[-1] if i == 0 else preds[-1]
        preds.append(pred_val)

        refit_now = (refit_every is not None) and ((i + 1) % refit_every == 0)
        history_model = history_model.append([test_series.iloc[i]], exog=exog_step, refit=refit_now)
    inference_time = time.time() - t0

    return np.array(preds), train_time, inference_time