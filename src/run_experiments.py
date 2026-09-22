"""
Run all 3 models (SARIMA, LSTM, Transformer) on the top-3 traffic squares,
forecasting Dec 16-22, and produce:
  - 9 plots (3 models x 3 squares): actual vs predicted
  - 3 tables (one per square): MAE, MAPE, RMSE for all models
  - timing stats for training and inference

Usage:
    python src/run_experiments.py
"""
import sys
import time
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.sarima_model import fit_predict_sarima
from models.lstm_model import fit_predict_lstm
from models.transformer_model import fit_predict_transformer

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EVAL_START = "2013-12-16"
EVAL_END = "2013-12-23"  # exclusive upper bound


def mae(a, p): return np.mean(np.abs(a - p))
def rmse(a, p): return np.sqrt(np.mean((a - p) ** 2))
def mape(a, p):
    mask = a != 0
    return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100


def get_top3_squares(df):
    totals = df.groupby("square_id")["internet"].sum().sort_values(ascending=False)
    return totals.head(3).index.tolist()


def get_square_series(df, square_id):
    sub = df[df.square_id == square_id].drop_duplicates(
        subset=["square_id", "timestamp"]
    ).set_index("timestamp")["internet"].asfreq("10min")
    return sub.interpolate()


def train_test_split_for_eval(series):
    """Train = everything before the eval week; test = the eval week itself."""
    train = series[series.index < EVAL_START]
    test = series[(series.index >= EVAL_START) & (series.index < EVAL_END)]
    return train, test


def run():
    print("Loading data...")
    df = pd.read_parquet("processed/internet_traffic.parquet")
    top3 = get_top3_squares(df)
    print(f"Top 3 squares: {top3}")

    all_results = {}   # {square_id: {model_name: {metrics, preds, times}}}
    hardware = f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}"

    for square_id in top3:
        print(f"\n=== Square {square_id} ===")
        series = get_square_series(df, square_id)
        train, test = train_test_split_for_eval(series)
        print(f"  train: {len(train)} points, test: {len(test)} points")

        results = {}

        print("  Fitting SARIMA...")
        try:
            preds, t_train, t_infer = fit_predict_sarima(train, test)
            results["SARIMA"] = dict(preds=preds, train_time=t_train, infer_time=t_infer)
        except Exception as e:
            print(f"    SARIMA failed: {e}")

        print("  Fitting LSTM...")
        preds, t_train, t_infer = fit_predict_lstm(train, test)
        results["LSTM"] = dict(preds=preds, train_time=t_train, infer_time=t_infer)

        print("  Fitting Transformer...")
        preds, t_train, t_infer = fit_predict_transformer(train, test)
        results["Transformer"] = dict(preds=preds, train_time=t_train, infer_time=t_infer)

        # metrics + plots
        actual = test.values
        rows = []
        for model_name, res in results.items():
            preds = res["preds"]
            m = dict(Model=model_name,
                     MAE=mae(actual, preds), MAPE=mape(actual, preds), RMSE=rmse(actual, preds),
                     TrainTime_s=res["train_time"], InferTime_s=res["infer_time"])
            rows.append(m)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(test.index, actual, label="Actual", linewidth=1)
            ax.plot(test.index, preds, label=f"{model_name} prediction", linewidth=1, linestyle="--")
            ax.set_title(f"Square {square_id} — {model_name} — Dec 16-22")
            ax.set_ylabel("Internet traffic")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"square{square_id}_{model_name}.png", dpi=150)
            plt.close(fig)

        table = pd.DataFrame(rows)
        table.to_csv(TABLE_DIR / f"square{square_id}_metrics.csv", index=False)
        print(table.to_string(index=False))

        # save raw predictions too, needed for cross-model failure analysis in evaluate.py
        pred_rows = []
        for model_name, res in results.items():
            for ts, act, pred in zip(test.index, actual, res["preds"]):
                pred_rows.append(dict(square_id=square_id, model=model_name,
                                       timestamp=ts, actual=act, predicted=pred))
        pd.DataFrame(pred_rows).to_csv(TABLE_DIR / f"square{square_id}_predictions.csv", index=False)

        all_results[square_id] = results

    with open(TABLE_DIR / "hardware.txt", "w") as f:
        f.write(f"Hardware used for timing statistics: {hardware}\n")
        f.write("Timing measured with time.time() around model fit/inference, per square.\n")

    print(f"\nDone. Figures in {FIG_DIR}/, tables in {TABLE_DIR}/")


if __name__ == "__main__":
    run()