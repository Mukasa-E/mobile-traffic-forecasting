"""
Reads the per-square outputs from run_experiments.py and produces the
required cross-model comparative analysis:
  - one combined summary table across all 3 squares
  - identification of the best-performing model (by average rank across metrics)
  - at least one documented failure case: the period/model/square with the
    largest prediction error, plotted for discussion

Run AFTER run_experiments.py has completed.

Usage:
    python src/evaluate.py
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")


def main():
    metric_files = sorted(glob.glob(str(TABLE_DIR / "square*_metrics.csv")))
    pred_files = sorted(glob.glob(str(TABLE_DIR / "square*_predictions.csv")))

    if not metric_files:
        print("No metrics files found in results/tables/ — run run_experiments.py first.")
        return

    # --- 1. Combined summary table across all squares ---
    all_metrics = []
    for f in metric_files:
        square_id = f.split("square")[1].split("_")[0]
        df = pd.read_csv(f)
        df["square_id"] = square_id
        all_metrics.append(df)
    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined[["square_id", "Model", "MAE", "MAPE", "RMSE", "TrainTime_s", "InferTime_s"]]
    combined.to_csv(TABLE_DIR / "combined_comparison.csv", index=False)

    print("=== Combined comparison across all squares ===")
    print(combined.to_string(index=False))

    # --- 2. Best model overall: average rank across MAE/MAPE/RMSE, averaged over squares ---
    ranked = combined.copy()
    for metric in ["MAE", "MAPE", "RMSE"]:
        ranked[f"{metric}_rank"] = ranked.groupby("square_id")[metric].rank()
    ranked["avg_rank"] = ranked[["MAE_rank", "MAPE_rank", "RMSE_rank"]].mean(axis=1)
    overall = ranked.groupby("Model")["avg_rank"].mean().sort_values()

    print("\n=== Overall ranking (lower avg rank = better, across MAE/MAPE/RMSE and all squares) ===")
    print(overall.to_string())
    overall.to_csv(TABLE_DIR / "overall_ranking.csv")
    best_model = overall.index[0]
    print(f"\nBest-performing model overall: {best_model}")

    # --- 3. Failure case: largest single prediction error anywhere ---
    if pred_files:
        all_preds = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in pred_files],
                               ignore_index=True)
        all_preds["abs_error"] = (all_preds["actual"] - all_preds["predicted"]).abs()
        worst = all_preds.loc[all_preds["abs_error"].idxmax()]
        print(f"\n=== Largest single error ===")
        print(f"Square {worst.square_id}, model {worst.model}, at {worst.timestamp}: "
              f"actual={worst.actual:.2f}, predicted={worst.predicted:.2f}, "
              f"abs_error={worst.abs_error:.2f}")

        # plot a window around the worst error for discussion in the report
        window = all_preds[
            (all_preds.square_id == worst.square_id) &
            (all_preds.model == worst.model) &
            (all_preds.timestamp >= worst.timestamp - pd.Timedelta(hours=12)) &
            (all_preds.timestamp <= worst.timestamp + pd.Timedelta(hours=12))
        ].sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window.timestamp, window.actual, label="Actual", linewidth=1)
        ax.plot(window.timestamp, window.predicted, label="Predicted", linewidth=1, linestyle="--")
        ax.axvline(worst.timestamp, color="red", linestyle=":", label="Largest error")
        ax.set_title(f"Failure case — Square {worst.square_id}, {worst.model}\n"
                      f"Largest single-step error in the evaluation week")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "failure_case.png", dpi=150)
        plt.close(fig)
        print(f"Failure case plot saved to {FIG_DIR / 'failure_case.png'}")
    else:
        print("\nNo prediction CSVs found (square*_predictions.csv) — "
              "failure case analysis skipped. Re-run run_experiments.py with "
              "the updated version that saves predictions.")

    print("\nDone. See results/tables/combined_comparison.csv and overall_ranking.csv")


if __name__ == "__main__":
    main()