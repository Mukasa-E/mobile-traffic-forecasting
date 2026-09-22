"""
Exploratory analysis on the aggregated Milan internet-traffic dataset.
Reads processed/internet_traffic.parquet (built by pipeline.py) and produces:

  1. Distribution of total 2-month traffic per square           -> fig1_distribution.png
  2. Top-3 squares by total traffic (+ ranking printed)
  3. First-two-weeks time series for top-3 + square 4159 + 4556  -> fig2_timeseries.png
  4. ACF/PACF of the top-traffic square                          -> fig3_acf.png
  5. STL seasonal decomposition of the top-traffic square        -> fig4_decomposition.png

Run after you've ingested at least the first two weeks + full period
totals via pipeline.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet("processed/internet_traffic.parquet")
df = df.drop_duplicates(subset=["square_id", "timestamp"])

# ---------------------------------------------------------------------
# 1. Distribution of total traffic per square over the full period
# ---------------------------------------------------------------------
total_per_square = df.groupby("square_id")["internet"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(total_per_square.values, bins=80, color="#3b6ea5")
ax.set_xlabel("Total internet traffic (2-month sum)")
ax.set_ylabel("Number of squares")  
ax.set_title("Distribution of total internet traffic across 10,000 areas")
fig.tight_layout()
fig.savefig(OUT / "fig1_distribution.png", dpi=150)
plt.close(fig)

print("Skew:", total_per_square.skew())
print("Top 3 squares by total traffic:")
print(total_per_square.head(3))

top3 = total_per_square.head(3).index.tolist()
target_squares = top3 + [4159, 4556]

# ---------------------------------------------------------------------
# 2. First two weeks, 5 squares
# ---------------------------------------------------------------------
start = df["timestamp"].min()
two_weeks_end = start + pd.Timedelta(days=14)

fig, axes = plt.subplots(len(target_squares), 1, figsize=(10, 2.2 * len(target_squares)), sharex=True)
for ax, sq in zip(axes, target_squares):
    sub = df[(df.square_id == sq) & (df.timestamp < two_weeks_end)]
    ax.plot(sub["timestamp"], sub["internet"], linewidth=0.8)
    label = f"Square {sq}" + (" (top traffic)" if sq in top3 else "")
    ax.set_ylabel(label, fontsize=8)
axes[-1].set_xlabel("Time")
fig.suptitle("First two weeks: internet traffic time series")
fig.tight_layout()
fig.savefig(OUT / "fig2_timeseries.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# 3 & 4. ACF/PACF and STL decomposition for the single highest-traffic square
# ---------------------------------------------------------------------
top_square = top3[0]
series = df[df.square_id == top_square].set_index("timestamp")["internet"].asfreq("10min")
series = series.interpolate()  # fill any gaps from missing intervals

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(series, lags=288, ax=axes[0])  # 2 days of lags at 10-min resolution
    plot_pacf(series, lags=50, ax=axes[1])
    axes[0].set_title(f"ACF — square {top_square}")
    axes[1].set_title(f"PACF — square {top_square}")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_acf.png", dpi=150)
    plt.close(fig)
except ImportError:
    print("statsmodels not installed — skipping ACF/PACF plot")

try:
    from statsmodels.tsa.seasonal import STL
    stl = STL(series, period=144, robust=True)  # 144 = one day at 10-min resolution
    res = stl.fit()
    fig = res.plot()
    fig.set_size_inches(9, 7)
    fig.suptitle(f"STL decomposition — square {top_square}")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_decomposition.png", dpi=150)
    plt.close(fig)
except ImportError:
    print("statsmodels not installed — skipping STL decomposition")

print("\nFigures written to:", OUT.resolve())
