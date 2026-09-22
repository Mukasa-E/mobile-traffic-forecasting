# Mobile Traffic Forecasting

This project processes the Milan Telecom mobile traffic dataset and compares
three forecasting models for internet traffic:

- SARIMA with Fourier terms for daily seasonality
- A small LSTM
- A small Transformer encoder

The target is total internet traffic for each square and 10-minute interval.

## Project Structure

```text
data/raw/                 Raw daily tab-separated files
processed/                Aggregated parquet dataset
src/pipeline.py           Memory-efficient raw-data ingestion
src/eda.py                Exploratory analysis and figures
src/run_experiments.py    Model training and forecasting
src/evaluate.py           Cross-model comparison and failure analysis
results/                  Metrics, predictions, figures, and experiment log
```

## Requirements

Use Python 3.10 or newer, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The project has been tested with pandas, PyArrow, statsmodels, scikit-learn,
PyTorch, NumPy, and Matplotlib.

## Get the Data

The raw Milan Telecom dataset is available from Harvard Dataverse:

<https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV>

Download individual daily files and place them in `data/raw/`. The files are
tab-separated, have no header, and use names such as:

```text
sms-call-internet-mi-2013-11-01.txt
```

For the complete experiment, collect the period needed by the assignment,
including the evaluation week from December 16 through December 22.

## Build the Parquet Dataset

Run commands from the repository root. Process one raw day at a time:

```bash
python src/pipeline.py data/raw/sms-call-internet-mi-2013-11-01.txt --compare-naive
python src/pipeline.py data/raw/sms-call-internet-mi-2013-11-02.txt
```

The first command reports a naive whole-file memory estimate. The optimized
pipeline reads 500,000-row chunks, keeps only the required columns, aggregates
country-level records immediately, and writes to:

```text
processed/internet_traffic.parquet
```

Repeat the command for each daily file. Do not process the same day twice. In
particular, files such as `2013-11-20.txt` and
`2013-11-20 (1).txt` may be duplicate downloads. The pipeline now removes
duplicate `(square_id, timestamp)` keys before writing, but avoiding duplicate
inputs prevents unnecessary work and double-counting concerns.

The final parquet contains these columns:

```text
square_id, internet, timestamp
```

## Run Exploratory Analysis

After building the parquet dataset:

```bash
python src/eda.py
```

This creates figures in `figures/`:

- `fig1_distribution.png`: traffic distribution across squares
- `fig2_timeseries.png`: selected square time series
- `fig3_acf.png`: autocorrelation and partial autocorrelation
- `fig4_decomposition.png`: STL seasonal decomposition

## Run Forecasting Experiments

```bash
python src/run_experiments.py
```

The experiment selects the three squares with the highest total traffic and
forecasts the week of December 16 through December 22. Each model uses
one-step-ahead evaluation with the actual previous observation available for
the next prediction.

The model settings are documented in
`results/experiment_log.md`. The final SARIMA baseline uses an ARIMA model with
Fourier terms rather than a full period-144 seasonal state-space model. This
keeps the daily seasonal component while avoiding impractical fitting times.

## Evaluate and Compare Models

Run this after the forecasting experiment:

```bash
python src/evaluate.py
```

It writes:

- `results/tables/combined_comparison.csv`
- `results/tables/overall_ranking.csv`
- `results/figures/failure_case.png`

Per-square metrics and prediction files are saved in
`results/tables/square*_metrics.csv` and
`results/tables/square*_predictions.csv`.

## Current Results

In the recorded run, the LSTM ranked best overall by average rank across MAE,
MAPE, and RMSE. The Transformer was strongest on some individual metrics, and
SARIMA provided a useful classical baseline but had much slower rolling
inference.

Detailed metrics, timings, preprocessing notes, and limitations are recorded
in [`results/experiment_log.md`](results/experiment_log.md).

## Reproducibility Notes

- The pipeline aggregates traffic across `CountryCode` rows for each
  `(SquareId, TimeInterval)` pair.
- Missing 10-minute intervals are interpolated before forecasting.
- Neural models normalize using training-set statistics only.
- Random seeds are set in the LSTM and Transformer modules.
- All evaluation metrics are calculated on the held-out evaluation week.
