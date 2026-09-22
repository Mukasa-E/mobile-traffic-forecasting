# Mobile Traffic Forecasting Experiment Log

## Run Summary

**Run date:** 2026-09-22  
**Dataset:** Milan Telecom traffic data  
**Target:** Total internet traffic per square and 10-minute interval  
**Evaluation period:** 2013-12-16 through 2013-12-22, inclusive  
**Evaluation granularity:** 10 minutes  
**Hardware:** CPU-only execution; exact hardware is recorded in `results/tables/hardware.txt` when generated.

The experiment compared three one-step-ahead forecasting models on the three
highest-traffic squares: `5161`, `5059`, and `5259`.

## Data Processing

Raw files were read in chunks of 500,000 rows. The pipeline retained only
`square_id`, `time_interval`, and `internet`, downcast traffic values to
`float32`, and aggregated country-level rows by `(square_id, time_interval)`.

The generated parquet initially contained duplicate keys because the same raw
day had been ingested more than once. Duplicate records had identical traffic
values, so the parquet was rewritten with one record per
`(square_id, timestamp)` using the first copy:

- Rows before deduplication: `166,976,670`
- Rows after deduplication: `82,048,368`
- Duplicate rows removed: `84,928,302`
- Final parquet size: approximately `376 MB`

The forecasting helpers also remove duplicate keys before resampling. Future
pipeline appends deduplicate the combined dataset before writing.

## Experimental Procedure

For each selected square:

1. Build a regular 10-minute series and interpolate missing intervals.
2. Split the series at `2013-12-16`.
3. Train on all earlier observations.
4. Forecast the evaluation week one step at a time, using the actual previous
	observation for the next input window.
5. Calculate MAE, MAPE, and RMSE.

The evaluation contains `6,486` training points and `1,008` test points per
square.

## Model Configurations

### SARIMA baseline

The initial seasonal SARIMA configuration used a seasonal period of 144
observations per day. Its state-space optimization was too expensive and was
interrupted during fitting.

The final baseline uses ARIMA `(2, 0, 2)` with three Fourier harmonics for the
daily cycle. This preserves daily seasonality without creating the very large
state vector produced by a full period-144 seasonal SARIMA. The model is
refit every 144 forecast steps.

### LSTM

- Sequence length: 24 observations (4 hours)
- Hidden units: 32
- Layers: 1
- Epochs: 15
- Batch size: 64
- Learning rate: `0.001`
- Optimizer: Adam

### Transformer

- Sequence length: 24 observations (4 hours)
- Model dimension: 32
- Attention heads: 4
- Encoder layers: 2
- Feed-forward dimension: 64
- Epochs: 15
- Batch size: 64
- Learning rate: `0.001`
- Optimizer: Adam

Both neural models normalized values using training-set statistics only and
used rolling one-step-ahead prediction during evaluation.

## Results

Metrics are lower-is-better.

| Square | Model | MAE | MAPE (%) | RMSE | Train (s) | Inference (s) |
|---:|---|---:|---:|---:|---:|---:|
| 5059 | SARIMA | 70.21 | 8.47 | 94.49 | 4.22 | 70.54 |
| 5059 | LSTM | 58.82 | 5.98 | 88.57 | 4.46 | 0.22 |
| 5059 | Transformer | 59.00 | 6.37 | 89.70 | 27.67 | 0.52 |
| 5161 | SARIMA | 84.65 | 10.83 | 124.32 | 3.91 | 72.19 |
| 5161 | LSTM | 84.09 | 11.64 | 122.78 | 4.48 | 0.22 |
| 5161 | Transformer | 79.83 | 7.79 | 129.31 | 26.73 | 0.49 |
| 5259 | SARIMA | 62.49 | 7.96 | 89.42 | 9.06 | 68.44 |
| 5259 | LSTM | 60.25 | 7.11 | 88.97 | 4.75 | 0.21 |
| 5259 | Transformer | 74.62 | 11.44 | 98.53 | 27.48 | 0.48 |

Average rank across all three metrics and squares:

| Model | Average rank |
|---|---:|
| LSTM | 1.33 |
| Transformer | 2.22 |
| SARIMA | 2.44 |

## Findings

- **LSTM is the strongest overall model** by average rank and has the lowest
  average computational cost among the three models.
- **Transformer performs best on square 5161 by MAPE** and is competitive on
	square 5059, but it takes substantially longer to train.
- **SARIMA is a useful classical baseline**, but rolling inference is much
	slower than either neural model because each step updates a statsmodels state
	space model.
- The best model depends somewhat on the square and metric. The LSTM is the
	recommended general-purpose model for this experiment, while the Transformer
	is a reasonable alternative when accuracy on square 5161 is prioritized.

## Reproduction Commands

From the repository root:

```bash
python src/eda.py
python src/run_experiments.py
python src/evaluate.py
```

Main outputs:

- `results/tables/combined_comparison.csv`
- `results/tables/overall_ranking.csv`
- `results/tables/square*_metrics.csv`
- `results/tables/square*_predictions.csv`
- `results/figures/`
