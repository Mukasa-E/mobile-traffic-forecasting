# Video Presentation Script

## Target length: 8-9 minutes

### Slide 1: Title and research question

**On screen:** Forecasting Mobile Network Traffic in Milan

Hello, my name is [your name], and this presentation describes my study of short-term mobile internet-traffic forecasting using the Milan Telecom dataset.

The research question is: which of three approaches, a classical ARIMA-based model, an LSTM, or a Transformer, provides the most accurate and computationally useful forecast for high-traffic areas?

This is important because mobile-network traffic changes over time and differs across locations. Reliable forecasts can help with capacity planning, network operations, and allocation of computing and radio resources.

The project is also a data-engineering problem. The raw dataset contains country-level records for many spatial squares and ten-minute intervals, so loading everything into memory at once is inefficient. My first objective was therefore to create a memory-aware processing pipeline before comparing forecasting models.

### Slide 2: Dataset and data challenge

**On screen:** Raw file example and project pipeline

The dataset comes from the Milan Telecom mobile-traffic collection hosted on Harvard Dataverse. Each daily file is tab-separated and has no header. The columns include a square identifier, a millisecond timestamp, country code, SMS and call measures, and Internet traffic.

For this study, I modeled only Internet traffic. Multiple rows can represent the same square and ten-minute interval because the traffic is separated by country code. These records must be aggregated before forecasting.

The pipeline reads each raw file in chunks of 500,000 rows. It selects only the square ID, timestamp, and Internet columns, uses compact numeric types, and immediately sums traffic across country codes for each square and interval.

The resulting data are stored as `processed/internet_traffic.parquet`, with one record per square and timestamp.

A practical data-quality issue occurred during processing: the same daily input had been included more than once. This produced duplicate square-timestamp keys. The duplicated records had identical traffic values, so I removed duplicate keys rather than summing them again. The parquet was reduced from 166,976,670 rows to 82,048,368 rows, and its final size was approximately 376 megabytes.

This was important for both correctness and memory use. Duplicate timestamps would also cause regular time-series reindexing to fail.

### Slide 3: Exploratory analysis

**On screen:** Figures 1 and 2

The first exploratory result is the distribution of total traffic by square. The distribution is strongly right-skewed, with skewness of approximately 4.25. This means traffic is not evenly distributed across the city. A relatively small number of areas account for much larger totals.

The three highest-traffic squares were 5161, 5059, and 5259. I selected these squares for the forecasting comparison because they are operationally important high-volume areas.

The second figure shows traffic over the first two weeks for these three squares and two additional reference squares. The series shows recurring within-day movement. This repeated daily structure is the main reason that the models include either an explicit daily seasonal component or a temporal input window capable of learning recent patterns.

### Slide 4: Time-series structure

**On screen:** Figures 3 and 4

The autocorrelation and partial autocorrelation plots show dependence at short lags and recurring daily structure. In this dataset, one day corresponds to 144 ten-minute observations.

The STL decomposition provides a similar interpretation. The seasonal component captures the recurring daily pattern, while the remainder contains local variation and occasional spikes. Those spikes are difficult to predict from recent values alone, which becomes important when interpreting the model failures later.

These analyses informed two design choices. First, the neural models use a recent temporal window of 24 observations, representing four hours. Second, the classical model includes Fourier terms that represent the daily cycle compactly.

### Slide 5: Forecasting setup

**On screen:** Train/test timeline

The evaluation period is December 16 through December 22, 2013. Observations before December 16 are used for training, and the held-out week is used for testing.

Each square has 6,486 training observations and 1,008 test observations at ten-minute resolution.

The evaluation is one-step-ahead. After each prediction, the actual observed value is used to construct the next input. This avoids compounding error from repeatedly feeding predictions back into the model. However, it is also an easier setting than a fully open-loop multi-step forecast, and that is one of the limitations of the study.

I evaluated the models using mean absolute error, mean absolute percentage error, and root mean squared error. Lower values are better. I also recorded training and inference time to compare predictive quality with computational cost.

### Slide 6: Models and technical decision

**On screen:** Three-model comparison

The first model is an ARIMA baseline with order `(2, 0, 2)` and three sine and cosine Fourier harmonics for the 144-observation daily cycle. I initially attempted a full seasonal state-space SARIMA model with seasonal period 144. That approach was computationally impractical because the seasonal state representation became expensive during optimization.

The important technical decision was to replace the full seasonal state-space component with Fourier terms as exogenous regressors. This preserved an explicit representation of daily seasonality while keeping the model computationally manageable. The ARIMA model was refit every 144 forecast steps.

The second model is a small LSTM. It uses a 24-observation input sequence, one recurrent layer, 32 hidden units, 15 training epochs, batch size 64, and the Adam optimizer with learning rate 0.001.

The third model is a small Transformer encoder. It uses the same 24-observation window, model dimension 32, four attention heads, two encoder layers, and a feed-forward dimension of 64. It uses the same optimizer, learning rate, epochs, and batch size as the LSTM.

Both neural models normalize using training-set statistics only. The parameters were selected as compact CPU-feasible settings rather than through an exhaustive hyperparameter search.

### Slide 7: Results

**On screen:** Results table

The results are relatively close, which makes the comparison more informative than a complete model collapse.

For square 5059, the LSTM performs best across all three accuracy metrics, with MAE 58.82, MAPE 5.98 percent, and RMSE 88.57.

For square 5161, the Transformer has the lowest MAE and MAPE, with MAE 79.83 and MAPE 7.79 percent. The LSTM is close in MAE at 84.09, while SARIMA has MAE 84.65. SARIMA is therefore close to the LSTM on MAE and RMSE for this square, although it is worse on MAPE.

For square 5259, the LSTM is again best across all three metrics, with MAE 60.25, MAPE 7.11 percent, and RMSE 88.97.

Across all squares and metrics, the average ranks are 1.33 for LSTM, 2.22 for Transformer, and 2.44 for SARIMA with Fourier terms. The LSTM is therefore the best general-purpose model in this experiment, but its advantage is narrow rather than dominant.

### Slide 8: Computational comparison

**On screen:** Timing table or bar chart

The timing results show a different set of trade-offs.

SARIMA with Fourier terms fits relatively quickly, taking approximately four to nine seconds depending on the square. However, its inference takes approximately 67 to 72 seconds because the forecast is updated sequentially through the state-space model.

The LSTM trains in approximately four to five seconds and has very fast inference, around 0.2 seconds for the evaluation week.

The Transformer takes longer to train, approximately 27 seconds, but its inference is still below one second for the full evaluation week.

This means that the LSTM offers the best balance in this experiment. SARIMA remains useful as an interpretable classical baseline, while the Transformer may be appropriate when its accuracy advantage for a particular square is more important than training cost.

### Slide 9: Failure analysis

**On screen:** Failure-case plot

The largest single error in the final comparison is a Transformer prediction for square 5161 on December 22 at 16:20.

The actual traffic value is 4,704.43, while the predicted value is 5,466.06, producing an absolute error of 761.62.

This is a meaningful failure case rather than a numerical breakdown such as a negative prediction. The Transformer overshoots a sharp traffic change. One possible explanation is that attention to recent momentum causes the model to react too strongly before an abrupt change in the local pattern. The available four-hour window may also be too short to identify whether the spike is part of a longer event or an isolated anomaly.

This example illustrates why average metrics alone are not enough. A model can perform well overall and still make costly errors during sudden traffic changes.

### Slide 10: Limitations and future work

**On screen:** Limitations and next steps

There are several limitations.

First, the evaluation uses one held-out week and only three high-traffic squares, so the results should not be generalized to every location or season.

Second, the forecasts are one-step-ahead and use the actual previous value. A multi-step open-loop evaluation would be more difficult and more representative of some operational planning tasks.

Third, the hyperparameters were selected manually as compact CPU-feasible settings. This is an informed baseline comparison, not an exhaustive tuning study.

Fourth, missing ten-minute intervals are filled by linear interpolation, and unusual spikes are not modeled explicitly.

Future work should use rolling-origin evaluation across multiple weeks, include more spatial squares, test direct multi-step forecasts, tune sequence lengths and Fourier harmonics systematically, and add spatial information so nearby squares can inform one another. Probabilistic forecasts would also be useful because uncertainty is especially important during sharp traffic changes.

### Slide 11: Conclusion

**On screen:** Main findings

To conclude, this project developed a memory-aware pipeline for processing a large mobile-traffic dataset and compared classical, recurrent, and attention-based forecasting models.

The most important engineering decision was chunked ingestion with early aggregation and duplicate-key handling. The most important modeling decision was replacing an impractical full seasonal SARIMA state-space model with a compact ARIMA model using Fourier terms for daily seasonality.

The LSTM ranked first overall, but the result was close. The Transformer was strongest for square 5161 on MAE and MAPE, and SARIMA remained competitive for selected metrics while being much slower at sequential inference.

The main conclusion is therefore not that one model always wins. It is that model selection involves a trade-off between accuracy, computational cost, interpretability, and robustness to sudden traffic changes.

Thank you for watching. The source code, setup instructions, experiment log, figures, and results are available in the project repository.
