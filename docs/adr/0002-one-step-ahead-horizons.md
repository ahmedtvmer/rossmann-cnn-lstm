# Horizons are one-step-ahead forecasts scored over the first 30 and 90 test days

The paper reports "30, 90 and 360 days ahead" but builds 7-lag inputs with next-day labels and never describes recursive forecasting, so we read each Horizon as one-step-ahead predictions (using true lags) scored over the first N days of the Test Period. The 360-day Horizon is dropped because the New Dataset covers only 942 days, leaving a 92-day Test Period. Metrics are computed on Open Days only, because closed days have zero sales and make MAPE undefined.

## Considered Options

- Recursive multi-step forecasting (feeding predictions back as lags): rejected as not described in the paper and harder to explain.
