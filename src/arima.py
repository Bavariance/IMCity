import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 1. Load data
df = pd.read_csv(
    "",
    parse_dates=["timestamp"]
)

# 2. Set datetime index and sort
df = df.set_index("timestamp").sort_index()
df = df.asfreq("15T")  # 15-minute grid

# 3. Select the series to model (temperature in Fahrenheit)
y = df["temp_F"]

# 4. Use only data before the IMC game starts for training
game_start = pd.Timestamp("2025-11-22 10:00:00", tz=None)  # adjust if needed
y_train = y[y.index < game_start]

# 5. Fit an ARIMA(p,d,q) model, e.g. ARIMA(2,1,2)
model = ARIMA(y_train, order=(2, 1, 2))
res = model.fit()

print(res.summary())

# 6. Forecast the next 48 half-hour steps (24 hours)
steps = 48
forecast = res.forecast(steps=steps)

print(forecast)
