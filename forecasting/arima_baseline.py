import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

HORIZONS = [6, 24, 72]
TEST_HOURS = 72
MIN_POINTS = 150

df = pd.read_csv("data/clean.csv", parse_dates=["timestamp"])
rows = []

for (city, sid), g in df.groupby(["city", "station_id"]):
    s = g.set_index("timestamp")["pm25"].sort_index()
    s = s[~s.index.duplicated()]
    s = s.asfreq("h").interpolate(limit=6, limit_area="inside")

    # keep only the latest unbroken stretch
    gaps = s[s.isna()]
    if len(gaps):
        s = s[s.index > gaps.index.max()]

    if len(s) < MIN_POINTS:
        print(f"skip {city} {sid}: only {len(s)} points")
        continue

    train, test = s[:-TEST_HOURS], s[-TEST_HOURS:]
    fc = np.clip(ARIMA(train, order=(2, 1, 2)).fit().forecast(TEST_HOURS).values, 0, None)
    persist = np.full(TEST_HOURS, train.iloc[-1])

    for h in HORIZONS:
        y = test.values[:h]
        for name, pred in [("arima", fc[:h]), ("persistence", persist[:h])]:
            rows.append({
                "city": city, "station_id": sid, "model": name, "horizon_h": h,
                "mae": np.mean(np.abs(y - pred[:h])),
                "rmse": np.sqrt(np.mean((y - pred[:h]) ** 2)),
            })

res = pd.DataFrame(rows)
res.to_csv("data/arima_metrics.csv", index=False)
print("Stations evaluated:", res.station_id.nunique())
print(res.groupby(["model", "horizon_h"])[["mae", "rmse"]].mean().round(2))
