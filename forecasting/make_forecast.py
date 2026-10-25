import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

POLL = ["pm25", "pm10", "no2", "co", "o3"]
HORIZONS = list(range(6, 73, 6))  # 6, 12, ..., 72
MIN_POINTS = 120
MAX_STALE_H = 48  # skip stations that stopped reporting

df = pd.read_csv("data/clean.csv", parse_dates=["timestamp"])
latest = df["timestamp"].max()
print("Latest data in clean.csv:", latest)
rows = []

for (city, sid), g in df.groupby(["city", "station_id"]):
    lat, lon = g["lat"].iloc[0], g["lon"].iloc[0]
    for pol in POLL:
        s = g.set_index("timestamp")[pol].sort_index()
        s = s[~s.index.duplicated()].dropna()
        if s.empty or s.index.max() < latest - pd.Timedelta(hours=MAX_STALE_H):
            continue
        s = s.asfreq("h").interpolate(limit=24, limit_area="inside")
        gaps = s[s.isna()]
        if len(gaps):
            s = s[s.index > gaps.index.max()]
        if len(s) < MIN_POINTS:
            continue

        origin = s.index[-1]
        fc = np.clip(ARIMA(s, order=(2, 1, 2)).fit().forecast(max(HORIZONS)).values, 0, None)
        for h in HORIZONS:
            rows.append({
                "forecast_time": (origin + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
                "city": city, "station_id": sid, "lat": lat, "lon": lon,
                "pollutant": pol, "value": round(float(fc[h - 1]), 2),
                "horizon_hours": h,
            })

out = pd.DataFrame(rows)
out.to_csv("data/forecast_arima.csv", index=False)
print("Saved", len(out), "rows to data/forecast_arima.csv")
print(out.groupby(["city", "pollutant"])["station_id"].nunique())
