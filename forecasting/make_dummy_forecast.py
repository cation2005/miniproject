import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

CITIES = {"Bangalore": (12.9716, 77.5946), "Delhi": (28.6139, 77.2090)}
CITY_FACTOR = {"Bangalore": 1.0, "Delhi": 1.8}
BASE = {"pm25": 60, "pm10": 110, "no2": 40, "co": 1.2, "o3": 35}
HORIZONS = [6, 24, 72]
N_STATIONS = 25

start = pd.Timestamp.now(tz="UTC").floor("h")
rows = []

for city, (clat, clon) in CITIES.items():
    lats = clat + rng.normal(0, 0.06, N_STATIONS)
    lons = clon + rng.normal(0, 0.06, N_STATIONS)
    for i in range(N_STATIONS):
        station_id = f"{city[:3].upper()}_{i+1:02d}"
        dist = np.hypot(lats[i] - clat, lons[i] - clon)
        center_boost = 1 + 1.2 * np.exp(-dist / 0.04)  # busier near the centre
        for pol, base in BASE.items():
            for h in HORIZONS:
                value = base * CITY_FACTOR[city] * center_boost * rng.uniform(0.8, 1.2)
                rows.append({
                    "forecast_time": (start + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
                    "city": city,
                    "station_id": station_id,
                    "lat": round(lats[i], 5),
                    "lon": round(lons[i], 5),
                    "pollutant": pol,
                    "value": round(value, 2),
                    "horizon_hours": h,
                })

Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("data/dummy_forecast.csv", index=False)
print(f"Wrote {len(rows)} rows to data/dummy_forecast.csv")
