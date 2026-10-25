import numpy as np
import pandas as pd

RAW = "data/raw_openaq.csv"
OUT = "data/clean.csv"
MOLAR_VOL = 24.45  # L/mol at 25 C, 1 atm

POLL = ["pm25", "pm10", "no2", "co", "o3"]
WEATHER = ["temp", "humidity", "wind_speed", "wind_dir"]
COLS = ["timestamp", "city", "station_id", "lat", "lon"] + POLL + WEATHER

df = pd.read_csv(RAW)
df["timestamp"] = (
    pd.to_datetime(df["timestamp"], utc=True).dt.floor("h").dt.tz_localize(None)
)

# ---- unit conversion to schema units ----
no2_ppb = (df.parameter == "no2") & (df.units == "ppb")
df.loc[no2_ppb, "value"] *= 46.01 / MOLAR_VOL            # ppb -> ug/m3

co_ppb = (df.parameter == "co") & (df.units == "ppb")
df.loc[co_ppb, "value"] *= 28.01 / MOLAR_VOL / 1000      # ppb -> mg/m3
co_ppm = (df.parameter == "co") & (df.units == "ppm")
df.loc[co_ppm, "value"] *= 28.01 / MOLAR_VOL             # ppm -> mg/m3
co_ug = (df.parameter == "co") & df.units.str.contains("g/m", na=False)
df.loc[co_ug, "value"] /= 1000                           # ug/m3 -> mg/m3

# ---- station metadata ----
meta = (
    df.groupby("station_id")
    .agg(city=("city", "first"), lat=("lat", "first"), lon=("lon", "first"))
    .reset_index()
)

# ---- long -> wide ----
wide = df.pivot_table(
    index=["station_id", "timestamp"], columns="parameter",
    values="value", aggfunc="mean",
).reset_index()
wide = wide.rename(columns={
    "temperature": "temp", "relativehumidity": "humidity",
    "wind_direction": "wind_dir",
})
for c in POLL + WEATHER:
    if c not in wide:
        wide[c] = np.nan

# ---- remove impossible values (never replace with 0) ----
for c in POLL + ["wind_speed", "humidity"]:
    wide.loc[wide[c] < 0, c] = np.nan
wide.loc[wide["humidity"] > 100, "humidity"] = np.nan

# ---- full hourly grid per station, fill short gaps only ----
frames = []
for sid, g in wide.groupby("station_id"):
    g = g.set_index("timestamp").sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
    g["station_id"] = sid
    fill = POLL + ["temp", "humidity", "wind_speed"]
    g[fill] = g[fill].interpolate(limit=3, limit_area="inside")
    g["wind_dir"] = g["wind_dir"].ffill(limit=3)
    frames.append(g.rename_axis("timestamp").reset_index())

clean = pd.concat(frames, ignore_index=True)
clean = clean.merge(meta, on="station_id", how="left")
clean = clean.dropna(subset=["pm25"])               # PM2.5 is required
clean["timestamp"] = clean["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
clean = clean[COLS].sort_values(["city", "station_id", "timestamp"])
clean.to_csv(OUT, index=False)

print("Saved", len(clean), "rows to", OUT)
print(clean.groupby("city")["station_id"].nunique().rename("stations"))
print("\nShare of missing values per column:")
print(clean[COLS].isna().mean().round(2))
