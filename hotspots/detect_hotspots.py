"""
hotspots/detect_hotspots.py  (Member 2)

Input : data/forecast.csv
        forecast_time, city, station_id, lat, lon, pollutant, value, horizon_hours
Output: data/hotspots.csv
        forecast_time, city, station_id, lat, lon,
        cluster_id, gi_zscore, gi_pvalue, is_significant, alert_flag

For each city and forecast_time:
  - Getis-Ord Gi* on PM2.5 gives gi_zscore / gi_pvalue / is_significant
    (stations significantly higher than their neighbours).
  - DBSCAN groups the top-quartile PM2.5 stations into cluster_id
    (-1 = not in a cluster). This is a RELATIVE hotspot, not a significance test.
alert_flag is left False here; hotspots/alerts.py fills it in.

Usage:
    python hotspots/detect_hotspots.py --city Delhi
    python hotspots/detect_hotspots.py              # all cities
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from libpysal.weights import KNN
from esda.getisord import G_Local

IN_PATH = "data/forecast.csv"
OUT_PATH = "data/hotspots.csv"

K_NEIGHBOURS = 3         # small k: Delhi has only ~10 stations
P_THRESHOLD = 0.05       # Gi* significance level
DBSCAN_EPS_KM = 10.0     # high-PM2.5 stations within 10 km join one cluster
HIGH_QUANTILE = 0.75     # "high" = top quartile of PM2.5 in that city/time
DBSCAN_MIN_SAMPLES = 2
PERMUTATIONS = 999
MIN_STATIONS = 4         # below this Gi* is meaningless

OUT_COLS = [
    "forecast_time", "city", "station_id", "lat", "lon",
    "cluster_id", "gi_zscore", "gi_pvalue", "is_significant", "alert_flag",
]


def to_km(lat, lon):
    """Approximate planar km coordinates (fine at city scale)."""
    lat0 = np.deg2rad(np.mean(lat))
    x = np.asarray(lon) * 111.320 * np.cos(lat0)
    y = np.asarray(lat) * 110.574
    return np.column_stack([x, y])


def detect_one_snapshot(snap: pd.DataFrame) -> pd.DataFrame:
    """snap = all stations of one city at one forecast_time."""
    snap = snap.copy()
    snap["gi_zscore"] = np.nan
    snap["gi_pvalue"] = np.nan
    snap["is_significant"] = False
    snap["cluster_id"] = -1
    snap["alert_flag"] = False

    n = len(snap)
    if n < MIN_STATIONS:
        return snap

    xy = to_km(snap["lat"].values, snap["lon"].values)
    y = snap["value"].values.astype(float)
    if np.isnan(y).any():
        return snap  # missing values: skip rather than guess

    w = KNN.from_array(xy, k=min(K_NEIGHBOURS, n - 1))
    w.transform = "B"
    gi = G_Local(y, w, star=True, permutations=PERMUTATIONS, seed=42)

    snap["gi_zscore"] = gi.Zs
    snap["gi_pvalue"] = gi.p_sim
    # significant HIGH cluster = positive z and small p
    snap["is_significant"] = (gi.Zs > 0) & (gi.p_sim < P_THRESHOLD)

    # cluster_id = DBSCAN groups of the relatively highest-PM2.5 stations
    # (top quartile of this city/time). Independent of Gi* significance, so the
    # map still shows relative hotspots when nothing is statistically significant.
    high_idx = np.where(y >= np.quantile(y, HIGH_QUANTILE))[0]
    if len(high_idx) >= DBSCAN_MIN_SAMPLES:
        labels = DBSCAN(eps=DBSCAN_EPS_KM,
                        min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy[high_idx])
        snap.iloc[high_idx, snap.columns.get_loc("cluster_id")] = labels
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None, help="e.g. Delhi (default: all)")
    args = ap.parse_args()

    df = pd.read_csv(IN_PATH, parse_dates=["forecast_time"])
    df = df[df["pollutant"] == "pm25"]
    if args.city:
        df = df[df["city"].str.lower() == args.city.lower()]
    if df.empty:
        raise SystemExit("No pm25 rows found for that selection in " + IN_PATH)

    parts = [
        detect_one_snapshot(g)
        for _, g in df.groupby(["city", "forecast_time"])
    ]
    out = pd.concat(parts, ignore_index=True)[OUT_COLS]
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(f"Significant station-times: {int(out['is_significant'].sum())}")
   