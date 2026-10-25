"""
hotspots/alerts.py  (Member 2)

Fills the alert_flag column of data/hotspots.csv.

Rule: alert_flag = True when the forecast PM2.5 for that station is above
ALERT_THRESHOLD and the forecast is within ALERT_HORIZON_HOURS from now.

Run AFTER detect_hotspots.py:
    python hotspots/detect_hotspots.py --city Delhi
    python hotspots/alerts.py
"""
import pandas as pd

FORECAST_PATH = "data/forecast.csv"
HOTSPOTS_PATH = "data/hotspots.csv"

ALERT_THRESHOLD = 90       # ug/m3 PM2.5. India CPCB AQI: above 90 = "Poor"
ALERT_HORIZON_HOURS = 6    # only warn for forecasts up to 6 hours ahead


def main():
    fc = pd.read_csv(FORECAST_PATH, parse_dates=["forecast_time"])
    fc = fc[fc["pollutant"] == "pm25"][
        ["forecast_time", "city", "station_id", "value", "horizon_hours"]
    ]

    hs = pd.read_csv(HOTSPOTS_PATH, parse_dates=["forecast_time"])
    hs = hs.drop(columns=["alert_flag"], errors="ignore")

    merged = hs.merge(fc, on=["forecast_time", "city", "station_id"], how="left")
    merged["alert_flag"] = (
        (merged["value"] > ALERT_THRESHOLD)
        & (merged["horizon_hours"] <= ALERT_HORIZON_HOURS)
    )

    # keep exactly the shared-schema columns, in schema order
    cols = ["forecast_time", "city", "station_id", "lat", "lon",
            "cluster_id", "gi_zscore", "gi_pvalue", "is_significant", "alert_flag"]
    merged[cols].to_csv(HOTSPOTS_PATH, index=False)

    print(f"Rows: {len(merged)}")
    print(f"Alerts raised: {int(merged['alert_flag'].sum())}")
    print(merged[merged["alert_flag"]].groupby("city").size().to_string()
          if merged["alert_flag"].any() else "No alerts (no value above threshold).")


if __name__ == "__main__":
    main()

