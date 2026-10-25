import pandas as pd

KEY = ["city", "station_id", "pollutant", "horizon_hours"]
LSTM_MAX_H = 24

a = pd.read_csv("data/forecast_arima.csv")
l = pd.read_csv("data/forecast_lstm.csv")
l = l[l.horizon_hours <= LSTM_MAX_H][KEY + ["forecast_time", "value"]]

m = a.merge(l, on=KEY, how="left", suffixes=("", "_lstm"))
same = m["forecast_time"] == m["forecast_time_lstm"]   # only swap when origins match
use = m["value_lstm"].notna() & same
m.loc[use, "value"] = m.loc[use, "value_lstm"]

cols = ["forecast_time", "city", "station_id", "lat", "lon", "pollutant", "value", "horizon_hours"]
m[cols].to_csv("data/forecast.csv", index=False)
print("Rows:", len(m), "| values taken from LSTM:", int(use.sum()))
print("LSTM rows skipped (time mismatch):", int((m["value_lstm"].notna() & ~same).sum()))
