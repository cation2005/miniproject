import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---- 1. Model comparison (same stations for every model) ----
a = pd.read_csv("data/arima_metrics.csv")
a = a[a.model == "arima"]
l = pd.read_csv("data/lstm_metrics.csv")
common = set(zip(l.city, l.station_id)) & set(zip(a.city, a.station_id))
keep = lambda d: d[[k in common for k in zip(d.city, d.station_id)]]
tab = (pd.concat([keep(l), keep(a)])
       .groupby(["model", "horizon_h"])[["mae", "rmse"]].mean().round(2).reset_index())
tab.to_csv("report/model_comparison.csv", index=False)
print(f"Stations compared: {len(common)}\n", tab)

ax = tab.pivot(index="horizon_h", columns="model", values="mae").plot(kind="bar", figsize=(7, 4))
ax.set_xlabel("Forecast horizon (hours)")
ax.set_ylabel("MAE (µg/m³)")
ax.set_title(f"PM2.5 forecast error ({len(common)} stations)")
ax.set_ylim(0, 65)
ax.legend(ncol=3, loc="upper center")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("report/model_comparison.png", dpi=200)
plt.close()

# ---- 2. Station coverage ----
fc = pd.read_csv("data/forecast.csv", parse_dates=["forecast_time"])
fc.groupby(["city", "pollutant"]).station_id.nunique().rename("stations") \
  .to_csv("report/station_coverage.csv")

# ---- 3. Example forecast for one Delhi station ----
cl = pd.read_csv("data/clean.csv", parse_dates=["timestamp"])
d = fc[(fc.city == "Delhi") & (fc.pollutant == "pm25")]
sid = d.station_id.iloc[0]
hist = cl[cl.station_id == sid].set_index("timestamp")["pm25"].sort_index()
hist = hist[hist.index >= hist.index.max() - pd.Timedelta(days=5)]
fut = d[d.station_id == sid].sort_values("forecast_time")
plt.figure(figsize=(8, 4))
plt.plot(hist.index, hist.values, label="observed")
plt.plot(fut.forecast_time, fut.value, marker="o", label="forecast")
plt.ylabel("PM2.5 (µg/m³)")
plt.title(f"Delhi station {sid}: last 5 days and 72 h forecast")
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("report/forecast_example.png", dpi=200)
plt.close()

# ---- 4. Exposure map (24 h ahead) per city ----
ex = pd.read_csv("data/exposure.csv")
for city in ["Delhi", "Bangalore"]:
    e = ex[ex.city == city]
    t = sorted(e.forecast_time.unique())[3]          # about 24 h ahead
    e = e[e.forecast_time == t]
    st = fc[(fc.city == city) & (fc.pollutant == "pm25") & (fc.horizon_hours == 24)]
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(e.lon, e.lat, c=e.exposure_score, s=6, cmap="YlOrRd")
    plt.colorbar(sc, label="Exposure score (population × PM2.5)")
    plt.scatter(st.lon, st.lat, c="black", marker="^", s=30, label="stations")
    plt.legend()
    plt.title(f"{city}: forecast exposure at {t}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"report/exposure_map_{city.lower()}.png", dpi=200)
    plt.close()
print("Figures saved in report/")
