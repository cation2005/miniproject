import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

RASTER = "data/worldpop/ind_ppp_2020_1km_Aggregated.tif"
# city: (centre lat, centre lon, half-width in degrees)
CITIES = {"Bangalore": (12.9716, 77.5946, 0.20), "Delhi": (28.6139, 77.2090, 0.25)}

fc = pd.read_csv("data/forecast.csv", parse_dates=["forecast_time"])
fc = fc[fc.pollutant == "pm25"]
out = []

with rasterio.open(RASTER) as src:
    for city, (clat, clon, r) in CITIES.items():
        win = from_bounds(clon - r, clat - r, clon + r, clat + r, src.transform)
        pop = src.read(1, window=win).astype(float)
        tr = src.window_transform(win)
        if src.nodata is not None:
            pop[pop == src.nodata] = np.nan
        pop[pop < 0] = np.nan

        rows, cols = np.indices(pop.shape)
        lons = (tr.c + (cols + 0.5) * tr.a).ravel()
        lats = (tr.f + (rows + 0.5) * tr.e).ravel()
        ids = np.array([f"{city[:3].upper()}_{i}_{j}" for i, j in zip(rows.ravel(), cols.ravel())])
        popf = pop.ravel()
        ok = ~np.isnan(popf)
        lons, lats, ids, popf = lons[ok], lats[ok], ids[ok], popf[ok]
        cosf = np.cos(np.radians(clat))

        g = fc[fc.city == city]
        for t, gt in g.groupby("forecast_time"):
            d2 = (lats[:, None] - gt.lat.values[None, :]) ** 2 \
               + ((lons[:, None] - gt.lon.values[None, :]) * cosf) ** 2
            w = 1.0 / (d2 + 1e-8)                       # IDW, power 2
            pm = (w * gt.value.values).sum(1) / w.sum(1)
            out.append(pd.DataFrame({
                "forecast_time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "city": city, "cell_id": ids,
                "lat": lats.round(5), "lon": lons.round(5),
                "population": popf.round(1),
                "pollution_value": pm.round(2),
                "exposure_score": (popf * pm).round(1),
                "schools_count": np.nan, "hospitals_count": np.nan,  # filled later (OSM)
            }))

res = pd.concat(out, ignore_index=True)
res.to_csv("data/exposure.csv", index=False)
print("Saved", len(res), "rows to data/exposure.csv")
print(res.groupby("city").agg(cells=("cell_id", "nunique"),
                              avg_pm25=("pollution_value", "mean"),
                              avg_exposure=("exposure_score", "mean")).round(1))
