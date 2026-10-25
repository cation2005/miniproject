import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

URLS = ["https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"]
exposure = pd.read_csv("data/exposure.csv")
cells = exposure[["city", "cell_id", "lat", "lon"]].drop_duplicates()


def overpass(s, w, n, e):
    q = (f'[out:json][timeout:120];'
         f'nwr["amenity"~"^(school|hospital)$"]({s},{w},{n},{e});out center tags;')
    for url in URLS:
        try:
            r = requests.post(url, data={"data": q}, timeout=180,
                              headers={"User-Agent": "miniproject-airquality/1.0"})
            r.raise_for_status()
            return r.json()["elements"]
        except Exception as err:
            print("  failed:", url, err)
    return None


parts = []
for city, c in cells.groupby("city"):
    c = c.reset_index(drop=True)
    els = overpass(c.lat.min() - 0.01, c.lon.min() - 0.01,
                   c.lat.max() + 0.01, c.lon.max() + 0.01)
    if els is None:
        print("Skipping", city, "(left blank)")
        continue
    cosf = np.cos(np.radians(c.lat.mean()))
    tree = cKDTree(np.c_[c.lat.values, c.lon.values * cosf])
    sch = np.zeros(len(c), int)
    hos = np.zeros(len(c), int)
    for el in els:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        kind = el.get("tags", {}).get("amenity")
        if lat is None:
            continue
        d, i = tree.query([lat, lon * cosf])
        if d > 0.007:      # outside the grid
            continue
        if kind == "school":
            sch[i] += 1
        elif kind == "hospital":
            hos[i] += 1
    print(city, "-> schools:", sch.sum(), "hospitals:", hos.sum())
    parts.append(pd.DataFrame({"city": city, "cell_id": c.cell_id,
                               "schools_count": sch, "hospitals_count": hos}))

if parts:
    cnt = pd.concat(parts)
    exposure = exposure.drop(columns=["schools_count", "hospitals_count"]) \
                       .merge(cnt, on=["city", "cell_id"], how="left")
    exposure.to_csv("data/exposure.csv", index=False)
    print("Updated data/exposure.csv")

