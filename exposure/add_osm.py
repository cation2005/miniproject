import time

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

URLS = ["https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"]
TILES = 3  # each city is split into TILES x TILES requests

exposure = pd.read_csv("data/exposure.csv")
cells = exposure[["city", "cell_id", "lat", "lon"]].drop_duplicates()


def overpass(s, w, n, e):
    q = (f'[out:json][timeout:90];'
         f'nwr["amenity"~"^(school|hospital)$"]({s},{w},{n},{e});out center tags;')
    for attempt in range(3):
        for url in URLS:
            try:
                r = requests.post(url, data={"data": q}, timeout=120,
                                  headers={"User-Agent": "miniproject-airquality/1.0"})
                r.raise_for_status()
                return r.json()["elements"]
            except Exception as err:
                print("   failed:", url.split("/")[2], str(err)[:60])
        time.sleep(15)
    return None


def tiles(s, w, n, e, k):
    la, lo = np.linspace(s, n, k + 1), np.linspace(w, e, k + 1)
    for i in range(k):
        for j in range(k):
            yield la[i], lo[j], la[i + 1], lo[j + 1]


for city, c in cells.groupby("city"):
    if exposure.loc[exposure.city == city, "schools_count"].notna().any():
        print(city, "already filled - skipping")
        continue
    c = c.reset_index(drop=True)
    seen, ok = {}, True
    box = (c.lat.min() - 0.01, c.lon.min() - 0.01, c.lat.max() + 0.01, c.lon.max() + 0.01)
    for t, (s, w, n, e) in enumerate(tiles(*box, TILES), 1):
        els = overpass(s, w, n, e)
        if els is None:
            print(f"  tile {t} failed")
            ok = False
            break
        for el in els:
            seen[(el["type"], el["id"])] = el
        print(f"  {city} tile {t}/{TILES ** 2} ok ({len(seen)} features so far)")
        time.sleep(2)
    if not ok:
        print("Skipping", city, "(left blank)")
        continue

    cosf = np.cos(np.radians(c.lat.mean()))
    tree = cKDTree(np.c_[c.lat.values, c.lon.values * cosf])
    sch, hos = np.zeros(len(c), int), np.zeros(len(c), int)
    for el in seen.values():
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        kind = el.get("tags", {}).get("amenity")
        if lat is None:
            continue
        d, i = tree.query([lat, lon * cosf])
        if d > 0.007:
            continue
        if kind == "school":
            sch[i] += 1
        elif kind == "hospital":
            hos[i] += 1
    print(city, "-> schools:", sch.sum(), "hospitals:", hos.sum())
    cnt = pd.DataFrame({"cell_id": c.cell_id, "schools_count": sch, "hospitals_count": hos})
    mask = exposure.city == city
    m = exposure.loc[mask, ["cell_id"]].merge(cnt, on="cell_id", how="left")
    exposure.loc[mask, "schools_count"] = m.schools_count.values
    exposure.loc[mask, "hospitals_count"] = m.hospitals_count.values

exposure.to_csv("data/exposure.csv", index=False)
print("Saved data/exposure.csv")
