import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
HEADERS = {"X-API-Key": os.getenv("OPENAQ_KEY")}
BASE = "https://api.openaq.org/v3"

CITIES = {"Bangalore": (12.9716, 77.5946), "Delhi": (28.6139, 77.2090)}
POLLUTANTS = ["pm25", "pm10", "no2", "co", "o3"]
WEATHER = ["temperature", "relativehumidity", "wind_speed", "wind_direction"]
DAYS = 14
MAX_STATIONS = 12
PAUSE = 1.1  # stay under the API rate limit


def get(url, params=None):
    for attempt in range(6):
        r = requests.get(url, headers=HEADERS, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(PAUSE)
        return r.json()
    raise RuntimeError("Rate limited too many times")


def pick_sensors(loc):
    """One sensor per parameter; prefer ug/m3 units for pollutants."""
    chosen = {}
    for s in loc.get("sensors", []):
        p = s["parameter"]["name"]
        units = s["parameter"].get("units", "")
        if p not in POLLUTANTS + WEATHER:
            continue
        if p not in chosen or (p in POLLUTANTS and "g/m" in units and "g/m" not in chosen[p][1]):
            chosen[p] = (s["id"], units)
    return chosen


def find_stations(lat, lon):
    data = get(f"{BASE}/locations", {"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 100})
    stations = []
    for loc in data.get("results", []):
        sensors = pick_sensors(loc)
        n_poll = sum(p in sensors for p in POLLUTANTS)
        if "pm25" in sensors and n_poll >= 4:
            stations.append((n_poll + sum(w in sensors for w in WEATHER), loc, sensors))
    stations.sort(key=lambda x: -x[0])
    return stations[:MAX_STATIONS]


def fetch_sensor(sensor_id, start, end):
    rows, page = [], 1
    while True:
        data = get(
            f"{BASE}/sensors/{sensor_id}/hours",
            {"datetime_from": start, "datetime_to": end, "limit": 1000, "page": page},
        )
        res = data.get("results", [])
        rows.extend(res)
        if len(res) < 1000:
            return rows
        page += 1


end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=DAYS)
start_s, end_s = start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")

all_rows = []
for city, (lat, lon) in CITIES.items():
    stations = find_stations(lat, lon)
    print(f"{city}: using {len(stations)} stations")
    for _, loc, sensors in stations:
        lat_s, lon_s = loc["coordinates"]["latitude"], loc["coordinates"]["longitude"]
        print("  ", loc["id"], loc["name"])
        for param, (sid, units) in sensors.items():
            try:
                for r in fetch_sensor(sid, start_s, end_s):
                    all_rows.append({
                        "timestamp": r["period"]["datetimeFrom"]["utc"],
                        "city": city,
                        "station_id": loc["id"],
                        "station_name": loc["name"],
                        "lat": lat_s,
                        "lon": lon_s,
                        "parameter": param,
                        "value": r["value"],
                        "units": units,
                    })
            except Exception as e:
                print("     skipped", param, sid, "->", e)

df = pd.DataFrame(all_rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_openaq.csv", index=False)
print("Saved", len(df), "rows to data/raw_openaq.csv")
print(df.groupby(["parameter", "units"]).size())
