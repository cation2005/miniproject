import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAQ_KEY")
print("Key loaded:", bool(key))
headers = {"X-API-Key": key}

cities = {"Bangalore": (12.9716, 77.5946), "Delhi": (28.6139, 77.2090)}

for city, (lat, lon) in cities.items():
    r = requests.get(
        "https://api.openaq.org/v3/locations",
        headers=headers,
        params={"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 20},
        timeout=30,
    )
    print(city, r.status_code)
    for loc in r.json().get("results", []):
        params = [s["parameter"]["name"] for s in loc.get("sensors", [])]
        print(" ", loc["id"], loc["name"], params)
