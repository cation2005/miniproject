import os

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Air Quality Risk Map", page_icon="🌫️", layout="wide")

st.markdown("""

<style>
.stApp {
    background: radial-gradient(ellipse at top left, #102943, #08111f 55%, #060b14);
    color: #eaf4ff;
}
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}
[data-testid="stHeader"] {
    background: rgba(8,17,31,.82);
}
.hero {
    background: linear-gradient(115deg, #122b47, #124b61 55%, #087f8c);
    padding: 1.5rem 1.7rem;
    border: 1px solid #35d8d0;
    border-radius: 18px;
    margin-bottom: 1.3rem;
    box-shadow: 0 12px 38px #00c6ff20;
}
.hero h1 {
    margin: 0;
    font-size: clamp(1.4rem, 3vw, 2rem);
    color: #ffffff;
    font-weight: 850;
    letter-spacing: -.03em;
}
.hero p {
    margin: .45rem 0 0;
    color: #d5f5ff;
    font-size: .98rem;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #13253a, #0d192a);
    border: 1px solid #294760;
    padding: .9rem 1rem;
    border-radius: 14px;
    box-shadow: 0 7px 22px #00000020;
    min-height: 110px;
}
[data-testid="stMetricLabel"] {
    color: #b5cce0 !important;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #32e6d0 !important;
    font-weight: 800;
}
button[data-baseweb="tab"] {
    font-weight: 700;
    color: #b8cce0;
    border-radius: 10px 10px 0 0;
    padding-top: .7rem;
    padding-bottom: .7rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #32e6d0;
    background: #153149;
    border-bottom: 2px solid #32e6d0;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: .35rem;
    border-bottom: 1px solid #263b50;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1d30, #091321);
    border-right: 1px solid #20364c;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #eaf4ff;
}
[data-testid="stSidebar"] label {
    color: #c5d8e9 !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #263e54;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
hr {
    border-color: #263b50;
}
@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero {
        padding: 1.1rem;
    }
}
</style>

<div class="hero"><h1>🌫️ Spatiotemporal Atmospheric Risk Dashboard</h1>
<p>Forecast PM2.5 · hotspots · population exposure · Bangalore &amp; Delhi</p></div>
""", unsafe_allow_html=True)


@st.cache_data
def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


fc = load("data/forecast.csv")
ex = load("data/exposure.csv")
hs = load("data/hotspots.csv")
cmp_ = load("report/model_comparison.csv")
if fc is None:
    st.error("data/forecast.csv not found.")
    st.stop()

CENTRES = {"Bangalore": (12.9716, 77.5946), "Delhi": (28.6139, 77.2090)}
BINS = [0, 30, 60, 90, 120, 250, 10000]
LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
COLOURS = ["#00e676", "#c6ff00", "#ffea00", "#ff9100", "#ff1744", "#d500f9"]



def pm25_aqi(v):
    ranges = [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ]
    v = max(0.0, float(v))
    for lo, hi, ilo, ihi in ranges:
        if v <= hi:
            return round(ilo + (v - lo) * (ihi - ilo) / (hi - lo))
    return 500


def pm25_category(v):
    if v <= 30:
        return "Good"
    if v <= 60:
        return "Satisfactory"
    if v <= 90:
        return "Moderate"
    if v <= 120:
        return "Poor"
    if v <= 250:
        return "Very Poor"
    return "Severe"


station_names = {}
try:
    _stations = pd.read_csv("data/stations.csv")
    if {"station_id", "station_name"}.issubset(_stations.columns):
        station_names = dict(zip(
            _stations["station_id"].astype(str),
            _stations["station_name"].astype(str)
        ))
except Exception:
    pass

def colour(v):
    return COLOURS[min(int(pd.cut([v], BINS, labels=False)[0]), 5)]


# ---------- sidebar ----------
st.sidebar.header("Controls")
city = st.sidebar.selectbox("City", list(CENTRES))
f = fc[(fc.city == city) & (fc.pollutant == "pm25")]
times = sorted(f.forecast_time.unique())
t = st.sidebar.select_slider("Forecast time", options=times, value=times[0])
layers = st.sidebar.multiselect(
    "Map layers", ["Forecast PM2.5", "Exposure", "Hotspots"],
    default=["Forecast PM2.5", "Exposure", "Hotspots"])
ft = f[f.forecast_time == t]
e = ex[(ex.city == city) & (ex.forecast_time == t)] if ex is not None else None

tab_map, tab_fc, tab_ex, tab_model, tab_alert, tab_plan, tab_about = st.tabs(
    ["🗺️ Risk map", "📈 Forecast", "👥 Exposure", "🧪 Model comparison",
     "🚨 Alerts", "🧭 Rescue Planner", "ℹ️ About"])

# ---------- risk map ----------
with tab_map:
    st.subheader(f"📊 Risk Overview — {city}")

    c1, c2, c3, c4, c5 = st.columns(5)

    station_count = ft.station_id.nunique()
    average_pm25 = ft.value.mean()
    highest_pm25 = ft.value.max()
    stations_over_15 = int(
        ft.groupby("station_id").value.max().gt(15).sum()
    )

    if hs is not None and {
        "city", "forecast_time", "is_significant"
    }.issubset(hs.columns):
        selected_hotspots = hs[
            (hs["city"] == city)
            & (hs["forecast_time"] == t)
            & (
                hs["is_significant"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )
        ]
        hotspot_count = len(selected_hotspots)
    else:
        hotspot_count = 0

    c1.metric("📍 Monitoring Stations", f"{station_count}")
    c2.metric("🌫️ Average PM2.5", f"{average_pm25:.1f} µg/m³")
    c3.metric("📈 Highest Forecast", f"{highest_pm25:.1f} µg/m³")
    c4.metric("⚠️ Stations Above 15", f"{stations_over_15}")
    c5.metric("🔥 Significant Hotspots", f"{hotspot_count}")

    if e is not None and len(e):
        poor = e[e.pollution_value > 90].population.sum()
        st.metric("👥 Population in Poor-or-Worse Areas", f"{poor:,.0f}")

    st.caption(
        f"Selected forecast time: {t} · "
        "Station values are forecasts, not guaranteed real-time measurements."
    )

    m = folium.Map(location=CENTRES[city], zoom_start=11, tiles="OpenStreetMap")
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) "
        "brightness(1.1) contrast(.9);}</style>"))
    if "Exposure" in layers and e is not None and len(e):
        eh = e[e.exposure_score > 0.05 * e.exposure_score.max()]
        w = (eh.exposure_score / eh.exposure_score.max()).values
        HeatMap([[a, b, c] for a, b, c in zip(eh.lat, eh.lon, w)],
                radius=14, blur=18, min_opacity=0.35,
                gradient={0.2: "#00e5ff", 0.5: "#ffea00", 0.8: "#ff6d00",
                          1.0: "#ff1744"}).add_to(m)
    if "Forecast PM2.5" in layers:
        for r in ft.itertuples():
            folium.CircleMarker(
                (r.lat, r.lon), radius=9, color="#ffffff", weight=2,
                fill_color=colour(r.value), fill=True, fill_opacity=1,
                popup=folium.Popup(
                    f"<b>{station_names.get(str(r.station_id), chr(83) + chr(116) + chr(97) + chr(116) + chr(105) + chr(111) + chr(110) + chr(32) + str(r.station_id))}</b><br>"
                    f"PM2.5: {float(r.value):.1f} µg/m³<br>"
                    f"Estimated PM2.5 AQI sub-index: {pm25_aqi(r.value)}<br>"
                    f"Category: {pm25_category(float(r.value))}",
                    max_width=300
                ),
                tooltip=f"Click for details: {station_names.get(str(r.station_id), str(r.station_id))}").add_to(m)
    if "Hotspots" in layers:
        if hs is None:
            st.info("hotspots.csv not available yet")
        else:
            h = hs[(hs.city == city) & (hs.forecast_time == t)
                   & (hs.is_significant.astype(str).str.lower() == "true")]
            for r in h.itertuples():
                folium.CircleMarker(
                    (r.lat, r.lon), radius=18, color="#ff00e5", weight=4, fill=False,
                    tooltip=f"Hotspot (Gi* z={r.gi_zscore:.1f})").add_to(m)
    st_folium(m, height=600, use_container_width=True)

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #10243a, #0b1727);
        border: 1px solid #294760;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 12px;
    ">
      <h4 style="margin:0 0 12px 0;color:#eaf4ff;">
        🗺️ Map Legend
      </h4>
      <div style="display:flex;flex-wrap:wrap;gap:12px 22px;color:#d5e6f5;">
        <span>🟢 <b>0–30</b> · Good</span>
        <span>🟡 <b>31–60</b> · Satisfactory</span>
        <span>🟨 <b>61–90</b> · Moderate</span>
        <span>🟠 <b>91–120</b> · Poor</span>
        <span>🔴 <b>121–250</b> · Very Poor</span>
        <span>🟣 <b>Above 250</b> · Severe</span>
        <span>💗 <b>Pink ring</b> · Significant hotspot</span>
        <span>🌡️ <b>Heat layer</b> · Relative population exposure</span>
      </div>
      <p style="margin:12px 0 0;color:#a9c1d6;font-size:0.85rem;">
        Marker colours show PM2.5 concentration categories.
        Hotspots are shown only when supported by the hotspot dataset.
        The heat layer represents relative exposure intensity.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Forecasts are estimates, not live measurements. "
        "Map coverage depends on available monitoring stations and data."
    )

# ---------- forecast ----------
with tab_fc:
    st.subheader(f"PM2.5 forecast by station: {city}")
    top = f[f.horizon_hours == f.horizon_hours.max()].nlargest(5, "value").station_id.tolist()
    pick = st.multiselect("Stations", sorted(f.station_id.astype(str).unique()),
                          default=[str(s) for s in top])
    piv = (f.assign(station_id=f.station_id.astype(str))
             .pivot_table(index="forecast_time", columns="station_id", values="value"))
    if pick:
        st.line_chart(piv[pick])
    st.markdown(f"**Stations at {t}**")
    tbl = ft[["station_id", "value", "lat", "lon"]].sort_values("value", ascending=False).copy()
    tbl["category"] = pd.cut(tbl.value, BINS, labels=LABELS)
    st.dataframe(tbl.rename(columns={"value": "PM2.5 (µg/m³)"}),
                 use_container_width=True, hide_index=True)

# ---------- exposure ----------
with tab_ex:
    st.subheader(f"Population exposure: {city}")
    if e is None or not len(e):
        st.info("exposure.csv not available.")
    else:
        e = e.copy()
        e["category"] = pd.cut(e.pollution_value, BINS, labels=LABELS)
        a, b = st.columns(2)
        a.metric("Population in the mapped area", f"{e.population.sum():,.0f}")
        b.metric("Schools / hospitals mapped",
                 f"{int(e.schools_count.sum())} / {int(e.hospitals_count.sum())}")
        st.markdown("**People by air-quality category**")
        st.bar_chart(e.groupby("category", observed=False).population.sum())
        st.markdown("**10 highest-exposure cells**")
        st.dataframe(
            e.nlargest(10, "exposure_score")[
                ["lat", "lon", "population", "pollution_value", "exposure_score",
                 "schools_count", "hospitals_count"]],
            use_container_width=True, hide_index=True)

# ---------- model comparison ----------
with tab_model:
    st.subheader("Forecast accuracy (PM2.5, same stations for every model)")
    if cmp_ is None:
        st.info("report/model_comparison.csv not found.")
    else:
        st.bar_chart(cmp_.pivot(index="horizon_h", columns="model", values="mae"))
        st.dataframe(cmp_, use_container_width=True, hide_index=True)
        st.markdown("The LSTM is best at short range, ARIMA at 72 h; at 24 h they tie. "
                    "Results come from one 72-hour test window per station, so treat "
                    "them as indicative.")

# ---------- alerts ----------
with tab_alert:
    st.subheader(f"🚨 Air Quality Alert Centre — {city}")
    st.caption(
        "Alerts are calculated from available forecast data. "
        "Thresholds below are configurable project settings."
    )

    a1, a2 = st.columns(2)
    lead = a1.selectbox(
        "Look-ahead window (hours)",
        [6, 12, 24, 48, 72],
        index=0,
        key="alert_lead"
    )
    thr = a2.slider(
        "Alert threshold, PM2.5 (µg/m³)",
        min_value=5,
        max_value=30,
        value=15,
        step=1,
        key="alert_threshold"
    )

    win = f[f.horizon_hours <= lead]

    if win.empty:
        st.warning("No forecast records are available for this time window.")
    else:
        peak = (
            win.groupby("station_id")["value"]
            .max()
            .rename("peak")
            .reset_index()
        )

        first = (
            win[win.value > thr]
            .sort_values("horizon_hours")
            .groupby("station_id")["horizon_hours"]
            .first()
            .rename("first_exceeds_h")
            .reset_index()
        )

        res = peak.merge(first, on="station_id", how="left")
        res["category"] = res["peak"].apply(pm25_category)
        res["severity"] = res["peak"].apply(
            lambda v: (
                "Very High" if v > 90
                else "High" if v > 60
                else "Moderate" if v > 30
                else "Low"
            )
        )

        alerts = (
            res[res["peak"] > thr]
            .sort_values("peak", ascending=False)
            .copy()
        )

        total_stations = res["station_id"].nunique()
        alert_count = alerts["station_id"].nunique()
        highest = res["peak"].max()

        m1, m2, m3 = st.columns(3)
        m1.metric("📍 Stations checked", total_stations)
        m2.metric("🚨 Stations above threshold", alert_count)
        m3.metric("📈 Highest forecast PM2.5", f"{highest:.1f} µg/m³")

        if alerts.empty:
            st.success(
                f"No station exceeds {thr} µg/m³ within the next {lead} hours."
            )
        else:
            st.warning(
                f"{alert_count} of {total_stations} stations exceed "
                f"{thr} µg/m³ within the next {lead} hours."
            )

            severity_counts = (
                alerts["severity"]
                .value_counts()
                .reindex(["Low", "Moderate", "High", "Very High"], fill_value=0)
            )

            st.markdown("#### Alert severity breakdown")
            st.bar_chart(severity_counts)

            st.markdown("#### Stations requiring attention")

            display = alerts.rename(
                columns={
                    "station_id": "Station",
                    "peak": "Peak PM2.5 (µg/m³)",
                    "first_exceeds_h": "First exceedance (hours)",
                    "category": "PM2.5 category",
                    "severity": "Project alert level",
                }
            )

            columns = [
                "Station",
                "Peak PM2.5 (µg/m³)",
                "First exceedance (hours)",
                "PM2.5 category",
                "Project alert level",
            ]

            st.dataframe(
                display[columns],
                use_container_width=True,
                hide_index=True
            )

    if hs is not None and {
        "city", "alert_flag", "station_id"
    }.issubset(hs.columns):
        flagged = hs[
            (hs.city == city)
            & (hs.alert_flag.astype(str).str.lower().isin(["true", "1"]))
        ]
        st.markdown("#### 🔥 Hotspot module summary")
        st.metric(
            "Stations flagged by hotspot module",
            flagged.station_id.nunique()
        )

    st.caption(
        "Look-ahead hours are measured from the forecast start time. "
        "Alert levels are project-defined and should not be interpreted "
        "as official medical or regulatory advice."
    )


# ---------- clean-air rescue planner ----------
with tab_plan:
    st.subheader("🧭 Clean-Air Rescue Planner")
    st.write(
        "Explore which mapped locations should receive attention first "
        "under different, transparent prioritisation strategies."
    )

    st.info(
        "This is a decision-support demonstration. Scores are relative "
        "to the selected dataset and are not official health-risk scores "
        "or real-world emergency dispatch instructions."
    )

    if ex is None or e is None or e.empty:
        st.warning(
            "No exposure-grid records are available for this city and "
            "forecast time. Check data/exposure.csv and its timestamps."
        )
    else:
        import numpy as np

        plan = e.copy().reset_index(drop=True)

        required = {"lat", "lon", "population", "pollution_value"}
        missing = required - set(plan.columns)

        if missing:
            st.error(
                "The exposure data is missing required columns: "
                + ", ".join(sorted(missing))
            )
        else:
            # Clean numeric inputs.
            for col in [
                "lat", "lon", "population", "pollution_value",
                "exposure_score", "schools_count", "hospitals_count"
            ]:
                if col in plan.columns:
                    plan[col] = pd.to_numeric(plan[col], errors="coerce").fillna(0)

            plan["population"] = plan["population"].clip(lower=0)
            plan["pollution_value"] = plan["pollution_value"].clip(lower=0)

            if "exposure_score" not in plan.columns:
                plan["exposure_score"] = (
                    plan["population"] * plan["pollution_value"]
                )

            if "schools_count" not in plan.columns:
                plan["schools_count"] = 0

            if "hospitals_count" not in plan.columns:
                plan["hospitals_count"] = 0

            # Scale each component to 0–1 so the components are comparable.
            def normalise(series):
                series = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
                maximum = series.max()
                if maximum <= 0:
                    return pd.Series(0.0, index=series.index)
                return series / maximum

            plan["pollution_component"] = normalise(plan["pollution_value"])
            plan["exposure_component"] = normalise(plan["exposure_score"])

            facilities = (
                plan["schools_count"] + plan["hospitals_count"]
            )
            plan["facility_component"] = normalise(facilities)

            # Give locations near significant hotspots a proximity component.
            plan["hotspot_component"] = 0.0
            hotspot_count = 0

            if hs is not None and {
                "city", "forecast_time", "is_significant", "lat", "lon"
            }.issubset(hs.columns):
                hotspot_rows = hs[
                    (hs["city"] == city)
                    & (hs["forecast_time"] == t)
                    & (
                        hs["is_significant"]
                        .astype(str).str.lower().isin(["true", "1"])
                    )
                ].copy()

                hotspot_rows["lat"] = pd.to_numeric(
                    hotspot_rows["lat"], errors="coerce"
                )
                hotspot_rows["lon"] = pd.to_numeric(
                    hotspot_rows["lon"], errors="coerce"
                )
                hotspot_rows = hotspot_rows.dropna(subset=["lat", "lon"])
                hotspot_count = len(hotspot_rows)

                if hotspot_count:
                    # Approximate distance in kilometres to the nearest hotspot.
                    cell_lat = np.radians(plan["lat"].to_numpy())[:, None]
                    cell_lon = np.radians(plan["lon"].to_numpy())[:, None]
                    hot_lat = np.radians(hotspot_rows["lat"].to_numpy())[None, :]
                    hot_lon = np.radians(hotspot_rows["lon"].to_numpy())[None, :]

                    dlat = hot_lat - cell_lat
                    dlon = hot_lon - cell_lon

                    a = (
                        np.sin(dlat / 2) ** 2
                        + np.cos(cell_lat) * np.cos(hot_lat)
                        * np.sin(dlon / 2) ** 2
                    )
                    distance_km = 6371 * 2 * np.arcsin(
                        np.sqrt(np.clip(a, 0, 1))
                    )

                    nearest_km = distance_km.min(axis=1)
                    plan["hotspot_component"] = np.exp(-nearest_km / 3.0)

            strategies = {
                "Balanced priority": {
                    "pollution_component": 0.35,
                    "exposure_component": 0.30,
                    "facility_component": 0.20,
                    "hotspot_component": 0.15,
                },
                "Highest pollution first": {
                    "pollution_component": 0.75,
                    "exposure_component": 0.10,
                    "facility_component": 0.05,
                    "hotspot_component": 0.10,
                },
                "Population exposure first": {
                    "pollution_component": 0.15,
                    "exposure_component": 0.65,
                    "facility_component": 0.10,
                    "hotspot_component": 0.10,
                },
                "Protect schools and hospitals": {
                    "pollution_component": 0.20,
                    "exposure_component": 0.20,
                    "facility_component": 0.45,
                    "hotspot_component": 0.15,
                },
            }

            p1, p2 = st.columns(2)

            strategy = p1.selectbox(
                "Prioritisation strategy",
                list(strategies.keys()),
                key="rescue_strategy"
            )

            max_teams = max(1, min(20, len(plan)))
            teams = p2.slider(
                "Available response teams / priority slots",
                min_value=1,
                max_value=max_teams,
                value=min(3, max_teams),
                key="rescue_team_count"
            )

            weights = strategies[strategy]

            plan["priority_score"] = sum(
                plan[column] * weight
                for column, weight in weights.items()
            ) * 100

            plan = plan.sort_values(
                ["priority_score", "pollution_value"],
                ascending=False
            ).reset_index(drop=True)

            plan["priority_rank"] = np.arange(1, len(plan) + 1)

            # Label selected rows as planning candidates, not confirmed emergencies.
            plan["planning_status"] = np.where(
                plan["priority_rank"] <= teams,
                "Selected for priority review",
                "Lower relative priority"
            )

            selected = plan.head(teams)

            k1, k2, k3 = st.columns(3)
            k1.metric("📍 Mapped locations", len(plan))
            k2.metric("🎯 Priority slots", teams)
            k3.metric("🔥 Significant hotspots nearby", hotspot_count)

            st.markdown("### 🎯 Recommended priority locations")
            st.caption(
                "The ranking changes with the selected strategy. "
                "A higher score means higher relative priority within this dataset."
            )

            display = selected[
                [
                    "priority_rank", "lat", "lon", "pollution_value",
                    "population", "schools_count", "hospitals_count",
                    "priority_score", "planning_status"
                ]
            ].rename(
                columns={
                    "priority_rank": "Rank",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "pollution_value": "PM2.5 (µg/m³)",
                    "population": "Population estimate",
                    "schools_count": "Mapped schools",
                    "hospitals_count": "Mapped hospitals",
                    "priority_score": "Relative score / 100",
                    "planning_status": "Planning status",
                }
            )

            st.dataframe(
                display.round(2),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📊 Compare prioritisation strategies")

            comparison_rows = []
            for name, weights_for_strategy in strategies.items():
                scored = sum(
                    plan[column] * weight
                    for column, weight in weights_for_strategy.items()
                ) * 100

                temp = plan.copy()
                temp["comparison_score"] = scored
                temp = temp.sort_values(
                    ["comparison_score", "pollution_value"],
                    ascending=False
                ).head(teams)

                comparison_rows.append({
                    "Strategy": name,
                    "Priority locations selected": len(temp),
                    "Average PM2.5 in selected locations": round(
                        temp["pollution_value"].mean(), 2
                    ),
                    "Estimated population across selected cells": round(
                        temp["population"].sum()
                    ),
                    "Mapped schools across selected cells": int(
                        temp["schools_count"].sum()
                    ),
                    "Mapped hospitals across selected cells": int(
                        temp["hospitals_count"].sum()
                    ),
                })

            comparison = pd.DataFrame(comparison_rows)
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 🧮 Why were these locations prioritised?")

            explanation = (
                f"The selected strategy is **{strategy}**. "
                f"It assigns these weights: pollution "
                f"{weights['pollution_component']:.0%}, population exposure "
                f"{weights['exposure_component']:.0%}, mapped schools/hospitals "
                f"{weights['facility_component']:.0%}, and hotspot proximity "
                f"{weights['hotspot_component']:.0%}. "
                "Each component is scaled relative to the current dataset. "
                "These are design choices for comparison, not scientifically "
                "validated emergency-response weights."
            )
            st.write(explanation)

            st.download_button(
                "⬇️ Download strategy comparison (CSV)",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="clean_air_strategy_comparison.csv",
                mime="text/csv"
            )

            st.caption(
                "Important: population and facility counts are estimates from "
                "the mapped cells. Adjacent cells may represent the same "
                "population or facilities, so totals should not be interpreted "
                "as unique people or unique facilities without deduplication. "
                "This prototype does not model travel time, road access, "
                "response capacity, or the effect of interventions."
            )

# ---------- about ----------
with tab_about:
    st.markdown("""
**Data:** OpenAQ station measurements (PM2.5 and weather), WorldPop 2020 1 km
population, OpenStreetMap school and hospital features.

**Method:** ARIMA and LSTM forecasts, DBSCAN and Getis-Ord Gi* hotspots,
inverse-distance interpolation, exposure = population × PM2.5.

**Limitations:**
- Newest data is from 23 Oct 2026, so forecasts start in the past.
- Bangalore has only 4 usable stations, so its surface is heavily interpolated.
- OpenStreetMap counts are feature counts, not official facility numbers.
""")
