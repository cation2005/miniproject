# Shared Data Schema

Cities: Bangalore, Delhi
Time: hourly, UTC, ISO format (2026-10-23 14:00:00)
Coordinates: WGS84 (lat, lon)

## 1. Clean dataset (Member 1 output) - data/clean.csv
timestamp, city, station_id, lat, lon,
pm25, pm10, no2, co, o3,
temp, humidity, wind_speed, wind_dir

## 2. Forecast (Member 1 -> Members 2, 3, 4) - data/forecast.csv
forecast_time, city, station_id, lat, lon, pollutant, value, horizon_hours

## 3. Hotspots (Member 2 -> Members 3, 4) - data/hotspots.csv
forecast_time, city, station_id, lat, lon,
cluster_id, gi_zscore, gi_pvalue, is_significant, alert_flag

## 4. Exposure (Member 3 -> Member 4) - data/exposure.csv
forecast_time, city, cell_id, lat, lon,
population, pollution_value, exposure_score,
schools_count, hospitals_count

## Rules
- Column names are lowercase with underscores
- Units: pollutants in µg/m³ (CO in mg/m³), temp in °C, wind in m/s
- Missing values are left blank (NaN), never 0
- Each person works only in their own folder
- Any schema change must be announced to the group first
