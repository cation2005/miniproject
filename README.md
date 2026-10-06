# AI-Driven Spatiotemporal Intelligence Framework for Atmospheric Risk Characterization and Exposure Analytics

College mini-project (AI & Data Science). Forecasts air pollution, detects statistically significant hotspots, estimates population exposure, and shows everything on one interactive risk-map dashboard.

## Modules
| Folder | Owner | Work |
|---|---|---|
| pipeline/, forecasting/ | Member 1 | Data pipeline, ARIMA, LSTM |
| hotspots/ | Member 2 | DBSCAN, Gi*, alerts |
| exposure/ | Member 3 | WorldPop, OSM, exposure score |
| dashboard/ | Member 4 | Streamlit + Folium, integration |

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Workflow
Never push to main. Work on a feat/ branch and open a pull request.