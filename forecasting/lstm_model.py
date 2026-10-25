import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

torch.manual_seed(0)
np.random.seed(0)

L, H = 48, 72                      # 48 h of history in, 72 h out
HORIZONS = [6, 24, 72]
OUT_H = list(range(6, 73, 6))
EPOCHS = 25
WEATHER = ["temp", "humidity", "wind_speed"]

df = pd.read_csv("data/clean.csv", parse_dates=["timestamp"])
latest = df["timestamp"].max()

raw = {}
for (city, sid), g in df.groupby(["city", "station_id"]):
    lat, lon = g["lat"].iloc[0], g["lon"].iloc[0]
    g = g.set_index("timestamp").sort_index()
    g = g[~g.index.duplicated()].asfreq("h")
    if g["pm25"].last_valid_index() < latest - pd.Timedelta(hours=48):
        continue
    f = g[["pm25"] + WEATHER].copy()
    f["pm25"] = f["pm25"].interpolate(limit=24, limit_area="inside")
    for c in WEATHER:
        f[c] = f[c].interpolate(limit=24, limit_area="inside").ffill(limit=24).bfill(limit=24)
    raw[(city, sid)] = (f, lat, lon)
print("Stations used:", len(raw))

# ---- scaling ----
allpm = np.log1p(np.clip(np.concatenate([f["pm25"].dropna().values for f, _, _ in raw.values()]), 0, None))
PM_MU, PM_SD = allpm.mean(), allpm.std()
W_MU, W_SD = {}, {}
for c in WEATHER:
    v = np.concatenate([f[c].values for f, _, _ in raw.values()])
    W_MU[c], W_SD[c] = np.nanmean(v), np.nanstd(v) + 1e-6


def prep(f):
    hrs = np.asarray(f.index.hour)
    cols = [(np.log1p(np.clip(f["pm25"].values, 0, None)) - PM_MU) / PM_SD]
    for c in WEATHER:
        cols.append(np.nan_to_num((f[c].values - W_MU[c]) / W_SD[c], nan=0.0))
    cols += [np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24)]
    return np.stack(cols, axis=1).astype(np.float32), f["pm25"].values


def pairs(X):
    n = len(X)
    m = n - L - H + 1
    if m <= 0:
        return None, None
    Xw = np.transpose(sliding_window_view(X, L, axis=0), (0, 2, 1))[:m]
    Yw = sliding_window_view(X[:, 0], H)[L:L + m]
    ok = ~np.isnan(Xw[:, :, 0]).any(1) & ~np.isnan(Yw).any(1)
    return Xw, Yw, ok


class Net(nn.Module):
    def __init__(self, nf=6, h=64):
        super().__init__()
        self.lstm = nn.LSTM(nf, h, batch_first=True)
        self.fc = nn.Linear(h, H)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])


def train(X, Y):
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    X, Y = torch.tensor(X), torch.tensor(Y)
    for ep in range(EPOCHS):
        perm, tot = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(X[b]), Y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 4:
            print(f"  epoch {ep + 1}: loss {tot / len(X):.4f}")
    return net.eval()


def inv(p):
    return np.clip(np.expm1(p * PM_SD + PM_MU), 0, None)


# ---- build windows ----
data = {}
for key, (f, lat, lon) in raw.items():
    X, pm_raw = prep(f)
    out = pairs(X)
    if out[0] is None:
        continue
    data[key] = (X, pm_raw, *out)

# ---- 1) train on everything except each station's last 72 h, then test there ----
trX, trY = [], []
for key, (X, pm_raw, Xw, Yw, ok) in data.items():
    n = len(X)
    m_tr = n - 2 * H - L + 1
    if m_tr > 0:
        keep = ok[:m_tr]
        trX.append(Xw[:m_tr][keep])
        trY.append(Yw[:m_tr][keep])
trX, trY = np.concatenate(trX), np.concatenate(trY)
print("Training windows:", len(trX))
net = train(trX, trY)

rows = []
for (city, sid), (X, pm_raw, Xw, Yw, ok) in data.items():
    n = len(X)
    if not ok[-1] or np.isnan(pm_raw[n - H:]).any() or np.isnan(pm_raw[n - H - 1]):
        continue
    with torch.no_grad():
        pred = inv(net(torch.tensor(Xw[-1:])).numpy()[0])
    truth, pers = pm_raw[n - H:], np.full(H, pm_raw[n - H - 1])
    for h in HORIZONS:
        for name, p in [("lstm", pred), ("persistence", pers)]:
            e = truth[:h] - p[:h]
            rows.append({"city": city, "station_id": sid, "model": name, "horizon_h": h,
                         "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean())})
res = pd.DataFrame(rows)
res.to_csv("data/lstm_metrics.csv", index=False)

# ---- compare with ARIMA on the same stations ----
am = pd.read_csv("data/arima_metrics.csv")
common = set(zip(res.city, res.station_id)) & set(zip(am.city, am.station_id))
res_c = res[[k in common for k in zip(res.city, res.station_id)]]
am_c = am[am.model == "arima"]
am_c = am_c[[k in common for k in zip(am_c.city, am_c.station_id)]]
both = pd.concat([res_c, am_c])
print("\nStations compared:", len(common))
print(both.groupby(["model", "horizon_h"])[["mae", "rmse"]].mean().round(2))

# ---- 2) retrain on all windows, forecast from the latest data ----
allX = np.concatenate([Xw[ok] for _, _, Xw, Yw, ok in data.values()])
allY = np.concatenate([Yw[ok] for _, _, Xw, Yw, ok in data.values()])
print("\nFinal model, windows:", len(allX))
final = train(allX, allY)

out_rows = []
for (city, sid), (f, lat, lon) in raw.items():
    X, _ = prep(f)
    win = X[-L:]
    if len(win) < L or np.isnan(win[:, 0]).any():
        continue
    with torch.no_grad():
        pred = inv(final(torch.tensor(win[None])).numpy()[0])
    origin = f.index[-1]
    for h in OUT_H:
        out_rows.append({
            "forecast_time": (origin + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
            "city": city, "station_id": sid, "lat": lat, "lon": lon,
            "pollutant": "pm25", "value": round(float(pred[h - 1]), 2), "horizon_hours": h})
pd.DataFrame(out_rows).to_csv("data/forecast_lstm.csv", index=False)
print("Saved", len(out_rows), "rows to data/forecast_lstm.csv")
