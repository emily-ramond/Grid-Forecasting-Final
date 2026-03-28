from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

CLUSTER_ID = 1
PRED_LEN   = 24
TEST_START = "2014-01-01 00:00"
MODEL_DIR  = "./models/cluster_1"

# ── Load model and training series once at startup ──
model        = joblib.load(f"{MODEL_DIR}/gb_model.pkl")
train_series = pd.read_csv(f"{MODEL_DIR}/train_series.csv", index_col=0, parse_dates=True).squeeze()
test_series  = pd.read_csv(f"{MODEL_DIR}/test_series.csv",  index_col=0, parse_dates=True).squeeze()
full_series  = pd.concat([train_series, test_series])

def build_features(history, timestamp):
    return {
        'lag_1':        history[-1]   if len(history) >= 1   else history[0],
        'lag_24':       history[-24]  if len(history) >= 24  else history[0],
        'lag_168':      history[-168] if len(history) >= 168 else history[0],
        'roll_mean_24': history[-24:].mean() if len(history) >= 24 else history.mean(),
        'roll_std_24':  history[-24:].std()  if len(history) >= 24 else history.std(),
        'hour':         timestamp.hour,
        'dow':          timestamp.dayofweek,
        'month':        timestamp.month,
        'is_weekend':   int(timestamp.dayofweek >= 5),
    }

@app.post("/forecast")
def forecast(item: dict):
    client_id = item["client_id"]

    # Use the last 168 hours of training data as the seed window
    window     = train_series.values[-168:]
    start_ts   = pd.Timestamp(TEST_START)
    preds      = []
    window_ext = window.copy()

    for step in range(PRED_LEN):
        ts    = start_ts + pd.Timedelta(hours=step)
        feats = build_features(window_ext, ts)
        arr   = np.array(list(feats.values())).reshape(1, -1)
        p     = max(np.expm1(model.predict(arr)[0]), 0.0)
        preds.append(p)
        window_ext = np.append(window_ext[1:], p)

    forecast_index = pd.date_range(start=TEST_START, periods=PRED_LEN, freq="h")

    return {
        "client_id": client_id,
        "cluster": CLUSTER_ID,
        "model": "GradientBoosting",
        "forecast_start": TEST_START,
        "forecast_hours": forecast_index.strftime("%Y-%m-%d %H:%M").tolist(),
        "forecast_values": preds
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
