# Grid-Forecasting-Final
An adaptive forecasting ecosystem for the Portuguese power grid using BasisFormer, PatchTST, and Gradient Boosting models

## Introduction
Motivates the cluster-first forecasting architecture. The core argument is that the 370 clients on the grid are behaviorally heterogeneous, and treating them as a single average or as 370 independent problems are both suboptimal strategies. The report introduces the four-stage solution: preprocessing, clustering, specialized model deployment, and an Agentic AI layer for real-time querying.

## Data Extraction & Preprocessing
Covers the full pipeline from raw UCI data to model-ready input:
- **Source:** UCI ElectricityLoadDiagrams20112014, 15-minute readings for 370 clients (MT_001–MT_370)
- **Cleaning:** 2011 data excluded due to high volumes of inactive meters and null values
- **Resampling:** Downsampled from 15-minute to hourly frequency, yielding 26,304 observations per client
- **Normalization:** StandardScaler applied per client to handle the large scale difference between residential and industrial consumers
- **Temporal split:** 2012–2013 used for training, 2014 held out as the out-of-sample test set

## Data Diagnostics & Clustering
Describes the exploratory analysis and clustering process that produced the four behavioral profiles:
- Visual and statistical inspection confirmed extreme diversity across clients in volatility, magnitude, and periodicity
- 32 features were engineered per client: hourly load profiles, STL decomposition components, autocorrelation at 24h and 168h lags, and behavioral scalars (coefficient of variation, peak-to-mean ratio, weekend/weekday ratio)
- PCA reduced the feature space for visualization; hierarchical clustering and silhouette analysis validated k=4 as the optimal number of clusters

**The four resulting clusters:**

| Cluster | Profile | # Clients |
|---------|---------|-----------|
| 0 | Residential — clear morning and evening peaks | 215 |
| 1 | Industrial — 8am start, sustained plateau load | 35 |
| 2 | Mixed-Use — low baseline, high intra-day variance | 91 |
| 3 | Critical Infrastructure — smooth baseload, near-zero overnight | 29 |

## Models
Six models were evaluated, spanning statistical baselines through state-of-the-art deep learning:

- **Seasonal Naive:** Repeats the value from exactly 168 hours prior (same hour, prior week). No fitting required. Serves as the lower-bound benchmark.
- **SARIMA(1,1,1)(1,0,1,24):** Classical seasonal time series model fitted once per cluster on the training mean series. Linear structure limits its ability to capture non-linear industrial load patterns.
- **Prophet:** Meta's additive decomposition model with piecewise trend, Fourier-series seasonality, and a holiday component. Best suited to smooth, regular profiles.
- **BasisFormer:** A self-supervised transformer that learns a library of consumption "bases" from the full client dataset, then represents each forecast as a weighted combination of those bases. Trained with AdaBelief optimizer using prediction, contrastive (InfoNCE), and smoothness losses.
- **Gradient Boosting (HistGradientBoostingRegressor):** Treats forecasting as tabular regression on engineered features — three lag values, rolling mean/std, and calendar features (hour, day of week, month, weekend flag). Log-transforms the target for variance stabilization.
- **PatchTST (Nie et al., ICLR 2023):** Transformer-based model using two key innovations — subseries-level patching of the look-back window and channel-independence, where each client is processed as a separate univariate series through a shared backbone. Trained on all individual clients within each cluster simultaneously.

## Results
All models were evaluated on a 365-window sliding evaluation over the full 2014 test year, forecasting 24 hours ahead at each step. MAPE was the primary metric.

| | Seasonal Naive | SARIMA | Prophet | BasisFormer | Gradient Boosting | PatchTST |
|---|---|---|---|---|---|---|
| **Cluster 0 (Residential)** | 6.41% | 29.15% | 50.26% | 171% | 4.59% | **4.49%** |
| **Cluster 1 (Industrial)** | 11.16% | 45.34% | 258.94% | 107% | **8.63%** | 12.18% |
| **Cluster 2 (Mixed-Use)** | 15.11% | 136.6% | 42.64% | 136% | 14.59% | **13.60%** |
| **Cluster 3 (Critical Infra.)** | 117.41% | 1227.37% | 121.18% | 432% | 110.73% | **110.17%** |

Key findings:
- No single model wins across all clusters, validating the cluster-specialized routing approach
- **PatchTST** is the strongest general-purpose model, winning on Clusters 0, 2, and 3
- **Gradient Boosting** outperforms all models on Cluster 1, where calendar-driven shift schedules favor explicit feature engineering over pattern learning
- Cluster 3 is an open challenge for all models; MAPE is misleadingly high due to near-zero denominators — sMAPE scores for the best models on that cluster are comparable to their performance elsewhere

## Future Work & Limitations
- The dataset lacks exogenous variables (temperature, pricing, industrial output) that would likely improve accuracy significantly in a real grid environment
- Both transformer models were trained on cluster-mean series rather than per-client series due to compute constraints; per-client training would likely yield further gains
- Cluster 3 warrants a metric rethink — sMAPE or MAE may be more appropriate than MAPE for near-zero load profiles, and event-driven models with exogenous triggers may be needed to forecast it reliably
- A production deployment would require automating cluster assignment for new clients and establishing a retraining schedule as the 2012–2013 frozen models degrade over time

## Running the Models

### Seasonal Naive, SARIMA, Prophet, Gradient Boosting, and PatchTST

All of these models can be reproduced end-to-end by running `project.ipynb`. The notebook handles data loading, preprocessing, clustering, training, evaluation, and visualization in sequence. Make sure `electricity.txt` is in the root folder and dependencies are installed before running.

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

### BasisFormer

A lot of files present in BasisFormer are too big for git. The pre-trained models are available here:
[Google Drive — BasisFormer models](https://drive.google.com/drive/folders/1H1bb-iVZ03b_npWnUqEihi3DHlhBhIZr?usp=drive_link)

However, we highly suggest setting up and running BasisFormer locally to explore how it performed. Running `project.ipynb` should install and configure BasisFormer for you.

You can read more about BasisFormer and clone the repo here:
[github.com/nzl5116190/Basisformer](https://github.com/nzl5116190/Basisformer)

> **Note:** BasisFormer requires significant GPU resources for training. For immediate review, all performance plots and metrics (sMAPE, MAE, and RMSE) are pre-rendered in `project.ipynb` and the technical report.


### File Structure:
```
Grid-Forecasting-Final/
├── Basisformer/                # [CLONE MANUALLY] Deep Learning model source code
│   ├── main.py                 # Primary execution script
│   ├── records/                # Outputs from training (if you train)
├── models/                     # Pre-trained models for n8n inference
│   ├── cluster_0/
│   │   ├── patchtst.pt         # Saved PatchTST weights + config
│   │   └── clients.json        # List of client IDs in this cluster
│   ├── cluster_1/
│   │   ├── gb_model.pkl        # Saved Gradient Boosting model
│   │   ├── train_series.csv    # Cluster mean training series
│   │   └── test_series.csv     # Cluster mean test series
│   ├── cluster_2/
│   │   ├── patchtst.pt
│   │   └── clients.json
│   └── cluster_3/              # TBD
├── apis/                       # FastAPI servers for n8n inference
│   ├── cluster_0_api.py        # PatchTST — port 5000
│   ├── cluster_1_api.py        # Gradient Boosting — port 5001
│   ├── cluster_2_api.py        # PatchTST — port 5003
│   ├── cluster_3_api.py        # TBD — port 5004
│   └── cluster_mapping.json    # Maps MT_001–MT_370 to cluster 0–3
├── technical-documentation/    # Technical documentation and final report
├── slides/                     # Final presentation (PowerPoint/PDF)
├── .gitignore                  # Instructions to ignore large .npy/weight files
├── electricity.txt             # [USER PROVIDED] Raw UCI Electricity Dataset LD2011_2014.txt renamed
├── project.ipynb               # Main analysis and visualization notebook
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
└── README.md                   # Project overview and setup instructions
```

## N8N Front-End Instructions

The `/models` and `/apis` folders are for running the n8n environment to interact with our AI agent.

### Overview

The n8n agent is an agentic AI forecasting assistant for electricity usage. Instead of a single one-size-fits-all model, 370 electricity clients are grouped into 4 clusters based on their usage patterns, with a separate forecasting model trained per cluster. The AI agent routes each request to the correct model automatically.

A business manager can open the n8n chat interface, type a client ID (e.g. `MT_035` or just `35`), and receive a natural language summary of that client's electricity usage forecast — without needing to know which model or cluster applies.

### How It Works

```
User (Chat UI)
     ↓
n8n AI Agent (Gemini)
     ↓
lookup_cluster tool          ← Code node: maps client ID → cluster (0–3)
     ↓
forecast_cluster_X tool      ← HTTP Request node: calls the correct FastAPI endpoint
     ↓
FastAPI (cluster_X_api.py)   ← Loads saved model, runs inference, returns forecast
     ↓
Agent formats and returns natural language response to user
```

### Cluster Summary

| Cluster | # Clients | Model | Port |
|---------|-----------|-------|------|
| 0 | 215 | PatchTST | 5001 |
| 1 | 35 | Gradient Boosting | 5002 |
| 2 | 91 | PatchTST | 5003 |
| 3 | 29 | PatchTST | 5004 |

### Setup & Running

**1. Install dependencies:**
```bash
pip install fastapi uvicorn torch pandas numpy scikit-learn joblib
```

**2. Place `electricity.txt` in the root folder** (see Data section above).

**3. Start the FastAPI servers** — open a separate terminal for each cluster:
```bash
cd apis/

# Terminal 1
python cluster_0_api.py   # port 5000

# Terminal 2
python cluster_1_api.py   # port 5001

# Terminal 3
python cluster_2_api.py   # port 5003

# Terminal 4
python cluster_3_api.py   # port 5004
```

Each server loads its model into memory at startup. You will see:
```
INFO: Uvicorn running on http://0.0.0.0:500X
```

**4. Start n8n via Docker:**
```bash
docker run -it --rm \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

Open `http://localhost:5678` in your browser and import the workflow.

**5. Use the chat interface** — click Chat in the n8n workflow and type a query:
```
Give me the forecast for client 35
What is the forecast for MT_035 on 2014-06-01?
```

### API Request & Response Format

**Request:**
```json
{
  "client_id": "MT_035"
}
```

**Response:**
```json
{
  "client_id": "MT_035",
  "cluster": 2,
  "model": "PatchTST",
  "forecast_date": "2014-01-01 00:00",
  "forecast_hours": ["2014-16-01 00:00", "2014-01-01 01:00", "..."],
  "forecast_values": [245.3, 238.1, 231.7, "..."]
}
```

> **Note:** n8n HTTP Request nodes use `http://host.docker.internal:PORT/forecast` to reach FastAPI servers running on the host machine from inside Docker.
