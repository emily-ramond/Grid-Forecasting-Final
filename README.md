# Grid-Forecasting-Final
An adaptive forecasting ecosystem for the Portuguese power grid using BasisFormer, PatchTST, and Gradient Boosting models

# Instructions

## File Structure:
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

## Data
The raw dataset is the UCI ElectricityLoadDiagrams20112014 (https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) and rename to electricity.txt. Due to GitHub size constraints, please download the LD2011_2014.txt file from the UCI Machine Learning Repository and place it in the root folder before running the notebook.

## BasisFormer
A lot of files present in BasisFormer are too big for git. This git: https://drive.google.com/drive/folders/1H1bb-iVZ03b_npWnUqEihi3DHlhBhIZr?usp=drive_link contains the best models. However, we highly suggest setting up and running BasisFormer locally to explore how it performed. Running the notebook should install and configure BasisFormer for you.

You can read more about BasisFormer and clone the repo here: https://github.com/nzl5116190/Basisformer

## Outputs
Note: The BasisFormer model requires significant GPU resources for training. For immediate review, all performance plots and metrics (sMAPE, MAE, and RMSE) are pre-rendered in the provided Project_new.ipynb file and technical report.

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
