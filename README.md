# Grid-Forecasting-Final
An adaptive forecasting ecosystem for the Portuguese power grid using BasisFormer, PatchTST, and Gradient Boosting models

# Instructions
## File Structure:
Grid-Forecasting-Final/
├── Basisformer/                # [CLONE MANUALLY] Deep Learning model source code
│   ├── main.py                 # Primary execution script
│   ├── records/                # Outputs from training (if you train)
├── models/                     # Pre-processed models for n8n
│   ├── cluster_0.csv
│   ├── cluster_1.csv
│   ├── cluster_2.csv
│   └── cluster_3.csv
├── apis/                       # Pre-processed apis for n8n
│   ├── cluster_0_api.py
│   ├── cluster_1_api.py
│   ├── cluster_2_api.py
│   └── cluster_3_api.py
├── technical-documentation/    # Technical documentation and final report
├── slides/                     # Final presentation (PowerPoint/PDF)
├── .gitignore                  # Instructions to ignore large .npy/weight files
├── electricity.txt             # [USER PROVIDED] Raw UCI Electricity Dataset LD2011_2014.txt renamed
├── project.ipynb               # Main analysis and visualization notebook
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
└── README.md                   # Project overview and setup instructions
## Data
The raw dataset is the UCI ElectricityLoadDiagrams20112014 (https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) and rename to electricity.txt. Due to GitHub size constraints, please download the LD2011_2014.txt file from the UCI Machine Learning Repository and place it in the root folder before running the notebook.

## BasisFormer
A lot of files present in BasisFormer are too big for git. This git: https://drive.google.com/drive/folders/1H1bb-iVZ03b_npWnUqEihi3DHlhBhIZr?usp=drive_link contains the best models. However, we highly suggest setting up and running BasisFormer locally to explore how it performed. Running the notebook should install and configure BasisFormer for you.

You can read more about BasisFormer and clone the repo here: https://github.com/nzl5116190/Basisformer

## Outputs
Note: The BasisFormer model requires significant GPU resources for training. For immediate review, all performance plots and metrics (sMAPE, MAE, and RMSE) are pre-rendered in the provided Project_new.ipynb file and technical report.

## N8N Front-End Instructions
the /models and /api folders are for running the n8n environment to interact with our AI agent.
