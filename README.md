# Grid-Forecasting-Final
An adaptive forecasting ecosystem for the Portuguese power grid using BasisFormer, PatchTST, and Gradient Boosting models

# Instructions
## Data
The raw dataset is the UCI ElectricityLoadDiagrams20112014 (https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) and rename to electricity.txt. Due to GitHub size constraints, please download the LD2011_2014.txt file from the UCI Machine Learning Repository and place it in the root folder before running the notebook.

## BasisFormer
A lot of files present in BasisFormer are too big for git. This git: https://drive.google.com/drive/folders/1H1bb-iVZ03b_npWnUqEihi3DHlhBhIZr?usp=drive_link contains the best models. However, we highly suggest setting up and running BasisFormer locally to explore how it performed. Running the notebook should install and configure BasisFormer for you.

You can read more about BasisFormer and clone the repo here: https://github.com/nzl5116190/Basisformer

## Outputs
Note: The BasisFormer model requires significant GPU resources for training. For immediate review, all performance plots and metrics (sMAPE, MAE, and RMSE) are pre-rendered in the provided Project_new.ipynb file and technical report.

## N8N Front-End Instructions
the /models and /api folders are for running the n8n environment to interact with our AI agent.
