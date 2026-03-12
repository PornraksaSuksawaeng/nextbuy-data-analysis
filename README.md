# NextBuy — From Raw Data to Smart Decisions

**Team:** Léo BELLARD · Pornraksa SUKSAWAENG · Mathis MONNIN<br>
**School:** EPITECH — Bachelor Computer Science, Year 1 (B1).<br>
**Project duration:** 10 days.<br>

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%2B%20EC2-yellow)](https://aws.amazon.com/)

---

## Project Overview

NextBuy is an end-to-end retail analytics project built on the **Instacart dataset — 1.4 million orders, 14 million product rows** across 206 aisles and 21 departments.

The goal: turn raw transactional data into actionable business insights and deploy two production-ready machine learning models inside an interactive dashboard.

### Key Features

- **Data Cleaning Pipeline** — reproducible merge of 5 CSV files, NaN handling, dtype enforcement, and export to a single `cleaned_data.csv`
- **Exploratory Data Analysis** — 12 business questions answered with varied visualisations, written analysis, and correlation studies
- **ML Model 1 — Reorder Classifier** — XGBoost pipeline predicting whether a customer will reorder a product (AUC-ROC > 0.75)
- **ML Model 2 — Cart Size Regressor** — Random Forest pipeline predicting how many items a customer will add to their next order
- **Bonus Analysis** — KMeans customer segmentation (4 profiles) and Apriori association rules (product pairs by lift)
- **Interactive Dashboard** — Streamlit app with department filters, KPI cards, 3 chart tabs, 2 ML prediction panels, and a Groq AI global analysis feature
- **Cloud Deployment** — dataset on AWS S3, app on AWS EC2, credentials via IAM Role

---

## Requirements

- Python 3.12+
- pip / virtual environment
- ~2GB disk space (1.2GB dataset + figures + models)
- AWS credentials (optional — local mode works without them)
- Groq API key (optional — dashboard works without it)

---

## Quick Start

### 1. Clone the repository
```bash
git clone git@github.com:PornraksaSuksawaengEpitech/nextbuy.git
cd nextbuy
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your values
```

### 5. Add the dataset
Download the 5 CSV files and place them in `data/`:

```
data/
├── orders.csv
├── order_products.csv
├── products.csv
├── aisles.csv
└── departments.csv
```

### 6. Run notebooks in order
```bash
jupyter notebook
# Run: 01_cleaning → 02_eda → 03_models_leo → 03_models_mathis → 04_bonus
```

### 7. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`

---

## Project Structure

```
nextbuy/
├── data/                        # gitignored — 1.2GB, download manually
│   ├── orders.csv
│   ├── order_products.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
├── figures/                     # gitignored — charts exported by notebooks
├── models/                      # gitignored — model1.joblib, model2.joblib
├── notebooks/
│   ├── 01_cleaning.ipynb        # Pornraksa — load, merge, clean, export
│   ├── 02_eda.ipynb             # Pornraksa — 12 business questions + correlations
│   ├── 03_models_leo.ipynb      # Léo — reorder classifier (XGBoost pipeline)
│   ├── 03_models_mathis.ipynb   # Mathis — cart size regressor (RF pipeline)
│   ├── 04_bonus.ipynb           # Pornraksa — KMeans + Apriori
│   └── notebook.ipynb           # Final merged notebook for submission
├── dashboard/
│   ├── app.py                   # Main page — KPIs, charts, ML panels, AI analysis
│   └── pages/
│       └── 01_analysis.py       # Deep EDA page — Q7, Q8, Q9, Q11
├── requirements.txt
├── .env.example                 # Environment variable template
├── .gitignore
├── CONTRIBUTING.md              # Git workflow for the team
└── README.md
```

---

## Notebooks

| File | Owner | Content |
|---|---|---|
| `01_cleaning.ipynb` | Pornraksa | Load 5 CSV files, merge in memory-efficient order, clean, export |
| `02_eda.ipynb` | Pornraksa | 12 business analyses, correlation heatmaps for model feature selection |
| `03_models_leo.ipynb` | Léo | Feature engineering, XGBoost reorder classifier, AUC-ROC evaluation |
| `03_models_mathis.ipynb` | Mathis | Cart size regressor, leakage-free feature engineering, GridSearchCV |
| `04_bonus.ipynb` | Pornraksa | KMeans customer segmentation, Apriori association rules |
| `notebook.ipynb` | All | Final merged notebook for defense submission |

---

## Dashboard
 
The Streamlit dashboard (`dashboard/app.py`) provides an interactive interface for exploring the dataset and running live ML predictions.

### Features

```
Sidebar
├── Department + aisle filters (cascading)
└── Global AI Analysis button (Groq — cross-dataset insights)
 
Main page
├── 5 KPI cards (orders, products, reorder rate, avg cart size, organic rate)
├── Tab 1 — Top 20 bestsellers (Q1)
├── Tab 2 — Order heatmap by day × hour (Q2)
├── Tab 3 — Reorder rate by department (Q6)
├── ML Panel 1 — Reorder probability prediction
└── ML Panel 2 — Cart size prediction
 
Analysis page (pages/01_analysis.py)
├── Q7 — Reorder rate vs days since prior order
├── Q8 — Organic share by department
├── Q9 — Most common first cart item
└── Q11 — Reorder rate by hour of day
```

---

## Machine Learning Models

Both models are exported as **sklearn Pipelines** (StandardScaler + model) — no manual scaling needed on the dashboard side.

| Model | File | Type | Target | Key metric |
|---|---|---|---|---|
| Reorder Classifier | `model1.joblib` | Classification | `reordered` (0/1) | AUC-ROC |
| Cart Size Regressor | `model2.joblib` | Regression | items per order | RMSE, R² |

---

## Cloud Deployment (AWS)

The production version runs on AWS with the following architecture:

```
S3 bucket
└── cleaned_data.csv   (1.2GB)
└── model1.joblib
└── model2.joblib
 
EC2 instance
└── streamlit run dashboard/app.py
    └── USE_S3=true → reads all files from S3 via s3fs
    └── IAM Role → no hardcoded credentials
```

Set `USE_S3=true` in your environment to switch from local files to S3. All notebooks and the dashboard support both modes transparently.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```
USE_S3=false                  # true on EC2, false locally
S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=...         # not needed on EC2 (IAM Role)
AWS_SECRET_ACCESS_KEY=...     # not needed on EC2 (IAM Role)
AWS_DEFAULT_REGION=eu-west-3
GROQ_API_KEY=...              # optional — AI analysis feature
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) |
| Raw size | ~14 million product rows |
| Orders | ~1.4 million |
| Products | 49,688 unique |
| Departments | 21 |
| Aisles | 206 |
| Files | 5 CSV files |

---

## Technical Stack

| Area | Tools |
|---|---|
| Data manipulation | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn, XGBoost |
| Unsupervised ML | KMeans, Apriori (mlxtend) |
| Dashboard | Streamlit |
| AI integration | Groq API (openai/gpt-oss-120b) |
| Cloud | AWS S3, EC2, IAM |
| Model serialisation | Joblib |
| Environment | Jupyter, VS Code |

---

## Team

This project was completed in **10 days** by three first-year Bachelor Computer Science students (B1) at Epitech Montpellier.

| Name | Contributions |
|---|---|
| **Pornraksa SUKSAWAENG** | Data cleaning, EDA, bonus analysis, dashboard, AWS deployment |
| **Léo BELLARD** | Reorder classifier, feature engineering, bonus analysis |
| **Mathis MONNIN** | Cart size regressor, GridSearchCV, model evaluation |

---

*Built on the dataset provide by Epitech — Epitech Montpellier 2026*