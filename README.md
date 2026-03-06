# NextBuy — From Raw Data to Smart Decisions

**Team:** Leo BELLARD, Pornraksa SUKSAWAENG, Mathis MONNIN, Nicolas LUSSIGNOL  
**School:** EPITECH — Bachelor Computer Science, Year 1 (B1).  
**Project duration:** 5 days.  

[![Python](https://img.shields.io/badge/Python-3.14+-blue)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0+-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)](https://scikit-learn.org/)

---

## Project Overview

NextBuy is an e-commerce data analysis project based on the Instacart dataset (~14 million rows).  
The goal: turn raw data into actionable business insights and build predictive ML models.

---

## Quick Start

### 1. Clone the repository
```bash
git clone git@github.com:PornraksaSuksawaengEpitech/nextbuy.git nextbuy
cd nextbuy
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
nextbuy/
├── data/                  # Manual download of csv file
│   ├── orders.csv
│   ├── order_products.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
├── figures/               # Charts exported from notebooks
├── notebooks/
│   ├── 01_cleaning.ipynb      # Pornraksa — Loading, merging, cleaning
│   ├── 02_eda.ipynb           # Pornraksa + Léo — EDA visualizations
│   ├── 03_models_leo.ipynb    # Léo — Feature engineering + ML models
│   ├── 03_models_mathis.ipynb # Mathis — ML review, tuning, GridSearch
│   ├── 04_bonus.ipynb         # Nicolas — Segmentation, rules, SHAP, network
│   └── notebook.ipynb         # Final merged notebook
├── dashboard/
│   └── app.py             # Dash application
├── requirements.txt
└── README.md
```

---

## Notebooks

| File | Owner | Content |
|---|---|---|
| `01_cleaning.ipynb` | Pornraksa | Load 5 CSV files, merge, clean, handle NaN |
| `02_eda.ipynb` | Pornraksa + Léo | 12 business analyses, varied visualizations |
| `03_models_leo.ipynb` | Léo | Feature engineering, 2 ML models, metrics |
| `03_models_mathis.ipynb` | Mathis | ML review, GridSearch tuning, evaluation |
| `04_bonus.ipynb` | Pornraksa | KMeans clustering, association rules, SHAP, co-purchase network |
| `notebook.ipynb` | Pornraksa | Final merged notebook for submission |

---

## Team & Git Workflow

Each member works on **their own branch** to avoid conflicts:

| Branch | Owner |
|---|---|
| `feat/data-cleaning` | Pornraksa |
| `feat/eda` | Pornraksa + Léo |
| `feat/models-leo` | Léo |
| `feat/models-mathis` | Mathis |
| `feat/bonus` | Pornraksa |
| `feat/dashboard` | Pornraksa + Mathis + Nicolas |

**Golden rule: never push directly to `main`. Always merge into `dev` first via a Pull Request.**

---

### Daily Routine — Start of Day

Every morning, before writing any code, sync your branch with the latest changes from `dev`.
This ensures you always have the most up-to-date version of the project.

```bash
git checkout feat/your-branch
git pull origin dev
```

If there are conflicts, resolve them in your files, then:

```bash
git add .
git commit -m "fix: resolve merge conflict"
```

---

### During the Day — Commit Often

After each meaningful section or chart, commit your work with a clear message:

```bash
git add notebook.ipynb
git commit -m "feat(eda): add heatmap orders by hour"
git push origin feat/your-branch
```

Commit conventions:
- `feat(eda):` — new analysis or visualization
- `feat(models):` — new model or metric
- `fix(data):` — bug fix or data correction
- `docs:` — README or comments update

---

### Pull Request — Merging Your Work into `dev`

When a section is complete and working, open a Pull Request (PR) to merge your branch into `dev`.
Never merge directly — always go through a PR so a teammate can review your code.

**On GitHub:**
1. Go to your repository -> click **"Pull requests"** -> **"New pull request"**
2. Set **base: `dev`** and **compare: `feat/your-branch`**
3. Write a short description of what you did
4. Assign a teammate as reviewer
5. Once approved, click **"Merge pull request"**

**Or via GitHub CLI:**
```bash
gh pr create --base dev --head feat/your-branch --title "feat(eda): EDA notebook complete" --body "Added all 7 charts with written analysis."
```

---

### After a PR is Merged — Everyone Pulls

Once any PR is merged into `dev`, all team members must sync the next morning:

```bash
git checkout feat/your-branch
git pull origin dev
```

This keeps everyone up to date and prevents conflicts from building up over days.

---

### End of Project — Merging `dev` into `main`

`main` is only updated once, on the code freeze day, after everything is validated:

```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```
