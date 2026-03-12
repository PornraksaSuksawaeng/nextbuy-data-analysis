# Contributing — Git Workflow

Internal reference for the NextBuy team. This document covers branch organisation, daily commit habits, and the pull request process.

---

## Branch Organisation

Each member works on their own branch to avoid conflicts. **Never push directly to `main` or `dev`.**

| Branch | Owner |
|---|---|
| `feat/data-cleaning` | Pornraksa |
| `feat/eda` | Pornraksa |
| `feat/models-leo` | Léo |
| `feat/models-mathis` | Mathis |
| `feat/bonus` | Léo + Mathis |
| `feat/dashboard` | Pornraksa |

```
main          ← production, updated once on code freeze day
└── dev       ← integration branch, always stable
    ├── feat/data-cleaning
    ├── feat/eda
    ├── feat/models-leo
    ├── feat/models-mathis
    ├── feat/bonus
    └── feat/dashboard
```

---

## Start of Day — Sync Your Branch

Every morning, before writing any code, pull the latest changes from `dev` into your branch. This ensures you are always working on the most up-to-date version and prevents conflicts from building up.

```bash
git checkout feat/your-branch
git pull origin dev
```

If there are conflicts, resolve them in your files, then:

```bash
git add .
git commit -m "fix: resolve merge conflict with dev"
```

---

## During the Day — Commit Often

After each meaningful section, chart, or feature, commit your work with a clear and descriptive message. Small, frequent commits are easier to review and easier to revert if something breaks.

```bash
git add notebooks/02_eda.ipynb # example 
git commit -m "feat(eda): add Q7 scatter reorder rate vs days" # example
git push origin feat/your-branch
```

### Commit message conventions

| Prefix | When to use |
|---|---|
| `feat(eda):` | New analysis or visualisation |
| `feat(models):` | New model, metric, or pipeline step |
| `feat(dashboard):` | New dashboard feature or chart |
| `fix(data):` | Bug fix or data correction |
| `fix(models):` | Model bug or leakage fix |
| `docs:` | README, comments, or markdown update |
| `chore:` | Dependencies, config, gitignore |

---

## Pull Request — Merging Your Work into `dev`

When a section is complete and working, open a Pull Request (PR) to merge your branch into `dev`. Never merge directly — always go through a PR so a teammate can review your code before it reaches the shared branch.

### On GitHub

1. Go to the repository → click **Pull requests** → **New pull request**
2. Set **base: `dev`** and **compare: `feat/your-branch`**
3. Write a short description of what you did and what was tested
4. Assign a teammate as reviewer
5. Once approved, click **Merge pull request**

### Via GitHub CLI

```bash
gh pr create \
  --base dev \
  --head feat/your-branch \
  --title "feat(eda): EDA notebook complete" \
  --body "Added all 12 charts with written analysis and correlation heatmaps."
```

---

## After a PR is Merged — Everyone Pulls

Once any PR is merged into `dev`, all team members must sync at the start of the next session:

```bash
git checkout feat/your-branch
git pull origin dev
```

This keeps everyone up to date and prevents conflicts from accumulating over days.

---

## Code Freeze — Merging `dev` into `main`

`main` is updated only once, on the code freeze day, after everything is validated and the final `notebook.ipynb` is merged.

```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```

Before merging, make sure:
- All notebooks run top-to-bottom without errors
- `notebook.ipynb` has been generated with `nbmerge`
- `model1.joblib` and `model2.joblib` are exported and tested
- The dashboard loads and all features work locally