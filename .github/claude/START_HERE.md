# Claude Code — Start Here

You are Claude Code. Your job is to autonomously implement the entire PCA_project.

## Step 1: Read These Files (In Order)

1. `.github/claude/CLAUDE.md` — Master project documentation
2. `.github/claude/IMPLEMENTATION_PLAN.md` — Your autonomous implementation checklist

## Step 2: Follow the Plan

Execute every phase in `.github/claude/IMPLEMENTATION_PLAN.md` from top to bottom:

- **PHASE 1**: Setup (config.yaml, requirements.txt, load_config)
- **PHASE 2**: Data pipeline (universe, downloader, preprocessor)
- **PHASE 3**: Base factor model (ABC)
- **PHASE 4**: PCA model
- **PHASE 5**: Autoencoder model
- **PHASE 6**: OU process & signals
- **PHASE 7**: Backtesting
- **PHASE 8**: Metrics
- **PHASE 9**: Grid search & main.py
- **PHASE 10**: Visualization
- **PHASE 11**: Single notebook

## Step 3: After Each Phase

```bash
git add -A
git commit -m "<type>: <message>"
git push origin main
```

## Step 4: When Done

All code is complete, all results are committed, and the project is ready for:
```bash
python main.py
```

Then the user opens `notebooks/analysis.ipynb` to see all visualizations.

---

## Quick Links

| What | Where |
|------|-------|
| Master instructions | `.github/claude/CLAUDE.md` |
| Implementation checklist | `.github/claude/IMPLEMENTATION_PLAN.md` |
| Detailed instruction files | `.github/claude/instructions/` (01–09) |
| Config template | `config.yaml` |
| Main entry point | `main.py` (create at project root) |
| Notebook | `notebooks/analysis.ipynb` |
| Package | `pca_project/` |

---

## Key Constraints (Never Forget)

✅ **Config-driven**: Every parameter in `config.yaml`, no magic numbers
✅ **Type hints**: All function signatures have type hints
✅ **Docstrings**: All classes and public methods have docstrings
✅ **OOP**: Everything is a class, organized by concern
✅ **No data leakage**: Test set never seen during training
✅ **Notebook visualization only**: Never runs experiments
✅ **Commit frequently**: After each phase

---

## Now Begin

Read `.github/claude/CLAUDE.md`, then `.github/claude/IMPLEMENTATION_PLAN.md`, then implement PHASE 1.
