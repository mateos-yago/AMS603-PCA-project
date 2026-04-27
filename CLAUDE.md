# CLAUDE.md — Project Instructions

## GitHub Workflow

This project uses GitHub for version control. Always follow these rules:

### Always push to GitHub
- After every meaningful change (new feature, bug fix, refactor, new analysis stage), commit and push to the `main` branch on GitHub.
- The remote is: `git@github.com:mateos-yago/AMS603-PCA-project.git`

### Commit message conventions
- Use the imperative mood: "Add PCA analysis" not "Added PCA analysis"
- First line: short summary (≤72 chars), no period at the end
- If more context is needed, leave a blank line then add bullet points
- Format:
  ```
  <type>: <short summary>

  - Optional detail bullet
  - Another detail
  ```
- Types: `feat` (new feature/analysis), `fix` (bug fix), `refactor`, `data` (data processing step), `docs`, `chore`

### Example commit messages
```
feat: implement PCA on covariance matrix

- Compute eigenvalues and eigenvectors manually
- Plot explained variance ratio
```
```
fix: correct normalization in preprocessing step
```

### Before every commit
1. Run `git status` to see what changed
2. Stage only relevant files (avoid large data files, IDE configs)
3. Write a clean commit message following the conventions above
4. Push immediately: `git push origin main`

## Project Context

- Course: AMS603
- Topic: Principal Component Analysis (PCA)
- Language: Python
- Environment: likely Jupyter notebooks or Python scripts

## What NOT to commit
- Large data files (`.csv`, `.npy`, `.pkl`, etc.) — add to `.gitignore`
- IDE configuration (`.idea/`, `.vscode/`)
- Checkpoint files (`.ipynb_checkpoints/`)
- Any secrets or API keys
