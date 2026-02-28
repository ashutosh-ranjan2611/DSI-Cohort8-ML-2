# Project Setup Guide (uv & pyproject.toml)

This guide explains how to set up the DSI-Cohort8-ML-2 project using the `uv` package manager and a `pyproject.toml` file for dependency management.

## 1. Clone the Repository

```bash
git clone https://github.com/ashutosh-ranjan2611/DSI-Cohort8-ML-2.git
cd DSI-Cohort8-ML-2
```

## 2. Install uv

If you don't have `uv` installed, run:

```bash
pip install uv
```

Or see the official docs: https://github.com/astral-sh/uv

## 3. Create and Activate Virtual Environment

```bash
uv venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

## 4. Install Dependencies from `pyproject.toml` (preferred)

This project uses `pyproject.toml` as the single source-of-truth for dependencies. Use `uv` to install and sync the environment.

- Install the project in editable mode (development):

```bash
uv pip install -e .
```

- Sync the virtual environment to match `pyproject.toml`:

```bash
uv pip sync
```

Note: Keep `pyproject.toml` as the authoritative dependency list for development and CI. Do not maintain a separate `requirements.txt` in the repository.

## 5. Using `uv` sync workflow (recommended)

When you want reproducible environments and CI-friendly installs, use `uv`'s sync workflow with `pyproject.toml` as the source-of-truth.

- Create and activate the virtual environment:

```bash
uv venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate # macOS / Linux
```

- Install the project in editable mode (development):

```bash
uv pip install -e .
```

- Sync the environment to match `pyproject.toml`:

```bash
uv pip sync
```

Notes:

- If you need a pinned deployment artifact, generate it from a synced environment in CI and store it as a build artifact. Prefer committing a lockfile if your tooling supports it.

## 6. Project Structure

```
data/
├── raw/
├── interim/
└── processed/
notebooks/
src/
├── data/
├── features/
├── models/
├── training/
├── evaluation/
└── deployment/
tests/
reports/
models/
.gitignore
README.md
pyproject.toml
setup.md
```

## 7. Version Control

- Ensure `.gitignore` is present to avoid committing unnecessary files.

## 8. Running Tests

```bash
uv pip install pytest  # If not already installed
pytest
```

## 9. Useful Commands

- To deactivate the virtual environment:
  ```bash
  deactivate
  ```
- To update dependencies:
  ```bash
  uv pip sync
  ```
- To add a new package:
  ```bash
  uv pip install <package-name>
  ```
- To sync dependencies from pyproject.toml:
  ```bash
  uv pip sync
  ```

---

For any issues, refer to the README.md or contact the repository owner.
