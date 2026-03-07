# tests/

This folder contains the **pytest** unit test suite for the **Call Smarter** project. All tests use synthetic fixtures and run independently — no raw data files are required.

---

## Files

| File               | Description                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `conftest.py`      | Shared pytest fixtures — 200-row synthetic dataset matching the UCI bank schema                                          |
| `test_clean.py`    | Tests for `src/clean.py` — unknown imputation, outlier clipping, duplicate removal, duration drop                        |
| `test_features.py` | Tests for `src/features.py` — `PdaysTransformer`, full pipeline fitting and output shape                                 |
| `test_train.py`    | Tests for `src/train.py` — model registry, `FIXED_PARAMS`, `train_final_model`, `tune_model`                             |
| `test_evaluate.py` | Tests for `src/evaluate.py` — `compute_metrics`, `find_optimal_threshold`, `business_cost_analysis`, `select_best_model` |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_clean.py -v
pytest tests/test_features.py -v
pytest tests/test_train.py -v
pytest tests/test_evaluate.py -v
```

---

## Fixtures (`conftest.py`)

The `synthetic_data` fixture generates a 200-row `DataFrame` that mirrors the UCI bank-additional-full schema (all 20 input features + target `y`). This means tests never depend on the real CSV file and can run in CI/CD without the data download step.

---

## Test Coverage Summary

| Module        | What is tested                                                                        |
| ------------- | ------------------------------------------------------------------------------------- |
| `clean.py`    | Unknown imputation per column, outlier clipping, duplicate removal, duration drop     |
| `features.py` | `PdaysTransformer` sentinel handling, log transform, full pipeline output shape       |
| `train.py`    | Model registry contents, hyperparameter config, training and tuning functions         |
| `evaluate.py` | Metric ranges, optimal threshold selection, business cost accounting, model selection |

---

## Notes

- The test suite contains **46 tests** in total.
- All tests use `seed=42` for reproducibility.
- `conftest.py` fixtures are auto-injected by pytest — no imports needed in test files.
