# automl-boilerplate-v3

Boilerplate for tabular regression experiments that don't lock you into one framework.

- **AutoML:** one `AutoMLRegressor` interface, with adapters for [FLAML](https://microsoft.github.io/FLAML/) and
  [AutoGluon](https://auto.gluon.ai/).
- **Experiment tracking:** one `ExperimentLogger` interface for params, metrics, artifacts and the model registry,
  with an adapter for [MLflow](https://mlflow.org/).

Calling code only uses the interfaces, so switching frameworks means changing one constructor.

## Install

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra flaml            # or: --extra autogluon, --extra mlflow, --all-extras
```

| Extra       | Installs                                           |
| ----------- | -------------------------------------------------- |
| `flaml`     | `flaml[automl]`                                    |
| `autogluon` | `autogluon-tabular` with CatBoost, LightGBM, XGBoost |
| `mlflow`    | `mlflow-skinny` (tracking client only, no server)  |

Importing `automl_boilerplate_v3` never imports a framework. Adapters live in their own modules and are imported
explicitly.

## Usage

Train, evaluate, log to MLflow, register the model, then load it back from the registry:

```python
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from automl_boilerplate_v3.flaml_regressor import FlamlConfig, FlamlRegressor
from automl_boilerplate_v3.mlflow_logger import MlflowConfig, MlflowLogger

features, target = fetch_california_housing(as_frame=True, return_X_y=True)
x_train, x_val, y_train, y_val = train_test_split(features, target, random_state=0)

regressor = FlamlRegressor(FlamlConfig(time_budget_s=5)).fit(x_train, y_train)
logger = MlflowLogger(MlflowConfig(experiment_name="california-housing", tracking_uri="sqlite:///mlflow.db"))

with logger.start_run("flaml-baseline") as run:
    run.log_params(regressor.params)
    run.log_metrics(regressor.validate(x_val, y_val).to_dict(prefix="val_"))
    regressor.save(Path("model"))
    run.log_model(Path("model"))
    version = run.register_model("california-housing")

restored = FlamlRegressor.load(logger.download_model(version, Path("downloads")))
predictions = restored.predict(x_val)
```

To use AutoGluon instead, swap the regressor for `AutoGluonRegressor(AutoGluonConfig(...))` from
`automl_boilerplate_v3.autogluon_regressor`. Nothing else changes.

### Choosing an MLflow backend

- **Tracking server:** set `tracking_uri="http://host:5000"` (or `MLFLOW_TRACKING_URI`). The `mlflow` extra is
  enough.
- **Local SQLite file** (`sqlite:///mlflow.db`, as above): needs the full `mlflow` package, which the dev group
  installs. Browse runs with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## What the interfaces guarantee

**`AutoMLRegressor`**

- `fit` rejects empty data, non-string column names, misaligned indexes and NaN targets.
- `predict` matches columns by name, ignores extra columns, and keeps the input index and target name.
- `validate` returns RMSE, MAE and R², computed the same way for every adapter.
- `save` writes a self-contained directory. `load` refuses a directory saved by a different adapter and warns when
  the framework version changed.
- `params` flattens the config into primitives that any tracker accepts.

**`ExperimentLogger`**

- One active run per logger. Used as a context manager, the run is marked `FAILED` if the block raises.
- Logging without an active run raises `RuntimeError` instead of silently creating one.
- `register_model` creates the registered model on first use and returns a `ModelVersion`.
- `download_model` doesn't need an active run.

> **Security:** `AutoMLRegressor.load` uses pickle. Only load models you trust.

## Adding an adapter

Both base classes follow the same pattern. Public methods are `final` and handle validation and bookkeeping. An
adapter only implements the private hooks:

| Base class         | Hooks to implement                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| `AutoMLRegressor`  | `_fit`, `_predict`, `_save`, `_load`, plus a `distribution_name` class attribute                              |
| `ExperimentLogger` | `_start_run`, `_end_run`, `_log_params`, `_log_metrics`, `_log_artifact`, `_register_model`, `_download_model` |

Each adapter takes a frozen dataclass as its config. See `flaml_regressor.py` and `mlflow_logger.py` for short
examples.

## Project layout

```text
src/automl_boilerplate_v3/
├── base.py                  # AutoMLRegressor interface, ValidationResult
├── flaml_regressor.py       # FLAML adapter
├── autogluon_regressor.py   # AutoGluon adapter
├── experiment_logger.py     # ExperimentLogger interface, ModelVersion
└── mlflow_logger.py         # MLflow adapter
tests/
├── test_contract.py         # AutoMLRegressor contract
└── test_experiment_logger.py
```

## Development

```bash
uv sync --all-extras
uv run ruff check && uv run ruff format --check
uv run pyright                 # strict mode
uv run pytest
```

Base-class tests use tiny in-memory adapters and run in milliseconds. Adapter tests skip themselves when their
framework isn't installed. MLflow tests use a throwaway SQLite store, so no server is needed.
