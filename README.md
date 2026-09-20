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
    run.log_table(regressor.leaderboard(), "leaderboard.csv")  # how every candidate scored
    regressor.save(Path("model.zip"))  # one file, best model only
    run.log_model(Path("model.zip"))
    version = run.register_model("california-housing")

restored = FlamlRegressor.load(logger.download_model(version, Path("downloads")))
predictions = restored.predict(x_val)
```

To use AutoGluon instead, swap the regressor for `AutoGluonRegressor(AutoGluonConfig(...))` from
`automl_boilerplate_v3.autogluon_regressor`. Nothing else changes.

### Choosing which models get trained

Each engine's config names the models it will search over. This one setting is deliberately not
shared between engines: FLAML tunes individual **learners**, AutoGluon trains model **families**
and stacks an ensemble on top, so each config uses its own engine's vocabulary.

```python
FlamlConfig(estimator_list=("lgbm", "catboost", "enet"))  # None lets FLAML choose
AutoGluonConfig(included_model_types=("GBM", "CAT", "XGB"))  # None leaves it to `presets`
```

Every valid name is listed, with a note on what it is, in the `FlamlEstimator` and
`AutoGluonModelType` types next to each config — so your editor offers them and a typo is a
type error rather than a wasted training run.

One AutoGluon quirk worth knowing: `included_model_types` **filters** the preset's model list
rather than replacing it, and the weighted ensemble is stacked on afterwards regardless. A run
restricted to `("GBM",)` still ends with a `WeightedEnsemble_L2` row in the leaderboard.

### The saved model

`save` writes **one file**, so the path you pass is the artifact — nothing is scattered into a
directory you did not expect. It is a zip archive, which keeps it inspectable:

```text
model.zip
├── metadata.json      # adapter, library version, feature names, target name, params
├── config.pkl         # the frozen config dataclass
├── leaderboard.csv    # one row per candidate model — performance only, no weights
└── payload/           # the framework's own files, pruned to the best model
```

```bash
unzip -l model.zip                      # what is in there
unzip -p model.zip leaderboard.csv      # how the candidates scored, without Python
```

Only the winning model's weights are kept. For AutoGluon that is the difference between copying
every trained model, every bagged fold and the training data, and copying one:

| | files | size |
| --- | --- | --- |
| full predictor directory | 25 | 24.1 MiB |
| `model.zip` | 1 | 2.5 MiB |

The losing candidates are not lost, just reduced to their numbers. `leaderboard()` returns them as
a dataframe — `model`, `loss` (lower is better), `metric` and `is_best`, plus adapter extras such as
AutoGluon's `fit_time_s` — sorted best first. It is captured during `fit`, because pruning a search
down to its winner destroys the framework's record of the rest, and it travels inside the archive,
so it survives a `load` too.

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
- `save` writes one self-contained file holding only the best model. `load` refuses a file saved by a different
  adapter and warns when the framework version changed.
- `leaderboard` ranks every candidate the search tried, and round-trips through `save` without losing precision.
- `params` flattens the config into primitives that any tracker accepts.

**`ExperimentLogger`**

- One active run per logger. Used as a context manager, the run is marked `FAILED` if the block raises.
- Logging without an active run raises `RuntimeError` instead of silently creating one.
- `log_table` uploads a dataframe as a CSV artifact, without the index.
- `register_model` creates the registered model on first use and returns a `ModelVersion`.
- `download_model` doesn't need an active run.

> **Security:** `AutoMLRegressor.load` unpacks an archive and uses pickle. Only load model files you trust.

## Adding an adapter

Both base classes follow the same pattern. Public methods are `final` and handle validation and bookkeeping. An
adapter only implements the private hooks:

| Base class         | Hooks to implement                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| `AutoMLRegressor`  | `_fit`, `_predict`, `_save`, `_load`, plus a `distribution_name` class attribute                              |
| `ExperimentLogger` | `_start_run`, `_end_run`, `_log_params`, `_log_metrics`, `_log_artifact`, `_register_model`, `_download_model` |

Each adapter takes a frozen dataclass as its config. See `flaml_regressor.py` and `mlflow_logger.py` for short
examples.

`AutoMLRegressor._leaderboard` is the one **optional** hook: return a frame with `model`, `loss`, `metric` and
`is_best` columns to have the base class normalise, sort and persist it. Adapters that train a single model can
leave it alone. `_save` and `_load` still receive a directory — a temporary one the base class packs into the
archive — so an adapter never deals with the file format itself.

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
