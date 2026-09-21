# scikit-learn ships no type information, so pyright cannot see through its dataset helpers.
# The untyped values are cast to pandas objects as soon as they arrive.
# pyright: reportUnknownMemberType=false
"""Train, log, register and re-load a model — the whole workflow in one file.

    uv run python examples/train_and_register.py
    uv run python examples/train_and_register.py --engine autogluon --time-budget 120
    uv run python examples/train_and_register.py --tracking-uri http://localhost:5000

Browse the result with ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from sklearn.datasets import load_diabetes

# `train_test_split` is unannotated all the way down; its result is cast in `_load_dataset`.
from sklearn.model_selection import train_test_split  # pyright: ignore[reportUnknownVariableType]

from automl_boilerplate_v3 import AutoMLRegressor, ExperimentLogger, ModelVersion
from automl_boilerplate_v3.mlflow_logger import MlflowConfig, MlflowLogger

logger = logging.getLogger(__name__)

#: Engines this example can drive; each adapter is imported only if it is picked.
type Engine = Literal["flaml", "autogluon"]

#: Code below only ever touches the base-class API, so the config type does not matter.
type AnyRegressor = AutoMLRegressor[Any]

# A database-backed store, not MLflow's default `mlruns/` file store: the model registry only
# exists behind a database. SQLite needs the full `mlflow` package, which the dev group installs.
_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_DEFAULT_TIME_BUDGET_S = 30.0
_EXPERIMENT = "diabetes-regression"
_MODEL_NAME = "diabetes-regressor"
_DATASET = "diabetes"
_HOLDOUT = 0.2
_SPLIT_SEED = 0


def _build_regressor(engine: Engine, time_budget_s: float) -> AnyRegressor:
    """Build the adapter for ``engine``.

    The framework import sits inside the branch, not at module level, so picking one engine
    never imports the other.

    Args:
        engine: Which AutoML framework to drive.
        time_budget_s: Wall-clock seconds the search may take.

    Returns:
        An unfitted regressor.
    """
    match engine:
        case "flaml":
            from automl_boilerplate_v3.flaml_regressor import FlamlConfig, FlamlRegressor

            return FlamlRegressor(FlamlConfig(time_budget_s=time_budget_s))
        case "autogluon":
            from automl_boilerplate_v3.autogluon_regressor import AutoGluonConfig, AutoGluonRegressor

            return AutoGluonRegressor(AutoGluonConfig(time_limit_s=time_budget_s))


def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the diabetes dataset and split it into a training and a validation half.

    The dataset ships inside scikit-learn, so this never touches the network.

    Returns:
        Training features, validation features, training target, validation target.
    """
    features, target = cast(tuple[pd.DataFrame, pd.Series], load_diabetes(as_frame=True, return_X_y=True))
    x_train, x_val, y_train, y_val = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(features, target, test_size=_HOLDOUT, random_state=_SPLIT_SEED),
    )
    return x_train, x_val, y_train, y_val


def _predictions_table(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Lay actual values, predictions and their error side by side, one row per sample."""
    table = pd.DataFrame({"actual": actual, "predicted": predicted, "error": predicted - actual})
    # `log_table` does not write the index, so carry the sample id as a real column.
    return table.rename_axis("sample").reset_index()


def train_and_register(*, engine: Engine, time_budget_s: float, tracking_uri: str) -> ModelVersion:
    """Search for a model, log everything worth keeping, and register the winner.

    Args:
        engine: Which AutoML framework to drive.
        time_budget_s: Wall-clock seconds the search may take.
        tracking_uri: MLflow tracking and registry store.

    Returns:
        The registry version just created.
    """
    x_train, x_val, y_train, y_val = _load_dataset()
    regressor = _build_regressor(engine, time_budget_s).fit(x_train, y_train)

    tracker = MlflowLogger(MlflowConfig(tracking_uri=tracking_uri, registry_uri=tracking_uri))
    with tracker.start_run(
        f"{engine}-baseline", experiment_name=_EXPERIMENT, tags={"engine": engine, "dataset": _DATASET}
    ) as run:
        # The engine's own config, plus what it was trained on: enough to explain the run later.
        run.log_params(
            {
                **regressor.params,
                "engine": engine,
                "dataset": _DATASET,
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_features": x_val.shape[1],
                "split_seed": _SPLIT_SEED,
            }
        )
        # Training scores next to validation scores: the gap between them is the overfitting.
        run.log_metrics(regressor.validate(x_train, y_train).to_dict(prefix="train_"))
        validation = regressor.validate(x_val, y_val)
        run.log_metrics(validation.to_dict(prefix="val_"))
        run.log_table(regressor.leaderboard(), "leaderboard.csv")
        run.log_table(_predictions_table(y_val, regressor.predict(x_val)), "predictions/validation.csv")

        with tempfile.TemporaryDirectory(prefix="example-model-") as staging:
            # Staged in a temporary directory: once it is uploaded the artifact store owns the
            # copy that matters. It is one zip file, so `unzip -l` works on whatever you download.
            model_file = Path(staging) / "model.zip"
            regressor.save(model_file)
            run.log_model(model_file)
        version = run.register_model(_MODEL_NAME)

    logger.info("Validation RMSE %.2f, MAE %.2f, R2 %.3f", validation.rmse, validation.mae, validation.r2)
    # Outside the run on purpose: fetching a registered model needs no active run.
    _check_registry_round_trip(tracker, version, regressor, x_val)
    return version


def _check_registry_round_trip(
    tracker: ExperimentLogger[Any], version: ModelVersion, regressor: AnyRegressor, features: pd.DataFrame
) -> None:
    """Download the registered version and prove it predicts exactly like the model in memory.

    This is what registering a model has to buy you: another process can fetch the version, load
    it and predict, without the code that trained it.

    Args:
        tracker: Logger holding the registry; no run has to be active.
        version: Version returned by `ExperimentLogger.register_model`.
        regressor: The fitted model this version was saved from.
        features: Rows to compare predictions on.
    """
    with tempfile.TemporaryDirectory(prefix="example-download-") as downloads:
        model_file = tracker.download_model(version, Path(downloads))
        # `load` is a classmethod: whichever adapter trained the model also loads it back.
        restored = type(regressor).load(model_file)
        difference = float((restored.predict(features) - regressor.predict(features)).abs().max())
    logger.info("Registry round-trip: largest prediction difference %g", difference)


def main() -> None:
    """Parse the command line and run the workflow once."""
    parser = argparse.ArgumentParser(description="Train a model, log it to MLflow and register it.")
    parser.add_argument(
        "--engine", choices=("flaml", "autogluon"), default="flaml", help="AutoML framework (default: %(default)s)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=_DEFAULT_TIME_BUDGET_S,
        metavar="SECONDS",
        help="Wall-clock seconds the search may take (default: %(default)s)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=_DEFAULT_TRACKING_URI,
        help="MLflow tracking and registry store (default: %(default)s)",
    )
    args = parser.parse_args()

    # INFO, so the library's own progress lines appear next to this script's.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    version = train_and_register(
        engine=cast(Engine, args.engine), time_budget_s=args.time_budget, tracking_uri=args.tracking_uri
    )
    logger.info("Registered %s version %s, ready for inference", version.name, version.version)


if __name__ == "__main__":
    main()
