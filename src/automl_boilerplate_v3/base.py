"""Framework-agnostic interface for AutoML regressors on tabular data.

Adapters for a concrete framework subclass `AutoMLRegressor` and implement the four
private hooks (`_fit`, `_predict`, `_save`, `_load`). Everything a caller touches lives
in this module, so swapping frameworks never changes calling code.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, TypedDict, cast, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import NotFittedError

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)

type ParamValue = str | int | float | bool | None
type FloatArray = npt.NDArray[np.float64]

_METADATA_FILE = "metadata.json"
_CONFIG_FILE = "config.pkl"
_DEFAULT_TARGET_NAME = "target"


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationResult:
    """Regression metrics computed identically for every adapter.

    Attributes:
        rmse: Root mean squared error.
        mae: Mean absolute error.
        r2: Coefficient of determination.
        n_samples: Number of rows the metrics were computed on.
    """

    rmse: float
    mae: float
    r2: float
    n_samples: int

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten into a dict ready for `mlflow.log_metrics` or `wandb.log`.

        Args:
            prefix: Prepended to every key, e.g. ``"val_"``.

        Returns:
            Metric name to value.
        """
        return {f"{prefix}{field.name}": float(getattr(self, field.name)) for field in dataclasses.fields(self)}


class _Metadata(TypedDict):
    adapter: str
    library_version: str
    feature_names: list[str]
    target_name: str
    params: dict[str, ParamValue]


class AutoMLRegressor[ConfigT: DataclassInstance](ABC):
    """Base class every AutoML regression adapter extends.

    Public methods are `final`: they validate inputs and keep pandas metadata (index,
    column order, target name) consistent, so adapters only deal with the framework.

    Attributes:
        config: Frozen dataclass holding the adapter's settings.
    """

    #: PyPI distribution whose version is recorded on save, e.g. ``"flaml"``.
    distribution_name: ClassVar[str]

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._feature_names: list[str] | None = None
        self._target_name: str = _DEFAULT_TARGET_NAME

    # ------------------------------------------------------------------ public API

    @final
    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        """Search for and train the best model.

        Args:
            features: One row per sample, string column names.
            target: Numeric target aligned with ``features`` by index.

        Returns:
            The fitted regressor itself.

        Raises:
            ValueError: If the inputs are empty, misaligned, or the target has NaNs.
            TypeError: If column names are not strings or the target is not numeric.
        """
        _check_features(features)
        _check_target(features, target)

        self._feature_names = [str(column) for column in features.columns]
        self._target_name = _DEFAULT_TARGET_NAME if target.name is None else str(target.name)
        logger.info("Fitting %s on %d rows, %d features", type(self).__name__, *features.shape)
        self._fit(features, target)
        return self

    @final
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict the target for each row.

        Columns are matched by name, so their order does not matter. Extra columns are ignored.

        Args:
            features: Must contain every column seen during `fit`.

        Returns:
            Float predictions with the same index as ``features``, named after the training target.

        Raises:
            NotFittedError: If called before `fit` or `load`.
            ValueError: If a training column is missing.
        """
        feature_names = self._require_fitted()
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        extra = [column for column in features.columns if column not in feature_names]
        if extra:
            logger.warning("Ignoring columns not seen during fit: %s", extra)

        predictions = self._predict(features[feature_names])
        return pd.Series(predictions, index=features.index, name=self._target_name, dtype=np.float64)

    @final
    def validate(self, features: pd.DataFrame, target: pd.Series) -> ValidationResult:
        """Score predictions against known targets.

        Args:
            features: Rows to predict.
            target: True values aligned with ``features`` by index.

        Returns:
            RMSE, MAE and R² on the given rows.
        """
        _check_target(features, target)
        predicted = self.predict(features).to_numpy(dtype=np.float64)
        return _score(target.to_numpy(dtype=np.float64), predicted)

    @final
    def save(self, path: Path) -> None:
        """Write a self-contained model directory.

        The directory can be logged as-is with `mlflow.log_artifacts` or `wandb.Artifact.add_dir`.

        Args:
            path: Directory to write into; created if it does not exist.
        """
        feature_names = self._require_fitted()
        path.mkdir(parents=True, exist_ok=True)
        model_metadata = _Metadata(
            adapter=type(self).__qualname__,
            library_version=metadata.version(self.distribution_name),
            feature_names=feature_names,
            target_name=self._target_name,
            params=dict(self.params),
        )
        (path / _METADATA_FILE).write_text(json.dumps(model_metadata, indent=2))
        # Pickle, not JSON, so config fields like `Path` or tuples round-trip with their real types.
        (path / _CONFIG_FILE).write_bytes(pickle.dumps(self.config))
        self._save(path)
        logger.info("Saved %s to %s", type(self).__name__, path)

    @final
    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a regressor written by `save`.

        Only load directories you trust: the config and some frameworks use pickle.

        Args:
            path: Directory previously passed to `save`.

        Returns:
            A fitted regressor ready to `predict`.

        Raises:
            ValueError: If the directory was saved by a different adapter.
        """
        model_metadata = cast(_Metadata, json.loads((path / _METADATA_FILE).read_text()))
        if model_metadata["adapter"] != cls.__qualname__:
            raise ValueError(f"{path} was saved by {model_metadata['adapter']}, not {cls.__qualname__}")

        installed_version = metadata.version(cls.distribution_name)
        if model_metadata["library_version"] != installed_version:
            logger.warning(
                "%s was saved with %s %s but %s is installed",
                path,
                cls.distribution_name,
                model_metadata["library_version"],
                installed_version,
            )

        config = cast(ConfigT, pickle.loads((path / _CONFIG_FILE).read_bytes()))
        regressor = cls(config)
        regressor._feature_names = model_metadata["feature_names"]
        regressor._target_name = model_metadata["target_name"]
        regressor._load(path)
        return regressor

    @property
    def is_fitted(self) -> bool:
        """Whether `predict` can be called."""
        return self._feature_names is not None

    @property
    def params(self) -> Mapping[str, ParamValue]:
        """Config as flat primitives, ready for `mlflow.log_params` or `wandb.config`."""
        return {field.name: _to_param(getattr(self.config, field.name)) for field in dataclasses.fields(self.config)}

    # ------------------------------------------------------------ adapter hooks

    @abstractmethod
    def _fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the framework on already-validated inputs."""

    @abstractmethod
    def _predict(self, features: pd.DataFrame) -> FloatArray:
        """Return one prediction per row; columns are already in training order."""

    @abstractmethod
    def _save(self, path: Path) -> None:
        """Persist framework state into the existing directory ``path``."""

    @abstractmethod
    def _load(self, path: Path) -> None:
        """Restore framework state from ``path`` onto this freshly constructed instance."""

    # ------------------------------------------------------------------ helpers

    def _require_fitted(self) -> list[str]:
        if self._feature_names is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() or load() first")
        return self._feature_names


def _check_features(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("features is empty")
    # pandas-stubs claims column labels are always `str`; at runtime they can be anything hashable.
    non_string = [column for column in features.columns if not isinstance(cast(object, column), str)]
    if non_string:
        raise TypeError(f"Feature column names must be strings, got: {non_string}")
    if features.columns.has_duplicates:
        raise ValueError("features has duplicate column names")


def _check_target(features: pd.DataFrame, target: pd.Series) -> None:
    if not features.index.equals(target.index):
        raise ValueError("features and target must share the same index")
    if not pd.api.types.is_numeric_dtype(target):
        raise TypeError(f"target must be numeric, got dtype {target.dtype}")
    if target.isna().any():
        raise ValueError("target contains NaN values")


def _score(actual: FloatArray, predicted: FloatArray) -> ValidationResult:
    # Plain numpy rather than sklearn.metrics: sklearn is untyped and would break pyright strict here.
    errors = actual - predicted
    total_sum_of_squares = float(np.sum((actual - actual.mean()) ** 2))
    residual_sum_of_squares = float(np.sum(errors**2))
    # A constant target makes R² undefined; report 0.0 like sklearn's `force_finite=True`.
    r2 = 1.0 - residual_sum_of_squares / total_sum_of_squares if total_sum_of_squares > 0 else 0.0
    return ValidationResult(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        r2=r2,
        n_samples=len(actual),
    )


def _to_param(value: object) -> ParamValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    # Paths, tuples, enums, ...: loggers only accept primitives, and str() is readable enough.
    return str(value)
