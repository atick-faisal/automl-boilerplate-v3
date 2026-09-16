"""Contract every `AutoMLRegressor` must honour.

Base-class behaviour is tested with a tiny stand-in adapter so it runs in milliseconds.
Real adapters are fitted once per module and run through the same public API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from automl_boilerplate_v3 import AutoMLRegressor, ValidationResult
from automl_boilerplate_v3.base import FloatArray


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n_rows = 300
    # A non-default index catches adapters that silently reset it.
    index = pd.Index([f"row-{i}" for i in range(n_rows)])
    features = pd.DataFrame(
        {
            "x1": rng.normal(size=n_rows),
            "x2": rng.normal(size=n_rows),
            "noise": rng.normal(size=n_rows),
        },
        index=index,
    )
    target = pd.Series(
        3 * features["x1"] - 2 * features["x2"] + rng.normal(scale=0.1, size=n_rows),
        index=index,
        name="price",
    )
    return features, target


# ----------------------------------------------------------------- base class


@dataclass(frozen=True, slots=True)
class _MeanConfig:
    offset: float = 0.0
    note: Path = Path("unused")


class _MeanRegressor(AutoMLRegressor[_MeanConfig]):
    """Predicts the training mean; enough to exercise everything the base class owns."""

    distribution_name = "numpy"

    def __init__(self, config: _MeanConfig | None = None) -> None:
        super().__init__(config or _MeanConfig())
        self.mean = 0.0

    @override
    def _fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.mean = float(target.mean()) + self.config.offset

    @override
    def _predict(self, features: pd.DataFrame) -> FloatArray:
        return np.full(len(features), self.mean, dtype=np.float64)

    @override
    def _save(self, path: Path) -> None:
        (path / "mean.txt").write_text(str(self.mean))

    @override
    def _load(self, path: Path) -> None:
        self.mean = float((path / "mean.txt").read_text())


def test_predict_before_fit_raises(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, _ = dataset
    with pytest.raises(NotFittedError):
        _MeanRegressor().predict(features)


def test_fit_rejects_misaligned_index(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = dataset
    with pytest.raises(ValueError, match="same index"):
        _MeanRegressor().fit(features, target.reset_index(drop=True))


def test_fit_rejects_nan_target(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = dataset
    with pytest.raises(ValueError, match="NaN"):
        _MeanRegressor().fit(features, target.where(target > 0))


def test_predict_matches_columns_by_name(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = dataset
    regressor = _MeanRegressor().fit(features, target)

    shuffled = features[["noise", "x2", "x1"]].assign(unseen=1.0)
    pd.testing.assert_series_equal(regressor.predict(shuffled), regressor.predict(features))

    with pytest.raises(ValueError, match="x1"):
        regressor.predict(features.drop(columns="x1"))


def test_validation_result_is_logger_friendly() -> None:
    result = ValidationResult(rmse=1.0, mae=0.5, r2=0.9, n_samples=10)
    assert result.to_dict(prefix="val_") == {"val_rmse": 1.0, "val_mae": 0.5, "val_r2": 0.9, "val_n_samples": 10.0}


def test_params_flatten_config_to_primitives() -> None:
    assert _MeanRegressor(_MeanConfig(offset=1.5)).params == {"offset": 1.5, "note": "unused"}


def test_load_rejects_directory_from_another_adapter(dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path) -> None:
    features, target = dataset
    _MeanRegressor().fit(features, target).save(tmp_path)

    class _OtherRegressor(_MeanRegressor):
        pass

    with pytest.raises(ValueError, match="saved by"):
        _OtherRegressor.load(tmp_path)


# -------------------------------------------------------------- real adapters

# Code that accepts any adapter doesn't care which config type it was built with.
type AnyRegressor = AutoMLRegressor[Any]


def _flaml() -> AnyRegressor:
    pytest.importorskip("flaml")
    from automl_boilerplate_v3.flaml_regressor import FlamlConfig, FlamlRegressor

    return FlamlRegressor(FlamlConfig(time_budget_s=5, estimator_list=("lgbm", "rf")))


def _autogluon() -> AnyRegressor:
    pytest.importorskip("autogluon.tabular")
    from automl_boilerplate_v3.autogluon_regressor import AutoGluonConfig, AutoGluonRegressor

    return AutoGluonRegressor(AutoGluonConfig(time_limit_s=20, presets="medium_quality"))


@pytest.fixture(scope="module", params=[_flaml, _autogluon], ids=["flaml", "autogluon"])
def fitted(request: pytest.FixtureRequest, dataset: tuple[pd.DataFrame, pd.Series]) -> AnyRegressor:
    factory: Callable[[], AnyRegressor] = request.param
    features, target = dataset
    regressor = factory()
    assert regressor.fit(features, target) is regressor
    return regressor


def test_adapter_predict_keeps_pandas_metadata(fitted: AnyRegressor, dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, _ = dataset
    predictions = fitted.predict(features)

    assert predictions.index.equals(features.index)
    assert predictions.name == "price"
    assert predictions.dtype == np.float64


def test_adapter_learns_a_simple_signal(fitted: AnyRegressor, dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = dataset
    assert fitted.validate(features, target).r2 > 0.5


def test_adapter_save_load_round_trip(
    fitted: AnyRegressor, dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    features, _ = dataset
    fitted.save(tmp_path)
    restored = type(fitted).load(tmp_path)

    pd.testing.assert_series_equal(restored.predict(features), fitted.predict(features))
    assert restored.params == fitted.params
