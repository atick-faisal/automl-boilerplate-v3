"""Contract every `AutoMLRegressor` must honour.

Base-class behaviour is tested with a tiny stand-in adapter so it runs in milliseconds.
Real adapters are fitted once per module and run through the same public API.
"""

from __future__ import annotations

import zipfile
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


class _SearchingRegressor(_MeanRegressor):
    """Reports a leaderboard, deliberately unsorted and with its columns out of order."""

    @override
    def _leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": ["ok", "broken", "good"],
                "fit_time_s": [1.0, 0.0, 2.0],
                "loss": [0.5, float("inf"), 0.1],
                "metric": "rmse",
                "is_best": [False, False, True],
            }
        )


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


def test_save_writes_a_single_self_contained_file(dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path) -> None:
    features, target = dataset
    path = tmp_path / "model.zip"
    _MeanRegressor().fit(features, target).save(path)

    assert path.is_file()
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["config.pkl", "leaderboard.csv", "metadata.json", "payload/mean.txt"]


def test_save_rejects_a_directory(dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path) -> None:
    features, target = dataset
    with pytest.raises(IsADirectoryError):
        _MeanRegressor().fit(features, target).save(tmp_path)


def test_load_rejects_a_file_save_did_not_write(tmp_path: Path) -> None:
    path = tmp_path / "junk.zip"
    path.write_bytes(b"definitely not an archive")
    with pytest.raises(ValueError, match="not a model file"):
        _MeanRegressor.load(path)


def test_load_rejects_a_file_from_another_adapter(dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path) -> None:
    features, target = dataset
    path = tmp_path / "model.zip"
    _MeanRegressor().fit(features, target).save(path)

    class _OtherRegressor(_MeanRegressor):
        pass

    with pytest.raises(ValueError, match="saved by"):
        _OtherRegressor.load(path)


def test_leaderboard_is_empty_when_the_adapter_does_not_search(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = dataset
    board = _MeanRegressor().fit(features, target).leaderboard()

    assert list(board.columns) == ["model", "loss", "metric", "is_best"]
    assert board.empty


def test_leaderboard_is_ordered_and_survives_a_round_trip(
    dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    features, target = dataset
    regressor = _SearchingRegressor().fit(features, target)
    board = regressor.leaderboard()

    # Guaranteed columns first, adapter extras after, best candidate on top.
    assert list(board.columns) == ["model", "loss", "metric", "is_best", "fit_time_s"]
    assert list(board["model"]) == ["good", "ok", "broken"]
    assert list(board["is_best"]) == [True, False, False]

    path = tmp_path / "model.zip"
    regressor.save(path)
    pd.testing.assert_frame_equal(_SearchingRegressor.load(path).leaderboard(), board, check_exact=True)


def test_leaderboard_copy_cannot_corrupt_what_save_writes(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = dataset
    regressor = _SearchingRegressor().fit(features, target)
    regressor.leaderboard().drop(columns="loss", inplace=True)

    assert "loss" in regressor.leaderboard().columns


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

    # Pinned to two fast families so the run is deterministic inside the time limit, and so the
    # whole adapter contract below is exercised against a narrowed search.
    return AutoGluonRegressor(AutoGluonConfig(time_limit_s=20, presets="medium_quality", model_types=("GBM", "XGB")))


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


def test_adapter_leaderboard_ranks_every_candidate(fitted: AnyRegressor) -> None:
    board = fitted.leaderboard()

    assert not board.empty
    assert list(board["is_best"]).count(True) == 1
    assert board["loss"].is_monotonic_increasing


def test_adapter_save_load_round_trip(
    fitted: AnyRegressor, dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    features, _ = dataset
    path = tmp_path / "model.zip"
    fitted.save(path)
    assert path.is_file()

    restored = type(fitted).load(path)
    pd.testing.assert_series_equal(restored.predict(features), fitted.predict(features))
    assert restored.params == fitted.params
    # Exact, not approximate: a leaderboard that drifts on save is a leaderboard you cannot trust.
    pd.testing.assert_frame_equal(restored.leaderboard(), fitted.leaderboard(), check_exact=True)


def test_adapter_can_be_saved_again_after_loading(
    fitted: AnyRegressor, dataset: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """A loaded model reads its files from a temporary directory it owns, and AutoGluon's copy is
    already pruned to the best model. Saving it a second time has to survive both."""
    features, _ = dataset
    first = tmp_path / "first.zip"
    fitted.save(first)

    restored = type(fitted).load(first)
    second = tmp_path / "second.zip"
    restored.save(second)

    twice = type(fitted).load(second)
    pd.testing.assert_series_equal(twice.predict(features), fitted.predict(features))


# ------------------------------------------------- autogluon model selection


def test_autogluon_trains_a_family_no_preset_carries(dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """`model_types` names the models to train rather than filtering the preset's own list.

    Regression test: while this went through AutoGluon's `included_model_types`, which only filters,
    asking for a family no preset carries left nothing to train and the fit died with
    "No models were trained successfully". "LR" is `sklearn.Ridge` here, so it is cheap to fit.
    """
    pytest.importorskip("autogluon.tabular")
    from automl_boilerplate_v3.autogluon_regressor import AutoGluonConfig, AutoGluonRegressor

    features, target = dataset
    regressor = AutoGluonRegressor(AutoGluonConfig(time_limit_s=60, model_types=("LR",)))

    board = regressor.fit(features, target).leaderboard()

    # The weighted ensemble is stacked on regardless, so LinearModel is not the only row.
    assert "LinearModel" in set(board["model"])
    # Nothing from the preset's default list leaked in alongside it.
    assert not {"NeuralNetTorch", "NeuralNetFastAI", "CatBoost", "XGBoost"} & set(board["model"])


def test_autogluon_default_model_types_all_reach_the_leaderboard(
    dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Every family in the default set survives training *and* being saved.

    Regression test for a silent one: a model that fits and then raises while AutoGluon saves it is
    dropped with only a log line, so it just goes missing from the leaderboard. XGBoost did exactly
    that until the `xgboost` override in `pyproject.toml` — see the comment there for why.
    """
    pytest.importorskip("autogluon.tabular")
    from automl_boilerplate_v3.autogluon_regressor import AutoGluonConfig, AutoGluonRegressor

    # AutoGluon's leaderboard spells families out; the config names them by AutoGluon's own keys.
    trained_names = {"GBM": "LightGBM", "CAT": "CatBoost", "XGB": "XGBoost", "RF": "RandomForest", "XT": "ExtraTrees"}
    assert set(trained_names) == set(AutoGluonConfig().model_types or ()), "default set changed; update this map"

    features, target = dataset
    board = AutoGluonRegressor(AutoGluonConfig(time_limit_s=120)).fit(features, target).leaderboard()

    assert set(trained_names.values()) <= set(board["model"])
