# FLAML ships no type information, so pyright sees many of its members as partially unknown.
# Untyped values are converted to typed numpy/pandas objects before leaving this module.
# pyright: reportUnknownMemberType=false
"""FLAML adapter: https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from flaml.automl.automl import AutoML
from sklearn.exceptions import NotFittedError

from automl_boilerplate_v3.base import AutoMLRegressor, FloatArray

_MODEL_FILE = "automl.pkl"


@dataclass(frozen=True, slots=True)
class FlamlConfig:
    """Settings passed to `flaml.AutoML.fit`.

    Attributes:
        time_budget_s: Wall-clock seconds for the whole search.
        metric: Metric FLAML optimizes, e.g. ``"rmse"``, ``"mae"``, ``"r2"``.
        estimator_list: Learners to try, e.g. ``("lgbm", "xgboost")``. ``None`` lets FLAML choose.
        seed: Random seed for reproducible searches.
        verbose: FLAML log level; 0 is silent.
    """

    time_budget_s: float = 60
    metric: str = "rmse"
    estimator_list: tuple[str, ...] | None = None
    seed: int = 0
    verbose: int = 0


class FlamlRegressor(AutoMLRegressor[FlamlConfig]):
    """AutoML regressor backed by FLAML's cost-frugal hyperparameter search."""

    distribution_name = "flaml"

    def __init__(self, config: FlamlConfig | None = None) -> None:
        super().__init__(config or FlamlConfig())
        self._automl: AutoML | None = None

    @override
    def _fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        automl = AutoML()
        automl.fit(
            X_train=features,
            y_train=target,
            task="regression",
            time_budget=self.config.time_budget_s,
            metric=self.config.metric,
            estimator_list=None if self.config.estimator_list is None else list(self.config.estimator_list),
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        self._automl = automl

    @override
    def _predict(self, features: pd.DataFrame) -> FloatArray:
        predictions = cast(npt.ArrayLike, self._require_automl().predict(features))
        return np.asarray(predictions, dtype=np.float64)

    @override
    def _save(self, path: Path) -> None:
        # FLAML's own pickle helper, which its docs recommend over plain `pickle.dump`.
        self._require_automl().pickle(str(path / _MODEL_FILE))

    @override
    def _load(self, path: Path) -> None:
        self._automl = AutoML.load_pickle(str(path / _MODEL_FILE))

    @override
    def _leaderboard(self) -> pd.DataFrame:
        automl = self._require_automl()
        # FLAML keeps the best configuration per learner, which is the granularity worth ranking.
        # A learner that never completed a trial is reported as `inf`, and sorts last.
        losses = cast(dict[str, float], automl.best_loss_per_estimator)
        best = cast(str, automl.best_estimator)
        return pd.DataFrame(
            {
                "model": list(losses),
                "loss": [float(loss) for loss in losses.values()],
                "metric": [self.config.metric] * len(losses),
                "is_best": [name == best for name in losses],
            }
        )

    def _require_automl(self) -> AutoML:
        if self._automl is None:
            raise NotFittedError("FlamlRegressor is not fitted")
        return self._automl
