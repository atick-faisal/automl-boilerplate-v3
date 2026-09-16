# AutoGluon ships no type information, so pyright sees many of its members as partially unknown.
# Untyped values are converted to typed numpy/pandas objects before leaving this module.
# pyright: reportUnknownMemberType=false
"""AutoGluon adapter: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.exceptions import NotFittedError

from automl_boilerplate_v3.base import AutoMLRegressor, FloatArray

_PREDICTOR_DIR = "predictor"
# AutoGluon wants features and target in one frame. A fixed internal name avoids
# clashing with however the caller named their target.
_LABEL = "__target__"


@dataclass(frozen=True, slots=True)
class AutoGluonConfig:
    """Settings passed to `TabularPredictor`.

    Attributes:
        time_limit_s: Wall-clock seconds for training; ``None`` means no limit.
        presets: Quality/speed trade-off, e.g. ``"medium_quality"``, ``"best_quality"``.
        eval_metric: Metric AutoGluon optimizes, e.g. ``"root_mean_squared_error"``.
        verbosity: AutoGluon log level from 0 (silent) to 4.
        work_dir: Where AutoGluon writes models while training. ``None`` uses a temporary
            directory that is removed with this object; call `save` to keep the model.
    """

    time_limit_s: float | None = 300
    presets: str = "medium_quality"
    eval_metric: str = "root_mean_squared_error"
    verbosity: int = 0
    work_dir: Path | None = None


class AutoGluonRegressor(AutoMLRegressor[AutoGluonConfig]):
    """AutoML regressor backed by AutoGluon's stacked ensembles of classical models."""

    distribution_name = "autogluon.tabular"

    def __init__(self, config: AutoGluonConfig | None = None) -> None:
        super().__init__(config or AutoGluonConfig())
        self._predictor: TabularPredictor | None = None
        # Held on the instance so the directory lives exactly as long as the predictor using it.
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    @override
    def _fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        if _LABEL in features.columns:
            raise ValueError(f"Feature column name {_LABEL!r} is reserved by {type(self).__name__}")

        train_data = features.assign(**{_LABEL: target.to_numpy()})
        predictor = TabularPredictor(
            label=_LABEL,
            problem_type="regression",
            eval_metric=self.config.eval_metric,
            path=str(self._training_dir()),
            verbosity=self.config.verbosity,
        )
        predictor.fit(
            train_data,
            # AutoGluon annotates `time_limit: float = None`; None (no limit) is its documented default.
            time_limit=self.config.time_limit_s,  # pyright: ignore[reportArgumentType]
            presets=self.config.presets,
        )
        self._predictor = predictor

    @override
    def _predict(self, features: pd.DataFrame) -> FloatArray:
        return np.asarray(self._require_predictor().predict(features), dtype=np.float64)

    @override
    def _save(self, path: Path) -> None:
        self._require_predictor().clone(str(path / _PREDICTOR_DIR), dirs_exist_ok=True)

    @override
    def _load(self, path: Path) -> None:
        self._predictor = TabularPredictor.load(str(path / _PREDICTOR_DIR), verbosity=self.config.verbosity)

    def _training_dir(self) -> Path:
        if self.config.work_dir is not None:
            return self.config.work_dir
        self._temp_dir = tempfile.TemporaryDirectory(prefix="autogluon-")
        return Path(self._temp_dir.name)

    def _require_predictor(self) -> TabularPredictor:
        if self._predictor is None:
            raise NotFittedError("AutoGluonRegressor is not fitted")
        return self._predictor
