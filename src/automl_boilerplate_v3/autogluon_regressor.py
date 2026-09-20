# AutoGluon ships no type information, so pyright sees many of its members as partially unknown.
# Untyped values are converted to typed numpy/pandas objects before leaving this module.
# pyright: reportUnknownMemberType=false
"""AutoGluon adapter: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.exceptions import NotFittedError

from automl_boilerplate_v3.base import AutoMLRegressor, FloatArray

_PREDICTOR_DIR = "predictor"
# AutoGluon wants features and target in one frame. A fixed internal name avoids
# clashing with however the caller named their target.
_LABEL = "__target__"

#: Every model family AutoGluon can train for regression, spelled as `included_model_types`
#: accepts it. Taken from `autogluon.tabular.registry.ag_model_registry`, minus the keys
#: AutoGluon manages itself (the weighted ensembles, "DUMMY") and the multimodal ones
#: ("AG_AUTOMM", "AG_IMAGE_NN", "AG_TEXT_NN"), which need installs this package does not pull in.
type AutoGluonModelType = Literal[
    # Gradient-boosted trees and forests: fast, and what the default presets lean on.
    "GBM",  # LightGBM
    "GBM_PREP",  # LightGBM with AutoGluon's extra feature preprocessing
    "CAT",  # CatBoost
    "XGB",  # XGBoost
    "RF",  # Random forest
    "XT",  # Extremely randomised trees
    # Simple baselines, cheap enough to always be worth a leaderboard row.
    "KNN",  # k-nearest neighbours
    "LR",  # Linear regression
    # Neural networks: slower, and the ones that benefit most from a GPU.
    "NN_TORCH",  # AutoGluon's own PyTorch tabular network
    "FASTAI",  # fast.ai tabular network
    "REALMLP",  # RealMLP
    "TABM",  # TabM
    "FT_TRANSFORMER",  # FT-Transformer
    # Interpretable models: you trade accuracy for a model you can read.
    "EBM",  # Explainable boosting machine
    "IM_RULEFIT",  # RuleFit
    "IM_FIGS",  # Fast interpretable greedy-tree sums
    "IM_GREEDYTREE",  # Greedy tree
    "IM_HSTREE",  # Hierarchical shrinkage tree
    "IM_BOOSTEDRULES",  # Boosted rule set
    # Pretrained tabular foundation models: large downloads, and they want a GPU. Only the
    # strongest presets reach for these, so naming one here is how you opt in deliberately.
    "TABPFN-2.6",
    "TABPFN-3",
    "TABPFNMIX",
    "REALTABPFN-V2",
    "REALTABPFN-V2.5",
    "TABICL",
    "TABDPT",
    "TABDPT-TURBO",
    "MITRA",
    "NORI",
]


@dataclass(frozen=True, slots=True)
class AutoGluonConfig:
    """Settings passed to `TabularPredictor`.

    Attributes:
        time_limit_s: Wall-clock seconds for training; ``None`` means no limit.
        presets: Quality/speed trade-off, e.g. ``"medium_quality"``, ``"best_quality"``.
        eval_metric: Metric AutoGluon optimizes, e.g. ``"root_mean_squared_error"``.
        included_model_types: Model families to train, e.g. ``("GBM", "CAT", "XGB")``;
            `AutoGluonModelType` lists every option. ``None`` leaves the choice to ``presets``.
        verbosity: AutoGluon log level from 0 (silent) to 4.
        work_dir: Where AutoGluon writes models while training. ``None`` uses a temporary
            directory that is removed with this object; call `save` to keep the model.
    """

    time_limit_s: float | None = 300
    presets: str = "medium_quality"
    eval_metric: str = "root_mean_squared_error"
    included_model_types: tuple[AutoGluonModelType, ...] | None = None
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
            # A filter over the preset's own model list, not a replacement for it, and the
            # weighted ensemble is stacked on afterwards either way: a run restricted to
            # ("GBM",) still finishes with a `WeightedEnsemble_L2` row in the leaderboard.
            included_model_types=(
                None if self.config.included_model_types is None else list(self.config.included_model_types)
            ),
        )
        self._predictor = predictor

    @override
    def _predict(self, features: pd.DataFrame) -> FloatArray:
        return np.asarray(self._require_predictor().predict(features), dtype=np.float64)

    @override
    def _save(self, path: Path) -> None:
        # `clone_for_deployment`, not `clone`: it keeps the winning model and deletes every other
        # candidate, the bagged folds and the copy of the training data. The clone can only
        # predict, which is all a saved model has to do.
        self._require_predictor().clone_for_deployment(str(path / _PREDICTOR_DIR), model="best", dirs_exist_ok=True)

    @override
    def _load(self, path: Path) -> None:
        self._predictor = TabularPredictor.load(str(path / _PREDICTOR_DIR), verbosity=self.config.verbosity)

    @override
    def _leaderboard(self) -> pd.DataFrame:
        # Read at fit time on purpose: `_save` prunes the predictor with `clone_for_deployment`,
        # which deletes the losing models and this table along with them.
        predictor = self._require_predictor()
        board = predictor.leaderboard()
        best = predictor.model_best
        return pd.DataFrame(
            {
                "model": board["model"],
                # AutoGluon always reports higher-is-better, flipping the sign of error metrics.
                # Flip it back so the column is a loss like every other adapter's.
                "loss": -board["score_val"],
                "metric": board["eval_metric"],
                "is_best": board["model"] == best,
                "fit_time_s": board["fit_time"],
                "predict_time_s": board["pred_time_val"],
            }
        )

    def _training_dir(self) -> Path:
        if self.config.work_dir is not None:
            return self.config.work_dir
        self._temp_dir = tempfile.TemporaryDirectory(prefix="autogluon-")
        return Path(self._temp_dir.name)

    def _require_predictor(self) -> TabularPredictor:
        if self._predictor is None:
            raise NotFittedError("AutoGluonRegressor is not fitted")
        return self._predictor
