# AutoGluon ships no type information, so pyright sees many of its members as partially unknown.
# Untyped values are converted to typed numpy/pandas objects before leaving this module.
# pyright: reportUnknownMemberType=false
"""AutoGluon adapter: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, override

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.exceptions import NotFittedError

from automl_boilerplate_v3.base import AutoMLRegressor, FloatArray

_PREDICTOR_DIR = "predictor"
# AutoGluon wants features and target in one frame. A fixed internal name avoids
# clashing with however the caller named their target.
_LABEL = "__target__"

#: Every model family AutoGluon can train for regression, spelled as `hyperparameters` accepts it.
#: Taken from `autogluon.tabular.registry.ag_model_registry`, minus the keys AutoGluon manages
#: itself (the weighted ensembles, "DUMMY") and the multimodal ones ("AG_AUTOMM", "AG_IMAGE_NN",
#: "AG_TEXT_NN"), which need installs this package does not pull in.
type AutoGluonModelType = Literal[
    # Gradient-boosted trees and forests: fast, and what the default presets lean on.
    "GBM",  # LightGBM
    "GBM_PREP",  # LightGBM with AutoGluon's extra feature preprocessing
    "CAT",  # CatBoost
    "XGB",  # XGBoost
    "RF",  # Random forest
    "XT",  # Extremely randomised trees
    # Simple baselines, cheap enough to always be worth a leaderboard row. No preset trains these,
    # so naming one here is the only way to get it.
    "KNN",  # k-nearest neighbours
    "LR",  # Linear regression; `sklearn.Ridge` for a regression task
    # Neural networks: slower, and the ones that benefit most from a GPU. Every one of these needs
    # `torch` (and "FASTAI" also `fastai`), which this package's `autogluon` extra does not install.
    "NN_TORCH",  # AutoGluon's own PyTorch tabular network
    "FASTAI",  # fast.ai tabular network
    "REALMLP",  # RealMLP
    "TABM",  # TabM
    "FT_TRANSFORMER",  # FT-Transformer
    # Interpretable models: you trade accuracy for a model you can read. These need their own
    # packages (`interpret`, `imodels`), which this package's `autogluon` extra does not install.
    "EBM",  # Explainable boosting machine
    "IM_RULEFIT",  # RuleFit
    "IM_FIGS",  # Fast interpretable greedy-tree sums
    "IM_GREEDYTREE",  # Greedy tree
    "IM_HSTREE",  # Hierarchical shrinkage tree
    "IM_BOOSTEDRULES",  # Boosted rule set
    # Pretrained tabular foundation models: large downloads, and they want a GPU. Each needs its own
    # install on top of the `autogluon` extra, so naming one here is how you opt in deliberately.
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

#: The families this package's `autogluon` extra can actually run: GBM, CAT and XGB come from the
#: extra's own lightgbm/catboost/xgboost, RF and XT from scikit-learn, which is a core dependency.
#: AutoGluon's own default list adds "NN_TORCH" and "FASTAI", which have no torch to import here
#: and so only cost a model slot and an ImportError in the log.
_INSTALLABLE_MODEL_TYPES: tuple[AutoGluonModelType, ...] = ("GBM", "CAT", "XGB", "RF", "XT")


class _ModelKwargs(TypedDict, total=False):
    """The one `TabularPredictor.fit` argument this adapter passes only sometimes.

    `total=False` is what makes "sometimes" expressible: an empty instance leaves the key off the
    call entirely, which is not the same as passing ``None`` (see `AutoGluonRegressor._model_kwargs`).
    """

    hyperparameters: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class AutoGluonConfig:
    """Settings passed to `TabularPredictor`.

    Attributes:
        time_limit_s: Wall-clock seconds for training; ``None`` means no limit.
        presets: Quality/speed trade-off, e.g. ``"medium_quality"``, ``"best_quality"``.
        eval_metric: Metric AutoGluon optimizes, e.g. ``"root_mean_squared_error"``.
        model_types: Model families to train, e.g. ``("GBM", "CAT", "XGB")``; `AutoGluonModelType`
            lists every option. Defaults to the families this package installs. Naming families
            here *replaces* the model list ``presets`` would have used, so it also forfeits that
            preset's tuned hyperparameters; ``None`` leaves the whole choice to ``presets`` and is
            the way to keep them.
        verbosity: AutoGluon log level from 0 (silent) to 4.
        work_dir: Where AutoGluon writes models while training. ``None`` uses a temporary
            directory that is removed with this object; call `save` to keep the model.
    """

    time_limit_s: float | None = 300
    presets: str = "medium_quality"
    eval_metric: str = "root_mean_squared_error"
    model_types: tuple[AutoGluonModelType, ...] | None = _INSTALLABLE_MODEL_TYPES
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
            **self._model_kwargs(),
        )
        self._predictor = predictor

    def _model_kwargs(self) -> _ModelKwargs:
        """Spell `config.model_types` the way `TabularPredictor.fit` understands it.

        `hyperparameters`, not `included_model_types`: the latter only *filters* the model list the
        preset already chose (`autogluon/common/model_filter/_model_filter.py`), so asking it for a
        family no preset carries — "LR", "KNN", any "IM_*" — leaves the list empty and the fit dies
        with "No models were trained successfully". `hyperparameters` names the families outright,
        and an explicit `fit` kwarg wins over the preset's own value
        (`autogluon/common/utils/decorators.py`). A weighted ensemble is stacked on afterwards
        either way: a run restricted to ("GBM",) still finishes with a `WeightedEnsemble_L2` row.

        Returns:
            The `hyperparameters` kwarg, or nothing at all when `presets` should decide. Empty
            rather than ``{"hyperparameters": None}`` on purpose — AutoGluon fills in a preset's
            values only for keys *absent* from the call, so passing ``None`` explicitly would stop
            ``best_quality`` from ever applying its own `zeroshot` portfolio.
        """
        if self.config.model_types is None:
            return _ModelKwargs()
        # An empty dict per family means "this model, with its own default hyperparameters".
        return _ModelKwargs(hyperparameters={name: {} for name in self.config.model_types})

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
