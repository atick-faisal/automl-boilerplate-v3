"""Framework-agnostic interface for experiment tracking.

Adapters for a concrete tracker subclass `ExperimentLogger` and implement the private hooks.
Everything a caller touches lives in this module, so swapping MLflow for another tracker never
changes calling code.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import TYPE_CHECKING, Self, SupportsFloat, cast, final

import pandas as pd

from automl_boilerplate_v3.base import ParamValue

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "model"
_DEFAULT_SAMPLING_INTERVAL_S = 10.0

#: Leaderboard columns `log_candidates` records as params rather than metrics. `is_best` is named
#: here because pandas counts a boolean column as numeric, and "is this the winner" is a label.
_CANDIDATE_PARAM_COLUMNS = ("model", "metric", "is_best")

#: Columns `log_candidates` needs; the same four `AutoMLRegressor.leaderboard` guarantees.
_CANDIDATE_REQUIRED_COLUMNS = ("model", "loss", "metric", "is_best")


@dataclasses.dataclass(frozen=True, slots=True)
class ModelVersion:
    """One version of a model in the tracker's registry.

    Attributes:
        name: Registered model name.
        version: Tracker-specific version label. A string because not every tracker numbers them.
    """

    name: str
    version: str


@dataclasses.dataclass(frozen=True, slots=True)
class _ActiveRun:
    """The open run, and the experiment it lives in — a child run needs both."""

    run_id: str
    experiment_name: str


class ExperimentLogger[ConfigT: DataclassInstance](ABC):
    """Base class every experiment tracking adapter extends.

    A logger tracks at most one run at a time, but every run names its own experiment, so one
    logger covers as many experiments as you like. Use it as a context manager so the run is always
    closed, and marked failed when the block raises::

        with MlflowLogger(config).start_run("baseline", experiment_name="housing") as run:
            run.log_params(regressor.params)

    An AutoML search trains many models, so a run can also be a *group*: `log_candidates` writes one
    child run per candidate under the active run, while the winner's metrics, model and registry
    version stay on the parent.

    Attributes:
        config: Frozen dataclass holding the adapter's settings.
    """

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._active: _ActiveRun | None = None

    # ------------------------------------------------------------------ run lifecycle

    @final
    def start_run(
        self,
        run_name: str | None = None,
        *,
        experiment_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Self:
        """Open a new run that subsequent logging calls write to.

        Args:
            run_name: Human-readable run name; ``None`` lets the tracker pick one.
            experiment_name: Experiment the run is grouped under; created on first use. Keyword-only
                so a stale positional call fails loudly instead of naming an experiment after a run.
            tags: Free-form string labels attached to the run.

        Returns:
            The logger itself, so it can be used in a ``with`` statement.

        Raises:
            RuntimeError: If a run is already active.
        """
        if self._active is not None:
            raise RuntimeError(f"Run {self._active.run_id} is still active; call end_run() first")
        run_id = self._start_run(experiment_name, run_name, dict(tags or {}), parent_run_id=None)
        self._active = _ActiveRun(run_id=run_id, experiment_name=experiment_name)
        logger.info("Started run %s", run_id)
        return self

    @final
    def end_run(self, *, failed: bool = False) -> None:
        """Close the active run.

        Args:
            failed: Mark the run as failed instead of finished.

        Raises:
            RuntimeError: If no run is active.
        """
        run_id = self.run_id
        self._end_run(run_id, failed=failed)
        self._active = None
        logger.info("Ended run %s (%s)", run_id, "failed" if failed else "finished")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Only close a run that is still open, so an explicit end_run() inside the block is allowed.
        if self._active is not None:
            self.end_run(failed=exc_type is not None)

    @property
    def run_id(self) -> str:
        """Identifier of the active run.

        Raises:
            RuntimeError: If no run is active.
        """
        return self._require_active().run_id

    # ------------------------------------------------------------------ logging

    @final
    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Record run settings, e.g. `AutoMLRegressor.params`.

        Args:
            params: Parameter name to primitive value.
        """
        self._log_params(self.run_id, dict(params))

    @final
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record numeric results, e.g. `ValidationResult.to_dict`.

        Args:
            metrics: Metric name to value.
            step: Training step or epoch, for metrics logged repeatedly.
        """
        self._log_metrics(self.run_id, dict(metrics), step)

    @final
    def log_artifact(self, path: Path, name: str | None = None) -> None:
        """Upload a file or directory to the active run.

        Args:
            path: Existing file or directory.
            name: Where it is stored inside the run; ``None`` keeps the file or directory name.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(path)
        self._log_artifact(self.run_id, path, name or path.name)
        logger.info("Logged artifact %s to run %s", path, self.run_id)

    @final
    def log_table(self, table: pd.DataFrame, name: str) -> None:
        """Upload a dataframe as a CSV artifact, e.g. `AutoMLRegressor.leaderboard`.

        Args:
            table: Rows to upload; the index is not written.
            name: Where it is stored inside the run, e.g. ``"leaderboard.csv"``.
        """
        run_id = self.run_id
        with tempfile.TemporaryDirectory() as staging:
            staged = Path(staging) / PurePosixPath(name).name
            table.to_csv(staged, index=False)
            self._log_artifact(run_id, staged, name)
        logger.info("Logged table %s (%d rows) to run %s", name, len(table), run_id)

    @final
    def log_model(self, model_file: Path, name: str = _DEFAULT_MODEL_NAME) -> None:
        """Upload the file written by `AutoMLRegressor.save`, so it can be registered.

        Args:
            model_file: Existing file holding the model.
            name: Where it is stored inside the run; pass the same value to `register_model`.

        Raises:
            FileNotFoundError: If ``model_file`` does not exist.
            IsADirectoryError: If ``model_file`` is a directory.
        """
        if model_file.is_dir():
            raise IsADirectoryError(f"{model_file} is a directory; AutoMLRegressor.save writes a single file")
        self.log_artifact(model_file, name)

    # ------------------------------------------------------------------ run groups

    @final
    def log_child_run(
        self,
        run_name: str,
        *,
        params: Mapping[str, ParamValue] | None = None,
        metrics: Mapping[str, float] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> str:
        """Write one finished run grouped under the active run.

        The child is opened, written and closed inside this call, so the parent is still the active
        run when it returns: `log_model` and `register_model` keep writing to the parent. The child
        lands in the parent's experiment, which is what makes the tracker group the two.

        Args:
            run_name: Human-readable name for the child, e.g. the candidate model's name.
            params: Settings describing the child, e.g. which learner it is.
            metrics: Numeric results for the child, e.g. its validation loss.
            tags: Free-form string labels attached to the child.

        Returns:
            Identifier of the child run.

        Raises:
            RuntimeError: If no run is active.
        """
        parent = self._require_active()
        child_run_id = self._start_run(parent.experiment_name, run_name, dict(tags or {}), parent_run_id=parent.run_id)
        if params:
            self._log_params(child_run_id, dict(params))
        if metrics:
            self._log_metrics(child_run_id, dict(metrics), None)
        self._end_run(child_run_id, failed=False)
        return child_run_id

    @final
    def log_candidates(self, leaderboard: pd.DataFrame) -> None:
        """Write one child run per row of `AutoMLRegressor.leaderboard`.

        This turns the candidates a search tried into runs you can sort, filter and chart next to
        each other, instead of a CSV you have to download. Only the winner's model file is worth
        keeping, so it — and the validation metrics — stay on the parent run.

        The rows are written after the search, not while it runs: no AutoML framework reports a
        candidate in a way the others also do.

        Args:
            leaderboard: One row per candidate. ``model`` names the child run, ``metric`` and
                ``is_best`` join it as params, ``loss`` and every other numeric column become
                metrics, and any remaining column becomes a param.

        Raises:
            RuntimeError: If no run is active.
            ValueError: If a required column is missing.
        """
        parent_run_id = self.run_id
        missing = [column for column in _CANDIDATE_REQUIRED_COLUMNS if column not in leaderboard.columns]
        if missing:
            raise ValueError(f"Leaderboard is missing columns: {missing}")

        extras = [column for column in leaderboard.columns if column not in _CANDIDATE_PARAM_COLUMNS]
        metric_columns = [column for column in extras if pd.api.types.is_numeric_dtype(leaderboard[column])]
        param_columns = [*_CANDIDATE_PARAM_COLUMNS, *(column for column in extras if column not in metric_columns)]

        for record in cast(list[dict[str, object]], leaderboard.to_dict("records")):
            self.log_child_run(
                str(record["model"]),
                # Every tracker stores params as strings anyway, and `is_best` is a numpy bool.
                params={column: str(record[column]) for column in param_columns},
                metrics=_candidate_metrics(record, metric_columns),
            )
        logger.info("Logged %d candidate runs under run %s", len(leaderboard), parent_run_id)

    @final
    @contextlib.contextmanager
    def monitor_system_metrics(self, sampling_interval_s: float = _DEFAULT_SAMPLING_INTERVAL_S) -> Generator[None]:
        """Record what the machine is doing, for as long as the block runs.

        Wrap the search in it to see what the search cost the machine::

            with run.monitor_system_metrics():
                regressor.fit(features, target)

        Sampling runs on a background thread and always stops when the block exits, including when
        it raises. An adapter that cannot sample warns and lets the block run unmonitored, so this
        never becomes the reason a training script fails.

        Args:
            sampling_interval_s: Seconds between samples.

        Yields:
            Nothing; the metrics land on the active run.

        Raises:
            RuntimeError: If no run is active.
        """
        run_id = self.run_id
        stop = self._start_system_metrics(run_id, sampling_interval_s)
        if stop is None:
            logger.warning("%s does not record system metrics", type(self).__name__)
        else:
            logger.info("Recording system metrics for run %s every %gs", run_id, sampling_interval_s)
        try:
            yield
        finally:
            if stop is not None:
                stop()

    # ------------------------------------------------------------------ registry

    @final
    def register_model(self, model_name: str, artifact_name: str = _DEFAULT_MODEL_NAME) -> ModelVersion:
        """Add a model logged in the active run to the registry as a new version.

        Args:
            model_name: Registered model name; created on first use.
            artifact_name: The ``name`` previously passed to `log_model`.

        Returns:
            The newly created version.
        """
        version = self._register_model(self.run_id, model_name, artifact_name)
        logger.info("Registered %s version %s", version.name, version.version)
        return version

    @final
    def download_model(self, version: ModelVersion, dst: Path) -> Path:
        """Fetch a registered model version; does not need an active run.

        Args:
            version: Version returned by `register_model` or built by hand.
            dst: Directory to download into; created if it does not exist.

        Returns:
            The downloaded model file, ready for `AutoMLRegressor.load`. A tracker that stores the
            artifact inside a directory returns that directory instead.
        """
        dst.mkdir(parents=True, exist_ok=True)
        return self._download_model(version, dst)

    # ------------------------------------------------------------ adapter hooks

    @abstractmethod
    def _start_run(
        self, experiment_name: str, run_name: str | None, tags: dict[str, str], *, parent_run_id: str | None
    ) -> str:
        """Create a run inside ``experiment_name`` and return its identifier.

        ``parent_run_id`` is another run in the same experiment that this one is grouped under, or
        ``None`` for a top-level run.
        """

    @abstractmethod
    def _end_run(self, run_id: str, *, failed: bool) -> None:
        """Mark the run finished or failed."""

    @abstractmethod
    def _log_params(self, run_id: str, params: dict[str, ParamValue]) -> None:
        """Write parameters to the run."""

    @abstractmethod
    def _log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None) -> None:
        """Write metrics to the run."""

    @abstractmethod
    def _log_artifact(self, run_id: str, path: Path, name: str) -> None:
        """Upload an existing file or directory under ``name``."""

    @abstractmethod
    def _register_model(self, run_id: str, model_name: str, artifact_name: str) -> ModelVersion:
        """Register the run's ``artifact_name`` directory as a new version of ``model_name``."""

    @abstractmethod
    def _download_model(self, version: ModelVersion, dst: Path) -> Path:
        """Download ``version`` into the existing directory ``dst`` and return the model directory."""

    def _start_system_metrics(self, run_id: str, sampling_interval_s: float) -> Callable[[], None] | None:
        """Start sampling machine metrics into ``run_id``; optional, unlike the hooks above.

        Returns:
            A callable that stops sampling and flushes whatever it collected, or ``None`` for a
            tracker that cannot sample — the caller is warned and carries on unmonitored.
        """
        return None

    # ------------------------------------------------------------------ helpers

    def _require_active(self) -> _ActiveRun:
        if self._active is None:
            raise RuntimeError(f"{type(self).__name__} has no active run; call start_run() first")
        return self._active


def _candidate_metrics(record: Mapping[str, object], columns: Sequence[str]) -> dict[str, float]:
    """Read the numeric cells of one leaderboard row as metrics.

    Non-finite values are dropped rather than logged: FLAML reports `inf` as the loss of a learner
    that never completed a trial, and trackers disagree about whether they can store one.
    """
    # Every column here passed `is_numeric_dtype`, so the cell is a number.
    values = {column: float(cast(SupportsFloat, record[column])) for column in columns}
    return {column: value for column, value in values.items() if math.isfinite(value)}
