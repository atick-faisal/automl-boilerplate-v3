"""Framework-agnostic interface for experiment tracking.

Adapters for a concrete tracker subclass `ExperimentLogger` and implement the private hooks.
Everything a caller touches lives in this module, so swapping MLflow for another tracker never
changes calling code.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self, final

from automl_boilerplate_v3.base import ParamValue

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "model"


@dataclasses.dataclass(frozen=True, slots=True)
class ModelVersion:
    """One version of a model in the tracker's registry.

    Attributes:
        name: Registered model name.
        version: Tracker-specific version label. A string because not every tracker numbers them.
    """

    name: str
    version: str


class ExperimentLogger[ConfigT: DataclassInstance](ABC):
    """Base class every experiment tracking adapter extends.

    A logger tracks at most one run at a time. Use it as a context manager so the run is always
    closed, and marked failed when the block raises::

        with MlflowLogger(config).start_run("baseline") as run:
            run.log_params(regressor.params)

    Attributes:
        config: Frozen dataclass holding the adapter's settings.
    """

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._run_id: str | None = None

    # ------------------------------------------------------------------ run lifecycle

    @final
    def start_run(self, run_name: str | None = None, tags: Mapping[str, str] | None = None) -> Self:
        """Open a new run that subsequent logging calls write to.

        Args:
            run_name: Human-readable run name; ``None`` lets the tracker pick one.
            tags: Free-form string labels attached to the run.

        Returns:
            The logger itself, so it can be used in a ``with`` statement.

        Raises:
            RuntimeError: If a run is already active.
        """
        if self._run_id is not None:
            raise RuntimeError(f"Run {self._run_id} is still active; call end_run() first")
        self._run_id = self._start_run(run_name, dict(tags or {}))
        logger.info("Started run %s", self._run_id)
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
        self._run_id = None
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
        if self._run_id is not None:
            self.end_run(failed=exc_type is not None)

    @property
    def run_id(self) -> str:
        """Identifier of the active run.

        Raises:
            RuntimeError: If no run is active.
        """
        if self._run_id is None:
            raise RuntimeError(f"{type(self).__name__} has no active run; call start_run() first")
        return self._run_id

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
    def log_model(self, model_dir: Path, name: str = _DEFAULT_MODEL_NAME) -> None:
        """Upload a model directory, e.g. one written by `AutoMLRegressor.save`, so it can be registered.

        Args:
            model_dir: Existing directory holding the model.
            name: Where it is stored inside the run; pass the same value to `register_model`.

        Raises:
            NotADirectoryError: If ``model_dir`` is not an existing directory.
        """
        if not model_dir.is_dir():
            raise NotADirectoryError(model_dir)
        self.log_artifact(model_dir, name)

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
            The downloaded model directory, ready for `AutoMLRegressor.load`.
        """
        dst.mkdir(parents=True, exist_ok=True)
        return self._download_model(version, dst)

    # ------------------------------------------------------------ adapter hooks

    @abstractmethod
    def _start_run(self, run_name: str | None, tags: dict[str, str]) -> str:
        """Create a run and return its identifier."""

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
