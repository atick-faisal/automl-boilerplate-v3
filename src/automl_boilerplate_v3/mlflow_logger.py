"""MLflow adapter: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import override

from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.entities import Metric, Param, RunStatus
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from automl_boilerplate_v3.base import ParamValue
from automl_boilerplate_v3.experiment_logger import ExperimentLogger, ModelVersion


@dataclass(frozen=True, slots=True)
class MlflowConfig:
    """Where runs are tracked. Which experiment they land in is named per run, in `start_run`.

    Attributes:
        tracking_uri: Tracking server or store, e.g. ``"http://localhost:5000"``. ``None`` uses the
            ``MLFLOW_TRACKING_URI`` environment variable, then MLflow's default.
        registry_uri: Model registry location. ``None`` uses the tracking URI.
        artifact_location: Artifact root for any experiment this logger creates; an experiment that
            already exists keeps the root it was created with.
    """

    tracking_uri: str | None = None
    registry_uri: str | None = None
    artifact_location: str | None = None


class MlflowLogger(ExperimentLogger[MlflowConfig]):
    """Experiment logger backed by the MLflow tracking client and model registry."""

    @cached_property
    def _client(self) -> MlflowClient:
        # An explicit client, not the fluent `mlflow.*` API, which keeps global state that
        # would leak between loggers pointing at different stores.
        return MlflowClient(tracking_uri=self.config.tracking_uri, registry_uri=self.config.registry_uri)

    @override
    def _start_run(
        self, experiment_name: str, run_name: str | None, tags: dict[str, str], *, parent_run_id: str | None
    ) -> str:
        if parent_run_id is not None:
            # A tag is all a nested run is: MLflow groups by `mlflow.parentRunId` and has no other
            # notion of a run group. The UI shows the children folded under the parent.
            tags = {**tags, MLFLOW_PARENT_RUN_ID: parent_run_id}
        run = self._client.create_run(self._experiment_id(experiment_name), tags=tags, run_name=run_name)
        return run.info.run_id

    @override
    def _start_system_metrics(self, run_id: str, sampling_interval_s: float) -> Callable[[], None] | None:
        """Sample CPU, memory, disk, network — and GPU, where one is visible — into the run."""
        # Imported here rather than at module scope: the monitor imports `psutil` eagerly, and the
        # rest of this adapter has to keep working for anyone who installed `mlflow-skinny` alone.
        from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

        # `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` in the environment overrides the argument, which
        # is MLflow's own precedence, not ours. GPU metrics appear when `pynvml` (NVIDIA) or
        # `pyrsmi` (AMD) imports, and are skipped silently otherwise.
        monitor = SystemMetricsMonitor(
            run_id,
            # MLflow leaves `sampling_interval` unannotated and defaults it to `10`, so it reads as
            # an int. It is only ever passed to `threading.Event.wait`, which takes a float.
            sampling_interval=sampling_interval_s,  # pyright: ignore[reportArgumentType]
            tracking_uri=self.config.tracking_uri,
        )
        monitor.start()
        # `finish` stops the thread and flushes what it has buffered.
        return monitor.finish

    @override
    def _end_run(self, run_id: str, *, failed: bool) -> None:
        status = RunStatus.FAILED if failed else RunStatus.FINISHED
        self._client.set_terminated(run_id, status=RunStatus.to_string(status))

    @override
    def _log_params(self, run_id: str, params: dict[str, ParamValue]) -> None:
        # MLflow stores every param as a string anyway; converting here makes None explicit.
        self._client.log_batch(run_id, params=[Param(key, str(value)) for key, value in params.items()])

    @override
    def _log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None) -> None:
        timestamp_ms = int(time.time() * 1000)
        self._client.log_batch(
            run_id,
            metrics=[Metric(key, value, timestamp_ms, step or 0) for key, value in metrics.items()],
        )

    @override
    def _log_artifact(self, run_id: str, path: Path, name: str) -> None:
        if path.is_dir():
            self._client.log_artifacts(run_id, str(path), artifact_path=name)
            return

        # `log_artifact` always keeps the local file name, so stage a copy under the requested one.
        destination = PurePosixPath(name)
        parent = None if destination.parent == PurePosixPath() else str(destination.parent)
        with tempfile.TemporaryDirectory() as staging_dir:
            staged = Path(staging_dir) / destination.name
            shutil.copyfile(path, staged)
            # `MlflowClient.log_artifact` leaves its parameters unannotated.
            self._client.log_artifact(run_id, str(staged), artifact_path=parent)  # pyright: ignore[reportUnknownMemberType]

    @override
    def _register_model(self, run_id: str, model_name: str, artifact_name: str) -> ModelVersion:
        try:
            self._client.create_registered_model(model_name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        model_version = self._client.create_model_version(
            model_name, source=f"runs:/{run_id}/{artifact_name}", run_id=run_id
        )
        # MLflow 3 returns the version as an int despite annotating it as str.
        return ModelVersion(name=model_name, version=str(model_version.version))

    @override
    def _download_model(self, version: ModelVersion, dst: Path) -> Path:
        # Resolve through our client: `download_artifacts("models:/...")` ignores `registry_uri`
        # and silently queries MLflow's default store instead.
        source_uri = self._client.get_model_version_download_uri(version.name, version.version)
        downloaded = download_artifacts(
            artifact_uri=source_uri, dst_path=str(dst), tracking_uri=self.config.tracking_uri
        )
        return Path(downloaded)

    def _experiment_id(self, experiment_name: str) -> str:
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return self._client.create_experiment(experiment_name, artifact_location=self.config.artifact_location)
