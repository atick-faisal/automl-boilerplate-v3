"""Contract every `ExperimentLogger` must honour.

Base-class behaviour is tested with an in-memory stand-in adapter. The MLflow adapter runs
against a throwaway SQLite store, which needs the full `mlflow` package from the dev group.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, override

import pandas as pd
import pytest

from automl_boilerplate_v3 import ExperimentLogger, ModelVersion, ParamValue

if TYPE_CHECKING:
    from automl_boilerplate_v3.mlflow_logger import MlflowLogger

# ----------------------------------------------------------------- base class


@dataclass(frozen=True, slots=True)
class _NoConfig:
    pass


@dataclass
class _Run:
    experiment: str = ""
    failed: bool | None = None
    params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    artifacts: dict[str, Path] = field(default_factory=dict[str, Path])
    contents: dict[str, str] = field(default_factory=dict[str, str])


class _RecordingLogger(ExperimentLogger[_NoConfig]):
    """Keeps runs in memory; enough to exercise everything the base class owns."""

    def __init__(self) -> None:
        super().__init__(_NoConfig())
        self.runs: dict[str, _Run] = {}

    @override
    def _start_run(self, experiment_name: str, run_name: str | None, tags: dict[str, str]) -> str:
        run_id = f"run-{len(self.runs)}"
        self.runs[run_id] = _Run(experiment=experiment_name)
        return run_id

    @override
    def _end_run(self, run_id: str, *, failed: bool) -> None:
        self.runs[run_id].failed = failed

    @override
    def _log_params(self, run_id: str, params: dict[str, ParamValue]) -> None:
        self.runs[run_id].params.update(params)

    @override
    def _log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None) -> None:
        pass

    @override
    def _log_artifact(self, run_id: str, path: Path, name: str) -> None:
        self.runs[run_id].artifacts[name] = path
        if path.is_file():
            # Snapshot now: `log_table` stages its CSV in a temp directory that is gone on return.
            self.runs[run_id].contents[name] = path.read_text()

    @override
    def _register_model(self, run_id: str, model_name: str, artifact_name: str) -> ModelVersion:
        return ModelVersion(model_name, "1")

    @override
    def _download_model(self, version: ModelVersion, dst: Path) -> Path:
        return dst


def test_logging_without_active_run_raises() -> None:
    with pytest.raises(RuntimeError, match="no active run"):
        _RecordingLogger().log_params({"a": 1})


def test_start_run_twice_raises() -> None:
    logger = _RecordingLogger().start_run(experiment_name="exp")
    with pytest.raises(RuntimeError, match="still active"):
        logger.start_run(experiment_name="exp")


def test_context_manager_finishes_run() -> None:
    with _RecordingLogger().start_run(experiment_name="exp") as logger:
        logger.log_params({"a": 1})

    assert logger.runs["run-0"] == _Run(experiment="exp", failed=False, params={"a": 1})
    with pytest.raises(RuntimeError):
        _ = logger.run_id


def test_exception_in_context_manager_fails_run_and_propagates() -> None:
    logger = _RecordingLogger()
    with pytest.raises(ZeroDivisionError), logger.start_run(experiment_name="exp"):
        _ = 1 / 0

    assert logger.runs["run-0"].failed is True


def test_one_logger_writes_runs_to_different_experiments() -> None:
    logger = _RecordingLogger()
    with logger.start_run("first", experiment_name="baseline"):
        pass
    with logger.start_run("second", experiment_name="sweep"):
        pass

    assert [run.experiment for run in logger.runs.values()] == ["baseline", "sweep"]


def test_log_artifact_defaults_name_to_file_name(tmp_path: Path) -> None:
    file = tmp_path / "report.txt"
    file.write_text("ok")
    with _RecordingLogger().start_run(experiment_name="exp") as logger:
        logger.log_artifact(file)

    assert logger.runs["run-0"].artifacts == {"report.txt": file}


def test_log_model_rejects_missing_or_directory_path(tmp_path: Path) -> None:
    with _RecordingLogger().start_run(experiment_name="exp") as logger:
        with pytest.raises(FileNotFoundError):
            logger.log_model(tmp_path / "missing.zip")
        with pytest.raises(IsADirectoryError):
            logger.log_model(tmp_path)


def test_log_table_stages_a_csv_under_the_given_name() -> None:
    table = pd.DataFrame({"model": ["lgbm", "rf"], "loss": [0.1, 0.25]})
    with _RecordingLogger().start_run(experiment_name="exp") as logger:
        logger.log_table(table, "reports/leaderboard.csv")

    # Reading it back also proves the index was not written as a stray column.
    csv = logger.runs["run-0"].contents["reports/leaderboard.csv"]
    pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(csv)), table)


# -------------------------------------------------------------------- mlflow


@pytest.fixture
def mlflow_logger(tmp_path: Path) -> MlflowLogger:
    # The SQLite store needs SQLAlchemy, which only the full `mlflow` package brings.
    pytest.importorskip("mlflow.server")
    from automl_boilerplate_v3.mlflow_logger import MlflowConfig, MlflowLogger

    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return MlflowLogger(
        MlflowConfig(
            tracking_uri=store,
            registry_uri=store,
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
    )


def test_mlflow_records_params_metrics_and_status(mlflow_logger: MlflowLogger) -> None:
    with mlflow_logger.start_run("ok", experiment_name="test", tags={"team": "ml"}) as run:
        run.log_params({"time_budget_s": 5, "estimator_list": None})
        run.log_metrics({"val_rmse": 0.5})
        run_id = run.run_id

    failed_run_id = mlflow_logger.start_run("broken", experiment_name="test").run_id
    with pytest.raises(ZeroDivisionError), mlflow_logger:
        _ = 1 / 0

    client = mlflow_logger._client  # pyright: ignore[reportPrivateUsage]
    finished = client.get_run(run_id)
    assert finished.info.status == "FINISHED"
    assert finished.data.params == {"time_budget_s": "5", "estimator_list": "None"}
    assert finished.data.metrics == {"val_rmse": 0.5}
    assert finished.data.tags["team"] == "ml"
    assert client.get_run(failed_run_id).info.status == "FAILED"


def test_mlflow_logs_file_under_custom_name(mlflow_logger: MlflowLogger, tmp_path: Path) -> None:
    file = tmp_path / "local.txt"
    file.write_text("hello")
    with mlflow_logger.start_run(experiment_name="test") as run:
        run.log_artifact(file, "reports/summary.txt")
        run_id = run.run_id

    client = mlflow_logger._client  # pyright: ignore[reportPrivateUsage]
    artifacts = client.list_artifacts(run_id, "reports")  # pyright: ignore[reportUnknownMemberType]
    assert [artifact.path for artifact in artifacts] == ["reports/summary.txt"]


def test_mlflow_model_register_download_round_trip(mlflow_logger: MlflowLogger, tmp_path: Path) -> None:
    model_file = tmp_path / "model.zip"
    model_file.write_bytes(b"PK\x03\x04 pretend archive")

    with mlflow_logger.start_run(experiment_name="test") as run:
        run.log_model(model_file)
        first = run.register_model("price-regressor")
        second = run.register_model("price-regressor")

    assert (first, second) == (ModelVersion("price-regressor", "1"), ModelVersion("price-regressor", "2"))

    downloaded = mlflow_logger.download_model(second, tmp_path / "download")
    assert downloaded.is_file()
    assert downloaded.read_bytes() == b"PK\x03\x04 pretend archive"


def test_mlflow_one_logger_writes_to_two_experiments(mlflow_logger: MlflowLogger) -> None:
    with mlflow_logger.start_run("first", experiment_name="baseline") as run:
        first_run_id = run.run_id
    with mlflow_logger.start_run("second", experiment_name="sweep") as run:
        second_run_id = run.run_id

    client = mlflow_logger._client  # pyright: ignore[reportPrivateUsage]
    experiment_ids = [client.get_run(run_id).info.experiment_id for run_id in (first_run_id, second_run_id)]
    assert experiment_ids[0] != experiment_ids[1]
    assert [client.get_experiment(experiment_id).name for experiment_id in experiment_ids] == ["baseline", "sweep"]
