"""Contract every `ExperimentLogger` must honour.

Base-class behaviour is tested with an in-memory stand-in adapter. The MLflow adapter runs
against a throwaway SQLite store, which needs the full `mlflow` package from the dev group.
"""

from __future__ import annotations

import io
import time
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
    name: str | None = None
    parent: str | None = None
    failed: bool | None = None
    params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    metrics: dict[str, float] = field(default_factory=dict[str, float])
    artifacts: dict[str, Path] = field(default_factory=dict[str, Path])
    contents: dict[str, str] = field(default_factory=dict[str, str])


class _RecordingLogger(ExperimentLogger[_NoConfig]):
    """Keeps runs in memory; enough to exercise everything the base class owns."""

    def __init__(self) -> None:
        super().__init__(_NoConfig())
        self.runs: dict[str, _Run] = {}

    @override
    def _start_run(
        self, experiment_name: str, run_name: str | None, tags: dict[str, str], *, parent_run_id: str | None
    ) -> str:
        run_id = f"run-{len(self.runs)}"
        self.runs[run_id] = _Run(experiment=experiment_name, name=run_name, parent=parent_run_id)
        return run_id

    @override
    def _end_run(self, run_id: str, *, failed: bool) -> None:
        self.runs[run_id].failed = failed

    @override
    def _log_params(self, run_id: str, params: dict[str, ParamValue]) -> None:
        self.runs[run_id].params.update(params)

    @override
    def _log_metrics(self, run_id: str, metrics: dict[str, float], step: int | None) -> None:
        self.runs[run_id].metrics.update(metrics)

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


def _leaderboard() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["lgbm", "rf", "sgd"],
            "loss": [0.1, 0.25, float("inf")],
            "metric": ["rmse", "rmse", "rmse"],
            "is_best": [True, False, False],
            "fit_time_s": [1.5, 2.0, 0.1],
        }
    )


def test_log_child_run_groups_under_the_active_run() -> None:
    with _RecordingLogger().start_run("parent", experiment_name="sweep") as logger:
        parent_id = logger.run_id
        child_id = logger.log_child_run("candidate", params={"model": "lgbm"}, metrics={"loss": 0.5})
        # The child is closed but the parent is not: logging after it still writes to the parent.
        assert logger.run_id == parent_id

    assert logger.runs[child_id] == _Run(
        experiment="sweep",
        name="candidate",
        parent=parent_id,
        failed=False,
        params={"model": "lgbm"},
        metrics={"loss": 0.5},
    )


def test_log_child_run_without_active_run_raises() -> None:
    with pytest.raises(RuntimeError, match="no active run"):
        _RecordingLogger().log_child_run("candidate")


def test_log_candidates_writes_one_child_run_per_row() -> None:
    with _RecordingLogger().start_run("search", experiment_name="sweep") as logger:
        parent_id = logger.run_id
        logger.log_candidates(_leaderboard())

    children = [run for run_id, run in logger.runs.items() if run_id != parent_id]
    assert [run.name for run in children] == ["lgbm", "rf", "sgd"]
    assert [run.parent for run in children] == [parent_id] * 3
    assert children[0].params == {"model": "lgbm", "metric": "rmse", "is_best": "True"}
    assert children[0].metrics == {"loss": 0.1, "fit_time_s": 1.5}
    # An `inf` loss is dropped: FLAML reports it for a learner that never completed a trial.
    assert children[2].metrics == {"fit_time_s": 0.1}


def test_log_candidates_rejects_an_incomplete_leaderboard() -> None:
    with (
        _RecordingLogger().start_run(experiment_name="sweep") as logger,
        pytest.raises(ValueError, match="missing columns"),
    ):
        logger.log_candidates(pd.DataFrame({"model": ["lgbm"], "loss": [0.1]}))


def test_monitor_system_metrics_warns_and_carries_on_without_a_hook(caplog: pytest.LogCaptureFixture) -> None:
    with _RecordingLogger().start_run(experiment_name="exp") as logger, logger.monitor_system_metrics():
        logger.log_params({"a": 1})

    assert "does not record system metrics" in caplog.text
    assert logger.runs["run-0"].params == {"a": 1}


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


def test_mlflow_log_candidates_creates_child_runs(mlflow_logger: MlflowLogger) -> None:
    with mlflow_logger.start_run("search", experiment_name="test") as run:
        run.log_candidates(_leaderboard())
        parent_run_id = run.run_id

    client = mlflow_logger._client  # pyright: ignore[reportPrivateUsage]
    parent = client.get_run(parent_run_id)
    children = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'"
    )

    assert sorted(str(child.info.run_name) for child in children) == ["lgbm", "rf", "sgd"]
    assert {child.info.status for child in children} == {"FINISHED"}
    best = next(child for child in children if child.data.params["is_best"] == "True")
    assert best.data.metrics == {"loss": 0.1, "fit_time_s": 1.5}


def test_mlflow_records_system_metrics_while_the_block_runs(
    mlflow_logger: MlflowLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psutil")
    # The environment variable wins over the argument, so a developer who exports it would
    # otherwise be waiting for their own sampling interval here.
    monkeypatch.delenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", raising=False)

    with mlflow_logger.start_run("monitored", experiment_name="test") as run:
        with run.monitor_system_metrics(sampling_interval_s=0.1):
            time.sleep(0.5)
        run_id = run.run_id

    metrics = mlflow_logger._client.get_run(run_id).data.metrics  # pyright: ignore[reportPrivateUsage]
    assert [key for key in metrics if key.startswith("system/")], metrics
