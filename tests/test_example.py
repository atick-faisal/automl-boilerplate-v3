"""Smoke test for `examples/train_and_register.py`.

The example is the first thing a reader runs, so it is checked the way they run it: as a script.
It writes into a temporary working directory, so the repo keeps no `mlflow.db` behind.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).parent.parent / "examples" / "train_and_register.py"


def test_example_trains_logs_and_registers(tmp_path: Path) -> None:
    pytest.importorskip("flaml")
    # The SQLite store the example defaults to needs SQLAlchemy, which only full `mlflow` brings.
    pytest.importorskip("mlflow.server")
    from mlflow import MlflowClient

    result = subprocess.run(
        [sys.executable, str(_EXAMPLE), "--engine", "flaml", "--time-budget", "5"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    # Assert on what landed in the store, not on log lines, so rewording a message is free.
    store = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=store, registry_uri=store)
    version = client.get_model_version("diabetes-regressor", "1")
    assert version.run_id is not None
    run = client.get_run(version.run_id)

    assert run.info.status == "FINISHED"
    assert "val_rmse" in run.data.metrics
    assert run.data.params["engine"] == "flaml"
    logged = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]  # pyright: ignore[reportUnknownMemberType]
    assert "leaderboard.csv" in logged
    assert "model" in logged
