"""Framework-agnostic AutoML regression and experiment tracking.

Adapters live in their own modules so importing this package never pulls in a framework:
`automl_boilerplate_v3.flaml_regressor`, `automl_boilerplate_v3.autogluon_regressor`,
`automl_boilerplate_v3.mlflow_logger`.
"""

from automl_boilerplate_v3.base import AutoMLRegressor, ParamValue, ValidationResult
from automl_boilerplate_v3.experiment_logger import ExperimentLogger, ModelVersion

__all__ = ["AutoMLRegressor", "ExperimentLogger", "ModelVersion", "ParamValue", "ValidationResult"]
