"""Framework-agnostic AutoML regression.

Adapters live in their own modules so importing this package never pulls in a framework:
`automl_boilerplate_v3.flaml_regressor`, `automl_boilerplate_v3.autogluon_regressor`.
"""

from automl_boilerplate_v3.base import AutoMLRegressor, ParamValue, ValidationResult

__all__ = ["AutoMLRegressor", "ParamValue", "ValidationResult"]
