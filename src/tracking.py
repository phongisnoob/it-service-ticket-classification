"""MLflow tracking helpers.

Tracking is env-driven:
  MLFLOW_TRACKING_URI  — required in production; defaults to sqlite:///mlruns.db
                         in development/test mode only.
  MLFLOW_EXPERIMENT_NAME — experiment label (default: ticket_classification)
  MLFLOW_TRACKING_ENABLED — set to "false"/"0"/"no" to disable entirely
  APP_ENV — "production" | "development" | "test"  (default: development)

In production mode (APP_ENV=production), a missing MLFLOW_TRACKING_URI raises
at import time so the pipeline fails fast rather than silently writing to a
local SQLite file.
"""

import os
from contextlib import contextmanager
from typing import Any, Generator

APP_ENV = os.environ.get("APP_ENV", "development").lower()

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

_PROD_URI_MISSING_MSG = (
    "MLFLOW_TRACKING_URI must be set when APP_ENV=production. "
    "Set the variable or switch to APP_ENV=development for local work."
)


def is_tracking_enabled() -> bool:
    if not MLFLOW_AVAILABLE:
        return False
    return os.environ.get("MLFLOW_TRACKING_ENABLED", "true").lower() in ("true", "1", "yes")


def _get_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    if APP_ENV == "production":
        raise RuntimeError(_PROD_URI_MISSING_MSG)
    # Development/test fallback only.
    return "sqlite:///mlruns.db"


@contextmanager
def start_run(run_name: str, model_backend: str) -> Generator[Any, None, None]:
    """Start or resume an MLflow run if tracking is enabled.

    The caller's exception is always propagated; tracking failures are logged
    but never suppress the original error.
    """
    if not is_tracking_enabled():
        yield None
        return

    tracking_uri = _get_tracking_uri()
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ticket_classification")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_id_path = os.path.join("artifacts", f".mlflow_run_id_{model_backend}")

    existing_run_id: str | None = None
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            existing_run_id = f.read().strip() or None

    caller_raised = False
    try:
        if existing_run_id:
            try:
                with mlflow.start_run(run_id=existing_run_id):
                    yield mlflow
                return
            except mlflow.exceptions.MlflowException:
                # Run deleted or inaccessible — fall through to create a new run.
                pass

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_backend", model_backend)
            os.makedirs("artifacts", exist_ok=True)
            with open(run_id_path, "w") as f:
                f.write(run.info.run_id)
            try:
                yield mlflow
            except BaseException:
                caller_raised = True
                raise
    except BaseException:
        if not caller_raised:
            # Tracking infrastructure failure — do not propagate.
            pass
        else:
            raise


def log_params(params: dict[str, Any]) -> None:
    if is_tracking_enabled():
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    if is_tracking_enabled():
        mlflow.log_metrics(metrics)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    if is_tracking_enabled() and os.path.exists(local_path):
        mlflow.log_artifact(local_path, artifact_path)


def log_dict_as_artifact(data: dict[str, Any], filename: str) -> None:
    if is_tracking_enabled():
        mlflow.log_dict(data, filename)
