import os
from contextlib import contextmanager
from typing import Any, Generator

# Check if mlflow is available
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def is_tracking_enabled() -> bool:
    if not MLFLOW_AVAILABLE:
        return False
    return os.environ.get("MLFLOW_TRACKING_ENABLED", "true").lower() in ("true", "1", "yes")

@contextmanager
def start_run(run_name: str, model_backend: str) -> Generator[Any, None, None]:
    """Start or resume an MLflow run if tracking is enabled."""
    if not is_tracking_enabled():
        yield None
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ticket_classification")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_id_path = os.path.join("artifacts", f".mlflow_run_id_{model_backend}")

    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
        try:
            with mlflow.start_run(run_id=run_id):
                yield mlflow
            return
        except Exception:
            pass # Run might be deleted, fallback to new run

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_backend", model_backend)
        os.makedirs("artifacts", exist_ok=True)
        with open(run_id_path, "w") as f:
            f.write(run.info.run_id)
        yield mlflow

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
