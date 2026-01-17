import mlflow
from pathlib import Path
from typing import Dict, Any


class MLflowLogger:
    def __init__(self, experiment_name: str):
        mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def end_run(self):
        mlflow.end_run()
