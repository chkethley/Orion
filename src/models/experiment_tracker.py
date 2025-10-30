"""Experiment tracking and model management."""
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False
    ):
        """Initialize experiment tracker."""
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb

        self.mlflow_client = None
        self.wandb_run = None

        self._initialize_trackers()

    def _initialize_trackers(self) -> None:
        """Initialize tracking backends."""
        if self.use_mlflow:
            try:
                import mlflow
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.mlflow_client = mlflow
                logger.info(f"MLflow initialized for experiment: {self.experiment_name}")
            except ImportError:
                logger.warning("MLflow not installed, skipping MLflow tracking")
                self.use_mlflow = False

        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.experiment_name,
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logger.info(f"W&B initialized for experiment: {self.experiment_name}")
            except ImportError:
                logger.warning("W&B not installed, skipping W&B tracking")
                self.use_wandb = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new experiment run."""
        if self.use_mlflow:
            self.mlflow_client.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run_name}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        if self.use_mlflow:
            self.mlflow_client.log_params(params)

        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update(params)

        logger.info(f"Logged parameters: {params}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log experiment metrics."""
        if self.use_mlflow:
            for key, value in metrics.items():
                self.mlflow_client.log_metric(key, value, step=step)

        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(metrics, step=step)

        logger.info(f"Logged metrics: {metrics}")

    def log_artifact(self, artifact_path: str) -> None:
        """Log an artifact (file, model, etc.)."""
        if self.use_mlflow:
            self.mlflow_client.log_artifact(artifact_path)

        if self.use_wandb and self.wandb_run:
            self.wandb_run.save(artifact_path)

        logger.info(f"Logged artifact: {artifact_path}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ) -> None:
        """Log a model."""
        if self.use_mlflow:
            self.mlflow_client.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model to: {artifact_path}")

    def end_run(self) -> None:
        """End the current run."""
        if self.use_mlflow:
            self.mlflow_client.end_run()

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        logger.info("Experiment run ended")


class ModelRegistry:
    """Model registry for managing trained models."""

    def __init__(self, storage_path: str = "data/models"):
        """Initialize model registry."""
        self.storage_path = storage_path
        self.models: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new model."""
        model_id = f"{model_name}:{model_version}"

        self.models[model_id] = {
            "name": model_name,
            "version": model_version,
            "path": model_path,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }

        logger.info(f"Registered model: {model_id}")
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        return self.models.get(model_id)

    def list_models(
        self,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all registered models."""
        if model_name:
            return [
                model for model_id, model in self.models.items()
                if model["name"] == model_name
            ]
        return list(self.models.values())

    def get_latest_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a model."""
        model_versions = self.list_models(model_name)
        if not model_versions:
            return None

        # Sort by version (assuming semantic versioning)
        model_versions.sort(
            key=lambda x: x["registered_at"],
            reverse=True
        )
        return model_versions[0]

    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate a model."""
        if model_id in self.models:
            self.models[model_id]["status"] = "inactive"
            logger.info(f"Deactivated model: {model_id}")
            return True
        return False


class ExperimentManager:
    """High-level experiment management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize experiment manager."""
        self.config = config or {}
        self.experiments: Dict[str, ExperimentTracker] = {}
        self.model_registry = ModelRegistry()

    def create_experiment(
        self,
        experiment_name: str,
        use_mlflow: bool = True,
        use_wandb: bool = False
    ) -> ExperimentTracker:
        """Create a new experiment."""
        tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri=self.config.get("mlflow_tracking_uri"),
            use_mlflow=use_mlflow,
            use_wandb=use_wandb
        )

        self.experiments[experiment_name] = tracker
        logger.info(f"Created experiment: {experiment_name}")

        return tracker

    def get_experiment(self, experiment_name: str) -> Optional[ExperimentTracker]:
        """Get an existing experiment."""
        return self.experiments.get(experiment_name)

    def list_experiments(self) -> List[str]:
        """List all experiments."""
        return list(self.experiments.keys())
