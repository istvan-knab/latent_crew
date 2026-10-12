import logging
from pathlib import Path
from typing import Any, Dict, Optional
import wandb

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Wrapper for Weights & Biases lifecycle and artifact management."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.run = None

    def start_run(
        self,
        project: str,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
        system_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            logger.info("W&B logging is disabled.")
            return

        full_config = {**config, "system": system_info or {}}
        try:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=full_config,
                reinit=True,
            )
            logger.info(f"W&B Run initialized: {self.run.url}")
        except Exception as e:
            logger.warning(f"Could not connect to W&B ({e}). Falling back to disabled mode.")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_artifact(self, file_path: Path, artifact_type: str, name: str) -> None:
        if self.enabled and self.run and file_path.exists():
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(str(file_path))
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self.run:
            wandb.finish()