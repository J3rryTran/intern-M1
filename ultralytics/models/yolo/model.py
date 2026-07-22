# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, PoseModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO26n face detection and pose estimation models.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, pose).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained YOLO26n detection model
        >>> model = YOLO("yolo26n.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo26n-repvit-face.yaml")

        Initialize a pose model from a YAML configuration
        >>> model = YOLO("yolo26n-repvit-face-pose-slim.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo26n.pt', 'yolo26n-repvit-face.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'pose'. Defaults to auto-detection based on
                model.
            verbose (bool): Display model info on load.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
        }
