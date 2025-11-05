"""Training helpers for fine-tuning YOLO models with event-aware metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - ultralytics optional for docs builds
    YOLO = None  # type: ignore

from ..config import DEFAULT_TILE_CLASSES, ProjectPaths
from ..data.annotation_store import AnnotationSession
from ..models.yolo_manager import ModelMetadata, ModelRegistry


@dataclass
class TrainingConfig:
    base_model: Path
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 1280
    learning_rate: float = 0.001
    augment: bool = True
    project_name: str = "mahjong_yolo"
    experiment_name: str = "default"


class Trainer:
    """High level training orchestrator built on top of Ultralytics YOLO."""

    def __init__(self, paths: ProjectPaths, registry: ModelRegistry) -> None:
        self.paths = paths
        self.registry = registry

    def prepare_dataset(self, sessions: Iterable[AnnotationSession]) -> Path:
        dataset_root = self.paths.data_root / "prepared"
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for session in sessions:
            video_name = session.video_path.stem
            image_dir = images_dir / video_name
            label_dir = labels_dir / video_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            session.export_yolo_labels(label_dir)
        yaml_path = dataset_root / "dataset.yaml"
        yaml_dump(
            {
                "path": str(dataset_root),
                "train": "images",
                "val": "images",
                "names": {idx: name for idx, name in enumerate(DEFAULT_TILE_CLASSES)},
            },
            yaml_path,
        )
        return yaml_path

    def train(self, config: TrainingConfig, dataset_yaml: Path) -> ModelMetadata:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        model = YOLO(str(config.base_model))
        results = model.train(
            data=str(dataset_yaml),
            epochs=config.epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            lr0=config.learning_rate,
            project=str(self.paths.export_root / config.project_name),
            name=config.experiment_name,
            exist_ok=True,
            augment=config.augment,
        )
        metrics: Dict[str, float] = {}
        if hasattr(results, "results_dict"):
            metrics = {key: float(value) for key, value in results.results_dict.items() if isinstance(value, (int, float))}
        best_model_path = Path(model.ckpt_path or model.weights)
        return self.registry.register_model(
            name=config.project_name,
            source="training",
            model_path=best_model_path,
            metrics=metrics,
        )


def yaml_dump(payload: dict, path: Path) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
