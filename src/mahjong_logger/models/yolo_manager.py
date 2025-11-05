"""Utilities for importing, exporting and using YOLO models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - ultralytics is optional during documentation builds
    YOLO = None  # type: ignore

from ..config import ProjectPaths


@dataclass
class ModelMetadata:
    """Metadata describing a stored model version."""

    name: str
    version: str
    created_at: datetime
    source: str
    path: Path
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "path": str(self.path),
            "metrics": self.metrics,
        }


@dataclass
class ModelRegistry:
    """Simple filesystem-based registry for YOLO models."""

    paths: ProjectPaths

    def _model_dir(self) -> Path:
        return self.paths.model_root

    def list_models(self) -> Iterable[ModelMetadata]:
        for meta_file in sorted(self._model_dir().glob("*/metadata.json")):
            data = json_load(meta_file)
            yield ModelMetadata(
                name=data["name"],
                version=data["version"],
                created_at=datetime.fromisoformat(data["created_at"]),
                source=data["source"],
                path=meta_file.parent / data["filename"],
                metrics=data.get("metrics", {}),
            )

    def register_model(
        self,
        name: str,
        source: str,
        model_path: Path,
        metrics: Optional[Dict[str, float]] = None,
    ) -> ModelMetadata:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version_dir = self._model_dir() / f"{name}_{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        target_model_path = version_dir / model_path.name
        if model_path.resolve() != target_model_path.resolve():
            target_model_path.write_bytes(model_path.read_bytes())
        metadata = {
            "name": name,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "source": source,
            "filename": model_path.name,
            "metrics": metrics or {},
        }
        json_dump(metadata, version_dir / "metadata.json")
        return ModelMetadata(
            name=name,
            version=version,
            created_at=datetime.fromisoformat(metadata["created_at"]),
            source=source,
            path=target_model_path,
            metrics=metadata["metrics"],
        )

    def load_model(self, metadata: ModelMetadata):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        return YOLO(str(metadata.path))

    def import_model(self, model_path: Path, name: Optional[str] = None) -> ModelMetadata:
        if name is None:
            name = model_path.stem
        return self.register_model(name=name, source="import", model_path=model_path)

    def latest(self, name: str) -> Optional[ModelMetadata]:
        candidates = [model for model in self.list_models() if model.name == name]
        if not candidates:
            return None
        return max(candidates, key=lambda meta: meta.version)


def json_load(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(payload: dict, path: Path) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
