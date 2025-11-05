"""High-level dataset management helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..config import ProjectPaths
from .annotation_store import AnnotationSession


@dataclass
class DatasetManager:
    """Manages dataset directories and annotation sessions."""

    paths: ProjectPaths

    def list_videos(self) -> Iterable[Path]:
        return sorted(self.paths.data_root.glob("raw_videos/**/*.mp4"))

    def list_sessions(self) -> Iterable[Path]:
        return sorted(self.paths.data_root.glob("annotations/**/*.json"))

    def load_session(self, session_path: Path) -> AnnotationSession:
        return AnnotationSession.load(session_path)

    def new_session(self, video_path: Path, image_width: int, image_height: int) -> AnnotationSession:
        return AnnotationSession(video_path=video_path, image_width=image_width, image_height=image_height)

    def save_session(self, session: AnnotationSession, target_name: Optional[str] = None) -> Path:
        target_dir = self.paths.data_root / "annotations"
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_name is None:
            target_name = f"{video_path_to_name(session.video_path)}.json"
        target_path = target_dir / target_name
        session.save(target_path)
        return target_path


def video_path_to_name(path: Path) -> str:
    return path.stem.replace(" ", "_")
