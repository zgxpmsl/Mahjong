"""Utilities to manage annotations, identities and dataset exports."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config import DEFAULT_TILE_CLASSES
from .log_schema import RoundLog


LABEL_TO_CLASS_ID: Dict[str, int] = {name: index for index, name in enumerate(DEFAULT_TILE_CLASSES)}


@dataclass
class BoundingBox:
    """Axis-aligned bounding box with identity tracking."""

    x: float
    y: float
    width: float
    height: float
    label: str
    track_id: str
    confidence: float = 1.0
    face_state: str = "visible"  # visible|face_down|unknown

    def as_yolo(self, image_width: int, image_height: int) -> List[float]:
        """Return YOLO-normalised bbox representation."""

        x_center = (self.x + self.width / 2) / image_width
        y_center = (self.y + self.height / 2) / image_height
        norm_width = self.width / image_width
        norm_height = self.height / image_height
        return [x_center, y_center, norm_width, norm_height]


@dataclass
class FrameAnnotation:
    """Collection of bounding boxes for a single frame."""

    frame_index: int
    boxes: List[BoundingBox] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"frame_index": self.frame_index, "boxes": [asdict(box) for box in self.boxes]}

    @classmethod
    def from_dict(cls, payload: dict) -> "FrameAnnotation":
        return cls(
            frame_index=payload["frame_index"],
            boxes=[BoundingBox(**box_data) for box_data in payload.get("boxes", [])],
        )


@dataclass
class AnnotationSession:
    """Aggregates per-frame annotations and related metadata."""

    video_path: Path
    image_width: int
    image_height: int
    frames: Dict[int, FrameAnnotation] = field(default_factory=dict)
    round_log: Optional[RoundLog] = None

    def ensure_frame(self, frame_index: int) -> FrameAnnotation:
        if frame_index not in self.frames:
            self.frames[frame_index] = FrameAnnotation(frame_index=frame_index)
        return self.frames[frame_index]

    def to_dict(self) -> dict:
        return {
            "video_path": str(self.video_path),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "frames": [frame.to_dict() for frame in sorted(self.frames.values(), key=lambda f: f.frame_index)],
            "round_log": self.round_log.to_dict() if self.round_log else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AnnotationSession":
        session = cls(
            video_path=Path(payload["video_path"]),
            image_width=payload["image_width"],
            image_height=payload["image_height"],
            round_log=RoundLog.from_dict(payload["round_log"]) if payload.get("round_log") else None,
        )
        for frame_payload in payload.get("frames", []):
            frame = FrameAnnotation.from_dict(frame_payload)
            session.frames[frame.frame_index] = frame
        return session

    def save(self, path: Path) -> None:
        import json

        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AnnotationSession":
        import json

        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def export_yolo_labels(self, target_dir: Path) -> None:
        """Export labels in YOLO txt format for all frames."""

        target_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in self.frames.items():
            label_path = target_dir / f"{frame_index:06d}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for box in frame.boxes:
                    if box.label not in LABEL_TO_CLASS_ID:
                        raise ValueError(f"Unknown label '{box.label}' encountered during export")
                    class_id = LABEL_TO_CLASS_ID[box.label]
                    yolo_data = box.as_yolo(self.image_width, self.image_height)
                    handle.write(f"{class_id} {' '.join(f'{value:.6f}' for value in yolo_data)}\n")
