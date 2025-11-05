"""Automatic labelling utilities powered by stored YOLO models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

from ..data.annotation_store import AnnotationSession, BoundingBox


@dataclass
class DetectionResult:
    frame_index: int
    boxes: List[BoundingBox]


class AutoLabeller:
    def __init__(self, model_path: Path):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        self.model = YOLO(str(model_path))

    def run_on_video(self, video_path: Path, step: int = 1) -> Iterable[DetectionResult]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if index % step != 0:
                index += 1
                continue
            height, width = frame.shape[:2]
            prediction = self.model(frame)
            boxes: List[BoundingBox] = []
            for result in prediction:
                for box in result.boxes:
                    xyxy = box.xyxy.squeeze().tolist()
                    label_idx = int(box.cls)
                    label = self.model.names[label_idx]
                    x1, y1, x2, y2 = xyxy
                    boxes.append(
                        BoundingBox(
                            x=float(x1),
                            y=float(y1),
                            width=float(x2 - x1),
                            height=float(y2 - y1),
                            label=label,
                            track_id=f"{label}_{index}_{len(boxes)}",
                            confidence=float(box.conf),
                        )
                    )
            yield DetectionResult(frame_index=index, boxes=boxes)
            index += 1
        capture.release()

    def populate_session(self, session: AnnotationSession, step: int = 1) -> AnnotationSession:
        for detection in self.run_on_video(session.video_path, step=step):
            frame = session.ensure_frame(detection.frame_index)
            frame.boxes = detection.boxes
        return session
