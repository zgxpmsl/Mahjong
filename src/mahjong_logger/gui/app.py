"""PySide6 based GUI for managing videos, annotations and training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import DEFAULT_TILE_CLASSES, ProjectPaths
from ..data.annotation_store import AnnotationSession, BoundingBox
from ..data.manager import DatasetManager
from ..models.yolo_manager import ModelMetadata, ModelRegistry
from ..pipeline.auto_label import AutoLabeller
from ..pipeline.training import Trainer, TrainingConfig


@dataclass
class VideoState:
    capture: cv2.VideoCapture
    total_frames: int
    fps: float


class VideoCanvas(QtWidgets.QLabel):
    """Widget that renders frames and overlays bounding boxes."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._boxes: list[BoundingBox] = []

    def set_frame(self, frame: np.ndarray, boxes: list[BoundingBox]) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        image = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._boxes = boxes
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # pragma: no cover - GUI paint
        super().paintEvent(event)
        if not self._pixmap:
            return
        painter = QtGui.QPainter(self)
        pixmap = self._pixmap.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        pixmap_rect = QtCore.QRect(QtCore.QPoint(0, 0), pixmap.size())
        pixmap_rect.moveCenter(self.rect().center())
        painter.drawPixmap(pixmap_rect.topLeft(), pixmap)
        if self._boxes:
            scale_x = pixmap_rect.width() / self._pixmap.width()
            scale_y = pixmap_rect.height() / self._pixmap.height()
            offset_x = pixmap_rect.left()
            offset_y = pixmap_rect.top()
            pen = QtGui.QPen(QtGui.QColor("#00FFAA"))
            pen.setWidth(2)
            painter.setPen(pen)
            for box in self._boxes:
                rect = QtCore.QRectF(
                    offset_x + box.x * scale_x,
                    offset_y + box.y * scale_y,
                    box.width * scale_x,
                    box.height * scale_y,
                )
                painter.drawRect(rect)
                label = f"{box.track_id}: {box.label}"
                painter.drawText(rect.topLeft() + QtCore.QPointF(0, -4), label)
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, paths: ProjectPaths, registry: ModelRegistry, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.registry = registry
        self.dataset_manager = DatasetManager(paths)
        self.session: Optional[AnnotationSession] = None
        self.video_state: Optional[VideoState] = None
        self.current_frame_index: int = 0
        self.current_frame: Optional[np.ndarray] = None
        self.loaded_model: Optional[ModelMetadata] = None

        self.setWindowTitle("Mahjong Logger Studio")
        self.resize(1280, 800)

        self.canvas = VideoCanvas()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Track ID", "Label", "X", "Y", "W", "H"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.on_table_changed)

        self.status_bar = self.statusBar()

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction("Open Video", self.open_video)
        toolbar.addAction("Load Session", self.load_session)
        toolbar.addAction("Save Session", self.save_session)
        toolbar.addSeparator()
        toolbar.addAction("Import Model", self.import_model)
        toolbar.addAction("Auto Label", self.auto_label)
        toolbar.addSeparator()
        toolbar.addAction("Train", self.train_model)

        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.addWidget(self.canvas, stretch=5)
        layout.addWidget(self.frame_slider)
        layout.addWidget(self.table, stretch=3)
        self.setCentralWidget(central_widget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)

    # --- Video handling -------------------------------------------------
    def open_video(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", str(self.paths.root), "Video Files (*.mp4 *.mov *.mkv)")
        if not file_name:
            return
        capture = cv2.VideoCapture(file_name)
        if not capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Unable to open video file")
            return
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_state = VideoState(capture=capture, total_frames=total_frames, fps=fps)
        self.frame_slider.setMaximum(max(total_frames - 1, 0))
        self.current_frame_index = 0
        success, frame = capture.read()
        if not success:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to read first frame")
            return
        height, width = frame.shape[:2]
        self.session = AnnotationSession(video_path=Path(file_name), image_width=width, image_height=height)
        self.current_frame = frame
        self.update_frame()
        self.status_bar.showMessage(f"Loaded video {file_name} ({total_frames} frames @ {fps:.2f} fps)")

    def next_frame(self) -> None:
        if not self.video_state:
            return
        next_index = self.current_frame_index + 1
        if next_index >= self.video_state.total_frames:
            self.timer.stop()
            return
        self.show_frame(next_index)

    def show_frame(self, index: int) -> None:
        if not self.video_state:
            return
        capture = self.video_state.capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if not success:
            return
        self.current_frame_index = index
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self.current_frame = frame
        self.update_frame()

    def on_slider_changed(self, value: int) -> None:
        self.show_frame(value)

    # --- Annotation table ------------------------------------------------
    def update_frame(self) -> None:
        boxes = []
        if self.session:
            frame = self.session.frames.get(self.current_frame_index)
            if frame:
                boxes = frame.boxes
        if self.current_frame is not None:
            self.canvas.set_frame(self.current_frame, boxes)
        self.populate_table(boxes)

    def populate_table(self, boxes: list[BoundingBox]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(boxes))
        for row, box in enumerate(boxes):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(box.track_id))
            label_item = QtWidgets.QTableWidgetItem(box.label)
            label_item.setFlags(label_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, label_item)
            for col, value in enumerate([box.x, box.y, box.width, box.height], start=2):
                item = QtWidgets.QTableWidgetItem(f"{value:.2f}")
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)

    def on_table_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if not self.session:
            return
        frame = self.session.ensure_frame(self.current_frame_index)
        if item.column() == 1:
            label = item.text()
            if label not in DEFAULT_TILE_CLASSES:
                QtWidgets.QMessageBox.warning(self, "Unknown label", f"Label '{label}' is not in the tile vocabulary")
                return
            frame.boxes[item.row()].label = label
        elif item.column() == 0:
            frame.boxes[item.row()].track_id = item.text()
        self.update_frame()

    # --- Session management ---------------------------------------------
    def load_session(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", str(self.paths.data_root), "JSON Files (*.json)")
        if not file_name:
            return
        session = AnnotationSession.load(Path(file_name))
        capture = cv2.VideoCapture(str(session.video_path))
        if not capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Unable to open session video")
            return
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_state = VideoState(capture, total_frames, fps)
        self.session = session
        self.frame_slider.setMaximum(max(total_frames - 1, 0))
        self.current_frame_index = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()
        if success:
            self.current_frame = frame
        self.update_frame()
        self.status_bar.showMessage(f"Loaded session {file_name}")

    def save_session(self) -> None:
        if not self.session:
            QtWidgets.QMessageBox.information(self, "No session", "Please load or create a session first")
            return
        target_dir = self.paths.data_root / "annotations"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save session",
            str(target_dir / f"{self.session.video_path.stem}.json"),
            "JSON Files (*.json)",
        )
        if not file_name:
            return
        self.session.save(Path(file_name))
        self.status_bar.showMessage(f"Saved session to {file_name}")

    # --- Model integration ----------------------------------------------
    def import_model(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import YOLO model", str(self.paths.root), "YOLO Weights (*.pt *.onnx *.engine)")
        if not file_name:
            return
        metadata = self.registry.import_model(Path(file_name))
        self.loaded_model = metadata
        QtWidgets.QMessageBox.information(self, "Model imported", f"Registered model {metadata.name} ({metadata.version})")

    def auto_label(self) -> None:
        if not self.session:
            QtWidgets.QMessageBox.information(self, "No session", "Load a video before auto labelling")
            return
        if not self.loaded_model:
            QtWidgets.QMessageBox.information(self, "No model", "Import or select a model first")
            return
        labeller = AutoLabeller(self.loaded_model.path)
        labeller.populate_session(self.session, step=1)
        self.update_frame()
        QtWidgets.QMessageBox.information(self, "Auto label", "Finished generating labels")

    def train_model(self) -> None:
        if not self.session:
            QtWidgets.QMessageBox.warning(self, "No data", "Annotate at least one session before training")
            return
        base_model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select base YOLO model",
            str(self.paths.model_root),
            "YOLO Weights (*.pt)",
        )
        if not base_model_path:
            return
        epochs, ok = QtWidgets.QInputDialog.getInt(self, "Epochs", "Training epochs", 50, 1, 300)
        if not ok:
            return
        trainer = Trainer(self.paths, self.registry)
        dataset_yaml = trainer.prepare_dataset([self.session])
        metadata = trainer.train(
            TrainingConfig(
                base_model=Path(base_model_path),
                epochs=epochs,
                batch_size=8,
                img_size=max(self.session.image_width, self.session.image_height),
            ),
            dataset_yaml,
        )
        self.loaded_model = metadata
        QtWidgets.QMessageBox.information(self, "Training complete", f"New model stored as {metadata.name} ({metadata.version})")


def launch_gui(paths: ProjectPaths, registry: ModelRegistry) -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow(paths, registry)
    window.show()
    app.exec()
