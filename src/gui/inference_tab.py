"""
    PySide6 inference tab for standalone YOLO scratch segmentation.

    Purpose:
        1. Load one surface image.
        2. Run YOLO26-seg on the full image or sliding-window tiles.
        3. Show the predicted scratch mask overlay.
        4. Save the overlay image for visual inspection or annotation review.
"""

from __future__ import annotations

import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INFERENCE_LOG_SEPARATOR = "-" * 90

from configs.yolo import (
    YOLO_COMPONENT_CONFIDENCE_THRESHOLD,
    YOLO_COMPONENT_IMAGE_SIZE,
    YOLO_COMPONENT_IOU_THRESHOLD,
    YOLO_COMPONENT_MODEL,
    YOLO_SCRATCH_CONFIDENCE_THRESHOLD,
    YOLO_SCRATCH_DEFAULT_IMAGE,
    YOLO_SCRATCH_IMAGE_SIZE,
    YOLO_SCRATCH_INFERENCE_OUTPUT,
    YOLO_SCRATCH_IOU_THRESHOLD,
    YOLO_SCRATCH_MASK_THRESHOLD,
    YOLO_SCRATCH_MODEL,
    YOLO_SCRATCH_OVERLAY_ALPHA,
    YOLO_SCRATCH_TILE_OVERLAP,
    YOLO_SCRATCH_TILE_SIZE,
)
from src.yolo.scratch.inference import (
    load_image,
    make_red_overlay,
    normalize_ultralytics_model_path,
    predict_full_image,
    predict_sliding_image,
    save_image,
)
from src.yolo.component.roi import (
    build_roi_boxes,
    detect_component_boxes,
    draw_boxes,
    paste_roi_mask,
    select_component_boxes,
)
from ultralytics import YOLO


YOLO_MODEL_CACHE_LIMIT = 2
YOLO_MODEL_CACHE: dict[tuple[str, int, str], Any] = {}
DEFAULT_IMAGE = YOLO_SCRATCH_DEFAULT_IMAGE


def load_yolo_model_cached(model_path: Path, task: str) -> tuple[Any, bool]:
    """
        Load a YOLO model once and reuse it across GUI runs.
    """

    model_path = normalize_ultralytics_model_path(model_path)
    stat = model_path.stat()
    cache_key = (
        str(model_path.resolve()),
        int(stat.st_mtime_ns),
        task,
    )

    if cache_key in YOLO_MODEL_CACHE:
        return YOLO_MODEL_CACHE[cache_key], True

    with redirect_stdout(StringIO()):
        model = YOLO(str(model_path), task=task)

    if len(YOLO_MODEL_CACHE) >= YOLO_MODEL_CACHE_LIMIT:
        oldest_key = next(iter(YOLO_MODEL_CACHE))
        YOLO_MODEL_CACHE.pop(oldest_key, None)

    YOLO_MODEL_CACHE[cache_key] = model
    return model, False


def display_model_run_name(model_path: Path) -> str:
    """
        Return a compact model run name for GUI logs.
    """

    if model_path.name == "best.pt" and model_path.parent.name == "weights":
        return model_path.parent.parent.name

    return model_path.stem


@dataclass
class YoloInferenceSettings:
    """
        Runtime settings collected from the YOLO inference tab.
    """

    image_path: Path
    component_model_path: Path
    model_path: Path
    output_dir: Path
    mode: str
    device_name: str
    imgsz: int
    confidence: float
    iou: float
    component_confidence: float
    mask_threshold: float
    tile_size: int
    overlap: float
    overlay_alpha: float


@dataclass
class YoloInferenceResult:
    """
        Result returned by the background YOLO inference worker.
    """

    image_path: Path
    image: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    component_boxes: list[Any]
    status_text: str


class YoloInferenceWorker(QObject):
    """
        Background worker that runs standalone YOLO scratch segmentation.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: YoloInferenceSettings) -> None:
        super().__init__()
        self.settings = settings

    def run(self) -> None:
        """
            Predict the scratch mask and build an overlay image.
        """

        try:
            total_start = time.perf_counter()
            settings = self.settings

            """Load input image and component detector."""
            image = load_image(settings.image_path)

            component_model_start = time.perf_counter()
            component_model, component_cached = load_yolo_model_cached(
                settings.component_model_path,
                task="detect",
            )
            component_model_seconds = time.perf_counter() - component_model_start

            """Detect one component ROI before scratch segmentation."""
            component_predict_start = time.perf_counter()
            component_boxes = detect_component_boxes(
                yolo_model=component_model,
                image_path=settings.image_path,
                imgsz=YOLO_COMPONENT_IMAGE_SIZE,
                conf=settings.component_confidence,
                iou=YOLO_COMPONENT_IOU_THRESHOLD,
            )
            selected_boxes = select_component_boxes(
                boxes=component_boxes,
                box_mode="best",
            )
            roi_boxes = build_roi_boxes(
                selected_boxes=selected_boxes,
                image_shape=image.shape,
            )
            component_predict_seconds = time.perf_counter() - component_predict_start

            """Run YOLO using the same standalone demo prediction helpers."""
            predict_args = SimpleNamespace(
                mode=settings.mode,
                imgsz=settings.imgsz,
                conf=settings.confidence,
                iou=settings.iou,
                device=None if settings.device_name == "auto" else settings.device_name,
                mask_threshold=settings.mask_threshold,
                tile_size=settings.tile_size,
                overlap=settings.overlap,
            )

            """Skip scratch segmentation when component detector finds no ROI."""
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            scratch_model_seconds = 0.0
            predict_seconds = 0.0
            tiles = 0
            instances = 0
            scratch_cached = False

            if roi_boxes:
                scratch_model_start = time.perf_counter()
                scratch_model, scratch_cached = load_yolo_model_cached(
                    settings.model_path,
                    task="segment",
                )
                scratch_model_seconds = time.perf_counter() - scratch_model_start

                """Run YOLO-seg only inside component ROI boxes."""
                for roi_box in roi_boxes:
                    roi_image = image[roi_box.y1:roi_box.y2, roi_box.x1:roi_box.x2]

                    if settings.mode == "full":
                        roi_mask, roi_instances, roi_predict_seconds = predict_full_image(
                            model=scratch_model,
                            image=roi_image,
                            args=predict_args,
                            box_canvas=None,
                        )
                        roi_tiles = 1
                    else:
                        (
                            roi_mask,
                            roi_instances,
                            roi_predict_seconds,
                            roi_tiles,
                        ) = predict_sliding_image(
                            model=scratch_model,
                            image=roi_image,
                            args=predict_args,
                            box_canvas=None,
                        )

                    paste_roi_mask(
                        full_mask=mask,
                        roi_mask=roi_mask,
                        roi_box=roi_box,
                    )
                    instances += roi_instances
                    predict_seconds += roi_predict_seconds
                    tiles += roi_tiles

            """Build the mask overlay; ROI visibility is handled by the GUI checkbox."""
            overlay = make_red_overlay(
                image=image,
                mask=mask,
                alpha=settings.overlay_alpha,
            )

            total_seconds = time.perf_counter() - total_start
            component_label = "component cached" if component_cached else "component load"
            scratch_label = "scratch cached" if scratch_cached else "scratch load"
            if not roi_boxes:
                scratch_label = "scratch skipped"
            positive_pixels = int((mask > 0).sum())
            tile_average_seconds = predict_seconds / tiles if tiles else 0.0
            device_label = settings.device_name
            scratch_model_name = display_model_run_name(settings.model_path)

            status_text = (
                f"Mode: yolo roi {settings.mode} | "
                f"device {device_label} | "
                f"scratch model {scratch_model_name} | "
                f"{component_label} {component_model_seconds:.2f}s | "
                f"component predict {component_predict_seconds:.2f}s | "
                f"{scratch_label} {scratch_model_seconds:.2f}s | "
                f"predict {predict_seconds:.2f}s | "
                f"component boxes {len(component_boxes)} | "
                f"ROI boxes {len(roi_boxes)} | "
                f"ROI tiles {tiles} | instances {instances} | "
                f"positive pixels {positive_pixels} | "
                f"tile avg {tile_average_seconds:.3f}s | "
                f"total {total_seconds:.2f}s"
            )

            self.finished.emit(
                YoloInferenceResult(
                    image_path=settings.image_path,
                    image=image,
                    mask=mask,
                    overlay=overlay,
                    component_boxes=selected_boxes,
                    status_text=status_text,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class ResultCanvas(QWidget):
    """
        Interactive result canvas with zoom and pan controls.
    """

    def __init__(self) -> None:
        super().__init__()

        """Canvas interaction setup."""
        self.setMinimumSize(760, 560)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        """Loaded display image state."""
        self.image_qt: QImage | None = None
        self.image_w = 0
        self.image_h = 0

        """Viewport state."""
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.view_x = 0
        self.view_y = 0
        self._pan_start: QPoint | None = None
        self._pan_view_start: tuple[int, int] | None = None

    def set_image(self, image_bgr: np.ndarray, preserve_view: bool = False) -> None:
        """
            Convert a BGR numpy image to a Qt image and optionally keep viewport.
        """

        had_same_size = (
            self.has_image()
            and self.image_w == image_bgr.shape[1]
            and self.image_h == image_bgr.shape[0]
        )
        old_zoom = self.zoom
        old_view_x = self.view_x
        old_view_y = self.view_y

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.image_qt = QImage(
            rgb.data,
            w,
            h,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.image_w = w
        self.image_h = h

        if preserve_view and had_same_size:
            self.fit_zoom = min(
                max(1, self.width()) / self.image_w,
                max(1, self.height()) / self.image_h,
            )
            self.zoom = max(old_zoom, self.fit_zoom)
            self.view_x = old_view_x
            self.view_y = old_view_y
            self._clamp_view()
            self.update()
            return

        self.fit_to_view()

    def fit_to_view(self) -> None:
        """
            Fit the full result image inside the current canvas size.
        """

        if not self.has_image():
            return

        self.fit_zoom = min(
            max(1, self.width()) / self.image_w,
            max(1, self.height()) / self.image_h,
        )
        self.zoom = self.fit_zoom
        self.view_x = 0
        self.view_y = 0
        self._clamp_view()
        self.update()

    def has_image(self) -> bool:
        """
            Return whether the canvas has a valid display image.
        """

        return self.image_qt is not None and self.image_w > 0 and self.image_h > 0

    def paintEvent(self, event) -> None:
        """
            Render the current result viewport.
        """

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if not self.has_image():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load an image to start YOLO inference",
            )
            return

        source_rect, target_rect = self._view_rects()
        painter.drawImage(target_rect, self.image_qt, source_rect)

    def resizeEvent(self, event) -> None:
        """
            Keep the viewport valid when the canvas size changes.
        """

        super().resizeEvent(event)
        if self.has_image() and self.zoom < self.fit_zoom:
            self.fit_to_view()
        else:
            self._clamp_view()

    def mousePressEvent(self, event) -> None:
        """
            Start panning the result viewport.
        """

        if not self.has_image():
            return

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._pan_start = event.position().toPoint()
            self._pan_view_start = (self.view_x, self.view_y)

    def mouseMoveEvent(self, event) -> None:
        """
            Update viewport position while panning.
        """

        if self._pan_start is None or self._pan_view_start is None:
            return

        point = event.position().toPoint()
        dx = int(round((point.x() - self._pan_start.x()) / self.zoom))
        dy = int(round((point.y() - self._pan_start.y()) / self.zoom))
        start_x, start_y = self._pan_view_start
        self.view_x = start_x - dx
        self.view_y = start_y - dy
        self._clamp_view()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        """
            Stop the active pan gesture.
        """

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._pan_start = None
            self._pan_view_start = None

    def wheelEvent(self, event) -> None:
        """
            Zoom around the mouse cursor.
        """

        if not self.has_image():
            return

        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom_at(event.position(), factor)

    def _view_rects(self) -> tuple[QRectF, QRectF]:
        """
            Return source image rect and centered widget target rect.
        """

        visible_w = min(self.image_w - self.view_x, self._visible_image_w())
        visible_h = min(self.image_h - self.view_y, self._visible_image_h())
        source = QRectF(self.view_x, self.view_y, visible_w, visible_h)

        target_w = visible_w * self.zoom
        target_h = visible_h * self.zoom
        offset_x = max(0.0, (self.width() - target_w) / 2.0)
        offset_y = max(0.0, (self.height() - target_h) / 2.0)
        target = QRectF(offset_x, offset_y, target_w, target_h)
        return source, target

    def _widget_to_image(self, point) -> tuple[int, int]:
        """
            Convert a widget mouse point to original image coordinates.
        """

        _, target_rect = self._view_rects()
        image_x = self.view_x + int(round((point.x() - target_rect.x()) / self.zoom))
        image_y = self.view_y + int(round((point.y() - target_rect.y()) / self.zoom))
        return (
            int(np.clip(image_x, 0, self.image_w - 1)),
            int(np.clip(image_y, 0, self.image_h - 1)),
        )

    def _visible_image_w(self) -> int:
        """
            Compute visible width in original image pixels.
        """

        return max(1, int(round(max(1, self.width()) / self.zoom)))

    def _visible_image_h(self) -> int:
        """
            Compute visible height in original image pixels.
        """

        return max(1, int(round(max(1, self.height()) / self.zoom)))

    def _clamp_view(self) -> None:
        """
            Keep the viewport inside image boundaries.
        """

        if not self.has_image():
            return

        max_x = max(0, self.image_w - self._visible_image_w())
        max_y = max(0, self.image_h - self._visible_image_h())
        self.view_x = int(np.clip(self.view_x, 0, max_x))
        self.view_y = int(np.clip(self.view_y, 0, max_y))

    def _zoom_at(self, widget_pos, factor: float) -> None:
        """
            Apply zoom while keeping the cursor's image point stable.
        """

        anchor_x, anchor_y = self._widget_to_image(widget_pos)
        old_zoom = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, self.fit_zoom, 16.0))
        if self.zoom == old_zoom:
            return

        self.view_x = anchor_x - int(round(widget_pos.x() / self.zoom))
        self.view_y = anchor_y - int(round(widget_pos.y() / self.zoom))
        self._clamp_view()
        self.update()


class YoloInferenceTab(QWidget):
    """
        Full YOLO inference tab containing configuration, preview, and save controls.
    """

    edit_prediction_requested = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()

        """Background worker state."""
        self.thread: QThread | None = None
        self.worker: YoloInferenceWorker | None = None
        self.result: YoloInferenceResult | None = None
        self.loaded_image: np.ndarray | None = None

        """Path controls."""
        self.image_edit = QLineEdit(str(DEFAULT_IMAGE))
        self.component_model_edit = QLineEdit(str(YOLO_COMPONENT_MODEL))
        self.model_edit = QLineEdit(str(YOLO_SCRATCH_MODEL))
        self.output_dir_edit = QLineEdit(str(YOLO_SCRATCH_INFERENCE_OUTPUT))

        """Inference configuration controls."""
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["sliding", "full"])

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText("auto")

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(YOLO_SCRATCH_IMAGE_SIZE)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(YOLO_SCRATCH_CONFIDENCE_THRESHOLD)

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(32, 4096)
        self.tile_size_spin.setSingleStep(32)
        self.tile_size_spin.setValue(YOLO_SCRATCH_TILE_SIZE)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.95)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setValue(YOLO_SCRATCH_TILE_OVERLAP)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(YOLO_SCRATCH_OVERLAY_ALPHA)

        self.draw_boxes_check = QCheckBox("Show ROI")
        self.draw_boxes_check.setChecked(True)
        self.show_prediction_check = QCheckBox("Show prediction")
        self.show_prediction_check.setChecked(True)

        """Preview and log widgets."""
        self.preview = ResultCanvas()
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("No YOLO inference result yet.")

        """Action buttons."""
        self.load_button = QPushButton("Load Image")
        self.component_model_button = QPushButton("Browse")
        self.model_button = QPushButton("Browse")
        self.output_button = QPushButton("Browse")
        self.run_button = QPushButton("Run Inference")
        self.save_button = QPushButton("Save Output")
        self.save_button.setEnabled(False)
        self.edit_prediction_button = QPushButton("Edit Annotation")
        self.edit_prediction_button.setEnabled(False)

        self._style_widgets()
        self._build_layout()
        self._connect_signals()
        self._update_mode_controls()

    def _style_widgets(self) -> None:
        """
            Apply tab-local visual styling.
        """

        controls = (
            self.image_edit,
            self.component_model_edit,
            self.model_edit,
            self.output_dir_edit,
            self.mode_combo,
            self.device_combo,
            self.imgsz_spin,
            self.conf_spin,
            self.tile_size_spin,
            self.overlap_spin,
            self.alpha_spin,
        )
        for widget in controls:
            widget.setMinimumHeight(28)

        for button in (
            self.load_button,
            self.component_model_button,
            self.model_button,
            self.output_button,
            self.run_button,
            self.save_button,
            self.edit_prediction_button,
        ):
            button.setMinimumHeight(28)

        path_button_width = 104
        for button in (
            self.load_button,
            self.component_model_button,
            self.model_button,
            self.output_button,
        ):
            button.setFixedWidth(path_button_width)

        for button in (
            self.run_button,
            self.save_button,
            self.edit_prediction_button,
        ):
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

    def _build_layout(self) -> None:
        """
            Build the full YOLO inference tab layout.
        """

        header = self._build_header()
        path_group = self._build_path_group()
        action_group = self._build_action_group()
        config_group = self._build_config_group()
        log_group = self._build_log_group()

        side_panel = QVBoxLayout()
        side_panel.setSpacing(8)
        side_panel.addWidget(path_group)
        side_panel.addWidget(action_group)
        side_panel.addWidget(config_group)
        side_panel.addWidget(log_group, stretch=1)

        preview_group = self._build_preview_group()
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        content_layout.addWidget(preview_group, stretch=5)
        content_layout.addLayout(side_panel, stretch=2)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addLayout(content_layout, stretch=1)
        self.setLayout(layout)

    def _build_header(self) -> QVBoxLayout:
        """
            Build the page title area.
        """

        title = QLabel("Run YOLO Inference")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(title)
        return layout

    def _build_path_group(self) -> QGroupBox:
        """
            Build image/model/output path controls.
        """

        group = QGroupBox("Inputs")
        layout = QVBoxLayout()
        layout.setSpacing(5)

        image_row = QHBoxLayout()
        image_row.setSpacing(6)
        image_row.addWidget(self.image_edit, stretch=1)
        image_row.addWidget(self.load_button)

        component_model_row = QHBoxLayout()
        component_model_row.setSpacing(6)
        component_model_row.addWidget(self.component_model_edit, stretch=1)
        component_model_row.addWidget(self.component_model_button)

        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(self.model_edit, stretch=1)
        model_row.addWidget(self.model_button)

        output_row = QHBoxLayout()
        output_row.setSpacing(6)
        output_row.addWidget(self.output_dir_edit, stretch=1)
        output_row.addWidget(self.output_button)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        form.addRow("Image", image_row)
        form.addRow("Component model", component_model_row)
        form.addRow("Scratch model", model_row)
        form.addRow("Output dir", output_row)

        layout.addLayout(form)
        group.setLayout(layout)
        return group

    def _build_action_group(self) -> QGroupBox:
        """
            Build inference action controls as a compact card.
        """

        group = QGroupBox("Actions")
        layout = QHBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.run_button, stretch=1)
        layout.addWidget(self.save_button, stretch=1)
        layout.addWidget(self.edit_prediction_button, stretch=1)
        group.setLayout(layout)
        return group

    def _build_config_group(self) -> QGroupBox:
        """
            Build YOLO inference configuration controls.
        """

        group = QGroupBox("Inference Config")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        form.addRow("Mode", self.mode_combo)
        form.addRow("Device", self.device_combo)
        form.addRow("Image size", self.imgsz_spin)
        form.addRow("Scratch conf", self.conf_spin)
        form.addRow("Tile size", self.tile_size_spin)
        form.addRow("Overlap", self.overlap_spin)
        form.addRow("Overlay alpha", self.alpha_spin)

        display_row = QHBoxLayout()
        display_row.setSpacing(12)
        display_row.addWidget(self.draw_boxes_check)
        display_row.addWidget(self.show_prediction_check)
        display_row.addStretch(1)
        form.addRow("", display_row)
        group.setLayout(form)
        return group

    def _build_preview_group(self) -> QGroupBox:
        """
            Build overlay preview card.
        """

        group = QGroupBox("Overlay Preview")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.preview, stretch=1)
        group.setLayout(layout)
        return group

    def _build_log_group(self) -> QGroupBox:
        """
            Build inference log card.
        """

        group = QGroupBox("Inference Log")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.log_edit, stretch=1)
        group.setLayout(layout)
        return group

    def _connect_signals(self) -> None:
        """
            Connect user actions to tab behavior.
        """

        self.load_button.clicked.connect(self.load_image)
        self.component_model_button.clicked.connect(self.browse_component_model)
        self.model_button.clicked.connect(self.browse_model)
        self.output_button.clicked.connect(self.browse_output_dir)
        self.run_button.clicked.connect(self.run_inference)
        self.save_button.clicked.connect(self.save_output)
        self.edit_prediction_button.clicked.connect(self.edit_prediction)
        self.mode_combo.currentTextChanged.connect(self._update_mode_controls)
        self.draw_boxes_check.toggled.connect(self._update_preview_image)
        self.show_prediction_check.toggled.connect(self._update_preview_image)

    def load_image(self) -> None:
        """
            Select and preview one input image.
        """

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select surface image",
            str(ROOT / "data" / "raw"),
            "Images (*.bmp *.png *.jpg *.jpeg *.webp)",
        )
        if not image_path:
            return

        self.image_edit.setText(image_path)
        self._load_preview_image(Path(image_path))

    def browse_component_model(self) -> None:
        """
            Select a YOLO component detector checkpoint.
        """

        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO component model",
            str(ROOT / "models" / "yolo" / "component"),
            "Models (*.pt *.onnx *.xml);;All files (*)",
        )
        if model_path:
            self.component_model_edit.setText(model_path)

    def browse_model(self) -> None:
        """
            Select a YOLO segmentation checkpoint.
        """

        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO segmentation model",
            str(ROOT / "models" / "yolo"),
            "Models (*.pt *.onnx *.xml);;All files (*)",
        )
        if model_path:
            self.model_edit.setText(model_path)

    def browse_output_dir(self) -> None:
        """
            Select output directory for saved overlay images.
        """

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.output_dir_edit.text() or str(YOLO_SCRATCH_INFERENCE_OUTPUT),
        )
        if output_dir:
            self.output_dir_edit.setText(output_dir)

    def run_inference(self) -> None:
        """
            Validate settings and run inference in the background.
        """

        if self.thread is not None:
            return

        try:
            settings = self._collect_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid inference settings", str(exc))
            return

        self.result = None
        self.save_button.setEnabled(False)
        self.edit_prediction_button.setEnabled(False)
        self._set_log("Running YOLO inference...")
        self._set_running(True)

        self.thread = QThread(self)
        self.worker = YoloInferenceWorker(settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._inference_finished)
        self.worker.failed.connect(self._inference_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def save_output(self) -> None:
        """
            Save the latest YOLO overlay result.
        """

        if self.result is None:
            QMessageBox.warning(self, "Save failed", "Run inference before saving.")
            return

        output_dir = self._resolve_path(self.output_dir_edit.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.result.image_path.stem}_yolo_prediction.jpg"

        try:
            output_image = self.result.overlay.copy()
            if self.draw_boxes_check.isChecked():
                output_image = draw_boxes(
                    image=output_image,
                    component_boxes=self.result.component_boxes,
                )

            save_image(output_path, output_image)
            self._append_log(f"Saved output: {output_path}", separate=True)
            QMessageBox.information(self, "Saved", f"Output: {output_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def edit_prediction(self) -> None:
        """
            Send the latest YOLO prediction to the SAM2 labeling tab.
        """

        if self.result is None:
            QMessageBox.warning(
                self,
                "Edit prediction failed",
                "Run inference before editing a prediction.",
            )
            return

        self.edit_prediction_requested.emit(
            self.result.image_path,
            self.result.mask.copy(),
        )
        self._append_log(
            "Sent YOLO prediction to Annotation tab for review.",
            separate=True,
        )

    def _collect_settings(self) -> YoloInferenceSettings:
        """
            Collect and validate current UI settings.
        """

        image_path = self._resolve_path(self.image_edit.text())
        component_model_path = normalize_ultralytics_model_path(
            self._resolve_path(self.component_model_edit.text())
        )
        model_path = normalize_ultralytics_model_path(
            self._resolve_path(self.model_edit.text())
        )
        output_dir = self._resolve_path(self.output_dir_edit.text())

        if not image_path.is_file():
            raise ValueError(f"Image not found: {image_path}")

        if not component_model_path.exists():
            raise ValueError(f"Component model not found: {component_model_path}")

        if not model_path.exists():
            raise ValueError(f"Scratch model not found: {model_path}")

        if not 0.0 <= self.overlap_spin.value() < 1.0:
            raise ValueError("Overlap must be in [0, 1).")

        return YoloInferenceSettings(
            image_path=image_path,
            component_model_path=component_model_path,
            model_path=model_path,
            output_dir=output_dir,
            mode=self.mode_combo.currentText(),
            device_name=self.device_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            confidence=self.conf_spin.value(),
            iou=YOLO_SCRATCH_IOU_THRESHOLD,
            component_confidence=YOLO_COMPONENT_CONFIDENCE_THRESHOLD,
            mask_threshold=YOLO_SCRATCH_MASK_THRESHOLD,
            tile_size=self.tile_size_spin.value(),
            overlap=self.overlap_spin.value(),
            overlay_alpha=self.alpha_spin.value(),
        )

    def _load_preview_image(self, image_path: Path) -> None:
        """
            Load selected image into preview before inference.
        """

        try:
            image = load_image(image_path)
            self.loaded_image = image
            self.result = None
            self.save_button.setEnabled(False)
            self.edit_prediction_button.setEnabled(False)
            self.preview.set_image(image)
            self._set_log(f"Loaded image: {image_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load image failed", str(exc))

    def _inference_finished(self, result: YoloInferenceResult) -> None:
        """
            Display overlay and enable saving.
        """

        self.result = result
        self._update_preview_image()
        self._set_log(result.status_text)
        self.save_button.setEnabled(True)
        self.edit_prediction_button.setEnabled(True)

    def _update_preview_image(self) -> None:
        """
            Toggle between the original image and predicted mask overlay.
        """

        if self.result is not None:
            preview = (
                self.result.overlay.copy()
                if self.show_prediction_check.isChecked()
                else self.result.image.copy()
            )

            if self.draw_boxes_check.isChecked():
                preview = draw_boxes(
                    image=preview,
                    component_boxes=self.result.component_boxes,
                )

            self.preview.set_image(preview, preserve_view=True)
            return

        if self.loaded_image is not None:
            self.preview.set_image(self.loaded_image, preserve_view=True)

    def _inference_failed(self, message: str) -> None:
        """
            Show inference error.
        """

        self._set_log(f"Error: {message}")
        QMessageBox.critical(self, "Inference failed", message)

    def _thread_finished(self) -> None:
        """
            Clear worker thread references and restore buttons.
        """

        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self._set_running(False)

    def _update_mode_controls(self) -> None:
        """
            Enable sliding-window settings only in sliding mode.
        """

        sliding = self.mode_combo.currentText() == "sliding"
        self.tile_size_spin.setEnabled(sliding)
        self.overlap_spin.setEnabled(sliding)

    def _set_running(self, running: bool) -> None:
        """
            Toggle action buttons while inference is active.
        """

        self.run_button.setEnabled(not running)
        self.load_button.setEnabled(not running)
        self.component_model_button.setEnabled(not running)
        self.model_button.setEnabled(not running)
        self.output_button.setEnabled(not running)
        self.edit_prediction_button.setEnabled(
            not running and self.result is not None
        )

    def _resolve_path(self, value: str) -> Path:
        """
            Convert user text into an absolute path.
        """

        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path

    def _append_log(
        self,
        text: str,
        separate: bool = False,
    ) -> None:
        """
            Append one line to the inference log.
        """

        if not text:
            return

        if separate:
            self._append_log_separator()

        self.log_edit.appendPlainText(text)

    def _append_log_separator(self) -> None:
        """
            Separate action logs from the previous inference/status block.
        """

        current_text = self.log_edit.toPlainText().rstrip()
        if not current_text:
            return

        if current_text.endswith(INFERENCE_LOG_SEPARATOR):
            return

        self.log_edit.appendPlainText(INFERENCE_LOG_SEPARATOR)

    def _set_log(self, text: str) -> None:
        """
            Replace inference log content with the current status.
        """

        self.log_edit.setPlainText(text)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = YoloInferenceTab()
    window.resize(1200, 850)
    window.show()
    sys.exit(app.exec())
