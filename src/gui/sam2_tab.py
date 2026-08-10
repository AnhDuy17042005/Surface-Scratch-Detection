"""Qt-based SAM2 scratch labeling tab."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject, QPoint, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "labeling"
DEFAULT_CHECKPOINT = ROOT / "models" / "sam2" / "checkpoint.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_ROI_SIZE = 512
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".webp"}

PROPOSAL_COLOR = QColor(255, 55, 55, 105)
ACCEPTED_COLOR = QColor(50, 220, 90, 105)
POSITIVE_COLOR = QColor(0, 255, 80)
NEGATIVE_COLOR = QColor(255, 60, 60)
BOX_COLOR = QColor(255, 220, 0)


class SAM2Worker(QObject):
    """Own the SAM2 predictor in a background Qt thread."""

    status_changed = Signal(str)
    prediction_ready = Signal(int, object, object, float, object)
    prediction_failed = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self.source_rgb: np.ndarray | None = None
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.roi_box: tuple[int, int, int, int] | None = None

    @Slot(object)
    def set_source_image(self, image_rgb: np.ndarray) -> None:
        """Set the current source image and invalidate its ROI embedding."""

        self.source_rgb = image_rgb
        self.roi_box = None

    @Slot()
    def preload_model(self) -> None:
        """Load the SAM2 model before the first prompt when possible."""

        if self.predictor is not None:
            return

        try:
            start = perf_counter()
            self._ensure_predictor()
            elapsed = perf_counter() - start
            self.status_changed.emit(
                f"SAM2 model ready on {self.device} | load {elapsed:.2f}s"
            )
        except Exception as exc:
            self.status_changed.emit(f"SAM2 model preload failed: {exc}")

    @Slot(int, object, object, object, object, object)
    def predict(
        self,
        request_id: int,
        roi_box: tuple[int, int, int, int],
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
        box: tuple[int, int, int, int] | None,
        previous_logits: np.ndarray | None,
    ) -> None:
        """Run one prompt prediction and return an ROI-local mask."""

        try:
            total_start = perf_counter()
            if self.source_rgb is None:
                raise RuntimeError("No image is loaded.")

            load_start = perf_counter()
            self._ensure_predictor()
            load_time = perf_counter() - load_start

            embedding_time = 0.0
            embedding_status = "cached"
            if self.roi_box != roi_box:
                x1, y1, x2, y2 = roi_box
                roi_image = self.source_rgb[y1:y2, x1:x2]
                self.status_changed.emit("Preparing SAM2 image embedding...")
                embedding_start = perf_counter()
                with torch.inference_mode():
                    self.predictor.set_image(roi_image)
                embedding_time = perf_counter() - embedding_start
                embedding_status = f"{embedding_time:.2f}s"
                self.roi_box = roi_box

            points = positive_points + negative_points
            point_coords = None
            point_labels = None
            if points:
                x1, y1, _, _ = roi_box
                point_coords = np.asarray(
                    [(x - x1, y - y1) for x, y in points],
                    dtype=np.float32,
                )
                point_labels = np.asarray(
                    [1] * len(positive_points) + [0] * len(negative_points),
                    dtype=np.int32,
                )

            roi_box_prompt = None
            if box is not None:
                x1, y1, _, _ = roi_box
                bx1, by1, bx2, by2 = box
                roi_box_prompt = np.asarray(
                    [bx1 - x1, by1 - y1, bx2 - x1, by2 - y1],
                    dtype=np.float32,
                )

            refine = previous_logits is not None
            multimask = not (
                refine
                or len(points) > 1
                or (box is not None and bool(points))
            )
            decode_start = perf_counter()
            with torch.inference_mode():
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=roi_box_prompt,
                    mask_input=previous_logits if refine else None,
                    multimask_output=multimask,
                    return_logits=False,
                    normalize_coords=True,
                )
            decode_time = perf_counter() - decode_start

            best_index = int(np.argmax(scores))
            score = float(scores[best_index])
            self.prediction_ready.emit(
                request_id,
                masks[best_index],
                logits[best_index][None, :, :],
                score,
                roi_box,
            )
            total_time = perf_counter() - total_start
            load_text = f" | load {load_time:.2f}s" if load_time >= 0.05 else ""
            self.status_changed.emit(
                f"SAM2 proposal ready | score {score:.3f}{load_text} | "
                f"embed {embedding_status} | decode {decode_time:.2f}s | "
                f"total {total_time:.2f}s"
            )
        except Exception as exc:  # Report model errors in the GUI instead of killing the thread.
            message = str(exc)
            self.prediction_failed.emit(request_id, message)
            self.status_changed.emit(f"SAM2 inference failed: {message}")

    def _ensure_predictor(self) -> None:
        """Load the fine-tuned predictor once per application session."""

        if self.predictor is not None:
            return

        if not DEFAULT_CHECKPOINT.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {DEFAULT_CHECKPOINT}")

        self.status_changed.emit("Loading SAM2 model...")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.evaluation.sam2 import build_predictor, setup_sam2_import

        setup_sam2_import(None)
        self.predictor = build_predictor(
            DEFAULT_CONFIG,
            DEFAULT_CHECKPOINT,
            self.device,
        )


class SAM2Canvas(QWidget):
    """Interactive Qt canvas for SAM2 point and box prompts."""

    prompt_requested = Signal(object, object, object, object)
    status_changed = Signal(str)
    annotation_changed = Signal()
    sessions_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.image_path: Path | None = None
        self.image_bgr: np.ndarray | None = None
        self.image_rgb: np.ndarray | None = None
        self.image_qt: QImage | None = None
        self.image_w = 0
        self.image_h = 0

        self.mode = "point"
        self.positive_points: list[tuple[int, int]] = []
        self.negative_points: list[tuple[int, int]] = []
        self.point_history: list[tuple[int, tuple[int, int]]] = []
        self.box: tuple[int, int, int, int] | None = None
        self.pending_box_start: tuple[int, int] | None = None
        self.pending_box_end: tuple[int, int] | None = None

        self.proposal_mask: np.ndarray | None = None
        self.proposal_logits: np.ndarray | None = None
        self.proposal_score: float | None = None
        self.active_roi_box: tuple[int, int, int, int] | None = None
        self.proposal_roi_box: tuple[int, int, int, int] | None = None
        self.roi_sessions: list[dict] = []
        self.active_session_index: int | None = None
        self.accepted_masks: list[np.ndarray] = []
        self.accepted_prompts: list[dict] = []
        self.saved_mask: np.ndarray | None = None
        self.initial_mask_source: str | None = None
        self.mask: np.ndarray | None = None
        self.show_mask_overlay = True
        self.brush_size = 2
        self._brush_active = False
        self._brush_history: list[dict] = []
        self._last_edit_action: str | None = None

        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.view_x = 0
        self.view_y = 0
        self._pan_start: QPoint | None = None
        self._pan_view_start: tuple[int, int] | None = None

    def load_image(self, image_path: Path) -> None:
        """Load one image and reset its annotation state."""

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_path = image_path
        self.image_bgr = image
        self.image_rgb = rgb
        self.image_h, self.image_w = image.shape[:2]
        self.image_qt = QImage(
            rgb.data,
            self.image_w,
            self.image_h,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._clear_annotation(emit=False)
        self._clear_brush_history()
        self.sessions_changed.emit()
        self.fit_to_view()
        self.status_changed.emit(f"Loaded {self.image_w}x{self.image_h}")
        self.update()

    def set_saved_mask(self, mask_path: Path | None) -> None:
        """Load an existing output mask for visual review."""

        self.saved_mask = None
        self.initial_mask_source = None
        if mask_path is not None and mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[:2] == (self.image_h, self.image_w):
                self.saved_mask = (mask > 0).astype(np.uint8) * 255
        self._rebuild_mask()
        self._clear_brush_history()
        self.update()

    def set_initial_mask(
        self,
        mask: np.ndarray | None,
        source: str,
    ) -> None:
        """
            Load a model prediction as the editable starting mask.
        """

        self.saved_mask = None
        self.initial_mask_source = None

        if mask is None:
            self._rebuild_mask()
            self.update()
            return

        if mask.shape[:2] != (self.image_h, self.image_w):
            raise ValueError(
                f"Initial mask shape {mask.shape[:2]} does not match "
                f"image shape {(self.image_h, self.image_w)}."
            )

        self.saved_mask = (mask > 0).astype(np.uint8) * 255
        self.initial_mask_source = source
        self._rebuild_mask()
        self._clear_brush_history()
        self.status_changed.emit(
            f"Loaded initial mask from {source} | "
            f"pixels {int((self.saved_mask > 0).sum())}"
        )
        self.update()

    def set_show_mask_overlay(self, enabled: bool) -> None:
        """Show or hide accepted, saved, and proposal masks."""

        self.show_mask_overlay = bool(enabled)
        self.update()

    def set_mode(self, mode: str) -> None:
        """Switch between point and box prompt modes."""

        self.mode = mode
        self.pending_box_start = None
        self.pending_box_end = None
        self.status_changed.emit(f"Mode: {mode}")
        self.update()

    def set_brush_size(self, size: int) -> None:
        """Set the brush/eraser radius in image pixels."""

        self.brush_size = max(1, int(size))
        self._emit_status()

    def fit_to_view(self) -> None:
        """Fit the original image inside the canvas."""

        if not self.has_image():
            return
        self.fit_zoom = min(self.width() / self.image_w, self.height() / self.image_h)
        self.zoom = max(0.01, self.fit_zoom)
        self.view_x = 0
        self.view_y = 0
        self._clamp_view()
        self._emit_status()
        self.update()

    def set_original_zoom(self) -> None:
        """Show the image at original-pixel scale."""

        if self.has_image():
            self.zoom = 1.0
            self._clamp_view()
            self._emit_status()
            self.update()

    def accept_proposal(self) -> None:
        """Merge the current proposal into the accepted mask."""

        if self.proposal_mask is None:
            self.status_changed.emit("Generate a proposal before accepting.")
            return

        self._snapshot_active_session()
        self._deactivate_session()
        self._last_edit_action = "accept"
        self.annotation_changed.emit()
        self.sessions_changed.emit()
        self.status_changed.emit(f"Accepted ROI session {len(self.roi_sessions)}")
        self.update()

    def discard_proposal(self) -> None:
        """Remove only the current proposal and keep its prompts."""

        self._clear_proposal()
        self._snapshot_active_session()
        self._rebuild_mask()
        self._last_edit_action = "discard"
        self.sessions_changed.emit()
        self.status_changed.emit("Proposal discarded.")
        self.update()

    def undo(self) -> None:
        """Undo the latest point, box, proposal, or accepted instance."""

        if self._last_edit_action == "brush" and self._brush_history:
            self._restore_brush_state(self._brush_history.pop())
            self._last_edit_action = "brush" if self._brush_history else None
            self.annotation_changed.emit()
            self.sessions_changed.emit()
            self.status_changed.emit("Brush/Eraser undone.")
            self.update()
            return

        if self.point_history:
            label, point = self.point_history.pop()
            points = self.positive_points if label == 1 else self.negative_points
            for index in range(len(points) - 1, -1, -1):
                if points[index] == point:
                    points.pop(index)
                    break
            self._clear_proposal()
            self._last_edit_action = "point"
        elif self.proposal_mask is not None:
            self._clear_proposal()
            self._last_edit_action = "proposal"
        elif self.box is not None:
            self.box = None
            self._last_edit_action = "box"
        elif self.roi_sessions:
            self.roi_sessions.pop()
            self.active_session_index = None
            self._rebuild_mask()
            self._last_edit_action = "session"
        else:
            return
        self._snapshot_active_session()
        self.annotation_changed.emit()
        self.sessions_changed.emit()
        if self.positive_points or self.box is not None:
            self._request_prediction()
        self._emit_status()
        self.update()

    def clear(self) -> None:
        """Clear prompts, proposals, and accepted masks."""

        self._clear_annotation(emit=True)
        self._clear_brush_history()
        self.sessions_changed.emit()
        self.status_changed.emit("Cleared annotations.")
        self.update()

    def save_outputs(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path]:
        """Save image and binary mask for the training dataset."""

        output_mask = self._merged_output_mask()
        if self.image_path is None or self.image_bgr is None or output_mask is None:
            raise RuntimeError("Create at least one SAM2 mask before saving.")

        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        for directory in (images_dir, masks_dir):
            directory.mkdir(parents=True, exist_ok=True)

        stem = self.image_path.stem
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"

        cv2.imwrite(str(image_path), self.image_bgr)
        cv2.imwrite(str(mask_path), ((output_mask > 0).astype(np.uint8) * 255))
        return image_path, mask_path

    def has_image(self) -> bool:
        """Return whether an image is loaded."""

        return self.image_qt is not None and self.image_w > 0 and self.image_h > 0

    def paintEvent(self, event) -> None:
        """Render image, masks, prompts, and box preview."""

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.has_image():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load an image to start labeling")
            return

        source_rect, target_rect = self._view_rects()
        painter.drawImage(target_rect, self.image_qt, source_rect)
        self._draw_mask_overlay(painter, source_rect, target_rect)
        self._draw_prompts(painter)

    def resizeEvent(self, event) -> None:
        """Keep the viewport valid after canvas resize."""

        super().resizeEvent(event)
        if self.has_image() and self.zoom <= self.fit_zoom:
            self.fit_to_view()
        else:
            self._clamp_view()

    def mousePressEvent(self, event) -> None:
        """Handle point prompts, box start, and panning."""

        if not self.has_image():
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position().toPoint()
            self._pan_view_start = (self.view_x, self.view_y)
            return

        image_point = self._widget_to_image(event.position())
        if self.mode in ("brush", "eraser") and event.button() == Qt.MouseButton.LeftButton:
            self._brush_active = True
            self._push_brush_state()
            self._apply_brush(image_point)
            return

        if self.mode == "box" and event.button() == Qt.MouseButton.LeftButton:
            self._snapshot_active_session()
            self._deactivate_session()
            self.pending_box_start = image_point
            self.pending_box_end = image_point
            self._last_edit_action = "box"
            self.update()
            return

        if self.mode == "point":
            if event.button() == Qt.MouseButton.LeftButton:
                label = 0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                self._add_point(image_point, label)
            elif event.button() == Qt.MouseButton.RightButton:
                self._add_point(image_point, 0)

    def mouseMoveEvent(self, event) -> None:
        """Update box preview or pan viewport."""

        if self._pan_start is not None and self._pan_view_start is not None:
            point = event.position().toPoint()
            dx = int(round((point.x() - self._pan_start.x()) / self.zoom))
            dy = int(round((point.y() - self._pan_start.y()) / self.zoom))
            self.view_x = self._pan_view_start[0] - dx
            self.view_y = self._pan_view_start[1] - dy
            self._clamp_view()
            self.update()
            return

        if self.pending_box_start is not None:
            self.pending_box_end = self._widget_to_image(event.position())
            self.update()
            return

        if self._brush_active and self.mode in ("brush", "eraser"):
            self._apply_brush(self._widget_to_image(event.position()))

    def mouseReleaseEvent(self, event) -> None:
        """Finish panning or a box prompt."""

        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self._pan_view_start = None
            return
        if event.button() == Qt.MouseButton.LeftButton and self._brush_active:
            self._brush_active = False
            return
        if (
            self.mode == "box"
            and event.button() == Qt.MouseButton.LeftButton
            and self.pending_box_start is not None
            and self.pending_box_end is not None
        ):
            self.box = self._normalized_box(self.pending_box_start, self.pending_box_end)
            self.pending_box_start = None
            self.pending_box_end = None
            self._clear_proposal(clear_active_roi=True)
            self.annotation_changed.emit()
            self._request_prediction()
            self.update()

    def wheelEvent(self, event) -> None:
        """Zoom around the mouse cursor."""

        if self.has_image():
            self._zoom_at(event.position(), 1.25 if event.angleDelta().y() > 0 else 0.8)

    def keyPressEvent(self, event) -> None:
        """Handle labeler shortcuts."""

        key = event.key()
        if key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo()
        elif key == Qt.Key.Key_U:
            self.undo()
        elif key == Qt.Key.Key_A:
            self.accept_proposal()
        elif key == Qt.Key.Key_D:
            self.discard_proposal()
        elif key == Qt.Key.Key_C:
            self.clear()
        elif key == Qt.Key.Key_P:
            self.set_mode("point")
        elif key == Qt.Key.Key_B:
            self.set_mode("box")
        elif key == Qt.Key.Key_R:
            self.set_mode("brush")
        elif key == Qt.Key.Key_E:
            self.set_mode("eraser")
        elif key == Qt.Key.Key_0:
            self.fit_to_view()
        elif key == Qt.Key.Key_1:
            self.set_original_zoom()
        elif key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._request_prediction()
        else:
            super().keyPressEvent(event)

    def _apply_brush(self, point: tuple[int, int]) -> None:
        """
            Paint directly on the editable mask draft.
        """

        if self.saved_mask is None:
            self.saved_mask = np.zeros((self.image_h, self.image_w), dtype=np.uint8)

        value = 255 if self.mode == "brush" else 0
        cv2.circle(
            self.saved_mask,
            center=point,
            radius=self.brush_size,
            color=value,
            thickness=-1,
        )

        if self.mode == "eraser":
            for session in self.roi_sessions:
                mask = session.get("mask")
                if mask is not None:
                    cv2.circle(
                        mask,
                        center=point,
                        radius=self.brush_size,
                        color=0,
                        thickness=-1,
                    )
            if self.proposal_mask is not None:
                cv2.circle(
                    self.proposal_mask,
                    center=point,
                    radius=self.brush_size,
                    color=0,
                    thickness=-1,
                )

        self._rebuild_mask()
        self._last_edit_action = "brush"
        self.annotation_changed.emit()
        self.status_changed.emit(
            f"{self.mode.title()} at {point} | size {self.brush_size}"
        )
        self.update()

    def _push_brush_state(self) -> None:
        """
            Save mask state before one brush/eraser stroke.
        """

        self._snapshot_active_session()
        self._brush_history.append(
            {
                "saved_mask": (
                    self.saved_mask.copy()
                    if self.saved_mask is not None
                    else None
                ),
                "proposal_mask": (
                    self.proposal_mask.copy()
                    if self.proposal_mask is not None
                    else None
                ),
                "roi_sessions": self._copy_roi_sessions(),
            }
        )

    def _restore_brush_state(self, state: dict) -> None:
        """
            Restore masks saved before a brush/eraser stroke.
        """

        self.saved_mask = (
            state["saved_mask"].copy()
            if state.get("saved_mask") is not None
            else None
        )
        self.proposal_mask = (
            state["proposal_mask"].copy()
            if state.get("proposal_mask") is not None
            else None
        )
        self.roi_sessions = self._copy_roi_sessions(state.get("roi_sessions", []))
        self._rebuild_mask()

    def _copy_roi_sessions(self, sessions: list[dict] | None = None) -> list[dict]:
        """
            Copy ROI sessions while preserving numpy masks safely.
        """

        source_sessions = self.roi_sessions if sessions is None else sessions
        copied_sessions: list[dict] = []
        for session in source_sessions:
            copied: dict = {}
            for key, value in session.items():
                if hasattr(value, "copy"):
                    copied[key] = value.copy()
                elif isinstance(value, list):
                    copied[key] = value.copy()
                else:
                    copied[key] = value
            copied_sessions.append(copied)

        return copied_sessions

    def _clear_brush_history(self) -> None:
        """
            Clear brush/eraser undo snapshots.
        """

        self._brush_history.clear()
        if self._last_edit_action == "brush":
            self._last_edit_action = None

    def _add_point(self, point: tuple[int, int], label: int) -> None:
        """Append one point and request a multi-point refinement."""

        session_index = self._session_index_for_point(point)
        if session_index is not None and session_index != self.active_session_index:
            self._snapshot_active_session()
            self._load_session(session_index)
            self.status_changed.emit(f"Editing ROI session {session_index + 1}.")
        elif self._point_outside_active_roi(point):
            if label == 0:
                self.status_changed.emit(
                    "Negative point outside current ROI. Left click to start a new ROI."
                )
                return
            self._snapshot_active_session()
            self._clear_current_prompt(remove_active_session=False)
            self._start_new_session()
            self.status_changed.emit("Started a new ROI from outside click.")
        elif self.active_session_index is None:
            if label == 0:
                self.status_changed.emit("Left click to start a new ROI before negative points.")
                return
            self._start_new_session()

        if label == 1:
            self.positive_points.append(point)
        else:
            self.negative_points.append(point)
        self.point_history.append((label, point))

        # Keep proposal logits so the next SAM2 call can refine the mask.
        self._last_edit_action = "point"
        self.annotation_changed.emit()
        self.sessions_changed.emit()
        self._emit_status()
        self.update()
        self._request_prediction()

    def _request_prediction(self) -> None:
        """Emit the current prompt for background SAM2 inference."""

        if not self.positive_points and self.box is None:
            self.status_changed.emit("Add a positive point or draw a box.")
            return
        if self.active_session_index is None:
            self._start_new_session()
        self._snapshot_active_session()
        self.prompt_requested.emit(
            self.positive_points.copy(),
            self.negative_points.copy(),
            self.box,
            self.proposal_logits,
        )

    def set_prediction(
        self,
        mask_roi: np.ndarray,
        logits: np.ndarray,
        score: float,
        roi_box: tuple[int, int, int, int],
    ) -> None:
        """Paste an ROI-local prediction back into full-image coordinates."""

        full_mask = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        x1, y1, x2, y2 = roi_box
        roi_h, roi_w = y2 - y1, x2 - x1
        mask = (mask_roi > 0).astype(np.uint8) * 255
        if mask.shape[:2] != (roi_h, roi_w):
            mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        full_mask[y1:y2, x1:x2] = mask
        self.proposal_mask = full_mask
        self.proposal_logits = logits
        self.proposal_score = score
        self.active_roi_box = roi_box
        self.proposal_roi_box = roi_box
        self._snapshot_active_session()
        self.status_changed.emit(f"Proposal ready | score {score:.3f}")
        self.sessions_changed.emit()
        self.update()

    def select_session(self, index: int) -> None:
        """Switch editing back to one existing ROI session."""

        if not 0 <= index < len(self.roi_sessions):
            return
        if index == self.active_session_index:
            return

        self._snapshot_active_session()
        self._load_session(index)
        self.status_changed.emit(f"Editing ROI session {index + 1}.")

    def _start_new_session(self) -> None:
        """Create an empty ROI session and make it active."""

        self._clear_proposal(clear_active_roi=True)
        self.roi_sessions.append(self._session_from_current_state())
        self.active_session_index = len(self.roi_sessions) - 1
        self._rebuild_mask()
        self.sessions_changed.emit()

    def _deactivate_session(self) -> None:
        """Keep the active session mask but leave edit mode."""

        self._snapshot_active_session()
        self.active_session_index = None
        self.positive_points.clear()
        self.negative_points.clear()
        self.point_history.clear()
        self.box = None
        self.pending_box_start = None
        self.pending_box_end = None
        self._clear_proposal(clear_active_roi=True)
        self._rebuild_mask()
        self.sessions_changed.emit()

    def _load_session(self, index: int) -> None:
        """Restore one ROI session for further prompt refinement."""

        session = self.roi_sessions[index]
        self.active_session_index = index
        self.positive_points = list(session.get("positive_points", []))
        self.negative_points = list(session.get("negative_points", []))
        self.point_history = list(session.get("point_history", []))
        self.box = session.get("box")
        self.proposal_mask = session.get("mask")
        self.proposal_logits = session.get("logits")
        self.proposal_score = session.get("score")
        self.active_roi_box = session.get("roi_box")
        self.proposal_roi_box = session.get("roi_box") if self.proposal_logits is not None else None
        self._rebuild_mask()
        self.sessions_changed.emit()
        self.update()

    def _snapshot_active_session(self) -> None:
        """Persist the current active prompt state into its ROI session."""

        if self.active_session_index is None:
            return
        if not 0 <= self.active_session_index < len(self.roi_sessions):
            return
        self.roi_sessions[self.active_session_index] = self._session_from_current_state()

    def _session_from_current_state(self) -> dict:
        """Build one editable ROI session from the current prompt state."""

        return {
            "roi_box": self.active_roi_box,
            "positive_points": self.positive_points.copy(),
            "negative_points": self.negative_points.copy(),
            "point_history": self.point_history.copy(),
            "box": self.box,
            "mask": self.proposal_mask.copy() if self.proposal_mask is not None else None,
            "logits": self.proposal_logits.copy() if hasattr(self.proposal_logits, "copy") else self.proposal_logits,
            "score": self.proposal_score,
        }

    def _clear_current_prompt(self, remove_active_session: bool = False) -> None:
        """Clear active prompts without removing accepted masks."""

        self.positive_points.clear()
        self.negative_points.clear()
        self.point_history.clear()
        self.box = None
        self.pending_box_start = None
        self.pending_box_end = None
        if remove_active_session and self.active_session_index is not None:
            if 0 <= self.active_session_index < len(self.roi_sessions):
                self.roi_sessions.pop(self.active_session_index)
            self.active_session_index = None
        self._clear_proposal(clear_active_roi=True)

    def _clear_annotation(self, emit: bool) -> None:
        """Clear all current and accepted annotations."""

        self._clear_current_prompt(remove_active_session=False)
        self.roi_sessions.clear()
        self.active_session_index = None
        self.accepted_masks.clear()
        self.accepted_prompts.clear()
        self.saved_mask = None
        self.initial_mask_source = None
        self.mask = None
        if emit:
            self.annotation_changed.emit()

    def _clear_proposal(self, clear_active_roi: bool = False) -> None:
        """Clear proposal mask, score, and refinement logits."""

        self.proposal_mask = None
        self.proposal_logits = None
        self.proposal_score = None
        if clear_active_roi:
            self.active_roi_box = None
        self.proposal_roi_box = None

    def _point_outside_active_roi(self, point: tuple[int, int]) -> bool:
        """Return whether a point should start a new ROI."""

        if self.active_roi_box is None:
            return False

        x, y = point
        x1, y1, x2, y2 = self.active_roi_box
        return not (x1 <= x < x2 and y1 <= y < y2)

    def _session_index_for_point(self, point: tuple[int, int]) -> int | None:
        """Find the latest ROI session containing the image point."""

        if (
            self.active_session_index is not None
            and self.active_roi_box is not None
            and self._point_in_roi(point, self.active_roi_box)
        ):
            return self.active_session_index

        for index in range(len(self.roi_sessions) - 1, -1, -1):
            roi_box = self.roi_sessions[index].get("roi_box")
            if roi_box is not None and self._point_in_roi(point, roi_box):
                return index

        return None

    def _point_in_roi(
        self,
        point: tuple[int, int],
        roi_box: tuple[int, int, int, int],
    ) -> bool:
        """Return whether an image point is inside an ROI box."""

        x, y = point
        x1, y1, x2, y2 = roi_box
        return x1 <= x < x2 and y1 <= y < y2

    def _session_prompt(self, session: dict) -> dict:
        """Convert an editable ROI session into saved prompt metadata."""

        mask = session.get("mask")
        positive_pixels = int((mask > 0).sum()) if mask is not None else 0
        return {
            "roi_box": session.get("roi_box"),
            "positive_points": list(session.get("positive_points", [])),
            "negative_points": list(session.get("negative_points", [])),
            "box": session.get("box"),
            "score": float(session.get("score") or 0.0),
            "positive_pixels": positive_pixels,
        }

    def _rebuild_mask(self) -> None:
        """Merge saved and accepted instance masks."""

        session_masks = [
            session.get("mask")
            for index, session in enumerate(self.roi_sessions)
            if index != self.active_session_index and session.get("mask") is not None
        ]
        self.accepted_masks = [mask for mask in session_masks if mask is not None]
        self.accepted_prompts = [
            self._session_prompt(session)
            for index, session in enumerate(self.roi_sessions)
            if index != self.active_session_index and session.get("mask") is not None
        ]

        if self.saved_mask is None and not session_masks:
            self.mask = None
            return
        merged = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        if self.saved_mask is not None:
            np.maximum(merged, self.saved_mask, out=merged)
        for session_mask in session_masks:
            np.maximum(merged, session_mask, out=merged)
        self.mask = merged

    def _merged_output_mask(self) -> np.ndarray | None:
        """Merge saved masks with every ROI session, including the active one."""

        session_masks = [
            session.get("mask")
            for session in self.roi_sessions
            if session.get("mask") is not None
        ]
        if self.proposal_mask is not None and self.active_session_index is None:
            session_masks.append(self.proposal_mask)

        if self.saved_mask is None and not session_masks:
            return None

        merged = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        if self.saved_mask is not None:
            np.maximum(merged, self.saved_mask, out=merged)
        for session_mask in session_masks:
            np.maximum(merged, session_mask, out=merged)
        return merged

    def _all_session_prompts(self) -> list[dict]:
        """Return prompt metadata for every ROI session with a mask."""

        return [
            self._session_prompt(session)
            for session in self.roi_sessions
            if session.get("mask") is not None
        ]

    def _make_overlay(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Create a red overlay for saved masks."""

        overlay = self.image_bgr.copy()
        display_mask = self.mask if mask is None else mask
        if display_mask is None:
            return overlay
        active = display_mask > 0
        color = np.zeros_like(overlay)
        color[:, :, 2] = 255
        overlay[active] = cv2.addWeighted(overlay[active], 0.65, color[active], 0.35, 0)
        return overlay

    def _draw_mask_overlay(self, painter: QPainter, source: QRectF, target: QRectF) -> None:
        """Paint accepted and proposal masks over the visible image crop."""

        if not self.show_mask_overlay:
            return

        source_x, source_y = int(source.x()), int(source.y())
        source_w, source_h = int(source.width()), int(source.height())
        accepted = np.zeros((source_h, source_w), dtype=np.uint8)
        if self.mask is not None:
            accepted = self.mask[source_y : source_y + source_h, source_x : source_x + source_w]
        proposal = np.zeros((source_h, source_w), dtype=np.uint8)
        if self.proposal_mask is not None:
            proposal = self.proposal_mask[source_y : source_y + source_h, source_x : source_x + source_w]
        if not np.any(accepted) and not np.any(proposal):
            return

        rgba = np.zeros((source_h, source_w, 4), dtype=np.uint8)
        accepted_active = accepted > 0
        proposal_active = proposal > 0
        rgba[accepted_active, :3] = [ACCEPTED_COLOR.red(), ACCEPTED_COLOR.green(), ACCEPTED_COLOR.blue()]
        rgba[accepted_active, 3] = ACCEPTED_COLOR.alpha()
        rgba[proposal_active, :3] = [PROPOSAL_COLOR.red(), PROPOSAL_COLOR.green(), PROPOSAL_COLOR.blue()]
        rgba[proposal_active, 3] = PROPOSAL_COLOR.alpha()
        image = QImage(
            rgba.data,
            source_w,
            source_h,
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()
        painter.drawImage(target, image)

    def _draw_prompts(self, painter: QPainter) -> None:
        """Paint small positive/negative points and the active box."""

        radius = max(1, min(3, int(round(1.5 * self.zoom))))
        for points, color in (
            (self.positive_points, POSITIVE_COLOR),
            (self.negative_points, NEGATIVE_COLOR),
        ):
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            for x, y in points:
                point = self._image_to_widget(x, y)
                painter.drawEllipse(point, radius, radius)

        active_box = self.box
        if self.pending_box_start is not None and self.pending_box_end is not None:
            active_box = self._normalized_box(self.pending_box_start, self.pending_box_end)
        if active_box is not None:
            x1, y1, x2, y2 = active_box
            p1 = self._image_to_widget(x1, y1)
            p2 = self._image_to_widget(x2, y2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(BOX_COLOR)
            painter.drawRect(QRectF(p1, p2))

    def _view_rects(self) -> tuple[QRectF, QRectF]:
        """Return source image and target widget rectangles."""

        visible_w = min(self.image_w - self.view_x, self._visible_image_w())
        visible_h = min(self.image_h - self.view_y, self._visible_image_h())
        source = QRectF(self.view_x, self.view_y, visible_w, visible_h)
        target_w, target_h = visible_w * self.zoom, visible_h * self.zoom
        target = QRectF(
            max(0.0, (self.width() - target_w) / 2.0),
            max(0.0, (self.height() - target_h) / 2.0),
            target_w,
            target_h,
        )
        return source, target

    def _widget_to_image(self, point) -> tuple[int, int]:
        """Convert widget coordinates into original-image coordinates."""

        _, target = self._view_rects()
        x = self.view_x + int(round((point.x() - target.x()) / self.zoom))
        y = self.view_y + int(round((point.y() - target.y()) / self.zoom))
        return int(np.clip(x, 0, self.image_w - 1)), int(np.clip(y, 0, self.image_h - 1))

    def _image_to_widget(self, x: int, y: int) -> QPoint:
        """Convert original-image coordinates into widget coordinates."""

        _, target = self._view_rects()
        return QPoint(
            int(round(target.x() + (x - self.view_x) * self.zoom)),
            int(round(target.y() + (y - self.view_y) * self.zoom)),
        )

    def _visible_image_w(self) -> int:
        return max(1, int(round(max(1, self.width()) / self.zoom)))

    def _visible_image_h(self) -> int:
        return max(1, int(round(max(1, self.height()) / self.zoom)))

    def _clamp_view(self) -> None:
        if not self.has_image():
            return
        self.view_x = int(np.clip(self.view_x, 0, max(0, self.image_w - self._visible_image_w())))
        self.view_y = int(np.clip(self.view_y, 0, max(0, self.image_h - self._visible_image_h())))

    def _zoom_at(self, position, factor: float) -> None:
        """Zoom while keeping the image point under the cursor stable."""

        anchor_x, anchor_y = self._widget_to_image(position)
        old_zoom = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, self.fit_zoom, 16.0))
        if self.zoom == old_zoom:
            return
        self.view_x = anchor_x - int(round(position.x() / self.zoom))
        self.view_y = anchor_y - int(round(position.y() / self.zoom))
        self._clamp_view()
        self._emit_status()
        self.update()

    def _normalized_box(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
        return (
            int(np.clip(min(start[0], end[0]), 0, self.image_w - 1)),
            int(np.clip(min(start[1], end[1]), 0, self.image_h - 1)),
            int(np.clip(max(start[0], end[0]), 0, self.image_w - 1)),
            int(np.clip(max(start[1], end[1]), 0, self.image_h - 1)),
        )

    def _emit_status(self) -> None:
        if not self.has_image():
            self.status_changed.emit("No image loaded")
            return
        self.status_changed.emit(
            f"{self.image_w}x{self.image_h} | mode: {self.mode} | "
            f"positive: {len(self.positive_points)} | negative: {len(self.negative_points)} | "
            f"accepted: {len(self.accepted_masks)} | zoom: {self.zoom:.2f}"
        )


class SAM2Tab(QWidget):
    """SAM2 prompt-labeling page using the same layout style as the old labeler."""

    model_preload_requested = Signal()
    source_image_requested = Signal(object)
    prediction_requested = Signal(int, object, object, object, object, object)

    def __init__(self) -> None:
        super().__init__()
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.image_paths: list[Path] = []
        self.current_index = -1
        self.annotation_dirty = False

        self.canvas = SAM2Canvas()
        self.canvas.status_changed.connect(self._set_status)
        self.canvas.annotation_changed.connect(self._mark_dirty)
        self.canvas.sessions_changed.connect(self._update_session_list)
        self.canvas.prompt_requested.connect(self._request_prediction)

        self.worker_thread = QThread(self)
        self.worker = SAM2Worker()
        self.worker.moveToThread(self.worker_thread)
        self.model_preload_requested.connect(self.worker.preload_model)
        self.source_image_requested.connect(self.worker.set_source_image)
        self.prediction_requested.connect(self.worker.predict)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker.prediction_ready.connect(self._prediction_ready)
        self.worker.prediction_failed.connect(self._prediction_failed)
        self.worker.status_changed.connect(self._worker_status)
        self.worker_thread.start()

        self.image_label = QLabel("No image selected")
        self.label_state = QLabel("Unlabeled")
        self.status_label = QLabel("No image loaded")
        self.sessions_list = QListWidget()

        self.page_title = QLabel("Annotation Scratches")
        self.load_button = QPushButton("Load Image")
        self.load_folder_button = QPushButton("Load Folder")
        self.previous_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.point_button = QPushButton("Point Mode")
        self.box_button = QPushButton("Box Mode")
        self.brush_button = QPushButton("Brush")
        self.eraser_button = QPushButton("Eraser")
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 80)
        self.brush_size_spin.setValue(self.canvas.brush_size)
        self.accept_button = QPushButton("Accept")
        self.discard_button = QPushButton("Discard")
        self.save_button = QPushButton("Save")
        self.undo_button = QPushButton("Undo")
        self.clear_button = QPushButton("Clear")
        self.show_mask_check = QCheckBox("Show mask")
        self.show_mask_check.setChecked(True)

        self._style_widgets()
        self._build_layout()
        self._connect_signals()
        self._install_shortcuts()
        self._set_mode_button("point")
        self._update_navigation_buttons()
        self._update_session_list()

    def _style_widgets(self) -> None:
        """Apply compact labeling-tab styling."""

        self.image_label.setWordWrap(True)
        self.image_label.setMinimumWidth(0)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.sessions_list.setMinimumHeight(150)
        self.sessions_list.setMinimumWidth(0)
        self.sessions_list.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        for button in (
            self.load_button,
            self.load_folder_button,
            self.previous_button,
            self.next_button,
            self.accept_button,
            self.discard_button,
            self.save_button,
            self.undo_button,
            self.clear_button,
        ):
            button.setMinimumHeight(28)
        for button in (
            self.point_button,
            self.box_button,
            self.brush_button,
            self.eraser_button,
        ):
            button.setMinimumHeight(34)
        self.point_button.setCheckable(True)
        self.box_button.setCheckable(True)
        self.brush_button.setCheckable(True)
        self.eraser_button.setCheckable(True)
        self.brush_size_spin.setMinimumHeight(28)

    def _build_layout(self) -> None:
        """Build a two-column SAM2 labeling layout."""

        header = self._build_header()
        canvas_group = self._build_canvas_group()
        tools_group = self._build_tools_group()
        sessions_group = self._build_sessions_group()
        status_group = self._build_status_group()

        side_panel = QVBoxLayout()
        side_panel.setSpacing(8)
        side_panel.addWidget(tools_group)
        side_panel.addWidget(sessions_group, stretch=1)
        side_panel.addWidget(status_group)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        content_layout.addWidget(canvas_group, stretch=5)
        content_layout.addLayout(side_panel, stretch=2)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addLayout(content_layout, stretch=1)
        self.setLayout(layout)

    def _build_header(self) -> QVBoxLayout:
        """Build the page title area."""

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.page_title)
        return layout

    def _build_canvas_group(self) -> QGroupBox:
        """Build the image canvas card."""

        group = QGroupBox("Preview")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.canvas, stretch=1)
        group.setLayout(layout)
        return group

    def _build_tools_group(self) -> QGroupBox:
        """Build SAM2 annotation controls for the right side panel."""

        group = QGroupBox("Annotation Tools")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        state_row = QHBoxLayout()
        state_row.setSpacing(8)
        state_row.addWidget(self.label_state)
        state_row.addWidget(self.show_mask_check)
        state_row.addStretch(1)

        load_row = QHBoxLayout()
        load_row.setSpacing(6)
        load_row.addWidget(self.load_button, stretch=1)
        load_row.addWidget(self.load_folder_button, stretch=1)

        navigation_row = QHBoxLayout()
        navigation_row.setSpacing(6)
        navigation_row.addWidget(self.previous_button, stretch=1)
        navigation_row.addWidget(self.next_button, stretch=1)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(self.point_button, stretch=1)
        mode_row.addWidget(self.box_button, stretch=1)

        mask_edit_row = QHBoxLayout()
        mask_edit_row.setSpacing(6)
        mask_edit_row.addWidget(self.brush_button, stretch=1)
        mask_edit_row.addWidget(self.eraser_button, stretch=1)
        mask_edit_row.addWidget(QLabel("Size"))
        mask_edit_row.addWidget(self.brush_size_spin)

        proposal_row = QHBoxLayout()
        proposal_row.setSpacing(6)
        proposal_row.addWidget(self.accept_button, stretch=1)
        proposal_row.addWidget(self.discard_button, stretch=1)

        edit_row = QHBoxLayout()
        edit_row.setSpacing(6)
        edit_row.addWidget(self.save_button, stretch=1)
        edit_row.addWidget(self.undo_button, stretch=1)
        edit_row.addWidget(self.clear_button, stretch=1)

        layout.addLayout(mode_row)
        layout.addLayout(mask_edit_row)
        layout.addWidget(self.image_label)
        layout.addLayout(state_row)
        layout.addLayout(load_row)
        layout.addLayout(navigation_row)
        layout.addLayout(proposal_row)
        layout.addLayout(edit_row)
        group.setLayout(layout)
        return group

    def _build_sessions_group(self) -> QGroupBox:
        """Build the editable ROI session list."""

        group = QGroupBox("ROI Sessions")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.sessions_list)
        group.setLayout(layout)
        return group

    def _build_status_group(self) -> QGroupBox:
        """Build the SAM2 status card."""

        group = QGroupBox("Status")
        group.setMinimumHeight(170)
        group.setMaximumHeight(190)
        group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        group.setLayout(layout)
        return group

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self.load_image)
        self.load_folder_button.clicked.connect(self.load_folder)
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        self.point_button.clicked.connect(lambda: self._set_mode_button("point"))
        self.box_button.clicked.connect(lambda: self._set_mode_button("box"))
        self.brush_button.clicked.connect(lambda: self._set_mode_button("brush"))
        self.eraser_button.clicked.connect(lambda: self._set_mode_button("eraser"))
        self.brush_size_spin.valueChanged.connect(self.canvas.set_brush_size)
        self.accept_button.clicked.connect(self.canvas.accept_proposal)
        self.discard_button.clicked.connect(self.canvas.discard_proposal)
        self.save_button.clicked.connect(self.save_outputs)
        self.undo_button.clicked.connect(self.canvas.undo)
        self.clear_button.clicked.connect(self.canvas.clear)
        self.show_mask_check.toggled.connect(self.canvas.set_show_mask_overlay)
        self.sessions_list.currentRowChanged.connect(self._select_session)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.load_image)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_outputs)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.canvas.undo)
        QShortcut(QKeySequence("A"), self, activated=self.canvas.accept_proposal)
        QShortcut(QKeySequence("D"), self, activated=self.canvas.discard_proposal)

    def _set_mode_button(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.point_button.setChecked(mode == "point")
        self.box_button.setChecked(mode == "box")
        self.brush_button.setChecked(mode == "brush")
        self.eraser_button.setChecked(mode == "eraser")
        self.canvas.setFocus()

    def _select_session(self, row: int) -> None:
        """Load one ROI session from the side-panel list."""

        if row < 0:
            return
        self.canvas.select_session(row)

    def _update_session_list(self) -> None:
        """Refresh the visible ROI session list."""

        self.sessions_list.blockSignals(True)
        try:
            self.sessions_list.clear()
            for index, session in enumerate(self.canvas.roi_sessions):
                self.sessions_list.addItem(self._session_list_text(index, session))
            if self.canvas.active_session_index is not None:
                self.sessions_list.setCurrentRow(self.canvas.active_session_index)
            else:
                self.sessions_list.setCurrentRow(-1)
        finally:
            self.sessions_list.blockSignals(False)

    def _session_list_text(self, index: int, session: dict) -> str:
        """Format one compact ROI session row."""

        state = "active" if index == self.canvas.active_session_index else "saved"
        positive_count = len(session.get("positive_points", []))
        negative_count = len(session.get("negative_points", []))
        has_box = " Box" if session.get("box") is not None else ""

        mask = session.get("mask")
        pixel_count = int((mask > 0).sum()) if mask is not None else 0
        score = session.get("score")
        score_text = f" | confidence: {float(score):.2f}" if score is not None else ""

        return (
            f"{index + 1}. {state} | P{positive_count} N{negative_count}{has_box} "
            f"| {pixel_count} pixel{score_text}"
        )

    def load_image(self) -> None:
        """Select and load one source image."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select surface image",
            str(ROOT / "data" / "raw"),
            "Images (*.bmp *.png *.jpg *.jpeg *.webp)",
        )
        if not path or not self._confirm_navigation():
            return
        self.image_paths = [Path(path)]
        self.current_index = 0
        self._load_current_image()

    def load_folder(self) -> None:
        """Load a folder and start from its first unlabeled image."""

        folder = QFileDialog.getExistingDirectory(self, "Select image folder", str(ROOT / "data" / "raw"))
        if not folder or not self._confirm_navigation():
            return
        paths = sorted(
            path for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            QMessageBox.warning(self, "Load folder failed", "No image files found.")
            return
        self.image_paths = paths
        self.current_index = self._first_unlabeled_index(paths)
        self._load_current_image()

    def load_prediction_for_labeling(
        self,
        image_path: Path,
        mask: np.ndarray,
        source: str = "yolo_prediction",
    ) -> bool:
        """
            Load a model prediction as an editable annotation draft.
        """

        if not self._confirm_navigation():
            return False

        path = Path(image_path)
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Load prediction failed",
                f"Image not found: {path}",
            )
            return False

        try:
            self._latest_request_id = getattr(self, "_latest_request_id", 0) + 1
            self.image_paths = [path]
            self.current_index = 0
            self.canvas.load_image(path)
            self.canvas.set_initial_mask(mask, source=source)
            self.source_image_requested.emit(self.canvas.image_rgb)
            self.model_preload_requested.emit()
            self.annotation_dirty = True
            self._update_image_label()
            self._update_navigation_buttons()
            self._update_session_list()
            self._set_status(
                "YOLO prediction loaded for review. "
                "Use SAM2 point/box prompts to refine, then Save."
            )
            self.canvas.setFocus()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Load prediction failed", str(exc))
            return False

    def _load_current_image(self) -> None:
        if not 0 <= self.current_index < len(self.image_paths):
            return
        try:
            self._latest_request_id = getattr(self, "_latest_request_id", 0) + 1
            self.canvas.load_image(self.image_paths[self.current_index])
            self._load_saved_mask_overlay()
            self.source_image_requested.emit(self.canvas.image_rgb)
            self.model_preload_requested.emit()
            self.annotation_dirty = False
            self._update_image_label()
            self._update_navigation_buttons()
            self.canvas.setFocus()
        except Exception as exc:
            QMessageBox.critical(self, "Load image failed", str(exc))

    def previous_image(self) -> None:
        target_index = self._previous_navigation_index()
        if target_index is not None and self._confirm_navigation():
            self.current_index = target_index
            self._load_current_image()

    def next_image(self) -> None:
        target_index = self._next_navigation_index()
        if target_index is not None and self._confirm_navigation():
            self.current_index = target_index
            self._load_current_image()

    def save_outputs(self) -> None:
        try:
            image_path, mask_path = self.canvas.save_outputs(self.output_dir)
            self.canvas.set_saved_mask(mask_path)
            self.annotation_dirty = False
            self._update_image_label()
            QMessageBox.information(self, "Saved", f"Image: {image_path}\nMask: {mask_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _request_prediction(
        self,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
        box: tuple[int, int, int, int] | None,
        previous_logits: np.ndarray | None,
    ) -> None:
        """Build a fixed hidden ROI and queue a background prediction."""

        if not self.canvas.has_image():
            return
        roi = self._build_roi(box, positive_points)
        self.canvas.active_roi_box = roi
        self.canvas._snapshot_active_session()

        refine_logits = previous_logits
        if self.canvas.proposal_roi_box != roi:
            refine_logits = None

        self._request_id = getattr(self, "_request_id", 0) + 1
        self._latest_request_id = self._request_id
        self.prediction_requested.emit(
            self._request_id,
            roi,
            positive_points,
            negative_points,
            box,
            refine_logits,
        )
        self._set_status("Predicting SAM2 proposal...")

    def _build_roi(
        self,
        box: tuple[int, int, int, int] | None,
        positive_points: list[tuple[int, int]],
    ) -> tuple[int, int, int, int]:
        """Center the fixed ROI on the active prompt."""

        if box is not None:
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
        else:
            cx, cy = positive_points[0]
        half = SAM2_ROI_SIZE // 2
        x1 = max(0, min(cx - half, self.canvas.image_w - SAM2_ROI_SIZE))
        y1 = max(0, min(cy - half, self.canvas.image_h - SAM2_ROI_SIZE))
        x2 = min(self.canvas.image_w, x1 + SAM2_ROI_SIZE)
        y2 = min(self.canvas.image_h, y1 + SAM2_ROI_SIZE)
        return int(x1), int(y1), int(x2), int(y2)

    @Slot(int, object, object, float, object)
    def _prediction_ready(self, request_id: int, mask, logits, score: float, roi_box) -> None:
        if request_id != getattr(self, "_latest_request_id", -1):
            return
        self.canvas.set_prediction(mask, logits, score, roi_box)

    @Slot(int, str)
    def _prediction_failed(self, request_id: int, message: str) -> None:
        if request_id == getattr(self, "_latest_request_id", -1):
            self._set_status(f"SAM2 error: {message}")

    @Slot(str)
    def _worker_status(self, text: str) -> None:
        if text:
            self._set_status(text)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _mark_dirty(self) -> None:
        self.annotation_dirty = True
        self._latest_request_id = getattr(self, "_latest_request_id", 0) + 1

    def _first_unlabeled_index(self, paths: list[Path]) -> int:
        for index, path in enumerate(paths):
            if not self._mask_path_for(path).is_file():
                return index
        return 0

    def _previous_navigation_index(self) -> int | None:
        """
            Return the previous image index, prioritizing unlabeled images.
        """

        if self.current_index <= 0:
            return None

        unlabeled_index = self._find_unlabeled_index(
            start=self.current_index - 1,
            stop=-1,
            step=-1,
        )
        if unlabeled_index is not None:
            return unlabeled_index

        return self.current_index - 1

    def _next_navigation_index(self) -> int | None:
        """
            Return the next image index, prioritizing unlabeled images.
        """

        if self.current_index >= len(self.image_paths) - 1:
            return None

        unlabeled_index = self._find_unlabeled_index(
            start=self.current_index + 1,
            stop=len(self.image_paths),
            step=1,
        )
        if unlabeled_index is not None:
            return unlabeled_index

        return self.current_index + 1

    def _find_unlabeled_index(
        self,
        start: int,
        stop: int,
        step: int,
    ) -> int | None:
        """
            Find the nearest unlabeled image index in one direction.
        """

        for index in range(start, stop, step):
            if not self._mask_path_for(self.image_paths[index]).is_file():
                return index

        return None

    def _mask_path_for(self, image_path: Path) -> Path:
        """Return expected output mask path for one image."""

        return self.output_dir / "masks" / f"{image_path.stem}.png"

    def _load_saved_mask_overlay(self) -> None:
        """Show the saved mask for the current image if it exists."""

        image_path = self.canvas.image_path
        if image_path is None:
            self.canvas.set_saved_mask(None)
            return
        self.canvas.set_saved_mask(self._mask_path_for(image_path))

    def _update_image_label(self) -> None:
        if not self.canvas.image_path:
            self.image_label.setText("No image selected")
            self._set_label_state(False)
            return
        path = self.canvas.image_path
        try:
            display_path = str(path.relative_to(ROOT))
        except ValueError:
            display_path = str(path)

        is_labeled = self._mask_path_for(path).is_file()
        self._set_label_state(is_labeled)
        if self.image_paths and self.current_index >= 0:
            self.image_label.setText(f"{self.current_index + 1}/{len(self.image_paths)} | {display_path}")
        else:
            self.image_label.setText(display_path)

    def _set_label_state(self, is_labeled: bool) -> None:
        """Update the visible labeled/unlabeled badge."""

        if is_labeled:
            self.label_state.setText("Labeled")
            return
        self.label_state.setText("Unlabeled")

    def _update_navigation_buttons(self) -> None:
        has_folder = len(self.image_paths) > 1
        self.previous_button.setEnabled(
            has_folder and self._previous_navigation_index() is not None
        )
        self.next_button.setEnabled(
            has_folder and self._next_navigation_index() is not None
        )

    def _confirm_navigation(self) -> bool:
        if not self.annotation_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved annotation",
            "Current image has unsaved annotation. Save before moving?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            self.save_outputs()
            return not self.annotation_dirty
        return reply == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        """Stop the background model thread with the tab."""

        self.worker_thread.quit()
        self.worker_thread.wait(3000)
        event.accept()
