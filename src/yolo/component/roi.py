"""
    ROI helpers for YOLO component-guided scratch segmentation.

    Purpose:
        1. Detect component boxes with the component YOLO model.
        2. Convert selected component boxes into scratch-segmentation ROIs.
        3. Paste ROI masks back to the full image.
        4. Draw component ROI boxes for GUI preview and saved outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from configs.yolo import YOLO_COMPONENT_TASK


@dataclass(frozen=True)
class ComponentBox:
    """
        One component bounding box detected by YOLO.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class RoiBox:
    """
        Region of interest where scratch segmentation should run.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """
            Return ROI width.
        """

        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """
            Return ROI height.
        """

        return self.y2 - self.y1


def load_component_model(model_path: Path) -> YOLO:
    """
        Load YOLO component detector.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO component model not found: {model_path}")

    return YOLO(str(model_path), task=YOLO_COMPONENT_TASK)


def detect_component_boxes(
    yolo_model: YOLO,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> list[ComponentBox]:
    """
        Detect component bounding boxes with YOLO.
    """

    result = yolo_model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    boxes: list[ComponentBox] = []
    if result.boxes is None:
        return boxes

    names = result.names

    """Convert Ultralytics boxes to simple Python dataclasses."""
    for box in result.boxes:
        raw_class = box.cls.item() if box.cls is not None else None
        class_id = int(raw_class) if raw_class is not None else 0

        raw_confidence = box.conf.item() if box.conf is not None else None
        confidence = float(raw_confidence) if raw_confidence is not None else 0.0

        class_name = "component"
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_name))

        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        boxes.append(
            ComponentBox(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return boxes


def select_component_boxes(
    boxes: list[ComponentBox],
    box_mode: str,
) -> list[ComponentBox]:
    """
        Select one best box or all YOLO boxes.
    """

    if not boxes:
        return []

    if box_mode == "all":
        return boxes

    return [max(boxes, key=lambda box: box.confidence)]


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
        Clamp a box to image boundaries.
    """

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2


def component_box_to_roi(
    box: ComponentBox,
    image_shape: tuple[int, ...],
) -> RoiBox:
    """
        Convert one YOLO component box into a scratch-segmentation ROI.
    """

    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = clamp_box(
        x1=box.x1,
        y1=box.y1,
        x2=box.x2,
        y2=box.y2,
        image_width=image_width,
        image_height=image_height,
    )

    return RoiBox(x1=x1, y1=y1, x2=x2, y2=y2)


def build_roi_boxes(
    selected_boxes: list[ComponentBox],
    image_shape: tuple[int, ...],
) -> list[RoiBox]:
    """
        Build scratch-segmentation ROI boxes from selected component boxes.
    """

    return [
        component_box_to_roi(
            box=box,
            image_shape=image_shape,
        )
        for box in selected_boxes
    ]


def paste_roi_mask(
    full_mask: np.ndarray,
    roi_mask: np.ndarray,
    roi_box: RoiBox,
) -> None:
    """
        Paste one ROI mask into the full-size mask.
    """

    target = full_mask[roi_box.y1:roi_box.y2, roi_box.x1:roi_box.x2]

    if target.shape[:2] != roi_mask.shape[:2]:
        roi_mask = cv2.resize(
            roi_mask,
            (roi_box.width, roi_box.height),
            interpolation=cv2.INTER_NEAREST,
        )

    target[roi_mask > 0] = 255


def draw_boxes(
    image: np.ndarray,
    component_boxes: list[ComponentBox],
) -> np.ndarray:
    """
        Draw YOLO component boxes on the output image.
    """

    drawn = image.copy()
    image_height, image_width = drawn.shape[:2]
    box_thickness = max(4, int(round(min(image_width, image_height) / 500)))
    outline_thickness = box_thickness + 4
    font_scale = max(0.8, min(image_width, image_height) / 2200)
    text_thickness = max(2, int(round(box_thickness / 2)))

    """Draw selected YOLO component boxes with a dark outline for GUI visibility."""
    for box in component_boxes:
        top_left = (box.x1, box.y1)
        bottom_right = (box.x2, box.y2)
        cv2.rectangle(
            drawn,
            top_left,
            bottom_right,
            (0, 0, 0),
            outline_thickness,
        )
        cv2.rectangle(
            drawn,
            top_left,
            bottom_right,
            (0, 255, 0),
            box_thickness,
        )

        label = f"{box.class_name} {box.confidence:.2f}"
        label_origin = (box.x1, max(box.y1 - box_thickness * 2, 24))
        cv2.putText(
            drawn,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            drawn,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    return drawn
