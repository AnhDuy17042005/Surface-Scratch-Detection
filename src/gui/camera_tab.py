"""
    PySide6 camera tab for HIKROBOT GigE image capture.

    Purpose:
        1. Enumerate and connect HIKROBOT/Hikvision GigE cameras through MVS.
        2. Configure common camera parameters from the GUI.
        3. Show live frames with zoom and pan support.
        4. Capture images into data/raw with a timestamped filename.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRectF, QSettings, QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_IMAGE_DIR = ROOT / "data" / "raw"
CAPTURE_FORMATS = ("bmp", "png", "jpg")
MVS_IMPORT_DIRS = (
    Path("/opt/MVS/Samples/64/Python/MvImport"),
    Path("/opt/MVS/Samples/32/Python/MvImport"),
)


def _decode_ctypes_chars(value: Any) -> str:
    """
        Decode MVS ctypes char arrays into a plain Python string.
    """

    raw = memoryview(value).tobytes()
    raw = raw.split(b"\x00", maxsplit=1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _gige_ip_to_text(ip_value: int) -> str:
    """
        Convert MVS integer GigE IP representation to dotted text.
    """

    return ".".join(
        str((ip_value >> shift) & 0xFF)
        for shift in (24, 16, 8, 0)
    )


def _load_mvs_sdk() -> ModuleType:
    """
        Load HIKROBOT MVS Python bindings from common installation paths.
    """

    for sdk_dir in MVS_IMPORT_DIRS:
        if sdk_dir.is_dir() and str(sdk_dir) not in sys.path:
            sys.path.append(str(sdk_dir))

        sdk_file = sdk_dir / "MvCameraControl_class.py"
        if not sdk_file.is_file():
            continue

        spec = importlib.util.spec_from_file_location(
            "MvCameraControl_class",
            sdk_file,
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["MvCameraControl_class"] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load HIKROBOT MVS SDK from {sdk_file}."
            ) from exc
        return module

    raise RuntimeError(
        "Cannot find HIKROBOT MVS SDK. "
        "Install MVS and check /opt/MVS/Samples/64/Python/MvImport."
    )


@dataclass
class CameraDevice:
    """
        One enumerated camera device and its MVS device info structure.
    """

    index: int
    model: str
    serial: str
    ip: str
    transport: str
    info: Any

    @property
    def label(self) -> str:
        """Return a compact label for the device combobox."""
        ip_text = self.ip or "no-ip"
        serial_text = self.serial or "no-serial"
        return f"{self.index}: {self.model} | {ip_text} | {serial_text}"


class HikrobotGigECamera:
    """
        Thin adapter around HIKROBOT MVS SDK for GigE preview and capture.
    """

    def __init__(self) -> None:
        self.mvs: ModuleType | None = None
        self.cam: Any | None = None
        self.connected_device: CameraDevice | None = None
        self.initialized = False
        self.grabbing = False

    def initialize(self) -> None:
        """
            Load SDK and initialize MVS once.
        """

        if self.initialized:
            return

        self.mvs = _load_mvs_sdk()
        ret = self.mvs.MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise RuntimeError(f"Initialize MVS SDK failed: 0x{ret:x}")
        self.initialized = True

    def enumerate_devices(self) -> list[CameraDevice]:
        """
            Enumerate GigE cameras available through MVS.
        """

        self.initialize()
        assert self.mvs is not None

        device_list = self.mvs.MV_CC_DEVICE_INFO_LIST()
        layer_type = self.mvs.MV_GIGE_DEVICE
        if hasattr(self.mvs, "MV_GENTL_GIGE_DEVICE"):
            layer_type |= self.mvs.MV_GENTL_GIGE_DEVICE

        ret = self.mvs.MvCamera.MV_CC_EnumDevices(layer_type, device_list)
        if ret != 0:
            raise RuntimeError(f"Enum GigE devices failed: 0x{ret:x}")

        devices: list[CameraDevice] = []
        for index in range(device_list.nDeviceNum):
            info = cast(
                device_list.pDeviceInfo[index],
                POINTER(self.mvs.MV_CC_DEVICE_INFO),
            ).contents
            if info.nTLayerType not in (
                self.mvs.MV_GIGE_DEVICE,
                getattr(self.mvs, "MV_GENTL_GIGE_DEVICE", self.mvs.MV_GIGE_DEVICE),
            ):
                continue

            gige_info = info.SpecialInfo.stGigEInfo
            devices.append(
                CameraDevice(
                    index=index,
                    model=_decode_ctypes_chars(gige_info.chModelName),
                    serial=_decode_ctypes_chars(gige_info.chSerialNumber),
                    ip=_gige_ip_to_text(int(gige_info.nCurrentIp)),
                    transport="GigE",
                    info=copy.deepcopy(info),
                )
            )

        return devices

    def connect(self, device: CameraDevice) -> None:
        """
            Open one camera, optimize GigE packet size, and start grabbing.
        """

        self.close()
        self.initialize()
        assert self.mvs is not None

        self.cam = self.mvs.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(device.info)
        if ret != 0:
            self.cam = None
            raise RuntimeError(f"Create camera handle failed: 0x{ret:x}")

        ret = self.cam.MV_CC_OpenDevice(self.mvs.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.close()
            raise RuntimeError(f"Open camera failed: 0x{ret:x}")

        self.connected_device = device
        self._optimize_gige_packet_size()
        self._set_enum("TriggerMode", "Off")

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.close()
            raise RuntimeError(f"Start grabbing failed: 0x{ret:x}")

        self.grabbing = True

    def apply_config(
        self,
        exposure_time_us: float,
        gain: float,
        frame_rate: float,
        packet_size: int,
        auto_exposure: bool,
    ) -> list[str]:
        """
            Apply common camera parameters and return non-fatal warnings.
        """

        self._require_connected()
        warnings: list[str] = []

        exposure_auto = "Continuous" if auto_exposure else "Off"
        self._try_set_enum("ExposureAuto", exposure_auto, warnings)
        if not auto_exposure:
            self._try_set_float("ExposureTime", exposure_time_us, warnings)

        self._try_set_float("Gain", gain, warnings)
        self._try_set_bool("AcquisitionFrameRateEnable", True, warnings)
        self._try_set_float("AcquisitionFrameRate", frame_rate, warnings)

        if packet_size > 0:
            self._try_set_int("GevSCPSPacketSize", packet_size, warnings)

        return warnings

    def grab_frame(self, timeout_ms: int = 1000) -> np.ndarray:
        """
            Grab one frame and convert it to OpenCV BGR uint8 image.
        """

        self._require_connected()
        assert self.mvs is not None and self.cam is not None

        frame = self.mvs.MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))

        ret = self.cam.MV_CC_GetImageBuffer(frame, int(timeout_ms))
        if ret != 0 or frame.pBufAddr is None:
            raise RuntimeError(f"Get image buffer failed: 0x{ret:x}")

        try:
            width = int(frame.stFrameInfo.nWidth)
            height = int(frame.stFrameInfo.nHeight)
            bgr_size = width * height * 3

            convert_param = self.mvs.MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(convert_param), 0, sizeof(convert_param))
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.pSrcData = frame.pBufAddr
            convert_param.nSrcDataLen = frame.stFrameInfo.nFrameLen
            convert_param.enSrcPixelType = frame.stFrameInfo.enPixelType
            convert_param.enDstPixelType = self.mvs.PixelType_Gvsp_BGR8_Packed
            convert_param.pDstBuffer = (c_ubyte * bgr_size)()
            convert_param.nDstBufferSize = bgr_size

            ret = self.cam.MV_CC_ConvertPixelTypeEx(convert_param)
            if ret != 0:
                raise RuntimeError(f"Convert pixel type failed: 0x{ret:x}")

            data = np.ctypeslib.as_array(
                convert_param.pDstBuffer,
                shape=(int(convert_param.nDstLen),),
            ).copy()
            return data.reshape((height, width, 3))
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    def close(self) -> None:
        """
            Stop grabbing and close the current camera handle.
        """

        if self.cam is not None:
            if self.grabbing:
                self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()

        self.cam = None
        self.connected_device = None
        self.grabbing = False

    def finalize(self) -> None:
        """
            Close the camera and release MVS SDK resources.
        """

        self.close()
        if self.initialized and self.mvs is not None:
            self.mvs.MvCamera.MV_CC_Finalize()
        self.initialized = False

    def _require_connected(self) -> None:
        """
            Raise when no camera is currently open.
        """

        if self.cam is None or not self.grabbing:
            raise RuntimeError("Camera is not connected and grabbing.")

    def _optimize_gige_packet_size(self) -> None:
        """
            Ask MVS for the recommended GigE packet size.
        """

        assert self.cam is not None
        packet_size = self.cam.MV_CC_GetOptimalPacketSize()
        if int(packet_size) > 0:
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet_size))

    def _set_enum(self, name: str, value: str) -> None:
        """Set one enum node and fail loudly."""
        assert self.cam is not None
        ret = self.cam.MV_CC_SetEnumValueByString(name, value)
        if ret != 0:
            raise RuntimeError(f"Set {name}={value} failed: 0x{ret:x}")

    def _try_set_enum(self, name: str, value: str, warnings: list[str]) -> None:
        """Set one enum node and collect failures as warnings."""
        assert self.cam is not None
        ret = self.cam.MV_CC_SetEnumValueByString(name, value)
        if ret != 0:
            warnings.append(f"{name}={value} failed: 0x{ret:x}")

    def _try_set_float(self, name: str, value: float, warnings: list[str]) -> None:
        """Set one float node and collect failures as warnings."""
        assert self.cam is not None
        ret = self.cam.MV_CC_SetFloatValue(name, float(value))
        if ret != 0:
            warnings.append(f"{name}={value} failed: 0x{ret:x}")

    def _try_set_int(self, name: str, value: int, warnings: list[str]) -> None:
        """Set one integer node and collect failures as warnings."""
        assert self.cam is not None
        ret = self.cam.MV_CC_SetIntValue(name, int(value))
        if ret != 0:
            warnings.append(f"{name}={value} failed: 0x{ret:x}")

    def _try_set_bool(self, name: str, value: bool, warnings: list[str]) -> None:
        """Set one boolean node and collect failures as warnings."""
        assert self.cam is not None
        ret = self.cam.MV_CC_SetBoolValue(name, bool(value))
        if ret != 0:
            warnings.append(f"{name}={value} failed: 0x{ret:x}")


class CameraCanvas(QWidget):
    """
        Interactive image canvas for camera preview with zoom and pan.
    """

    def __init__(self) -> None:
        super().__init__()

        """Canvas interaction setup"""
        self.setMinimumSize(520, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        """Loaded image state"""
        self.image_qt: QImage | None = None
        self.image_w = 0
        self.image_h = 0

        """Viewport state"""
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.view_x = 0
        self.view_y = 0
        self._pan_start: QPoint | None = None
        self._pan_view_start: tuple[int, int] | None = None

    def set_image(self, image_bgr: np.ndarray) -> None:
        """
            Convert one OpenCV BGR frame to QImage for display.
        """

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

        if self.zoom <= self.fit_zoom or self.fit_zoom == 1.0:
            self.fit_to_view()
        else:
            self._clamp_view()
            self.update()

    def fit_to_view(self) -> None:
        """
            Fit the full camera image inside the canvas.
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
            Return whether the canvas has a valid frame.
        """

        return self.image_qt is not None and self.image_w > 0 and self.image_h > 0

    def paintEvent(self, event) -> None:
        """
            Paint the current camera frame viewport.
        """

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if not self.has_image():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Connect camera and start preview",
            )
            return

        source_rect, target_rect = self._view_rects()
        painter.drawImage(target_rect, self.image_qt, source_rect)

    def resizeEvent(self, event) -> None:
        """
            Keep viewport valid after resizing.
        """

        super().resizeEvent(event)
        if self.has_image() and self.zoom < self.fit_zoom:
            self.fit_to_view()
        else:
            self._clamp_view()

    def mousePressEvent(self, event) -> None:
        """
            Start panning by left or middle mouse drag.
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
            Pan the current viewport.
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
            Stop active pan gesture.
        """

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._pan_start = None
            self._pan_view_start = None

    def wheelEvent(self, event) -> None:
        """
            Zoom around the cursor.
        """

        if not self.has_image():
            return

        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom_at(event.position(), factor)

    def _view_rects(self) -> tuple[QRectF, QRectF]:
        """
            Return source image rect and centered target widget rect.
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
            Convert widget position to original image coordinates.
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
            Compute visible image width in source pixels.
        """

        return max(1, int(round(max(1, self.width()) / self.zoom)))

    def _visible_image_h(self) -> int:
        """
            Compute visible image height in source pixels.
        """

        return max(1, int(round(max(1, self.height()) / self.zoom)))

    def _clamp_view(self) -> None:
        """
            Keep viewport inside image boundaries.
        """

        if not self.has_image():
            return

        max_x = max(0, self.image_w - self._visible_image_w())
        max_y = max(0, self.image_h - self._visible_image_h())
        self.view_x = int(np.clip(self.view_x, 0, max_x))
        self.view_y = int(np.clip(self.view_y, 0, max_y))

    def _zoom_at(self, widget_pos, factor: float) -> None:
        """
            Apply zoom while preserving cursor anchor.
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


class CameraTab(QWidget):
    """
        Full camera tab with GigE connection, preview, config, and capture.
    """

    def __init__(self) -> None:
        super().__init__()

        """Persistent GUI settings"""
        self.settings = QSettings("SurfaceScratchDetection", "CameraTab")

        """Camera state"""
        self.camera = HikrobotGigECamera()
        self.devices: list[CameraDevice] = []
        self.latest_frame: np.ndarray | None = None

        """Preview timer"""
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._grab_preview_frame)

        """Camera selection controls"""
        self.device_combo = QComboBox()
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("Optional camera IP, e.g. 192.168.1.64")
        self.output_dir_edit = QLineEdit(str(RAW_IMAGE_DIR))
        self.format_combo = QComboBox()
        self.format_combo.addItems(CAPTURE_FORMATS)
        self.grayscale_check = QCheckBox("Save grayscale")
        self.grayscale_check.setChecked(True)

        """Camera parameter controls"""
        self.auto_exposure_check = QCheckBox("Auto exposure")
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(1.0, 10_000_000.0)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSingleStep(100.0)
        self.exposure_spin.setValue(10_000.0)
        self.exposure_spin.setSuffix(" us")

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 48.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setValue(0.0)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 240.0)
        self.frame_rate_spin.setDecimals(1)
        self.frame_rate_spin.setSingleStep(1.0)
        self.frame_rate_spin.setValue(10.0)
        self.frame_rate_spin.setSuffix(" fps")

        self.packet_size_spin = QSpinBox()
        self.packet_size_spin.setRange(0, 16_384)
        self.packet_size_spin.setSingleStep(100)
        self.packet_size_spin.setValue(0)
        self.packet_size_spin.setSpecialValueText("Auto")

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 5000)
        self.timeout_spin.setSingleStep(50)
        self.timeout_spin.setValue(1000)
        self.timeout_spin.setSuffix(" ms")

        self.preview_interval_spin = QSpinBox()
        self.preview_interval_spin.setRange(30, 2000)
        self.preview_interval_spin.setSingleStep(10)
        self.preview_interval_spin.setValue(100)
        self.preview_interval_spin.setSuffix(" ms")

        """Preview and log widgets"""
        self.canvas = CameraCanvas()
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlainText("Camera not connected.")
        self.log_edit.setMaximumHeight(140)

        """Action buttons"""
        self.refresh_button = QPushButton("Refresh")
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.apply_button = QPushButton("Apply Config")
        self.start_preview_button = QPushButton("Start Preview")
        self.stop_preview_button = QPushButton("Stop Preview")
        self.capture_button = QPushButton("Capture")
        self.output_button = QPushButton("Browse")

        self.disconnect_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.start_preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(False)
        self.capture_button.setEnabled(False)

        self._style_widgets()
        self._build_layout()
        self._connect_signals()
        self._install_shortcuts()
        self._load_settings()

    def _style_widgets(self) -> None:
        """
            Apply tab-local Roboflow-like styling.
        """

        controls = (
            self.device_combo,
            self.ip_edit,
            self.output_dir_edit,
            self.format_combo,
            self.exposure_spin,
            self.gain_spin,
            self.frame_rate_spin,
            self.packet_size_spin,
            self.timeout_spin,
            self.preview_interval_spin,
        )
        for widget in controls:
            widget.setMinimumHeight(28)

        for button in (
            self.refresh_button,
            self.connect_button,
            self.disconnect_button,
            self.apply_button,
            self.start_preview_button,
            self.stop_preview_button,
            self.capture_button,
            self.output_button,
        ):
            button.setMinimumHeight(28)

    def _build_layout(self) -> None:
        """
            Build the full camera tab layout.
        """

        header = self._build_header()
        preview_group = self._build_preview_group()

        side_widget = QWidget()
        side_panel = QVBoxLayout()
        side_panel.setContentsMargins(0, 0, 0, 0)
        side_panel.setSpacing(6)
        side_panel.addWidget(self._build_connection_group())
        side_panel.addWidget(self._build_config_group())
        side_panel.addWidget(self._build_capture_group())
        side_panel.addWidget(self._build_log_group())
        side_panel.addStretch(1)
        side_widget.setLayout(side_panel)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side_widget)
        side_scroll.setMinimumWidth(320)

        content = QHBoxLayout()
        content.setSpacing(10)
        content.addWidget(preview_group, stretch=5)
        content.addWidget(side_scroll, stretch=2)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addLayout(content, stretch=1)
        self.setLayout(layout)

    def _build_header(self) -> QVBoxLayout:
        """
            Build title area.
        """

        title = QLabel("Camera Capture")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(title)
        return layout

    def _build_connection_group(self) -> QGroupBox:
        """
            Build camera selection and connection controls.
        """

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.connect_button)
        actions.addWidget(self.disconnect_button)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        form.addRow("Device", self.device_combo)
        form.addRow("IP", self.ip_edit)

        group = QGroupBox("Connection")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addLayout(form)
        layout.addLayout(actions)
        group.setLayout(layout)
        return group

    def _build_config_group(self) -> QGroupBox:
        """
            Build camera parameter controls.
        """

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        form.addRow("", self.auto_exposure_check)
        form.addRow("Exposure", self.exposure_spin)
        form.addRow("Gain", self.gain_spin)
        form.addRow("Frame rate", self.frame_rate_spin)
        form.addRow("Packet size", self.packet_size_spin)
        form.addRow("Grab timeout", self.timeout_spin)
        form.addRow("Preview interval", self.preview_interval_spin)

        group = QGroupBox("Camera Config")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addLayout(form)
        layout.addWidget(self.apply_button)
        group.setLayout(layout)
        return group

    def _build_capture_group(self) -> QGroupBox:
        """
            Build preview/capture controls and output folder selector.
        """

        preview_actions = QHBoxLayout()
        preview_actions.setSpacing(8)
        preview_actions.addWidget(self.start_preview_button)
        preview_actions.addWidget(self.stop_preview_button)

        capture_actions = QHBoxLayout()
        capture_actions.setSpacing(8)
        capture_actions.addWidget(self.capture_button)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        output_row.addWidget(self.output_dir_edit, stretch=1)
        output_row.addWidget(self.output_button)

        format_row = QFormLayout()
        format_row.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        format_row.setHorizontalSpacing(10)
        format_row.setVerticalSpacing(7)
        format_row.addRow("Format", self.format_combo)
        format_row.addRow("", self.grayscale_check)

        group = QGroupBox("Capture")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addLayout(preview_actions)
        layout.addLayout(capture_actions)
        layout.addLayout(output_row)
        layout.addLayout(format_row)
        group.setLayout(layout)
        return group

    def _build_preview_group(self) -> QGroupBox:
        """
            Build live preview card.
        """

        group = QGroupBox("Live Preview")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.canvas, stretch=1)
        group.setLayout(layout)
        return group

    def _build_log_group(self) -> QGroupBox:
        """
            Build camera log card.
        """

        group = QGroupBox("Camera Log")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.log_edit, stretch=1)
        group.setLayout(layout)
        return group

    def _connect_signals(self) -> None:
        """
            Connect UI actions.
        """

        self.refresh_button.clicked.connect(self.refresh_devices)
        self.connect_button.clicked.connect(self.connect_camera)
        self.disconnect_button.clicked.connect(self.disconnect_camera)
        self.apply_button.clicked.connect(self.apply_config)
        self.start_preview_button.clicked.connect(self.start_preview)
        self.stop_preview_button.clicked.connect(self.stop_preview)
        self.capture_button.clicked.connect(self.capture_image)
        self.output_button.clicked.connect(self.browse_output_dir)
        self.auto_exposure_check.toggled.connect(self.exposure_spin.setDisabled)
        self.ip_edit.editingFinished.connect(self._save_settings)
        self.output_dir_edit.editingFinished.connect(self._handle_output_dir_changed)
        self.format_combo.currentTextChanged.connect(self._save_settings)
        self.grayscale_check.toggled.connect(self._save_settings)
        self.auto_exposure_check.toggled.connect(self._save_settings)
        self.exposure_spin.valueChanged.connect(self._save_settings)
        self.gain_spin.valueChanged.connect(self._save_settings)
        self.frame_rate_spin.valueChanged.connect(self._save_settings)
        self.packet_size_spin.valueChanged.connect(self._save_settings)
        self.timeout_spin.valueChanged.connect(self._save_settings)
        self.preview_interval_spin.valueChanged.connect(self._save_settings)

    def _install_shortcuts(self) -> None:
        """
            Install capture shortcut.
        """

        QShortcut(QKeySequence("S"), self, activated=self.capture_image)

    def refresh_devices(self) -> None:
        """
            Enumerate available GigE cameras and fill the combobox.
        """

        try:
            self.devices = self.camera.enumerate_devices()
            self.device_combo.clear()
            for device in self.devices:
                self.device_combo.addItem(device.label)

            if self.devices:
                self._set_log(f"Found {len(self.devices)} GigE camera(s).")
            else:
                self._set_log("No GigE camera found.")
        except Exception as exc:
            self._set_log(f"Refresh failed: {exc}")
            QMessageBox.warning(self, "Refresh failed", str(exc))

    def connect_camera(self) -> None:
        """
            Connect to selected camera or a camera matching the entered IP.
        """

        try:
            if not self.devices:
                self.devices = self.camera.enumerate_devices()

            device = self._selected_device()
            self.camera.connect(device)
            self.apply_config(show_dialog=False)
            self._set_connected_state(True)
            self._set_log(f"Connected: {device.label}")
        except Exception as exc:
            self._set_log(f"Connect failed: {exc}")
            QMessageBox.warning(self, "Connect failed", str(exc))

    def disconnect_camera(self) -> None:
        """
            Stop preview and close camera.
        """

        self.stop_preview()
        self.camera.close()
        self._set_connected_state(False)
        self._set_log("Camera disconnected.")

    def apply_config(self, show_dialog: bool = True) -> None:
        """
            Apply current parameter controls to the connected camera.
        """

        try:
            warnings = self.camera.apply_config(
                exposure_time_us=self.exposure_spin.value(),
                gain=self.gain_spin.value(),
                frame_rate=self.frame_rate_spin.value(),
                packet_size=self.packet_size_spin.value(),
                auto_exposure=self.auto_exposure_check.isChecked(),
            )
            if warnings:
                self._set_log("Config applied with warnings:\n" + "\n".join(warnings))
            else:
                self._set_log("Camera config applied.")
            if show_dialog and warnings:
                QMessageBox.warning(self, "Config warnings", "\n".join(warnings))
        except Exception as exc:
            self._set_log(f"Apply config failed: {exc}")
            if show_dialog:
                QMessageBox.warning(self, "Apply config failed", str(exc))

    def start_preview(self) -> None:
        """
            Start live preview timer.
        """

        if self.preview_timer.isActive():
            return

        self.preview_timer.start(self.preview_interval_spin.value())
        self.start_preview_button.setEnabled(False)
        self.stop_preview_button.setEnabled(True)
        self._set_log("Preview started.")

    def stop_preview(self) -> None:
        """
            Stop live preview timer.
        """

        if self.preview_timer.isActive():
            self.preview_timer.stop()
            self._set_log("Preview stopped.")
        if self.camera.grabbing:
            self.start_preview_button.setEnabled(True)
        self.stop_preview_button.setEnabled(False)

    def capture_image(self) -> None:
        """
            Save the latest frame to data/raw with Image_YYYYMMDD_HHMMSS name.
        """

        if not self.camera.grabbing:
            QMessageBox.warning(self, "Capture failed", "Connect camera before capture.")
            return

        try:
            frame = self.latest_frame
            if frame is None:
                frame = self.camera.grab_frame(self.timeout_spin.value())
                self.latest_frame = frame
                self.canvas.set_image(frame)

            output_dir = Path(self.output_dir_edit.text()).expanduser()
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = self.format_combo.currentText().lower()
            output_path = output_dir / f"Image_{timestamp}.{suffix}"
            frame_to_save = self._prepare_frame_for_save(frame)

            if not cv2.imwrite(str(output_path), frame_to_save):
                raise RuntimeError(f"Cannot write image: {output_path}")

            self._save_settings()
            self._set_log(
                "Saved image: "
                f"{output_path}\n"
                f"Mode: {'grayscale' if self.grayscale_check.isChecked() else 'color'} | "
                f"Format: {suffix} | "
                f"Size: {self._format_bytes(output_path.stat().st_size)}"
            )
        except Exception as exc:
            self._set_log(f"Capture failed: {exc}")
            QMessageBox.warning(self, "Capture failed", str(exc))

    def browse_output_dir(self) -> None:
        """
            Choose output directory for captured raw images.
        """

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select raw image output directory",
            self.output_dir_edit.text() or str(RAW_IMAGE_DIR),
        )
        if output_dir:
            self.output_dir_edit.setText(output_dir)
            self._handle_output_dir_changed()

    def closeEvent(self, event) -> None:
        """
            Release camera resources when the tab/window is closed.
        """

        self.preview_timer.stop()
        self.camera.finalize()
        super().closeEvent(event)

    def _grab_preview_frame(self) -> None:
        """
            Grab and display one preview frame.
        """

        try:
            frame = self.camera.grab_frame(self.timeout_spin.value())
            self.latest_frame = frame
            self.canvas.set_image(frame)
        except Exception as exc:
            self.preview_timer.stop()
            self.start_preview_button.setEnabled(True)
            self.stop_preview_button.setEnabled(False)
            self._set_log(f"Preview stopped: {exc}")

    def _selected_device(self) -> CameraDevice:
        """
            Return device matching IP text or current combobox index.
        """

        if not self.devices:
            raise RuntimeError("No camera device available.")

        ip_text = self.ip_edit.text().strip()
        if ip_text:
            for device in self.devices:
                if device.ip == ip_text:
                    return device
            raise RuntimeError(f"No enumerated GigE camera has IP {ip_text}.")

        index = self.device_combo.currentIndex()
        if index < 0 or index >= len(self.devices):
            raise RuntimeError("Select a camera device.")
        return self.devices[index]

    def _prepare_frame_for_save(self, frame: np.ndarray) -> np.ndarray:
        """
            Convert the current frame according to capture save settings.
        """

        if self.grayscale_check.isChecked() and frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(frame)

    def _handle_output_dir_changed(self) -> None:
        """
            Persist output directory changes.
        """

        self._save_settings()

    def _load_settings(self) -> None:
        """
            Restore camera and capture settings from the previous session.
        """

        self.ip_edit.setText(self.settings.value("ip", "", str))
        self.output_dir_edit.setText(
            self.settings.value("output_dir", str(RAW_IMAGE_DIR), str)
        )
        self.auto_exposure_check.setChecked(
            self.settings.value("auto_exposure", False, bool)
        )
        self.exposure_spin.setValue(
            float(self.settings.value("exposure_us", self.exposure_spin.value()))
        )
        self.gain_spin.setValue(float(self.settings.value("gain", self.gain_spin.value())))
        self.frame_rate_spin.setValue(
            float(self.settings.value("frame_rate", self.frame_rate_spin.value()))
        )
        self.packet_size_spin.setValue(
            int(self.settings.value("packet_size", self.packet_size_spin.value()))
        )
        self.timeout_spin.setValue(
            int(self.settings.value("timeout_ms", self.timeout_spin.value()))
        )
        self.preview_interval_spin.setValue(
            int(self.settings.value("preview_interval_ms", self.preview_interval_spin.value()))
        )
        suffix = self.settings.value("format", "bmp", str)
        index = self.format_combo.findText(suffix)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        self.grayscale_check.setChecked(
            self.settings.value("save_grayscale", True, bool)
        )

    def _save_settings(self) -> None:
        """
            Persist camera and capture settings for the next run.
        """

        self.settings.setValue("ip", self.ip_edit.text().strip())
        self.settings.setValue("output_dir", self.output_dir_edit.text().strip())
        self.settings.setValue("auto_exposure", self.auto_exposure_check.isChecked())
        self.settings.setValue("exposure_us", self.exposure_spin.value())
        self.settings.setValue("gain", self.gain_spin.value())
        self.settings.setValue("frame_rate", self.frame_rate_spin.value())
        self.settings.setValue("packet_size", self.packet_size_spin.value())
        self.settings.setValue("timeout_ms", self.timeout_spin.value())
        self.settings.setValue(
            "preview_interval_ms",
            self.preview_interval_spin.value(),
        )
        self.settings.setValue("format", self.format_combo.currentText())
        self.settings.setValue("save_grayscale", self.grayscale_check.isChecked())

    def _format_bytes(self, value: int) -> str:
        """
            Format byte count for UI messages.
        """

        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _set_connected_state(self, connected: bool) -> None:
        """
            Enable and disable controls based on connection state.
        """

        self.connect_button.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.apply_button.setEnabled(connected)
        self.start_preview_button.setEnabled(connected)
        self.stop_preview_button.setEnabled(False)
        self.capture_button.setEnabled(connected)

    def _set_log(self, text: str) -> None:
        """
            Replace camera log content.
        """

        self.log_edit.setPlainText(text)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = CameraTab()
    window.resize(1200, 850)
    window.show()
    sys.exit(app.exec())
