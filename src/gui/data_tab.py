"""
    PySide6 data preparation tab for the scratch YOLO dataset pipeline.

    Purpose:
        1. Configure the metadata-first YOLO dataset preparation pipeline.
        2. Run outputs/labeling -> data/scratch_yolo_seg directly.
        3. Check the final YOLO dataset before training.
        4. Stream process logs without blocking the GUI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data import IMAGE_EXTENSIONS, LABELING_DATASET, SPLITS


CHECK_LOG_SEPARATOR = "-" * 112
DEFAULT_YOLO_OUTPUT_ROOT = ROOT / "data" / "scratch_yolo_seg"
DEFAULT_SPLIT_OUTPUT_ROOT = ROOT / "data" / "scratch"


class DataTab(QWidget):
    """
        YOLO data preparation dashboard tab.

        The old intermediate split and patch folders are no longer exposed in
        the GUI. The button launches:

            outputs/labeling -> src.dataset.prepare_yolo_dataset -> data/scratch_yolo_seg
    """

    def __init__(self) -> None:
        super().__init__()

        """Background process state"""
        self.process: QProcess | None = None
        self._active_job_name = ""
        self._last_log_group = ""

        """Input and output controls"""
        self.source_edit = QLineEdit(str(LABELING_DATASET))
        self.output_edit = QLineEdit(str(DEFAULT_YOLO_OUTPUT_ROOT))
        self.split_output_edit = QLineEdit(str(DEFAULT_SPLIT_OUTPUT_ROOT))
        self.split_output_button = QPushButton("Browse")
        self.save_split_check = QCheckBox("Save full-image split for evaluation")
        self.save_split_check.setChecked(False)

        """Split controls"""
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.0, 1.0)
        self.train_ratio_spin.setDecimals(2)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.70)

        self.valid_ratio_spin = QDoubleSpinBox()
        self.valid_ratio_spin.setRange(0.0, 1.0)
        self.valid_ratio_spin.setDecimals(2)
        self.valid_ratio_spin.setSingleStep(0.05)
        self.valid_ratio_spin.setValue(0.20)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.0, 1.0)
        self.test_ratio_spin.setDecimals(2)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.10)

        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 999999)
        self.split_seed_spin.setValue(42)

        """Patch and negative-sampling controls"""
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(32, 4096)
        self.patch_size_spin.setSingleStep(32)
        self.patch_size_spin.setValue(512)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.95)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(0.25)

        self.train_negative_spin = QDoubleSpinBox()
        self.train_negative_spin.setRange(0.0, 100.0)
        self.train_negative_spin.setDecimals(2)
        self.train_negative_spin.setSingleStep(0.25)
        self.train_negative_spin.setValue(1.00)

        self.valid_negative_spin = QDoubleSpinBox()
        self.valid_negative_spin.setRange(0.0, 100.0)
        self.valid_negative_spin.setDecimals(2)
        self.valid_negative_spin.setSingleStep(0.25)
        self.valid_negative_spin.setValue(1.50)

        self.test_negative_spin = QDoubleSpinBox()
        self.test_negative_spin.setRange(0.0, 100.0)
        self.test_negative_spin.setDecimals(2)
        self.test_negative_spin.setSingleStep(0.25)
        self.test_negative_spin.setValue(1.50)

        self.patch_seed_spin = QSpinBox()
        self.patch_seed_spin.setRange(0, 999999)
        self.patch_seed_spin.setValue(42)

        self.overwrite_check = QCheckBox("Overwrite existing YOLO dataset output")
        self.overwrite_check.setChecked(False)

        """Log and progress widgets"""
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

        self.progress_label = QLabel("YOLO dataset preparation not started.")

        """Action buttons"""
        self.prepare_button = QPushButton("Prepare YOLO Dataset")
        self.check_button = QPushButton("Check YOLO Dataset")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        """Create UI and connect behavior"""
        self._style_widgets()
        self._build_layout()
        self._connect_signals()

    def _style_widgets(self) -> None:
        """
            Apply PySide6-only widget sizing.
        """

        inputs = (
            self.source_edit,
            self.output_edit,
            self.split_output_edit,
            self.train_ratio_spin,
            self.valid_ratio_spin,
            self.test_ratio_spin,
            self.split_seed_spin,
            self.patch_size_spin,
            self.overlap_spin,
            self.train_negative_spin,
            self.valid_negative_spin,
            self.test_negative_spin,
            self.patch_seed_spin,
        )
        for widget in inputs:
            widget.setMinimumHeight(28)

        for edit in (self.source_edit, self.output_edit, self.split_output_edit):
            edit.setMinimumWidth(0)
            edit.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )

        for button in (
            self.prepare_button,
            self.check_button,
            self.stop_button,
            self.split_output_button,
        ):
            button.setMinimumHeight(30)

        self.progress_bar.setMinimumHeight(12)
        self._sync_split_output_controls()

    def _build_layout(self) -> None:
        """
            Build the full YOLO data preparation tab layout.
        """

        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)
        top_layout.addWidget(self._build_dataset_group(), stretch=2)
        top_layout.addWidget(self._build_split_group(), stretch=1)
        top_layout.addWidget(self._build_patch_group(), stretch=1)

        layout = QVBoxLayout()
        layout.setContentsMargins(22, 14, 22, 14)
        layout.setSpacing(8)
        layout.addLayout(self._build_header())
        layout.addLayout(top_layout)
        layout.addWidget(self._build_log_group(), stretch=1)
        self.setLayout(layout)

    def _build_header(self) -> QVBoxLayout:
        """
            Build the page title area.
        """

        title = QLabel("Prepare YOLO Dataset")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(title)
        return layout

    def _build_dataset_group(self) -> QGroupBox:
        """
            Build input and output path controls.
        """

        group = QGroupBox("Dataset Paths")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.addRow("Labeled root", self._path_row(self.source_edit, self._browse_source))
        form.addRow("YOLO output", self._path_row(self.output_edit, self._browse_output))
        form.addRow("Full split output", self._split_output_row())

        layout.addLayout(form)
        layout.addWidget(self.save_split_check)
        layout.addWidget(self.overwrite_check)
        group.setLayout(layout)
        return group

    def _build_split_group(self) -> QGroupBox:
        """
            Build split ratio controls.
        """

        group = QGroupBox("Split Config")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.addRow("Train ratio", self.train_ratio_spin)
        form.addRow("Valid ratio", self.valid_ratio_spin)
        form.addRow("Test ratio", self.test_ratio_spin)
        form.addRow("Seed", self.split_seed_spin)

        layout.addLayout(form)
        group.setLayout(layout)
        return group

    def _build_patch_group(self) -> QGroupBox:
        """
            Build patch and negative-sampling controls.
        """

        group = QGroupBox("Patch Config")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.addRow("Patch size", self.patch_size_spin)
        form.addRow("Overlap", self.overlap_spin)
        form.addRow("Train negatives", self.train_negative_spin)
        form.addRow("Valid negatives", self.valid_negative_spin)
        form.addRow("Test negatives", self.test_negative_spin)
        form.addRow("Seed", self.patch_seed_spin)

        layout.addLayout(form)
        group.setLayout(layout)
        return group

    def _build_log_group(self) -> QGroupBox:
        """
            Build process controls and log console.
        """

        group = QGroupBox("Data Log")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addWidget(self.prepare_button)
        actions.addWidget(self.check_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)

        layout.addLayout(actions)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_edit, stretch=1)
        group.setLayout(layout)
        return group

    def _path_row(self, edit: QLineEdit, slot) -> QHBoxLayout:
        """
            Build a path input row with a matching Browse button.
        """

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(edit, stretch=1)

        browse_button = QPushButton("Browse")
        browse_button.setMinimumHeight(28)
        browse_button.clicked.connect(slot)
        row.addWidget(browse_button)
        return row

    def _split_output_row(self) -> QHBoxLayout:
        """
            Build the optional full-image split output path row.
        """

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.split_output_edit, stretch=1)
        self.split_output_button.clicked.connect(self._browse_split_output)
        row.addWidget(self.split_output_button)
        return row

    def _connect_signals(self) -> None:
        """
            Connect buttons to tab behavior.
        """

        self.prepare_button.clicked.connect(self.start_yolo_pipeline)
        self.check_button.clicked.connect(self.check_yolo_dataset)
        self.stop_button.clicked.connect(self.stop_process)
        self.save_split_check.toggled.connect(self._sync_split_output_controls)

    def start_yolo_pipeline(self) -> None:
        """
            Launch the compact YOLO dataset preparation pipeline.
        """

        if self.process is not None:
            return

        try:
            args = self._build_yolo_args(validate=True)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid YOLO dataset settings", str(exc))
            return

        self._start_process(
            job_name="Prepare YOLO dataset",
            args=args,
            start_message="Preparing YOLO dataset from labeled images...",
        )

    def check_yolo_dataset(self) -> None:
        """
            Check the final train/valid/test YOLO output.
        """

        data_yaml = self._resolve_path(self.output_edit.text()) / "data.yaml"

        try:
            summary = self._collect_yolo_dataset_summary(data_yaml)
        except Exception as exc:
            self._append_check_log(
                "Checked YOLO dataset: Error\n"
                f"- {self._format_error_message(str(exc))}"
            )
            return

        status = "OK" if summary["ok"] else "Error"
        lines = [f"Checked YOLO dataset: {status}"]

        for split in SPLITS:
            row = summary["splits"][split]
            lines.append(
                f"- {self._format_split_name(split)}: "
                f"images: {row['images']} | labels: {row['labels']} | "
                f"missing labels: {row['missing_labels']} | "
                f"orphan labels: {row['orphan_labels']}"
            )

        self._append_check_log("\n".join(lines))

    def stop_process(self) -> None:
        """
            Stop the active data preparation process.
        """

        if self.process is None:
            return

        self._append_log(f"Stopping {self._active_job_name.lower()}...")
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()

    def _build_yolo_args(self, validate: bool = False) -> list[str]:
        """
            Build arguments for python -m src.dataset.prepare_yolo_dataset.
        """

        src_root = self._resolve_path(self.source_edit.text())
        output_root = self._resolve_path(self.output_edit.text())
        split_output_root = self._resolve_path(self.split_output_edit.text())

        train_ratio = self.train_ratio_spin.value()
        valid_ratio = self.valid_ratio_spin.value()
        test_ratio = self.test_ratio_spin.value()
        patch_size = self.patch_size_spin.value()
        overlap = self.overlap_spin.value()

        if train_ratio + valid_ratio + test_ratio <= 0:
            raise ValueError("At least one split ratio must be greater than zero.")

        if validate:
            if not src_root.is_dir():
                raise ValueError(f"Labeled dataset root not found: {src_root}")
            if output_root.exists() and not self.overwrite_check.isChecked():
                raise ValueError(
                    f"YOLO dataset output already exists: {output_root}\n"
                    "Enable overwrite or choose a new output folder."
                )
            if self.save_split_check.isChecked():
                if split_output_root.resolve() == output_root.resolve():
                    raise ValueError(
                        "Full split output must be different from YOLO output."
                    )
                if split_output_root.exists() and not self.overwrite_check.isChecked():
                    raise ValueError(
                        f"Full split output already exists: {split_output_root}\n"
                        "Enable overwrite or choose a new output folder."
                    )
            if not 0.0 <= overlap < 1.0:
                raise ValueError("Overlap must be in [0, 1).")

        args = [
            "-m",
            "src.dataset.prepare_yolo_dataset",
            "--src",
            str(src_root),
            "--output-root",
            str(output_root),
            "--train-ratio",
            f"{train_ratio:.4f}",
            "--valid-ratio",
            f"{valid_ratio:.4f}",
            "--test-ratio",
            f"{test_ratio:.4f}",
            "--patch-size",
            str(patch_size),
            "--overlap",
            f"{overlap:.4f}",
            "--train-negative-ratio",
            f"{self.train_negative_spin.value():.4f}",
            "--valid-negative-ratio",
            f"{self.valid_negative_spin.value():.4f}",
            "--test-negative-ratio",
            f"{self.test_negative_spin.value():.4f}",
            "--seed",
            str(self.split_seed_spin.value()),
            "--negative-seed",
            str(self.patch_seed_spin.value()),
        ]

        if self.overwrite_check.isChecked():
            args.append("--overwrite")

        if self.save_split_check.isChecked():
            args.extend(["--split-output-root", str(split_output_root)])

        return args

    def _start_process(self, job_name: str, args: list[str], start_message: str) -> None:
        """
            Start a processing command with shared QProcess wiring.
        """

        self.log_edit.clear()
        self._active_job_name = job_name
        self._last_log_group = ""
        self._append_log(start_message)
        self.progress_label.setText(start_message)
        self.progress_bar.setRange(0, 0)

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self._set_running(True)
        self.process.start(sys.executable, args)

    def _read_stdout(self) -> None:
        """
            Append process stdout to the log console.
        """

        if self.process is None:
            return

        text = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        self._append_log(text.rstrip())

    def _read_stderr(self) -> None:
        """
            Append process stderr to the log console.
        """

        if self.process is None:
            return

        text = bytes(self.process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )
        self._append_log(text.rstrip())

    def _process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """
            Handle process completion and refresh summaries.
        """

        job_name = self._active_job_name or "Process"

        if exit_code == 0:
            self.progress_label.setText(f"{job_name} finished.")
            self._append_log(f"{job_name} finished with exit code 0.")
            data_yaml = self._resolve_path(self.output_edit.text()) / "data.yaml"
            self._append_log(f"YOLO data.yaml: {data_yaml}")
            self.check_yolo_dataset()
        else:
            self.progress_label.setText(f"{job_name} failed.")
            self._append_log(f"{job_name} failed with exit code {exit_code}.")

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if exit_code == 0 else 0)
        self._set_running(False)
        self.process = None
        self._active_job_name = ""

    def _process_error(self, error: QProcess.ProcessError) -> None:
        """
            Show QProcess startup/runtime errors.
        """

        self._append_log(f"Process error: {error.name}")

    def _set_running(self, running: bool) -> None:
        """
            Enable or disable controls while a background job is active.
        """

        self.prepare_button.setEnabled(not running)
        self.check_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _append_log(self, text: str) -> None:
        """
            Append text to the log console and keep the newest lines visible.
        """

        if not text:
            return

        for line in text.splitlines():
            self._append_log_line(line.rstrip())

    def _append_log_line(self, line: str) -> None:
        """
            Append one log line and separate major output groups.
        """

        if not line:
            return

        group = self._log_group_key(line)
        if self._should_insert_separator(group):
            self._append_separator()

        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        if self.log_edit.toPlainText():
            self.log_edit.insertPlainText("\n")
        self.log_edit.insertPlainText(line)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        self._last_log_group = group

    def _log_group_key(self, line: str) -> str:
        """
            Classify one process log line into a visual group.
        """

        if line.startswith("Preparing YOLO dataset"):
            return "start"

        if line.startswith(
            (
                "[train] source_images=",
                "[valid] source_images=",
                "[test] source_images=",
            )
        ):
            return "patch_summary"

        if line.startswith(
            (
                "[train] exported full split images:",
                "[valid] exported full split images:",
                "[test] exported full split images:",
                "[train] exported source images:",
                "[valid] exported source images:",
                "[test] exported source images:",
            )
        ):
            return "export_progress"

        if line.startswith(
            (
                "DONE:",
                "Source:",
                "Output root:",
                "YOLO data:",
                "Full split output:",
            )
        ):
            return "output_summary"

        if line.startswith(("train:", "valid:", "test:")):
            return "output_summary"

        if line.startswith("Train with:") or line.startswith("  .venv/bin/python"):
            return "train_command"

        if "finished with exit code" in line or "failed with exit code" in line:
            return "process_result"

        if line.startswith("Checked YOLO dataset:") or line.startswith(("- Train:", "- Valid:", "- Test:")):
            return "dataset_check"

        if line == CHECK_LOG_SEPARATOR:
            return "separator"

        return self._last_log_group or "misc"

    def _should_insert_separator(self, group: str) -> bool:
        """
            Return True when the next log line starts a new major group.
        """

        if not group or group == "separator":
            return False

        if not self.log_edit.toPlainText():
            return False

        if not self._last_log_group or self._last_log_group == "separator":
            return False

        return group != self._last_log_group

    def _append_separator(self) -> None:
        """
            Append the shared long separator line once.
        """

        current_text = self.log_edit.toPlainText()
        if current_text.endswith(CHECK_LOG_SEPARATOR):
            return

        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        if current_text:
            self.log_edit.insertPlainText("\n")
        self.log_edit.insertPlainText(CHECK_LOG_SEPARATOR)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        self._last_log_group = "separator"

    def _append_check_log(self, text: str) -> None:
        """
            Append one dataset check block with a visible separator.
        """

        self._append_log(f"{text}\n{CHECK_LOG_SEPARATOR}")

    def _collect_yolo_dataset_summary(self, data_yaml: Path) -> dict[str, Any]:
        """
            Load data.yaml and count YOLO image-label pairs.
        """

        data = self._load_yaml(data_yaml)
        dataset_root = self._resolve_dataset_root(data_yaml=data_yaml, data=data)

        missing_keys = []
        for split in SPLITS:
            try:
                self._get_split_value(data=data, split=split)
            except KeyError:
                missing_keys.append(split)

        if missing_keys:
            raise ValueError(f"data.yaml missing split keys: {', '.join(missing_keys)}")

        if not data.get("names"):
            raise ValueError("data.yaml missing class names.")

        summary: dict[str, Any] = {
            "ok": True,
            "splits": {},
        }

        for split in SPLITS:
            image_dir = self._resolve_split_image_dir(
                dataset_root=dataset_root,
                data_yaml=data_yaml,
                split_value=self._get_split_value(data=data, split=split),
            )
            label_dir = self._image_dir_to_label_dir(image_dir)

            if not image_dir.is_dir():
                raise FileNotFoundError(
                    f"{split} images directory not found: "
                    f"{self._display_path(image_dir)}"
                )
            if not label_dir.is_dir():
                raise FileNotFoundError(
                    f"{split} labels directory not found: "
                    f"{self._display_path(label_dir)}"
                )

            images = sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            labels = sorted(path for path in label_dir.iterdir() if path.suffix == ".txt")

            image_stems = {path.stem for path in images}
            label_stems = {path.stem for path in labels}
            missing_labels = image_stems - label_stems
            orphan_labels = label_stems - image_stems

            if not images or missing_labels or orphan_labels:
                summary["ok"] = False

            summary["splits"][split] = {
                "images": len(images),
                "labels": len(labels),
                "missing_labels": len(missing_labels),
                "orphan_labels": len(orphan_labels),
            }

        return summary

    def _get_split_value(self, data: dict[str, Any], split: str) -> Any:
        """
            Return one split image directory from data.yaml.
        """

        keys = ("valid", "val") if split == "valid" else (split,)
        for key in keys:
            if key in data:
                return data[key]

        raise KeyError(split)

    def _load_yaml(self, data_yaml: Path) -> dict[str, Any]:
        """
            Load a YOLO data.yaml file.
        """

        if not data_yaml.is_file():
            raise FileNotFoundError(
                f"YOLO data YAML not found: {self._display_path(data_yaml)}"
            )

        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to inspect YOLO data.yaml.") from exc

        with data_yaml.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YOLO data YAML: {self._display_path(data_yaml)}")

        return data

    def _resolve_dataset_root(
        self,
        data_yaml: Path,
        data: dict[str, Any],
    ) -> Path:
        """
            Resolve YOLO dataset root from data.yaml.
        """

        root_value = data.get("path")
        if root_value is None:
            return data_yaml.parent.resolve()

        root = Path(str(root_value)).expanduser()
        if not root.is_absolute():
            root = data_yaml.parent / root

        return root.resolve()

    def _resolve_split_image_dir(
        self,
        dataset_root: Path,
        data_yaml: Path,
        split_value: Any,
    ) -> Path:
        """
            Resolve one split image directory from a YOLO data.yaml value.
        """

        split_path = Path(str(split_value)).expanduser()
        if split_path.is_absolute():
            return split_path

        root_candidate = (dataset_root / split_path).resolve()
        if root_candidate.exists():
            return root_candidate

        return (data_yaml.parent / split_path).resolve()

    def _image_dir_to_label_dir(self, image_dir: Path) -> Path:
        """
            Infer YOLO labels directory from an images directory.
        """

        if image_dir.name == "images":
            return image_dir.with_name("labels")

        return image_dir.parent / "labels"

    def _resolve_path(self, text: str) -> Path:
        """
            Resolve user-entered paths relative to the project root.
        """

        path = Path(text.strip()).expanduser()
        if path.is_absolute():
            return path
        return ROOT / path

    def _browse_source(self) -> None:
        """
            Browse for labeling output root.
        """

        self._browse_directory(self.source_edit, "Select labeled dataset root")

    def _browse_output(self) -> None:
        """
            Browse for YOLO dataset output root.
        """

        self._browse_directory(self.output_edit, "Select YOLO dataset output root")

    def _browse_split_output(self) -> None:
        """
            Browse for full-image split dataset output root.
        """

        self._browse_directory(
            self.split_output_edit,
            "Select full-image split dataset output root",
        )

    def _sync_split_output_controls(self) -> None:
        """
            Enable full-image split output controls only when requested.
        """

        enabled = self.save_split_check.isChecked()
        self.split_output_edit.setEnabled(enabled)
        self.split_output_button.setEnabled(enabled)

    def _browse_directory(self, edit: QLineEdit, title: str) -> None:
        """
            Open a folder picker and write the selected path into a line edit.
        """

        start_dir = self._resolve_path(edit.text())
        selected = QFileDialog.getExistingDirectory(self, title, str(start_dir))
        if selected:
            edit.setText(selected)

    def _format_split_name(self, split: str) -> str:
        """
            Format split names for user-facing log lines.
        """

        return split.strip().capitalize()

    def _display_path(self, path: Path) -> str:
        """
            Display project-relative paths when possible.
        """

        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    def _format_error_message(self, message: str) -> str:
        """
            Shorten verbose absolute-path errors for GUI display.
        """

        return message.replace(str(ROOT) + "/", "")
