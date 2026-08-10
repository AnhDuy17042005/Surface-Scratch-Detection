"""
    PySide6 training tab for scratch YOLO26n segmentation.

    Purpose:
        1. Configure the local scratch YOLO26n training run.
        2. Launch src.yolo.scratch.train in a background QProcess.
        3. Stream Ultralytics logs without blocking the GUI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QTextCursor
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
    QProgressBar,
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

YOLO26_SEG_BASE_MODELS = (
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolo26m-seg.pt",
    "yolo26l-seg.pt",
    "yolo26x-seg.pt",
)

from configs.data import SPLITS
from configs.yolo import (
    YOLO_SCRATCH_BASE_MODEL,
    YOLO_SCRATCH_BATCH_SIZE,
    YOLO_SCRATCH_DEVICE,
    YOLO_SCRATCH_EPOCHS,
    YOLO_SCRATCH_EXIST_OK,
    YOLO_SCRATCH_IMAGE_SIZE,
    YOLO_SCRATCH_LEARNING_RATE,
    YOLO_SCRATCH_PATIENCE,
    YOLO_SCRATCH_RUN_NAME,
    YOLO_SCRATCH_SEED,
    YOLO_SCRATCH_TRAIN_DATA,
    YOLO_SCRATCH_TRAIN_PROJECT,
    YOLO_SCRATCH_WORKERS,
)


class TrainingTab(QWidget):
    """
        Scratch YOLO26n training dashboard tab.

        This widget only starts the CLI training script in a background process.
        The model is still trained by Ultralytics, so the GUI remains a light
        control surface around the reproducible local command.
    """

    def __init__(self) -> None:
        super().__init__()

        """Process and log-format state"""
        self.process: QProcess | None = None
        self._printed_log_sections: set[str] = set()
        self._dynamic_log_line_active = False

        """Dataset controls"""
        self.data_yaml_edit = QLineEdit(str(YOLO_SCRATCH_TRAIN_DATA))

        """Run output controls"""
        self.project_edit = QLineEdit(str(YOLO_SCRATCH_TRAIN_PROJECT))
        self.run_name_edit = QLineEdit(YOLO_SCRATCH_RUN_NAME)
        self.exist_ok_check = QCheckBox("Allow writing into existing run folder")
        self.exist_ok_check.setChecked(YOLO_SCRATCH_EXIST_OK)

        """YOLO26n model controls"""
        self.model_edit = QComboBox()
        self.model_edit.addItems(YOLO26_SEG_BASE_MODELS)
        self.model_edit.setCurrentText(YOLO_SCRATCH_BASE_MODEL)
        self.pretrained_check = QCheckBox("Use pretrained weights")
        self.pretrained_check.setChecked(True)

        """Training hyperparameter controls"""
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(YOLO_SCRATCH_EPOCHS)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(YOLO_SCRATCH_BATCH_SIZE)

        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(32, 4096)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(YOLO_SCRATCH_IMAGE_SIZE)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.000001, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(YOLO_SCRATCH_LEARNING_RATE)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 10000)
        self.patience_spin.setValue(YOLO_SCRATCH_PATIENCE)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        self.num_workers_spin.setValue(YOLO_SCRATCH_WORKERS)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(YOLO_SCRATCH_SEED)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0"])
        self.device_combo.setCurrentText(
            "auto" if YOLO_SCRATCH_DEVICE is None else str(YOLO_SCRATCH_DEVICE)
        )

        """Validation controls"""
        self.validate_check = QCheckBox("Validate best checkpoint after training")
        self.validate_check.setChecked(True)

        self.plots_check = QCheckBox("Save Ultralytics plots")
        self.plots_check.setChecked(True)

        self.val_split_combo = QComboBox()
        self.val_split_combo.addItems(list(SPLITS))
        self.val_split_combo.setCurrentText("test")

        """Log output"""
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        """Progress widgets"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("YOLO training progress not started.")
        self.progress_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        """Action buttons"""
        self.start_button = QPushButton("Start YOLO Training")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        """Create UI and connect behavior"""
        self._style_widgets()
        self._build_layout()
        self._connect_signals()

    def _style_widgets(self) -> None:
        """
            Apply tab-local visual styling.
        """

        """Keep input heights compact and consistent"""
        inputs = (
            self.data_yaml_edit,
            self.project_edit,
            self.run_name_edit,
            self.model_edit,
            self.epochs_spin,
            self.batch_spin,
            self.image_size_spin,
            self.learning_rate_spin,
            self.patience_spin,
            self.num_workers_spin,
            self.seed_spin,
            self.device_combo,
            self.val_split_combo,
        )
        for widget in inputs:
            widget.setMinimumHeight(28)

        """Long paths should not force the left column to become too wide"""
        flexible_line_edits = (
            self.data_yaml_edit,
            self.project_edit,
            self.run_name_edit,
            self.model_edit,
        )
        for widget in flexible_line_edits:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )

        """Keep action button heights consistent with inputs"""
        for button in (
            self.start_button,
            self.stop_button,
        ):
            button.setMinimumHeight(28)

        self.progress_bar.setMinimumHeight(12)

    def _build_layout(self) -> None:
        """
            Build the full YOLO training tab layout.
        """

        """Right column contains compact configuration cards."""
        config_panel = QWidget()
        config_layout = QVBoxLayout()
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(10)
        config_layout.addWidget(self._build_dataset_group())
        config_layout.addWidget(self._build_run_config_group())
        config_layout.addWidget(self._build_training_config_group())
        config_layout.addWidget(self._build_action_group())
        config_layout.addStretch(1)
        config_panel.setLayout(config_layout)
        config_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )

        """Main content: large log on the left, compact config on the right."""
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        content_layout.addWidget(self._build_training_log_group(), stretch=3)
        content_layout.addWidget(config_panel, stretch=1)

        """Page layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(22, 14, 22, 14)
        layout.setSpacing(8)
        layout.addLayout(self._build_header())
        layout.addLayout(content_layout, stretch=1)
        self.setLayout(layout)

    def _build_header(self) -> QVBoxLayout:
        """
            Build the page title area.
        """

        title = QLabel("Train Scratch YOLO26n")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(title)
        return layout

    def _build_dataset_group(self) -> QGroupBox:
        """
            Build YOLO dataset controls.
        """

        group = QGroupBox("YOLO Dataset")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        data_row = self._path_row(
            edit=self.data_yaml_edit,
            button_text="Browse",
            slot=self._browse_data_yaml,
        )

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.addRow("Data YAML", data_row)

        layout.addLayout(form)
        group.setLayout(layout)
        return group

    def _build_run_config_group(self) -> QGroupBox:
        """
            Build output run controls.
        """

        group = QGroupBox("Run Output")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        project_row = self._path_row(
            edit=self.project_edit,
            button_text="Browse",
            slot=self._browse_project,
        )

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.addRow("Project dir", project_row)
        form.addRow("Run name", self.run_name_edit)
        form.addRow("Base model", self.model_edit)

        layout.addLayout(form)
        layout.addWidget(self.exist_ok_check)
        layout.addWidget(self.pretrained_check)
        group.setLayout(layout)
        return group

    def _build_training_config_group(self) -> QGroupBox:
        """
            Build YOLO26n training hyperparameter controls.
        """

        group = QGroupBox("Training Config")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(7)
        form.addRow("Epochs", self.epochs_spin)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("Image size", self.image_size_spin)
        form.addRow("Learning rate", self.learning_rate_spin)
        form.addRow("Patience", self.patience_spin)
        form.addRow("Num workers", self.num_workers_spin)
        form.addRow("Seed", self.seed_spin)
        form.addRow("Device", self.device_combo)
        form.addRow("Validation split", self.val_split_combo)

        layout.addLayout(form)
        layout.addWidget(self.validate_check)
        layout.addWidget(self.plots_check)
        group.setLayout(layout)
        return group

    def _build_action_group(self) -> QGroupBox:
        """
            Build training action buttons.
        """

        group = QGroupBox("Actions")
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        group.setLayout(layout)
        return group

    def _build_training_log_group(self) -> QGroupBox:
        """
            Build progress display and log console.
        """

        group = QGroupBox("Training Log")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_edit, stretch=1)
        group.setLayout(layout)
        return group

    def _path_row(self, edit: QLineEdit, button_text: str, slot) -> QHBoxLayout:
        """
            Build a path input row with a matching Browse button.
        """

        row = QHBoxLayout()
        row.setSpacing(10)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)

        browse_button = QPushButton(button_text)
        browse_button.setMinimumHeight(28)
        browse_button.clicked.connect(slot)
        row.addWidget(browse_button)
        return row

    def _connect_signals(self) -> None:
        """
            Connect user actions to tab behavior.
        """

        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)

    def start_training(self) -> None:
        """
            Validate settings and launch the YOLO training process.
        """

        if self.process is not None:
            return

        try:
            args = self._build_command_args(validate=True)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid YOLO training settings", str(exc))
            return

        """Reset log and progress state for the new run"""
        self.log_edit.clear()
        self._printed_log_sections.clear()
        self._dynamic_log_line_active = False
        self.progress_label.setText("Starting YOLO training...")
        self.progress_bar.setRange(0, 0)

        """Start src.yolo.scratch.train in a background process"""
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._training_finished)
        self.process.errorOccurred.connect(self._training_error)

        self._set_running(True)
        self.process.start(sys.executable, args)

    def stop_training(self) -> None:
        """
            Stop the active YOLO training process.
        """

        if self.process is None:
            return

        self._append_log("Stopping YOLO training process...")
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()

    def _build_command_args(self, validate: bool = False) -> list[str]:
        """
            Build arguments for python -m src.yolo.scratch.train.
        """

        data_yaml = self._resolve_path(self.data_yaml_edit.text())
        project_dir = self._resolve_path(self.project_edit.text())
        run_name = self.run_name_edit.text().strip()
        model_name = self.model_edit.currentText().strip()
        device = self.device_combo.currentText().strip()

        if validate:
            if not data_yaml.is_file():
                raise ValueError(
                    f"YOLO data YAML not found: {self._display_path(data_yaml)}"
                )
            if not run_name:
                raise ValueError("Run name cannot be empty.")
            if not model_name:
                raise ValueError("Base model cannot be empty.")

        args = [
            "-m",
            "src.yolo.scratch.train",
            "--data",
            str(data_yaml),
            "--model",
            model_name,
            "--project",
            str(project_dir),
            "--name",
            run_name,
            "--imgsz",
            str(self.image_size_spin.value()),
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch",
            str(self.batch_spin.value()),
            "--lr",
            f"{self.learning_rate_spin.value():.8f}",
            "--workers",
            str(self.num_workers_spin.value()),
            "--patience",
            str(self.patience_spin.value()),
            "--seed",
            str(self.seed_spin.value()),
            "--val-split",
            self.val_split_combo.currentText(),
        ]

        if device != "auto":
            args.extend(["--device", device])

        args.append("--exist-ok" if self.exist_ok_check.isChecked() else "--no-exist-ok")
        args.append(
            "--pretrained" if self.pretrained_check.isChecked() else "--no-pretrained"
        )
        args.append("--plots" if self.plots_check.isChecked() else "--no-plots")
        args.append("--no-check-data")
        args.append("--validate" if self.validate_check.isChecked() else "--no-validate")

        return args

    def _resolve_path(self, value: str) -> Path:
        """
            Convert user text into an absolute path.
        """

        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path

    def _browse_data_yaml(self) -> None:
        """
            Open a file dialog for selecting data.yaml.
        """

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO data.yaml",
            str(ROOT / "data"),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.data_yaml_edit.setText(path)

    def _browse_project(self) -> None:
        """
            Open a folder dialog for selecting the YOLO training project dir.
        """

        path = QFileDialog.getExistingDirectory(
            self,
            "Select YOLO project directory",
            self.project_edit.text() or str(ROOT / "models" / "yolo"),
        )
        if path:
            self.project_edit.setText(path)

    def _read_stdout(self) -> None:
        """
            Read stdout from the training process.
        """

        if self.process is None:
            return

        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self._handle_process_text(text)

    def _read_stderr(self) -> None:
        """
            Read stderr from the training process.
        """

        if self.process is None:
            return

        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        self._handle_process_text(text)

    def _training_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        """
            Handle process completion and restore button state.
        """

        status = "finished" if exit_status == QProcess.ExitStatus.NormalExit else "crashed"

        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_label.setText("YOLO training finished | 100%")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_label.setText(f"YOLO training {status} | exit code {exit_code}")

        self._append_log(f"YOLO training {status} with exit code {exit_code}.")
        self.process = None
        self._set_running(False)

    def _training_error(self, error: QProcess.ProcessError) -> None:
        """
            Log QProcess startup/runtime errors.
        """

        self._append_log(f"Process error: {error.name}")

    def _set_running(self, running: bool) -> None:
        """
            Toggle run buttons for active/inactive training states.
        """

        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

        if not running:
            self.progress_bar.setRange(0, 100)

    def _handle_process_text(self, text: str) -> None:
        """
            Route process text to progress parser and formatted log output.
        """

        for line in re.split(r"[\r\n]+", text):
            line = self._clean_log_line(line)
            if not line:
                continue

            self._update_training_progress(line)
            self._append_log(
                text=line,
                dynamic=self._is_dynamic_progress_line(line),
            )

    def _clean_log_line(self, line: str) -> str:
        """
            Remove terminal escape sequences from one process-output line.
        """

        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        return line.strip()

    def _update_training_progress(self, line: str) -> None:
        """
            Parse Ultralytics epoch progress when available.
        """

        batch_progress = self._parse_batch_progress(line)
        if batch_progress is not None:
            current_epoch, total_epochs, current_batch, total_batches = batch_progress
            epoch_fraction = current_batch / max(total_batches, 1)
            progress = ((current_epoch - 1) + epoch_fraction) / total_epochs
            percent = max(0, min(100, int(round(progress * 100))))

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(
                f"Epoch {current_epoch}/{total_epochs} | "
                f"batch {current_batch}/{total_batches} | {percent}%"
            )
            return

        """Ultralytics table rows commonly start with "epoch/epochs"."""
        epoch_match = re.match(r"^([0-9]+)\s*/\s*([0-9]+)\b", line)
        if not epoch_match:
            return

        current_epoch = int(epoch_match.group(1))
        total_epochs = int(epoch_match.group(2))
        if total_epochs <= 0:
            return

        percent = max(0, min(100, int(round(current_epoch * 100 / total_epochs))))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        self.progress_label.setText(
            f"Epoch {current_epoch}/{total_epochs} | {percent}%"
        )

    def _parse_batch_progress(self, line: str) -> tuple[int, int, int, int] | None:
        """
            Parse one Ultralytics tqdm-style training progress line.
        """

        match = re.match(
            r"^([0-9]+)\s*/\s*([0-9]+)\b.*?:\s*[0-9]+%.*?\b([0-9]+)\s*/\s*([0-9]+)\b",
            line,
        )
        if match is None:
            return None

        current_epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        current_batch = int(match.group(3))
        total_batches = int(match.group(4))
        if total_epochs <= 0 or total_batches <= 0:
            return None

        return current_epoch, total_epochs, current_batch, total_batches

    def _is_dynamic_progress_line(self, line: str) -> bool:
        """
            Return True for tqdm lines that should update in place.
        """

        return self._parse_batch_progress(line) is not None

    def _append_log(self, text: str, dynamic: bool = False) -> None:
        """
            Append one clean log line with section formatting.
        """

        if not text:
            return

        section = self._log_section(text)
        if section is not None:
            key, title = section
            if key not in self._printed_log_sections:
                self._append_blank_line_if_needed()
                self.log_edit.appendPlainText(title)
                self._printed_log_sections.add(key)
                self._dynamic_log_line_active = False

        if dynamic:
            self._append_dynamic_log_line(text)
            return

        self.log_edit.appendPlainText(text)
        self._dynamic_log_line_active = False
        self._scroll_log_to_end()

    def _append_dynamic_log_line(self, text: str) -> None:
        """
            Update the current tqdm progress row instead of creating a new row.
        """

        if not self._dynamic_log_line_active:
            self.log_edit.appendPlainText(text)
            self._dynamic_log_line_active = True
            self._scroll_log_to_end()
            return

        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(
            QTextCursor.MoveOperation.StartOfBlock,
            QTextCursor.MoveMode.KeepAnchor,
        )
        cursor.removeSelectedText()
        cursor.insertText(text)
        self.log_edit.setTextCursor(cursor)
        self._scroll_log_to_end()

    def _log_section(self, text: str) -> tuple[str, str] | None:
        """
            Map one log line to a visual section.
        """

        setup_prefixes = (
            "Checking YOLO scratch dataset",
            "data.yaml:",
            "dataset root:",
            "classes:",
            "Training YOLO scratch segmentation",
            "Base model:",
            "Data:",
            "Project:",
            "Run name:",
            "Device:",
            "Image size:",
            "Batch:",
            "Epochs:",
        )

        summary_prefixes = (
            "Run dir:",
            "Best checkpoint:",
            "Last checkpoint:",
            "Validating best checkpoint",
            "YOLO training finished",
            "YOLO training crashed",
        )

        if text.startswith(setup_prefixes):
            return "setup", "[Setup]"

        if re.match(r"^([0-9]+)\s*/\s*([0-9]+)\b", text):
            return "epochs", "[Epochs]"

        if text.startswith(summary_prefixes):
            return "summary", "[Summary]"

        return None

    def _append_blank_line_if_needed(self) -> None:
        """
            Add a blank line unless the log is still empty.
        """

        if self.log_edit.toPlainText():
            self.log_edit.appendPlainText("")
            self._dynamic_log_line_active = False

    def _scroll_log_to_end(self) -> None:
        """
            Keep the log console scrolled to the latest line.
        """

        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_edit.setTextCursor(cursor)

    def _display_path(self, path: Path) -> str:
        """
            Format paths compactly for GUI labels.
        """

        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = TrainingTab()
    window.resize(1100, 800)
    window.show()
    sys.exit(app.exec())
