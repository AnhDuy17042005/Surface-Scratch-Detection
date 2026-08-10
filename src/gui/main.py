from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.gui.camera_tab import CameraTab
    from src.gui.data_tab import DataTab
    from src.gui.sahi import SahiInferenceTab
    from src.gui.sam2_tab import SAM2Tab
    from src.gui.training_tab import TrainingTab
else:
    from .camera_tab import CameraTab
    from .data_tab import DataTab
    from .sahi import SahiInferenceTab
    from .sam2_tab import SAM2Tab
    from .training_tab import TrainingTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowTitle("Surface Scratch Detection")
        self.resize(1300, 900)

        self.inference_tab = SahiInferenceTab()
        self.training_tab = TrainingTab()
        self.data_tab = DataTab()
        self.annotation_tab = SAM2Tab()
        self.camera_tab = CameraTab()

        tabs = QTabWidget()
        tabs.addTab(self.inference_tab, "Inference")
        tabs.addTab(self.training_tab,  "Training")
        tabs.addTab(self.data_tab,      "Data Processing")
        tabs.addTab(self.annotation_tab, "Annotation")
        tabs.addTab(self.camera_tab,    "Camera")

        self.setCentralWidget(tabs)
        self.inference_tab.edit_prediction_requested.connect(
            self.open_prediction_editor
        )

    def open_prediction_editor(self, image_path: Path, mask: np.ndarray) -> None:
        """
            Open the Annotation tab with a SAHI prediction as draft label.
        """

        loaded = self.annotation_tab.load_prediction_for_labeling(
            image_path=Path(image_path),
            mask=mask,
            source="sahi_prediction",
        )
        if loaded:
            tabs = self.centralWidget()
            if isinstance(tabs, QTabWidget):
                tabs.setCurrentWidget(self.annotation_tab)

    def closeEvent(self, event) -> None:
        """Let child tabs stop background threads before the window closes."""

        tabs = self.centralWidget()
        if isinstance(tabs, QTabWidget):
            for index in range(tabs.count()):
                tabs.widget(index).close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
