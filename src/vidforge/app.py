"""GUI entry point: `vidforge-gui`."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .core import paths
from .core.jobs import JobQueue
from .core.settings import Settings
from .gui.main_window import MainWindow


def main() -> int:
    paths.ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("VidForge")
    window = MainWindow(JobQueue(workers=1), Settings.load())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
