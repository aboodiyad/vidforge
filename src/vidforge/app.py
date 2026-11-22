"""GUI entry point: `vidforge-gui`."""

from __future__ import annotations

import sys

from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .core import paths
from .core.jobs import JobQueue
from .core.settings import Settings
from .gui import i18n, theme
from .gui.main_window import MainWindow


APP_ID = "VidForge.VidForge"


def _taskbar_identity() -> None:
    """Show VidForge's own name and icon on the taskbar (not Python's)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except (AttributeError, OSError):
            pass


def main() -> int:
    _taskbar_identity()
    paths.ensure_dirs()
    settings = Settings.load()
    i18n.set_language(i18n.resolve(settings.language, QLocale.system().name()))
    app = QApplication(sys.argv)
    app.setApplicationName("VidForge")
    if i18n.is_rtl():
        app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    theme.apply(app, i18n.is_rtl())
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "gui" / "assets" / "vidforge.ico")))
    from .gui.first_run import FirstRunDialog

    if not FirstRunDialog().exec_if_needed():
        return 0
    window = MainWindow(JobQueue(workers=1), settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
