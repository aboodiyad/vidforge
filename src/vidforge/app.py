"""GUI entry point: `vidforge-gui`."""

from __future__ import annotations

import sys

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import QApplication

from .core import paths
from .core.jobs import JobQueue
from .core.settings import Settings
from .gui import i18n, theme
from .gui.main_window import MainWindow


def main() -> int:
    paths.ensure_dirs()
    settings = Settings.load()
    i18n.set_language(i18n.resolve(settings.language, QLocale.system().name()))
    app = QApplication(sys.argv)
    app.setApplicationName("VidForge")
    if i18n.is_rtl():
        app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    theme.apply(app, i18n.is_rtl())
    window = MainWindow(JobQueue(workers=1), settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
