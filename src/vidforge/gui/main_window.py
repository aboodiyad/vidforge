"""Main window: feature pages on the left, job queue at the bottom."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget, QHBoxLayout, QListWidget, QMainWindow, QStackedWidget, QWidget,
)

from .. import __version__
from ..core.jobs import JobQueue
from ..core.settings import Settings
from .bridge import JobBridge
from .pages.placeholder import PlaceholderPage
from .pages.system import SystemPage
from .pages.upscale import UpscalePage
from .widgets.job_queue import JobQueueWidget


class MainWindow(QMainWindow):
    def __init__(self, queue: JobQueue, settings: Settings | None = None):
        super().__init__()
        self.setWindowTitle(f"VidForge {__version__}")
        self.resize(1100, 780)
        self.queue = queue
        self.settings = settings or Settings.load()
        self.bridge = JobBridge(queue, self)

        pages = [
            ("Upscale", UpscalePage(self.bridge, self.settings)),
            ("Face Enhance", PlaceholderPage(
                "Face Enhance", "M2",
                "Face detection and tracking, GFPGAN or CodeFormer restoration, blended back "
                "into the upscaled frame.")),
            ("Subtitles", PlaceholderPage(
                "Subtitles", "M3", "Speech to SRT with whisper.cpp, optionally muxed as a soft track.")),
            ("Translate & Dub", PlaceholderPage(
                "Translate & Dub", "M4",
                "Whisper transcript, local LLM translation, voice-cloned TTS, muxed as the default "
                "audio track with a translated soft subtitle.")),
            ("System", SystemPage(self.bridge)),
        ]

        self.nav = QListWidget()
        self.nav.setFixedWidth(170)
        self.stack = QStackedWidget()
        for name, page in pages:
            self.nav.addItem(name)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        dock = QDockWidget("Jobs", self)
        dock.setObjectName("jobs")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(JobQueueWidget(self.bridge))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def closeEvent(self, event) -> None:
        self.queue.shutdown()
        super().closeEvent(event)
