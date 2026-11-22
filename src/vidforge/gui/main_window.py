"""Main window: navigation rail, the current page, and the job list under it."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.jobs import JobQueue
from ..core.settings import Settings
from . import icons, theme
from .bridge import JobBridge
from .i18n import tr
from .pages.faces import FacesPage
from .pages.placeholder import ComingSoonPage
from .pages.settings import SettingsPage
from .pages.subtitles import SubtitlesPage
from .pages.system import SystemPage
from .pages.upscale import UpscalePage
from .widgets.common import label
from .widgets.job_queue import JobQueueWidget


class NavButton(QPushButton):
    """A rail item: icon, name and an optional "Soon" tag, laid out with real labels so the
    order flips correctly in Arabic."""

    def __init__(self, icon_name: str, text: str, soon: bool = False):
        super().__init__()
        self.setObjectName("NavButton")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)  # focus ring for keyboard users only
        self.setAccessibleName(text)
        self.icon_name = icon_name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 10, 0)
        lay.setSpacing(11)
        self.icon_lb = QLabel()
        self.icon_lb.setFixedSize(20, 20)
        lay.addWidget(self.icon_lb)
        self.text_lb = label(text, "NavText")
        lay.addWidget(self.text_lb, 1)
        if soon:
            tag = label(tr("nav.soon"), "NavSoon")
            tag.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lay.addWidget(tag, 0, Qt.AlignmentFlag.AlignVCenter)
        for w in self.findChildren(QLabel):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.toggled.connect(self._paint)
        self._paint(False)

    def _paint(self, on: bool) -> None:
        self.icon_lb.setPixmap(icons.pixmap(self.icon_name, 20, theme.ACCENT if on else theme.TEXT_3))
        self.text_lb.setProperty("active", "true" if on else "false")
        theme.repolish(self.text_lb)


class MainWindow(QMainWindow):
    def __init__(self, queue: JobQueue, settings: Settings | None = None):
        super().__init__()
        self.setWindowTitle("VidForge")
        self.resize(1180, 820)
        self.setMinimumSize(940, 640)
        self.queue = queue
        self.settings = settings or Settings.load()
        self.bridge = JobBridge(queue, self)

        pages = [
            ("upscale", tr("nav.upscale"), False, UpscalePage(self.bridge, self.settings)),
            ("face", tr("nav.faces"), False, FacesPage(self.bridge, self.settings)),
            ("subtitles", tr("nav.subtitles"), False, SubtitlesPage(self.bridge, self.settings)),
            ("translate", tr("nav.dub"), True, ComingSoonPage("dub", "translate")),
        ]
        bottom_pages = [
            ("chip", tr("nav.system"), False, SystemPage(self.bridge)),
            ("settings", tr("nav.settings"), False, SettingsPage(self.bridge, self.settings)),
        ]

        rail = QFrame()
        rail.setObjectName("Rail")
        rail.setFixedWidth(252)
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(16, 24, 16, 18)
        rl.setSpacing(4)
        rl.addWidget(label("VidForge", "Brand"))
        tag = label(tr("app.tagline"), "BrandTag", wrap=True)
        rl.addWidget(tag)
        rl.addSpacing(26)

        self.stack = QStackedWidget()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        def add(entries) -> None:
            for icon_name, text, soon, page in entries:
                b = NavButton(icon_name, text, soon)
                idx = self.stack.addWidget(page)
                b.clicked.connect(lambda _=False, i=idx: self.stack.setCurrentIndex(i))
                self.group.addButton(b, idx)
                rl.addWidget(b)

        add(pages)
        rl.addStretch(1)
        add(bottom_pages)
        rl.addSpacing(8)
        rl.addWidget(label(f"v{__version__}", "BrandTag"))
        self.group.button(0).setChecked(True)

        content = QWidget()
        content.setObjectName("Content")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.stack, 1)
        cl.addWidget(JobQueueWidget(self.bridge))

        root = QWidget()
        hl = QHBoxLayout(root)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        hl.addWidget(rail)
        hl.addWidget(content, 1)
        self.setCentralWidget(root)

    def closeEvent(self, event) -> None:
        self.queue.shutdown()
        super().closeEvent(event)
