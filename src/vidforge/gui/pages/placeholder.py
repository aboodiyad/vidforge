"""Pages for parts that are coming soon: what they will do, in plain words."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from .. import theme
from ..i18n import tr
from ..widgets.common import Page, icon_label, label, panel


class ComingSoonPage(Page):
    def __init__(self, key: str, icon_name: str, parent=None):
        super().__init__(tr(f"{key}.title"), tr(f"{key}.desc"), parent)
        box, lay = panel(QHBoxLayout, margins=(20, 18, 20, 18), spacing=14)
        lay.addWidget(icon_label(icon_name, 28, theme.ACCENT), 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(4)
        texts.addWidget(label(tr("soon.badge"), "Strong"))
        texts.addWidget(label(tr("soon.note"), "Body", wrap=True))
        lay.addLayout(texts, 1)
        box.setMaximumWidth(640)
        self.add(box)
        self.finish()
