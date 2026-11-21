"""Page shown for features that arrive in a later milestone."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    def __init__(self, title: str, milestone: str, summary: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        head = QLabel(f"<h2>{title}</h2>")
        body = QLabel(f"{summary}<br><br><i>Arrives in milestone {milestone}.</i>")
        body.setWordWrap(True)
        for w in (head, body):
            w.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(w)
        layout.addStretch(1)
