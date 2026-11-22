"""Small building blocks shared by the pages."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

from .. import icons, theme


def label(text: str = "", name: str = "Body", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName(name)
    lb.setWordWrap(wrap)
    lb.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
    return lb


def icon_label(name: str, size: int = 20, color: str = theme.TEXT_2) -> QLabel:
    lb = QLabel()
    lb.setPixmap(icons.pixmap(name, size, color))
    lb.setFixedSize(size, size)
    return lb


def button(text: str, icon_name: str | None = None, name: str | None = None) -> QPushButton:
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    if icon_name:
        color = theme.ON_ACCENT if name == "Primary" else theme.TEXT_2
        b.setIcon(icons.icon(icon_name, color, active=theme.TEXT if name != "Primary" else color))
        b.setIconSize(icons.ICON_SIZE if name == "Primary" else icons.ICON_SIZE * 0.85)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def icon_button(icon_name: str, tooltip: str, color: str = theme.TEXT_3) -> QPushButton:
    b = QPushButton()
    b.setObjectName("IconButton")
    b.setIcon(icons.icon(icon_name, color, active=theme.TEXT))
    b.setIconSize(icons.ICON_SIZE * 0.9)
    b.setToolTip(tooltip)
    b.setAccessibleName(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class Page(QScrollArea):
    """A scrolling page with a title and a one-line explanation under it."""

    def __init__(self, title: str, lead: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("Content")
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(40, 32, 40, 28)
        self.body.setSpacing(0)
        self.body.addWidget(label(title, "PageTitle"))
        self.body.addSpacing(6)
        lead_lb = label(lead, "PageLead", wrap=True)
        lead_lb.setMaximumWidth(760)
        self.body.addWidget(lead_lb)
        self.body.addSpacing(28)
        self.setWidget(inner)

    def section(self, title: str, hint: str | None = None) -> None:
        self.body.addSpacing(8)
        self.body.addWidget(label(title, "SectionTitle"))
        if hint:
            self.body.addSpacing(3)
            self.body.addWidget(label(hint, "Hint", wrap=True))
        self.body.addSpacing(12)

    def add(self, widget: QWidget, spacing_after: int = 26) -> None:
        self.body.addWidget(widget)
        self.body.addSpacing(spacing_after)

    def finish(self) -> None:
        self.body.addStretch(1)


def panel(layout_cls=QVBoxLayout, margins=(18, 16, 18, 16), spacing=10) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Panel")
    lay = layout_cls(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return f, lay


class ChipGroup(QWidget):
    """A row of pill buttons where exactly one is chosen (for sizes)."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], current: str, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for value, text in options:
            b = QPushButton(text)
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))
            self.group.addButton(b)
            self.buttons[value] = b
            lay.addWidget(b)
        lay.addStretch(1)
        self.set_value(current if current in self.buttons else options[0][0])

    def set_value(self, value: str) -> None:
        if value in self.buttons:
            self.buttons[value].setChecked(True)

    def value(self) -> str:
        return next(v for v, b in self.buttons.items() if b.isChecked())


class Tile(QFrame):
    """A big clickable choice with an icon, a title, a plain explanation and an estimate."""

    clicked = Signal()

    def __init__(self, icon_name: str, title: str, desc: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Tile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._enabled = True
        self.icon_name = icon_name
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(10)
        self.icon = icon_label(icon_name, 22, theme.ACCENT)
        top.addWidget(self.icon)
        self.title = label(title, "TileTitle")
        top.addWidget(self.title)
        top.addStretch(1)
        self.check = icon_label("check", 18, theme.ACCENT)
        top.addWidget(self.check)
        lay.addLayout(top)
        self.desc = label(desc, "Body", wrap=True)
        lay.addWidget(self.desc)
        lay.addSpacing(4)
        self.estimate = label("", "TileEstimate", wrap=True)
        lay.addWidget(self.estimate)
        self.set_selected(False)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", "true" if on else "false")
        self.check.setVisible(on)
        theme.repolish(self)

    def set_available(self, on: bool, reason: str = "") -> None:
        self._enabled = on
        self.setProperty("disabled", "false" if on else "true")
        self.setCursor(Qt.CursorShape.PointingHandCursor if on else Qt.CursorShape.ArrowCursor)
        self.icon.setPixmap(icons.pixmap(self.icon_name, 22, theme.ACCENT if on else theme.TEXT_3))
        self.title.setStyleSheet("" if on else f"color: {theme.TEXT_3};")
        self.estimate.setProperty("muted", "false" if on else "true")
        if not on:
            self.estimate.setText(reason)
        theme.repolish(self)
        theme.repolish(self.estimate)

    def available(self) -> bool:
        return self._enabled

    def mouseReleaseEvent(self, event) -> None:
        if self._enabled and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._enabled and event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit()
        else:
            super().keyPressEvent(event)
