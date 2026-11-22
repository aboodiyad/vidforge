"""VidForge's look: a dark "projection booth". Graphite surfaces so the eye rests on the video
work, one warm amber (the projector lamp) for the main action and for progress.

All colours are defined once here as tokens; the stylesheet and custom-drawn widgets use them.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

ASSETS = Path(__file__).parent / "assets"
FONT_FAMILY = "IBM Plex Sans Arabic"  # one face for Latin and Arabic, OFL licence

# Surfaces step up in lightness; text steps down in emphasis.
BG = "#111215"         # window
RAIL = "#0C0D10"       # navigation rail
SURFACE = "#191B20"    # panels
RAISED = "#21242B"     # inputs, rows, tiles
RAISED_HOVER = "#282C34"
LINE = "#2D313A"       # hairlines
LINE_STRONG = "#3B404B"
TEXT = "#EEEEF0"
TEXT_2 = "#B4B8C2"     # 8.4:1 on SURFACE
TEXT_3 = "#8D929D"     # 5.0:1 on SURFACE: hints, captions
ACCENT = "#F3A93C"     # projector amber
ACCENT_HOVER = "#FFBD5A"
ACCENT_PRESS = "#D98F22"
ACCENT_SOFT = "#3A2C17"  # amber tint for selected tiles
ON_ACCENT = "#1B1206"
OK = "#62C98F"
DANGER = "#F07A72"
DANGER_SOFT = "#3A1F1E"

RADIUS = 10


def load_fonts() -> str:
    for f in sorted((ASSETS / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(f))
    return FONT_FAMILY if FONT_FAMILY in QFontDatabase.families() else QApplication.font().family()


def stylesheet(family: str, rtl: bool = False) -> str:
    return f"""
* {{ font-family: "{family}"; }}
QWidget {{ color: {TEXT}; font-size: 10pt; }}
QMainWindow, #Content {{ background: {BG}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE_STRONG}; padding: 6px 8px; border-radius: 6px; }}

/* navigation rail */
#Rail {{ background: {RAIL}; border: none; }}
#Brand {{ font-size: 17pt; font-weight: 700; color: {TEXT}; }}
#BrandTag {{ color: {TEXT_3}; font-size: 9pt; }}
#NavButton {{ border: none; border-radius: 8px; background: transparent; padding: 0; }}
#NavButton:hover {{ background: {SURFACE}; }}
#NavButton:checked {{ background: {RAISED}; }}
#NavButton:focus {{ background: {SURFACE}; }}  /* keyboard focus: soft, not mistaken for selection */
#NavText {{ color: {TEXT_2}; font-size: 10.5pt; font-weight: 500; }}
#NavText[active="true"] {{ color: {TEXT}; font-weight: 600; }}
#NavSoon {{ color: {TEXT_3}; font-size: 8pt; padding: 1px 6px; border: 1px solid {LINE_STRONG}; border-radius: 7px; }}

/* page text */
#PageTitle {{ font-size: 20pt; font-weight: 600; color: {TEXT}; }}
#PageLead {{ color: {TEXT_2}; font-size: 10.5pt; }}
#SectionTitle {{ font-size: 12pt; font-weight: 600; color: {TEXT}; }}
#Hint {{ color: {TEXT_3}; font-size: 9.5pt; }}
#Body {{ color: {TEXT_2}; }}
#Strong {{ color: {TEXT}; font-weight: 600; }}
#Ok {{ color: {OK}; }}
#Danger {{ color: {DANGER}; }}

/* panels */
#Panel {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: {RADIUS}px; }}
#DropZone {{ background: {SURFACE}; border: 1.5px dashed {LINE_STRONG}; border-radius: {RADIUS}px; }}
#DropZone[hover="true"] {{ border-color: {ACCENT}; background: {ACCENT_SOFT}; }}
#Row {{ background: {RAISED}; border-radius: 8px; }}
#Row #RowName {{ font-weight: 600; }}
#RowMeta {{ color: {TEXT_3}; font-size: 9.5pt; }}
#RowAfter {{ color: {TEXT}; font-weight: 600; }}
#RowNum {{ color: {TEXT_2}; }}

/* buttons */
QPushButton {{
    background: {RAISED}; border: 1px solid {LINE_STRONG}; border-radius: 8px;
    padding: 7px 14px; color: {TEXT}; font-weight: 500;
}}
QPushButton:hover {{ background: {RAISED_HOVER}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:disabled {{ color: {TEXT_3}; background: {SURFACE}; border-color: {LINE}; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
#Primary {{
    background: {ACCENT}; color: {ON_ACCENT}; border: none; font-weight: 700;
    font-size: 11.5pt; padding: 10px 30px; border-radius: 9px;
}}
#Primary:hover {{ background: {ACCENT_HOVER}; }}
#Primary:pressed {{ background: {ACCENT_PRESS}; }}
#Primary:disabled {{ background: {RAISED}; color: {TEXT_3}; }}
#Ghost {{ background: transparent; border: none; color: {TEXT_2}; padding: 6px 8px; }}
#Ghost:hover {{ color: {TEXT}; background: {RAISED}; }}
#IconButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px; }}
#IconButton:hover {{ background: {RAISED_HOVER}; }}

/* size chips */
#Chip {{
    background: {RAISED}; border: 1px solid {LINE}; border-radius: 17px;
    padding: 7px 16px; color: {TEXT_2}; font-weight: 500;
}}
#Chip:hover {{ color: {TEXT}; border-color: {LINE_STRONG}; }}
#Chip:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {TEXT}; font-weight: 600; }}

/* mode tiles */
#Tile {{ background: {SURFACE}; border: 1.5px solid {LINE}; border-radius: {RADIUS}px; }}
#Tile:hover {{ border-color: {LINE_STRONG}; }}
#Tile[selected="true"] {{ border-color: {ACCENT}; background: {ACCENT_SOFT}; }}
#Tile[disabled="true"] {{ background: {BG}; }}
#TileTitle {{ font-size: 12.5pt; font-weight: 700; }}
#TileEstimate {{ color: {ACCENT}; font-weight: 600; }}
#TileEstimate[muted="true"] {{ color: {TEXT_3}; font-weight: 500; }}

/* inputs */
QComboBox, QLineEdit, QSpinBox {{
    background: {RAISED}; border: 1px solid {LINE_STRONG}; border-radius: 7px;
    padding: 6px 10px; min-height: 20px; selection-background-color: {ACCENT}; selection-color: {ON_ACCENT};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {RAISED}; border: 1px solid {LINE_STRONG}; outline: none; padding: 4px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QRadioButton {{ spacing: 9px; padding: 4px 0; }}
QRadioButton::indicator {{ width: 16px; height: 16px; border-radius: 9px; border: 1px solid {LINE_STRONG}; background: {RAISED}; }}
QRadioButton::indicator:checked {{
    border: 1px solid {ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {ACCENT}, stop:0.48 {ACCENT}, stop:0.56 {RAISED}, stop:1 {RAISED});
}}
QSlider::groove:horizontal {{ height: 4px; background: {LINE_STRONG}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -7px 0; border-radius: 8px; background: {TEXT}; }}

/* progress */
QProgressBar {{ background: {LINE}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QProgressBar[state="done"]::chunk {{ background: {OK}; }}
QProgressBar[state="failed"]::chunk {{ background: {DANGER}; }}

/* jobs panel */
#JobsPanel {{ background: {SURFACE}; border-top: 1px solid {LINE}; }}
#JobRow {{ background: {RAISED}; border-radius: 8px; }}
#JobName {{ font-weight: 600; }}
#JobStatus {{ color: {TEXT_2}; font-size: 9.5pt; }}
#JobStatus[state="failed"] {{ color: {DANGER}; }}
#JobStatus[state="done"] {{ color: {OK}; }}
#JobTime {{ color: {TEXT}; font-size: 9.5pt; font-weight: 600; }}

/* scrolling */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {LINE_STRONG}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_3}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QPlainTextEdit {{ background: {BG}; border: 1px solid {LINE}; border-radius: 8px; color: {TEXT_2}; font-size: 9pt; }}
QDialog {{ background: {SURFACE}; }}
"""


def apply(app: QApplication, rtl: bool = False) -> None:
    app.setStyle("Fusion")
    family = load_fonts()
    app.setFont(QFont(family, 10))
    pal = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, RAISED),
        (QPalette.ColorRole.AlternateBase, SURFACE), (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Button, RAISED),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, ON_ACCENT), (QPalette.ColorRole.PlaceholderText, TEXT_3),
        (QPalette.ColorRole.ToolTipBase, RAISED), (QPalette.ColorRole.ToolTipText, TEXT),
        (QPalette.ColorRole.Link, ACCENT),
    ]:
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TEXT_3))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(TEXT_3))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TEXT_3))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(family, rtl))


def repolish(widget) -> None:
    """Re-apply the stylesheet after a dynamic property (e.g. selected) changes."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
