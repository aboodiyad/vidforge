"""Upscale page: pick videos, choose Fast / Quality and the output, queue jobs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ...core import gpu as gpu_mod
from ...core import sizing, tools
from ...core.settings import Settings
from ...features.upscale.models import QUALITY_MODELS
from ...features.upscale.nvencc_vsr import supports_av1_encode
from ...features.upscale.service import UpscaleRequest, run_upscale
from ..bridge import JobBridge
from ..tasks import run_in_background

VIDEO_FILTER = "Videos (*.mp4 *.mkv *.mov *.avi *.m4v *.ts *.m2ts *.webm *.wmv *.flv *.mpg *.mpeg *.vob);;All files (*)"
CUSTOM = "custom"


def _combo(items: list[tuple[str, str]], current: str) -> QComboBox:
    c = QComboBox()
    for value, label in items:
        c.addItem(label, value)
    i = c.findData(current)
    c.setCurrentIndex(max(0, i))
    return c


class UpscalePage(QWidget):
    def __init__(self, bridge: JobBridge, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.settings = settings
        self.gpu: gpu_mod.GpuInfo | None = None
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Upscale</h2>"))

        # --- input files ---
        files_box = QGroupBox("Videos (drag files here)")
        fl = QVBoxLayout(files_box)
        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        fl.addWidget(self.files)
        row = QHBoxLayout()
        for text, slot in (("Add videos…", self._add_files), ("Remove", self._remove_selected), ("Clear", self.files.clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        fl.addLayout(row)
        layout.addWidget(files_box, 1)

        # --- options ---
        opts = QGroupBox("Options")
        form = QFormLayout(opts)
        s = settings
        self.mode = _combo([
            ("auto", "Auto (Fast on RTX GPUs, otherwise Quality)"),
            ("fast", "Fast: NVIDIA RTX Video Super Resolution"),
            ("quality", "Quality: Real-ESRGAN (any GPU)"),
        ], s.upscale_mode)
        self.mode_note = QLabel("Detecting GPU…")
        self.mode_note.setStyleSheet("color: gray")
        self.size = _combo([(p, sizing.LABELS[p]) for p in sizing.PRESETS] + [(CUSTOM, "Custom…")],
                           s.upscale_size if s.upscale_size in sizing.PRESETS else CUSTOM)
        self.custom = QLineEdit(s.upscale_size if s.upscale_size not in sizing.PRESETS else "")
        self.custom.setPlaceholderText("e.g. 2560x1080")
        size_row = QHBoxLayout()
        size_row.addWidget(self.size, 1)
        size_row.addWidget(self.custom)
        self.model = _combo([(k, m.label) for k, m in QUALITY_MODELS.items()], s.upscale_model)
        self.vsr = QSpinBox()
        self.vsr.setRange(1, 4)
        self.vsr.setValue(s.upscale_vsr_quality)
        self.vsr.setToolTip("NGX VSR quality level: 4 is the highest quality, 1 the fastest")
        self.codec = _combo([("hevc", "HEVC (H.265) 10-bit"), ("h264", "H.264 8-bit"), ("av1", "AV1 10-bit")],
                            s.upscale_codec)
        self.quality = _combo([("high", "High (bigger file)"), ("balanced", "Balanced"), ("small", "Small file")],
                              s.upscale_quality)
        self.container = _combo([("mp4", "MP4"), ("mkv", "MKV (keeps every subtitle type)")], s.upscale_container)
        self.out_dir = QLineEdit(s.output_dir)
        self.out_dir.setPlaceholderText("Same folder as each video")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(browse)

        form.addRow("Mode", self.mode)
        form.addRow("", self.mode_note)
        form.addRow("Output size", size_row)
        form.addRow("Quality model", self.model)
        form.addRow("RTX VSR level", self.vsr)
        form.addRow("Video codec", self.codec)
        form.addRow("File size", self.quality)
        form.addRow("Container", self.container)
        form.addRow("Output folder", out_row)
        layout.addWidget(opts)

        self.start = QPushButton("Upscale")
        self.start.setDefault(True)
        self.start.clicked.connect(self._start)
        self.status = QLabel("")
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.start)
        layout.addLayout(bottom)

        for w in (self.mode, self.size):
            w.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()
        run_in_background(gpu_mod.detect_gpus, self._on_gpus)

    # --- GPU-dependent options ---
    def _on_gpus(self, gpus) -> None:
        gpus = gpus if isinstance(gpus, list) else []
        self.gpu = next((g for g in gpus if g.index == self.settings.gpu_index), gpus[0] if gpus else None)
        rtx = bool(self.gpu and self.gpu.vsr_capable)
        has_nvencc = tools.locate("nvencc")[0] is not None
        name = self.gpu.name if self.gpu else "no GPU found"
        if rtx and has_nvencc:
            note = f"{name}: Auto uses Fast mode."
        elif rtx:
            note = f"{name}: install NVEncC on the System page to enable Fast mode."
        else:
            note = f"{name}: Fast mode needs an NVIDIA RTX GPU, so Auto uses Quality mode."
        self.mode_note.setText(note)
        fast_item = self.mode.model().item(self.mode.findData("fast"))
        fast_item.setEnabled(rtx and has_nvencc)
        av1 = self.codec.model().item(self.codec.findData("av1"))
        av1.setEnabled(supports_av1_encode(self.gpu) or not rtx)
        self._update_enabled()

    def _update_enabled(self) -> None:
        mode = self.mode.currentData()
        self.custom.setVisible(self.size.currentData() == CUSTOM)
        self.model.setEnabled(mode != "fast")
        self.vsr.setEnabled(mode != "quality")

    # --- files ---
    def _add_paths(self, paths: list[str]) -> None:
        existing = {self.files.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files.count())}
        for p in paths:
            if p and p not in existing and Path(p).is_file():
                self.files.addItem(Path(p).name)
                item = self.files.item(self.files.count() - 1)
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setToolTip(p)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose videos", "", VIDEO_FILTER)
        self._add_paths(paths)

    def _remove_selected(self) -> None:
        for item in self.files.selectedItems():
            self.files.takeItem(self.files.row(item))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self._add_paths([u.toLocalFile() for u in event.mimeData().urls()])

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    # --- start ---
    def _size_value(self) -> str:
        return self.custom.text().strip() if self.size.currentData() == CUSTOM else self.size.currentData()

    def _start(self) -> None:
        if self.files.count() == 0:
            self.status.setText("Add at least one video.")
            return
        size = self._size_value()
        if self.size.currentData() == CUSTOM:
            try:
                sizing.parse_custom(size)
            except sizing.SizeError as e:
                self.status.setText(str(e))
                return
        s = self.settings
        s.upscale_mode = self.mode.currentData()
        s.upscale_size = size
        s.upscale_model = self.model.currentData()
        s.upscale_vsr_quality = self.vsr.value()
        s.upscale_codec = self.codec.currentData()
        s.upscale_quality = self.quality.currentData()
        s.upscale_container = self.container.currentData()
        s.output_dir = self.out_dir.text().strip()
        s.save()
        out_dir = Path(s.output_dir) if s.output_dir else None
        for i in range(self.files.count()):
            path = Path(self.files.item(i).data(Qt.ItemDataRole.UserRole))
            req = UpscaleRequest(
                input=path, output_dir=out_dir, size=size, mode=s.upscale_mode, codec=s.upscale_codec,
                quality=s.upscale_quality, container=s.upscale_container,
                vsr_quality=s.upscale_vsr_quality, model=s.upscale_model, gpu_index=s.gpu_index,
            )
            self.bridge.queue.submit(f"Upscale {path.name}", lambda ctx, r=req: run_upscale(r, ctx))
        self.status.setText(f"Queued {self.files.count()} video(s). Progress is in the Jobs panel.")
        self.files.clear()
