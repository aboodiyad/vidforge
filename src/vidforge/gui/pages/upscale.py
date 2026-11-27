"""Make bigger: pick videos, pick a size, pick Fast or Best quality, see what you'll get, start."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QSlider, QVBoxLayout, QWidget

from ...core import gpu as gpu_mod
from ...core import sizing, tools
from ...core.settings import Settings
from ...features.upscale import estimate as est
from ...features.upscale.nvencc_vsr import supports_av1_encode
from ...features.upscale.service import UpscaleRequest, output_size, run_upscale
from .. import i18n
from ..bridge import JobBridge
from ..i18n import tr
from ..tasks import run_in_background
from ..widgets.common import ChipGroup, Page, Tile, label
from ..widgets.options import MoreOptions, OutputRows, combo
from ..widgets.speedup import SpeedupCard
from ..widgets.video_list import ActionBar, VideoList

SIZE_ORDER = ("1080p", "1440p", "2160p", "2x", "3x", "4x")


def fast_status(gpu: gpu_mod.GpuInfo | None) -> tuple[bool, str]:
    """(Fast mode available, reason when it isn't), in plain words."""
    if not (gpu and gpu.vsr_capable):
        return False, tr("mode.fast.no_rtx", gpu=gpu.name if gpu else "—")
    if tools.locate("nvencc")[0] is None:
        return False, tr("mode.fast.no_tool")
    return True, ""


class UpscalePage(QWidget):
    def __init__(self, bridge: JobBridge, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.settings = settings
        self.gpu: gpu_mod.GpuInfo | None = None
        self.fast_ok = False
        self.mode = settings.upscale_mode if settings.upscale_mode in ("fast", "quality") else "auto"
        self.stats = est.load_stats()
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page = Page(tr("up.title"), tr("up.subtitle"))
        outer.addWidget(self.page, 1)

        self.page.section(tr("up.videos"))
        self.videos = VideoList()
        self.videos.changed.connect(self._recalc)
        self.page.add(self.videos)

        self.page.section(tr("up.size"))
        self.sizes = ChipGroup([(k, tr(f"size.{k}")) for k in SIZE_ORDER],
                               settings.upscale_size if settings.upscale_size in SIZE_ORDER else "2160p")
        self.sizes.changed.connect(self._size_changed)
        self.page.body.addWidget(self.sizes)
        self.page.body.addSpacing(8)
        self.size_desc = label("", "Hint", wrap=True)
        self.page.add(self.size_desc)

        self.page.section(tr("up.mode"))
        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.fast_tile = Tile("bolt", tr("mode.fast"), tr("mode.fast.desc"))
        self.quality_tile = Tile("sparkle", tr("mode.quality"), tr("mode.quality.desc"))
        self.fast_tile.clicked.connect(lambda: self._set_mode("fast"))
        self.quality_tile.clicked.connect(lambda: self._set_mode("quality"))
        for t in (self.fast_tile, self.quality_tile):
            t.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            tiles.addWidget(t, 1)
        holder = QWidget()
        holder.setLayout(tiles)
        self.page.add(holder, 16)
        self.fast_tile.estimate.setText(tr("mode.checking"))
        self.speedup = SpeedupCard(bridge, on_done=self._recalc)
        self.page.add(self.speedup.box, 16)

        self.more = MoreOptions()
        self.model = combo([(k, tr(f"model.{k}")) for k in ("general", "max", "anime")], settings.upscale_model,
                           self._options_changed)
        self.vsr = QSlider(Qt.Orientation.Horizontal)
        self.vsr.setRange(1, 4)
        self.vsr.setPageStep(1)
        self.vsr.setValue(settings.upscale_vsr_quality)
        self.vsr.setMaximumWidth(220)
        self.vsr_value = label(str(settings.upscale_vsr_quality), "Strong")
        self.vsr.valueChanged.connect(lambda v: (self.vsr_value.setText(str(v)), self._options_changed()))
        vsr_row = QHBoxLayout()
        vsr_row.addWidget(self.vsr)
        vsr_row.addWidget(self.vsr_value)
        vsr_row.addStretch(1)
        self.more.row(tr("more.model"), self.model)
        self.more.row(tr("more.vsr"), vsr_row, tr("more.vsr.help"))
        self.output = OutputRows(self.more, settings, self._recalc)
        self.page.body.addWidget(self.more.button, 0, Qt.AlignmentFlag.AlignLeading)
        self.page.body.addSpacing(10)
        self.page.body.addWidget(self.more.panel)
        self.page.finish()

        self.bar = ActionBar(self._start)
        outer.addWidget(self.bar)

        bridge.job_updated.connect(self._on_job)
        self._size_changed(self.sizes.value(), save=False)
        run_in_background(gpu_mod.detect_gpus, self._on_gpus)

    # --- drag and drop anywhere on the page ---
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.videos.drop.set_hover(True)

    def dragLeaveEvent(self, event) -> None:
        self.videos.drop.set_hover(False)

    def dropEvent(self, event) -> None:
        self.videos.drop.set_hover(False)
        self.videos.add_dropped(event.mimeData().urls())

    def add_paths(self, paths: list[str]) -> None:
        self.videos.add_paths(paths)

    # --- options ---
    def _options_changed(self) -> None:
        self.settings.upscale_model = self.model.currentData()
        self.settings.upscale_vsr_quality = self.vsr.value()
        self.settings.save()
        self._recalc()

    def _on_gpus(self, gpus) -> None:
        gpus = gpus if isinstance(gpus, list) else []
        self.gpu = next((g for g in gpus if g.index == self.settings.gpu_index), gpus[0] if gpus else None)
        self.fast_ok, reason = fast_status(self.gpu)
        self.fast_tile.set_available(self.fast_ok, reason)
        rtx = bool(self.gpu and self.gpu.vsr_capable)
        self.output.allow_av1(supports_av1_encode(self.gpu) or not rtx)
        if self.mode == "auto" or (self.mode == "fast" and not self.fast_ok):
            self.mode = "fast" if self.fast_ok else "quality"
        self._set_mode(self.mode, save=False)

    def _set_mode(self, mode: str, save: bool = True) -> None:
        if mode == "fast" and not self.fast_ok:
            return
        self.mode = mode
        self.fast_tile.set_selected(mode == "fast")
        self.quality_tile.set_selected(mode == "quality")
        self.model.setEnabled(mode == "quality")
        self.vsr.setEnabled(mode == "fast")
        self.speedup.refresh(self.gpu, mode == "quality")
        if save:
            self.settings.upscale_mode = mode
            self.settings.save()
        self._recalc()

    def _size_changed(self, value: str, save: bool = True) -> None:
        self.size_desc.setText(tr(f"size.{value}.desc"))
        if save:
            self.settings.upscale_size = value
            self.settings.save()
        self._recalc()

    # --- estimates ---
    def _recalc(self) -> None:
        if not hasattr(self, "bar"):
            return
        s = self.settings
        totals = {"fast": [0.0, 0.0], "quality": [0.0, 0.0]}
        download = 0.0
        ready = 0
        for v in self.videos.videos:
            v.estimates = {}
            v.size_error = False
            if v.info is None:
                continue
            try:
                w, h = output_size(v.info, self.sizes.value())
            except sizing.SizeError:
                v.size_error = True
                continue
            ready += 1
            for mode in ("fast", "quality"):
                e = est.estimate(v.info, w, h, mode=mode, model=s.upscale_model, codec=s.upscale_codec,
                                 quality=s.upscale_quality, container=s.upscale_container, stats=self.stats)
                v.estimates[mode] = e
                totals[mode][0] += e.seconds
                totals[mode][1] += e.bytes
                if mode == "quality":
                    download = max(download, e.download_mb)
        mode = self.mode if self.mode in ("fast", "quality") else "quality"
        self.videos.show_estimates(mode)
        for tile, m in ((self.fast_tile, "fast"), (self.quality_tile, "quality")):
            if tile.available():
                secs, size = totals[m]
                tile.estimate.setText(
                    f"{i18n.about_time(secs)}  ·  {i18n.about(i18n.size(est.round_nice(size)))}" if ready else "")
        secs, size = totals[mode]
        self.bar.show_totals(len(self.videos.videos), ready if self.mode in ("fast", "quality") else 0,
                             secs, size, download if mode == "quality" else 0)

    def _on_job(self, job) -> None:
        if job.state.finished:
            self.stats = est.load_stats()  # a finished job may have taught the estimator
            QTimer.singleShot(0, self._recalc)

    # --- start ---
    def _start(self) -> None:
        s = self.settings
        out_dir = Path(s.output_dir) if s.output_dir else None
        started = 0
        for row in list(self.videos.rows):
            v = row.video
            if v.info is None or v.size_error or not v.estimates:
                continue
            req = UpscaleRequest(
                input=v.path, output_dir=out_dir, size=self.sizes.value(), mode=self.mode,
                codec=s.upscale_codec, quality=s.upscale_quality, container=s.upscale_container,
                vsr_quality=s.upscale_vsr_quality, model=s.upscale_model, gpu_index=s.gpu_index,
            )
            self.bridge.queue.submit(v.path.name, lambda ctx, r=req: run_upscale(r, ctx))
            self.videos.remove_row(row)
            started += 1
        if started:
            self.bar.show_added()
