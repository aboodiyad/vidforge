"""Fix faces: pick videos, pick Natural or Strong repair, how much, and whether to also make
the video bigger. Shows the result size, file size and time before starting."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QSlider, QVBoxLayout, QWidget

from ...core import gpu as gpu_mod
from ...core import sizing
from ...core.settings import Settings
from ...features.face import estimate as face_est
from ...features.face.service import KEEP, FaceRequest, keep_size, run_faces
from ...features.upscale import estimate as up_est
from ...features.upscale.nvencc_vsr import supports_av1_encode
from ...features.upscale.service import output_size
from .. import i18n
from ..bridge import JobBridge
from ..i18n import tr
from ..tasks import run_in_background
from ..widgets.common import ChipGroup, Page, Tile, label
from ..widgets.options import MoreOptions, OutputRows, combo
from ..widgets.video_list import ActionBar, VideoList
from .upscale import SIZE_ORDER, fast_status

MODELS = ("gfpgan", "codeformer")


def _slider(lo: int, hi: int, value: float, scale: int = 10) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setPageStep(1)
    s.setValue(round(value * scale))
    s.setMinimumWidth(240)
    s.setMaximumWidth(360)
    return s


class FacesPage(QWidget):
    def __init__(self, bridge: JobBridge, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.settings = settings
        self.gpu: gpu_mod.GpuInfo | None = None
        self.fast_ok = False
        self.model = settings.face_model if settings.face_model in MODELS else "gfpgan"
        self.background = settings.face_background
        self.stats = up_est.load_stats()
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page = Page(tr("faces.title"), tr("faces.lead"))
        outer.addWidget(self.page, 1)

        self.page.section(tr("up.videos"))
        self.videos = VideoList()
        self.videos.changed.connect(self._recalc)
        self.page.add(self.videos)

        # which repair
        self.page.section(tr("fx.how"))
        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.tiles = {
            "gfpgan": Tile("face", tr("fx.gfpgan"), tr("fx.gfpgan.desc")),
            "codeformer": Tile("sparkle", tr("fx.codeformer"), tr("fx.codeformer.desc")),
        }
        for key, t in self.tiles.items():
            t.clicked.connect(lambda k=key: self._set_model(k))
            t.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            tiles.addWidget(t, 1)
        holder = QWidget()
        holder.setLayout(tiles)
        self.page.add(holder)

        # how much
        self.page.section(tr("fx.amount"), tr("fx.amount.help"))
        amount = QHBoxLayout()
        amount.setSpacing(12)
        self.strength = _slider(3, 10, settings.face_strength)
        self.strength.valueChanged.connect(self._amount_changed)
        amount.addWidget(label(tr("fx.amount.low"), "Body"))
        amount.addWidget(self.strength)
        amount.addWidget(label(tr("fx.amount.high"), "Body"))
        amount.addStretch(1)
        self.page.body.addLayout(amount)
        self.page.body.addSpacing(26)

        # bigger?
        self.page.section(tr("fx.bigger"))
        self.sizes = ChipGroup([(KEEP, tr("size.keep"))] + [(k, tr(f"size.{k}")) for k in SIZE_ORDER],
                               settings.face_size if settings.face_size in (KEEP, *SIZE_ORDER) else KEEP)
        self.sizes.changed.connect(self._size_changed)
        self.page.body.addWidget(self.sizes)
        self.page.body.addSpacing(8)
        self.size_desc = label("", "Hint", wrap=True)
        self.page.body.addWidget(self.size_desc)
        self.bg_box = QWidget()
        bg_lay = QVBoxLayout(self.bg_box)
        bg_lay.setContentsMargins(0, 14, 0, 0)
        bg_lay.setSpacing(6)
        bg_lay.addWidget(label(tr("fx.bg"), "Strong"))
        self.bg = ChipGroup([("fast", tr("fx.bg.fast")), ("quality", tr("fx.bg.quality"))],
                            self.background if self.background in ("fast", "quality") else "quality")
        self.bg.changed.connect(self._bg_changed)
        bg_lay.addWidget(self.bg)
        bg_lay.addWidget(label(tr("fx.bg.help"), "Hint", wrap=True))
        self.page.add(self.bg_box)

        # more options
        self.more = MoreOptions()
        self.fidelity = _slider(0, 10, settings.face_fidelity)
        self.fidelity.valueChanged.connect(self._amount_changed)
        self.more.row(tr("fx.fidelity"), self.fidelity, tr("fx.fidelity.help"))
        self.bg_model = combo([(k, tr(f"model.{k}")) for k in ("general", "max", "anime")], settings.upscale_model,
                              self._bg_model_changed)
        self.more.row(tr("more.model"), self.bg_model)
        self.output = OutputRows(self.more, settings, self._recalc)
        self.page.body.addWidget(self.more.button, 0, Qt.AlignmentFlag.AlignLeading)
        self.page.body.addSpacing(10)
        self.page.body.addWidget(self.more.panel)
        self.page.finish()

        self.bar = ActionBar(self._start)
        outer.addWidget(self.bar)

        bridge.job_updated.connect(self._on_job)
        self._set_model(self.model, save=False)
        self._size_changed(self.sizes.value(), save=False)
        run_in_background(gpu_mod.detect_gpus, self._on_gpus)

    # --- drag and drop ---
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

    # --- choices ---
    def _on_gpus(self, gpus) -> None:
        gpus = gpus if isinstance(gpus, list) else []
        self.gpu = next((g for g in gpus if g.index == self.settings.gpu_index), gpus[0] if gpus else None)
        self.fast_ok, reason = fast_status(self.gpu)
        fast_btn = self.bg.buttons["fast"]
        fast_btn.setEnabled(self.fast_ok)
        fast_btn.setToolTip(reason)
        rtx = bool(self.gpu and self.gpu.vsr_capable)
        self.output.allow_av1(supports_av1_encode(self.gpu) or not rtx)
        if self.background not in ("fast", "quality") or (self.background == "fast" and not self.fast_ok):
            self.background = "fast" if self.fast_ok else "quality"
        self.bg.set_value(self.background)
        self._recalc()

    def _set_model(self, key: str, save: bool = True) -> None:
        self.model = key
        for k, t in self.tiles.items():
            t.set_selected(k == key)
        self.fidelity.setEnabled(key == "codeformer")
        if save:
            self.settings.face_model = key
            self.settings.save()
        self._recalc()

    def _amount_changed(self, *_args) -> None:
        self.settings.face_strength = self.strength.value() / 10
        self.settings.face_fidelity = self.fidelity.value() / 10
        self.settings.save()

    def _size_changed(self, value: str, save: bool = True) -> None:
        self.size_desc.setText(tr("size.keep.desc") if value == KEEP else tr(f"size.{value}.desc"))
        self.bg_box.setVisible(value != KEEP)
        if save:
            self.settings.face_size = value
            self.settings.save()
        self._recalc()

    def _bg_changed(self, value: str) -> None:
        self.background = value
        self.settings.face_background = value
        self.settings.save()
        self._recalc()

    def _bg_model_changed(self) -> None:
        self.settings.upscale_model = self.bg_model.currentData()
        self.settings.save()
        self._recalc()

    # --- estimates ---
    def _target(self, v) -> tuple[int, int, str]:
        size = self.sizes.value()
        if size == KEEP:
            w, h = keep_size(v.info)
            return w, h, KEEP
        w, h = output_size(v.info, size)
        return w, h, self.background if self.background in ("fast", "quality") else "quality"

    def _recalc(self) -> None:
        if not hasattr(self, "bar"):
            return
        s = self.settings
        totals = {m: [0.0, 0.0] for m in MODELS}
        download = {m: 0.0 for m in MODELS}
        ready = 0
        for v in self.videos.videos:
            v.estimates = {}
            v.size_error = False
            if v.info is None:
                continue
            try:
                w, h, bg = self._target(v)
            except sizing.SizeError:
                v.size_error = True
                continue
            ready += 1
            for m in MODELS:
                e = face_est.estimate(v.info, w, h, background=bg, model=m, esrgan_model=s.upscale_model,
                                      codec=s.upscale_codec, quality=s.upscale_quality,
                                      container=s.upscale_container, stats=self.stats)
                v.estimates[m] = e
                totals[m][0] += e.seconds
                totals[m][1] += e.bytes
                download[m] = max(download[m], e.download_mb)
        self.videos.show_estimates(self.model)
        for m, tile in self.tiles.items():
            secs, size = totals[m]
            tile.estimate.setText(
                f"{i18n.about_time(secs)}  ·  {i18n.about(i18n.size(up_est.round_nice(size)))}" if ready else "")
        secs, size = totals[self.model]
        self.bar.show_totals(len(self.videos.videos), ready, secs, size, download[self.model])

    def _on_job(self, job) -> None:
        if job.state.finished:
            self.stats = up_est.load_stats()
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
            req = FaceRequest(
                input=v.path, output_dir=out_dir, size=self.sizes.value(),
                background=self.background if self.background in ("fast", "quality") else "auto",
                model=self.model, strength=s.face_strength, fidelity=s.face_fidelity,
                esrgan_model=s.upscale_model, vsr_quality=s.upscale_vsr_quality, codec=s.upscale_codec,
                quality=s.upscale_quality, container=s.upscale_container, gpu_index=s.gpu_index,
            )
            self.bridge.queue.submit(v.path.name, lambda ctx, r=req: run_faces(r, ctx))
            self.videos.remove_row(row)
            started += 1
        if started:
            self.bar.show_added()
