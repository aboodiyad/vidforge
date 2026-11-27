"""The "NVIDIA speed-up" card: offers the TensorRT add-on on RTX cards that don't have it yet."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from ...core.jobs import Job, JobContext
from ...features.upscale import speedup
from .. import theme
from ..bridge import JobBridge
from ..i18n import tr
from .common import button, icon_label, label, panel


class SpeedupCard:
    def __init__(self, bridge: JobBridge, on_done=None):
        self.bridge = bridge
        self.on_done = on_done
        self.gpu = None
        self.wanted = False
        self.job: Job | None = None
        self.box, lay = panel(QHBoxLayout, margins=(18, 16, 18, 16), spacing=14)
        lay.addWidget(icon_label("bolt", 24, theme.ACCENT), 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(4)
        texts.addWidget(label(tr("speedup.title"), "Strong"))
        self.text = label("", "Body", wrap=True)
        texts.addWidget(self.text)
        lay.addLayout(texts, 1)
        self.btn = button(tr("speedup.install"))
        self.btn.clicked.connect(self._install)
        lay.addWidget(self.btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.box.setVisible(False)
        bridge.job_updated.connect(self._on_job)

    def refresh(self, gpu=None, wanted: bool | None = None) -> None:
        """*wanted*: whether the page currently uses Best quality (the card only helps then)."""
        if gpu is not None:
            self.gpu = gpu
        if wanted is not None:
            self.wanted = wanted
        installing = self.job is not None and not self.job.state.finished
        show = self.wanted and speedup.supported(self.gpu) and not speedup.is_installed()
        self.box.setVisible(show)
        if show:
            mb = speedup.download_mb(self.gpu)
            self.text.setText(tr("speedup.installing") if installing else tr("speedup.desc", mb=f"{mb:.0f}"))
            self.btn.setEnabled(not installing)

    def _install(self) -> None:
        gpu = self.gpu

        def work(ctx: JobContext):
            def prog(done: int, total: int | None) -> None:
                ctx.report(done / total if total else None, tr("speedup.downloading"), phase="downloading")

            return speedup.install(gpu, cancel=ctx.cancel, progress=prog, on_log=ctx.log)

        self.job = self.bridge.queue.submit(tr("speedup.job"), work)
        self.refresh()

    def _on_job(self, job: Job) -> None:
        if self.job is not None and job.id == self.job.id and job.state.finished:
            self.refresh()
            if self.on_done:
                self.on_done()
