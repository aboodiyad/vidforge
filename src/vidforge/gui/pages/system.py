"""My PC: the graphics card, whether Fast mode can run, and the helper programs."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

from ...core import gpu, paths, tools
from ...core.jobs import Job, JobContext, JobState
from .. import i18n, icons, theme
from ..bridge import JobBridge
from ..i18n import tr
from ..tasks import run_in_background
from ..widgets.common import Page, button, icon_label, label, panel

SHOWN_TOOLS = ("ffmpeg", "nvencc", "sevenzip", "whisper", "llama")


class ToolRow(QFrame):
    def __init__(self, key: str, on_install, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Row")
        self.key = key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 10, 10)
        lay.setSpacing(14)
        self.dot = icon_label("check", 18, theme.TEXT_3)
        lay.addWidget(self.dot)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label(tools.TOOLS[key].display, "RowName"))
        text.addWidget(label(tr(f"tool.{key}"), "RowMeta", wrap=True))
        lay.addLayout(text, 1)
        self.state = label("", "Hint")
        lay.addWidget(self.state)
        self.btn = button(tr("tool.install"))
        self.btn.clicked.connect(lambda: on_install(key))
        lay.addWidget(self.btn)

    def show_status(self, st: tools.ToolStatus) -> None:
        if st.found:
            self.dot.setPixmap(icons.pixmap("check", 18, theme.OK))
            key = "tool.installed" if st.origin == "managed" else "tool.found"
            self.state.setText(tr(key, v=i18n.ltr(st.version or "?")))
            self.btn.setText(tr("tool.reinstall"))
        else:
            self.dot.setPixmap(icons.pixmap("alert", 18, theme.TEXT_3))
            self.state.setText(tr("tool.missing"))
            self.btn.setText(tr("tool.install"))


class SystemPage(Page):
    def __init__(self, bridge: JobBridge, parent=None):
        super().__init__(tr("sys.title"), tr("sys.subtitle"), parent)
        self.bridge = bridge

        self.section(tr("sys.gpu"))
        box, lay = panel()
        self.gpu_line = label("…", "Strong", wrap=True)
        self.fast_line = label("", "Body", wrap=True)
        lay.addWidget(self.gpu_line)
        lay.addWidget(self.fast_line)
        row = QHBoxLayout()
        self.test_btn = button(tr("sys.test"), "bolt")
        self.test_btn.clicked.connect(self._test)
        self.test_msg = label("", "Body", wrap=True)
        row.addWidget(self.test_btn)
        row.addWidget(self.test_msg, 1)
        lay.addSpacing(4)
        lay.addLayout(row)
        self.add(box)

        self.section(tr("sys.tools"), tr("sys.tools.desc"))
        self.tool_rows = {k: ToolRow(k, self._install) for k in SHOWN_TOOLS}
        for r in self.tool_rows.values():
            self.body.addWidget(r)
            self.body.addSpacing(8)
        self.body.addSpacing(20)

        self.body.addWidget(label(tr("sys.folder"), "Hint"))
        self.body.addSpacing(6)
        frow = QHBoxLayout()
        path = label(str(paths.data_root()), "Body")
        path.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        frow.addWidget(path, 1)
        open_btn = button(tr("sys.open"), "folder")
        open_btn.clicked.connect(lambda: os.startfile(paths.data_root()) if hasattr(os, "startfile") else None)
        frow.addWidget(open_btn)
        self.body.addLayout(frow)
        self.finish()

        bridge.job_updated.connect(self._on_job)
        self.refresh()

    def refresh(self) -> None:
        run_in_background(gpu.detect_gpus, self._show_gpus)
        run_in_background(tools.all_status, self._show_tools)

    def _show_gpus(self, gpus) -> None:
        gpus = gpus if isinstance(gpus, list) else []
        if not gpus:
            self.gpu_line.setText(tr("sys.gpu.none"))
            self.fast_line.setText(tr("sys.fast.no"))
            return
        lines = []
        for g in gpus:
            vram = tr("u.gb", n=round(g.vram_mb / 1024)) if g.vram_mb else "?"
            lines.append(tr("sys.gpu.line", name=g.name, vram=vram))
        self.gpu_line.setText("\n".join(lines))
        ok = any(g.vsr_capable for g in gpus)
        self.fast_line.setText(tr("sys.fast.ready") if ok else tr("sys.fast.no"))
        self.fast_line.setObjectName("Ok" if ok else "Body")
        theme.repolish(self.fast_line)
        self.test_btn.setVisible(ok)

    def _show_tools(self, statuses) -> None:
        if isinstance(statuses, Exception):
            return
        for st in statuses:
            if st.spec.key in self.tool_rows:
                self.tool_rows[st.spec.key].show_status(st)

    def _install(self, key: str) -> None:
        def work(ctx: JobContext):
            ctx.report(None, "resolving release", phase="preparing")

            def prog(done: int, total: int | None) -> None:
                ctx.report(done / total if total else None, f"downloading {done / 1e6:.0f} MB", phase="working")

            return tools.fetch(key, progress=prog, cancel=ctx.cancel)

        self.bridge.queue.submit(tr("job.install", name=tools.TOOLS[key].display), work)

    def _on_job(self, job: Job) -> None:
        if job.state == JobState.DONE:
            self.refresh()

    def _test(self) -> None:
        self.test_btn.setEnabled(False)
        self.test_msg.setObjectName("Body")
        self.test_msg.setText(tr("sys.testing"))

        def done(result) -> None:
            self.test_btn.setEnabled(True)
            if isinstance(result, Exception):
                ok, why = False, str(result)
            else:
                ok, why = result
            if ok is None:
                self.test_msg.setText(tr("sys.test.skip"))
            elif ok:
                self.test_msg.setText(tr("sys.test.ok"))
            else:
                self.test_msg.setText(tr("sys.test.fail", why=why))
            self.test_msg.setObjectName("Ok" if ok else "Body")
            theme.repolish(self.test_msg)

        run_in_background(gpu.probe_vsr, done)
