"""System page: detected GPUs, external tools (with install buttons) and the RTX VSR test."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...core import gpu, paths, tools
from ...core.jobs import Job, JobContext, JobState
from ..bridge import JobBridge
from ..tasks import run_in_background


class SystemPage(QWidget):
    def __init__(self, bridge: JobBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<h2>System</h2>Data folder: <code>{paths.data_root()}</code>"))

        gbox = QGroupBox("GPUs")
        gl = QVBoxLayout(gbox)
        self.gpu_table = self._table(["#", "Name", "Vendor", "Driver", "VRAM", "RTX VSR"])
        self.gpu_table.setMaximumHeight(110)
        gl.addWidget(self.gpu_table)
        row = QHBoxLayout()
        self.vsr_btn = QPushButton("Test RTX VSR")
        self.vsr_btn.clicked.connect(self._test_vsr)
        self.vsr_label = QLabel("")
        row.addWidget(self.vsr_btn)
        row.addWidget(self.vsr_label, 1)
        gl.addLayout(row)
        layout.addWidget(gbox)

        tbox = QGroupBox("External tools")
        tl = QVBoxLayout(tbox)
        self.tool_table = self._table(["Tool", "Needed from", "Version", "Location", ""])
        tl.addWidget(self.tool_table)
        layout.addWidget(tbox, 1)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh, 0, Qt.AlignmentFlag.AlignRight)
        bridge.job_updated.connect(self._on_job)
        self.refresh()

    @staticmethod
    def _table(cols: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    def refresh(self) -> None:
        run_in_background(gpu.detect_gpus, self._show_gpus)
        run_in_background(tools.all_status, self._show_tools)

    def _show_gpus(self, gpus) -> None:
        if isinstance(gpus, Exception):
            gpus = []
        self.gpu_table.setRowCount(len(gpus))
        for r, g in enumerate(gpus):
            vals = [str(g.index), g.name, g.vendor, g.driver or "?",
                    f"{g.vram_mb} MiB" if g.vram_mb else "?", "likely" if g.vsr_capable else "no"]
            for c, v in enumerate(vals):
                self.gpu_table.setItem(r, c, QTableWidgetItem(v))

    def _show_tools(self, statuses) -> None:
        if isinstance(statuses, Exception):
            return
        self.tool_table.setRowCount(len(statuses))
        for r, st in enumerate(statuses):
            vals = [st.spec.display, st.spec.milestone, st.version or "",
                    f"{st.path} [{st.origin}]" if st.found else "not installed"]
            for c, v in enumerate(vals):
                self.tool_table.setItem(r, c, QTableWidgetItem(v))
            btn = QPushButton("Reinstall" if st.origin == "managed" else "Install")
            btn.clicked.connect(lambda _=False, k=st.spec.key: self._install(k))
            self.tool_table.setCellWidget(r, 4, btn)

    def _install(self, key: str) -> None:
        def work(ctx: JobContext):
            ctx.report(None, "resolving release")

            def prog(done: int, total: int | None) -> None:
                ctx.report(done / total if total else None, f"downloading {done / 1e6:.0f} MB")

            exe = tools.fetch(key, progress=prog, cancel=ctx.cancel)
            ctx.report(1.0, f"installed {exe.name}")
            return exe

        self.bridge.queue.submit(f"Install {tools.TOOLS[key].display}", work)

    def _on_job(self, job: Job) -> None:
        if job.title.startswith("Install ") and job.state == JobState.DONE:
            self.refresh()

    def _test_vsr(self) -> None:
        self.vsr_btn.setEnabled(False)
        self.vsr_label.setText("Running a 1-second test encode...")

        def done(result) -> None:
            self.vsr_btn.setEnabled(True)
            if isinstance(result, Exception):
                self.vsr_label.setText(f"Error: {result}")
                return
            ok, detail = result
            prefix = {True: "OK", False: "Failed", None: "Skipped"}[ok]
            self.vsr_label.setText(f"{prefix}: {detail}")

        run_in_background(gpu.probe_vsr, done)
