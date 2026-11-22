"""Write the words: speech -> subtitles. Pick videos, the spoken language, same language or
English, how carefully to listen, and whether to also add them inside a copy of the video."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from ...core import gpu as gpu_mod
from ...core.settings import Settings
from ...features.subtitles import estimate as sub_est
from ...features.subtitles.service import SubtitleRequest, run_subtitles
from ...features.upscale import estimate as up_est
from ..bridge import JobBridge
from ..i18n import tr
from ..tasks import run_in_background
from ..widgets.common import ChipGroup, Page, Tile, label
from ..widgets.options import MoreOptions, OutputRows, combo
from ..widgets.video_list import ActionBar, VideoList

LEVELS = ("quick", "balanced", "best")
# Spoken-language choices: ISO 639-1 code and the language's own name.
LANGUAGES = [
    ("en", "English"), ("ar", "العربية"), ("fr", "Français"), ("es", "Español"), ("de", "Deutsch"),
    ("it", "Italiano"), ("pt", "Português"), ("tr", "Türkçe"), ("ru", "Русский"), ("uk", "Українська"),
    ("fa", "فارسی"), ("ur", "اردو"), ("hi", "हिन्दी"), ("bn", "বাংলা"), ("id", "Bahasa Indonesia"),
    ("ms", "Bahasa Melayu"), ("ja", "日本語"), ("ko", "한국어"), ("zh", "中文"), ("nl", "Nederlands"),
    ("pl", "Polski"), ("sv", "Svenska"), ("el", "Ελληνικά"), ("he", "עברית"),
    ("cs", "Čeština"), ("hu", "Magyar"), ("da", "Dansk"), ("fi", "Suomi"), ("no", "Norsk"), ("sw", "Kiswahili"),
]


class SubtitlesPage(QWidget):
    def __init__(self, bridge: JobBridge, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.settings = settings
        self.gpu: gpu_mod.GpuInfo | None = None
        self.stats = up_est.load_stats()
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page = Page(tr("subs.title"), tr("subs.lead"))
        outer.addWidget(self.page, 1)

        self.page.section(tr("up.videos"))
        self.videos = VideoList()
        self.videos.changed.connect(self._recalc)
        self.page.add(self.videos)

        self.page.section(tr("sb.lang"), tr("sb.lang.help"))
        self.language = combo([("auto", tr("sb.lang.auto"))] + LANGUAGES, settings.subs_language, self._changed)
        self.language.setMaximumWidth(420)
        self.page.add(self.language)

        self.page.section(tr("sb.which"))
        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.same = Tile("subtitles", tr("sb.same"), tr("sb.same.desc"))
        self.english = Tile("translate", tr("sb.english"), tr("sb.english.desc"))
        self.same.clicked.connect(lambda: self._set_translate(False))
        self.english.clicked.connect(lambda: self._set_translate(True))
        for t in (self.same, self.english):
            t.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            t.estimate.setVisible(False)
            tiles.addWidget(t, 1)
        holder = QWidget()
        holder.setLayout(tiles)
        self.page.add(holder)

        self.page.section(tr("sb.level"))
        self.level = ChipGroup([(k, tr(f"sb.level.{k}")) for k in LEVELS],
                               settings.subs_level if settings.subs_level in LEVELS else "balanced")
        self.level.changed.connect(self._level_changed)
        self.page.body.addWidget(self.level)
        self.page.body.addSpacing(8)
        self.level_desc = label("", "Hint", wrap=True)
        self.page.add(self.level_desc)

        self.inside = QCheckBox(tr("sb.inside"))
        self.inside.setChecked(settings.subs_add_to_video)
        self.inside.toggled.connect(self._changed)
        self.page.body.addWidget(self.inside)
        self.page.body.addSpacing(4)
        self.page.add(label(tr("sb.inside.help"), "Hint", wrap=True), spacing_after=12)
        self.burn = QCheckBox(tr("burn.check"))
        self.burn.setChecked(settings.subs_burn_in)
        self.burn.toggled.connect(self._changed)
        self.page.body.addWidget(self.burn)
        self.page.body.addSpacing(4)
        self.page.add(label(tr("burn.help"), "Hint", wrap=True))

        self.more = MoreOptions()
        self.track = combo([(i, tr("sb.track.n", n=i + 1)) for i in range(4)], settings.subs_audio_track, self._changed)
        self.more.row(tr("sb.track"), self.track, tr("sb.track.help"))
        self.output = OutputRows(self.more, settings, self._recalc, video=False)
        self.page.body.addWidget(self.more.button, 0, Qt.AlignmentFlag.AlignLeading)
        self.page.body.addSpacing(10)
        self.page.body.addWidget(self.more.panel)
        self.page.finish()

        self.bar = ActionBar(self._start)
        outer.addWidget(self.bar)
        bridge.job_updated.connect(self._on_job)
        self._sync_burn()
        self._set_translate(settings.subs_translate, save=False)
        self._level_changed(self.level.value(), save=False)
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
        self._recalc()

    def _sync_burn(self) -> None:
        """Writing onto the picture always makes a copy of the video, so that box is ticked then."""
        burning = self.burn.isChecked()
        self.inside.blockSignals(True)
        self.inside.setChecked(burning or self.settings.subs_add_to_video)
        self.inside.setEnabled(not burning)
        self.inside.blockSignals(False)

    def _set_translate(self, on: bool, save: bool = True) -> None:
        self.same.set_selected(not on)
        self.english.set_selected(on)
        self.settings.subs_translate = on
        if save:
            self.settings.save()
        self._recalc()

    def _level_changed(self, value: str, save: bool = True) -> None:
        self.level_desc.setText(tr(f"sb.level.{value}.desc"))
        self.settings.subs_level = value
        if save:
            self.settings.save()
        self._recalc()

    def _changed(self, *_args) -> None:
        s = self.settings
        s.subs_language = self.language.currentData()
        if self.inside.isEnabled():
            s.subs_add_to_video = self.inside.isChecked()
        s.subs_burn_in = self.burn.isChecked()
        self._sync_burn()
        s.subs_audio_track = int(self.track.currentData() or 0)
        s.save()
        self._recalc()

    # --- estimates ---
    def _recalc(self) -> None:
        if not hasattr(self, "bar"):
            return
        s = self.settings
        secs = size = download = 0.0
        ready = 0
        for v in self.videos.videos:
            v.estimates = {}
            v.size_error = False
            if v.info is None or not v.info.audio:
                if v.info is not None and not v.error:
                    v.error = tr("err.no_audio")
                continue
            e = sub_est.estimate(v.info, level=s.subs_level, gpu=self.gpu,
                                 add_to_video=s.subs_add_to_video, stats=self.stats, burn_in=s.subs_burn_in)
            v.estimates["subs"] = e
            secs += e.seconds
            size += e.bytes
            download = max(download, e.download_mb)
            ready += 1
        self.videos.show_estimates("subs")
        self.bar.show_totals(len(self.videos.videos), ready, secs, size, download)

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
            if v.info is None or not v.estimates:
                continue
            req = SubtitleRequest(
                input=v.path, output_dir=out_dir, audio_track=s.subs_audio_track, language=s.subs_language,
                translate=s.subs_translate, level=s.subs_level, add_to_video=s.subs_add_to_video, burn_in=s.subs_burn_in,
                gpu_index=s.gpu_index,
            )
            self.bridge.queue.submit(v.path.name, lambda ctx, r=req: run_subtitles(r, ctx))
            self.videos.remove_row(row)
            started += 1
        if started:
            self.bar.show_added()

