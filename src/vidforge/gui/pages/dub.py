"""Translate & dub: pick videos and a language; get translated subtitles, and if wanted a new
voice that sounds like each person speaking. The voice needs a one-time add-on, offered here."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from ...core import gpu as gpu_mod
from ...core.jobs import Job, JobContext
from ...core.settings import Settings
from ...features.dub import addon
from ...features.dub import estimate as dub_est
from ...features.dub.models import ENGINES, VOICE_LANGUAGES
from ...features.dub.service import LEVELS, DubRequest, run_dub
from ...features.subtitles.service import load_transcript
from ...features.upscale import estimate as up_est
from .. import theme
from ..bridge import JobBridge
from ..i18n import tr
from ..tasks import run_in_background
from ..widgets.common import ChipGroup, Page, Tile, button, icon_label, label, panel
from ..widgets.options import MoreOptions, OutputRows, combo
from ..widgets.video_list import ActionBar, VideoList
from .subtitles import LANGUAGES

ADDON_JOB = "voice-addon"


class DubPage(QWidget):
    def __init__(self, bridge: JobBridge, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.settings = settings
        self.gpu: gpu_mod.GpuInfo | None = None
        self.stats = up_est.load_stats()
        self.transcripts: dict[Path, object] = {}
        self.engine_name = settings.dub_engine if settings.dub_engine in ENGINES else "chatterbox"
        self.addon_job: Job | None = None
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.page = Page(tr("dub.title"), tr("dub.lead"))
        outer.addWidget(self.page, 1)

        self.page.section(tr("up.videos"))
        self.videos = VideoList()
        self.videos.changed.connect(self._recalc)
        self.page.add(self.videos)

        self.page.section(tr("db.to"), tr("db.to.help"))
        target = settings.dub_target if any(c == settings.dub_target for c, _ in LANGUAGES) else "ar"
        self.target = combo(LANGUAGES, target, self._target_changed)
        self.target.setMaximumWidth(420)
        self.page.add(self.target)

        self.page.section(tr("db.what"))
        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.subs_tile = Tile("subtitles", tr("db.subs"), tr("db.subs.desc"))
        self.voice_tile = Tile("voice", tr("db.voice"), tr("db.voice.desc"))
        self.subs_tile.clicked.connect(lambda: self._set_voice(False))
        self.voice_tile.clicked.connect(lambda: self._set_voice(True))
        for t in (self.subs_tile, self.voice_tile):
            t.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            t.estimate.setVisible(False)
            tiles.addWidget(t, 1)
        holder = QWidget()
        holder.setLayout(tiles)
        self.page.add(holder, spacing_after=14)

        # Which voice: only shown when a new voice is chosen.
        self.engine_box = QWidget()
        eb = QVBoxLayout(self.engine_box)
        eb.setContentsMargins(0, 0, 0, 0)
        eb.setSpacing(0)
        eb.addSpacing(8)
        eb.addWidget(label(tr("db.engine"), "SectionTitle"))
        eb.addSpacing(12)
        self.engine = ChipGroup([(k, tr(f"db.engine.{k}")) for k in ENGINES], self.engine_name)
        self.engine.changed.connect(self._engine_changed)
        eb.addWidget(self.engine)
        eb.addSpacing(8)
        self.engine_desc = label("", "Hint", wrap=True)
        eb.addWidget(self.engine_desc)
        self.page.add(self.engine_box, spacing_after=18)

        # The voice add-on, offered when a new voice is chosen and it isn't installed yet.
        self.addon_box, lay = panel(QHBoxLayout, margins=(18, 16, 18, 16), spacing=14)
        lay.addWidget(icon_label("voice", 24, theme.ACCENT), 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(4)
        texts.addWidget(label(tr("db.addon.title"), "Strong"))
        self.addon_text = label("", "Body", wrap=True)
        texts.addWidget(self.addon_text)
        lay.addLayout(texts, 1)
        self.addon_btn = button(tr("db.addon.install"))
        self.addon_btn.clicked.connect(self._install_addon)
        lay.addWidget(self.addon_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self.page.add(self.addon_box, spacing_after=14)

        self.keep = QCheckBox(tr("db.keep"))
        self.keep.setChecked(settings.dub_keep_original)
        self.keep.toggled.connect(self._changed)
        self.page.body.addWidget(self.keep)
        self.inside = QCheckBox(tr("sb.inside"))
        self.inside.setChecked(settings.dub_add_to_video)
        self.inside.toggled.connect(self._changed)
        self.page.body.addWidget(self.inside)
        self.page.body.addSpacing(4)
        self.check_help = label("", "Hint", wrap=True)
        self.page.add(self.check_help)

        self.page.section(tr("db.level"))
        self.level = ChipGroup([(k, tr(f"db.level.{k}")) for k in LEVELS],
                               settings.dub_level if settings.dub_level in LEVELS else "best")
        self.level.changed.connect(self._level_changed)
        self.page.body.addWidget(self.level)
        self.page.body.addSpacing(8)
        self.level_desc = label("", "Hint", wrap=True)
        self.page.add(self.level_desc)

        self.more = MoreOptions()
        self.spoken = combo([("auto", tr("sb.lang.auto"))] + LANGUAGES, settings.dub_language, self._changed)
        self.more.row(tr("sb.lang"), self.spoken, tr("db.spoken.help"))
        self.track = combo([(i, tr("sb.track.n", n=i + 1)) for i in range(4)], settings.dub_audio_track, self._changed)
        self.more.row(tr("sb.track"), self.track, tr("sb.track.help"))
        self.output = OutputRows(self.more, settings, self._recalc, video=False)
        self.page.body.addWidget(self.more.button, 0, Qt.AlignmentFlag.AlignLeading)
        self.page.body.addSpacing(10)
        self.page.body.addWidget(self.more.panel)
        self.page.finish()

        self.bar = ActionBar(self._start)
        outer.addWidget(self.bar)
        bridge.job_updated.connect(self._on_job)
        self._target_changed(save=False)
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
        self._show_addon()
        self._recalc()

    @property
    def addon_ready(self) -> bool:
        return addon.is_installed(self.engine_name)

    def voice_possible(self) -> bool:
        return self.target.currentData() in VOICE_LANGUAGES[self.engine_name]

    def voice_chosen(self) -> bool:
        """The new-voice tile is picked (and some voice maker speaks the language)."""
        return self.settings.dub_voice and any(self.target.currentData() in VOICE_LANGUAGES[e] for e in ENGINES)

    def wants_voice(self) -> bool:
        """A new voice will really be made: chosen, and the chosen voice maker speaks the language."""
        return self.voice_chosen() and self.voice_possible()

    def _target_changed(self, *_args, save: bool = True) -> None:
        self.settings.dub_target = self.target.currentData()
        ok = self.voice_possible()
        # The tile stays usable when the other voice can speak this language: pick that voice.
        any_ok = any(self.target.currentData() in VOICE_LANGUAGES[e] for e in ENGINES)
        self.voice_tile.set_available(any_ok, tr("db.voice.no_lang"))
        self.voice_tile.estimate.setVisible(not any_ok)
        if not ok and any_ok:
            self.engine.set_value(next(e for e in ENGINES if self.target.currentData() in VOICE_LANGUAGES[e]))
            self._engine_changed(self.engine.value(), save=save, refresh=False)
        else:
            self._engine_changed(self.engine_name, save=False, refresh=False)
        if save:
            self.settings.save()
        self._set_voice(self.settings.dub_voice, save=False)

    def _set_voice(self, on: bool, save: bool = True) -> None:
        if save:
            self.settings.dub_voice = on
            self.settings.save()
        voice = self.voice_chosen()
        self.subs_tile.set_selected(not voice)
        self.voice_tile.set_selected(voice)
        self.keep.setVisible(voice)
        self.engine_box.setVisible(voice)
        self.inside.setVisible(not voice)
        self.check_help.setText(tr("db.keep.help") if voice else tr("db.inside.help"))
        self._show_addon()
        self._recalc()

    def _show_addon(self) -> None:
        installing = self.addon_job is not None and not self.addon_job.state.finished
        self.addon_box.setVisible(self.wants_voice() and not self.addon_ready)
        gb = dub_est.addon_mb(self.gpu) / 1000
        self.addon_text.setText(tr("db.addon.installing") if installing else tr("db.addon.desc", gb=f"{gb:.0f}"))
        self.addon_btn.setEnabled(not installing)

    def _engine_changed(self, value: str, save: bool = True, refresh: bool = True) -> None:
        self.engine_name = value
        self.settings.dub_engine = value
        desc = tr(f"db.engine.{value}.desc")
        if self.target.currentData() not in VOICE_LANGUAGES[value]:
            desc += " " + tr("db.engine.no_lang")
        self.engine_desc.setText(desc)
        if save:
            self.settings.save()
        if refresh:
            self._set_voice(self.settings.dub_voice, save=False)

    def _level_changed(self, value: str, save: bool = True) -> None:
        self.level_desc.setText(tr(f"db.level.{value}.desc"))
        self.settings.dub_level = value
        if save:
            self.settings.save()
        self._recalc()

    def _changed(self, *_args) -> None:
        s = self.settings
        s.dub_keep_original = self.keep.isChecked()
        s.dub_add_to_video = self.inside.isChecked()
        s.dub_language = self.spoken.currentData()
        s.dub_audio_track = int(self.track.currentData() or 0)
        s.save()
        self._recalc()

    # --- the add-on ---
    def _install_addon(self) -> None:
        gpu = self.gpu

        def work(ctx: JobContext):
            def step(i: int, n: int, name: str) -> None:
                ctx.report(i / n, tr(f"db.addon.step.{name}"),
                           phase="downloading" if name in ("torch", "packages", "voice") else "working")
            return addon.install(gpu, cancel=ctx.cancel, on_log=ctx.log, on_step=step)

        self.addon_job = self.bridge.queue.submit(tr("db.addon.job"), work)
        self._show_addon()

    # --- estimates ---
    def _transcript(self, path: Path):
        if path not in self.transcripts:
            self.transcripts[path] = load_transcript(path)
        return self.transcripts[path]

    def _recalc(self) -> None:
        if not hasattr(self, "bar"):
            return
        s = self.settings
        voice = self.wants_voice()
        secs = size = download = 0.0
        ready = 0
        for v in self.videos.videos:
            v.estimates = {}
            v.size_error = False
            if v.info is None or not v.info.audio:
                if v.info is not None and not v.error:
                    v.error = tr("err.no_audio")
                continue
            e = dub_est.estimate(v.info, level=s.dub_level, dub=voice, add_to_video=s.dub_add_to_video,
                                 target=s.dub_target, gpu=self.gpu, transcript=self._transcript(v.path),
                                 stats=self.stats, voice=self.engine_name)
            v.estimates["dub"] = e
            secs += e.seconds
            size += e.bytes
            download = max(download, e.download_mb)
            ready += 1
        if self.voice_chosen() and not (voice and self.addon_ready):
            ready = 0  # nothing can start until the voice add-on is in and the voice speaks the language
        self.videos.show_estimates("dub")
        self.bar.show_totals(len(self.videos.videos), ready, secs, size, download)

    def _on_job(self, job) -> None:
        if not job.state.finished:
            return
        if self.addon_job is not None and job.id == self.addon_job.id:
            self._show_addon()
        self.stats = up_est.load_stats()
        self.transcripts.clear()  # a subtitles job may have just made one
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
            req = DubRequest(
                input=v.path, target=s.dub_target, dub=self.wants_voice(), add_to_video=s.dub_add_to_video,
                level=s.dub_level, keep_original=s.dub_keep_original, language=s.dub_language,
                audio_track=s.dub_audio_track, output_dir=out_dir, gpu_index=s.gpu_index, voice=self.engine_name,
            )
            self.bridge.queue.submit(v.path.name, lambda ctx, r=req: run_dub(r, ctx))
            self.videos.remove_row(row)
            started += 1
        if started:
            self.bar.show_added()
