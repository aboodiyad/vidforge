"""English and Arabic text for the interface, plus number, size and time formatting.

Every visible sentence lives here. Wording rule: say what happens for the person, in words a
child understands; keep technical names (H.265, MKV) only where the person has to pick one.
"""

from __future__ import annotations

LANGUAGES = ("system", "en", "ar")
_lang = "en"

S: dict[str, tuple[str, str]] = {
    # --- app / navigation ---
    "app.tagline": ("Better videos, made on your PC", "فيديوهات أوضح، تُصنع على جهازك"),
    "nav.upscale": ("Make bigger", "تكبير الفيديو"),
    "nav.faces": ("Fix faces", "تحسين الوجوه"),
    "nav.subtitles": ("Write the words", "كتابة الكلام"),
    "nav.dub": ("Translate & dub", "ترجمة ودبلجة"),
    "nav.settings": ("Settings", "الإعدادات"),
    "nav.system": ("My PC", "جهازي"),
    "nav.soon": ("Soon", "قريبًا"),

    # --- upscale page ---
    "up.title": ("Make videos bigger and sharper", "اجعل الفيديو أكبر وأوضح"),
    "up.subtitle": (
        "Pick a video, choose a size, and press Start. Your original video stays exactly as it is.",
        "اختر فيديو، ثم اختر الحجم، واضغط «ابدأ». الفيديو الأصلي يبقى كما هو تمامًا.",
    ),
    "up.videos": ("Your videos", "الفيديوهات"),
    "drop.title": ("Drop videos here", "اسحب الفيديوهات وأفلتها هنا"),
    "drop.or": ("or", "أو"),
    "drop.choose": ("Choose videos…", "اختر فيديوهات…"),
    "drop.dialog": ("Choose videos", "اختر فيديوهات"),
    "up.add_more": ("Add more", "أضف المزيد"),
    "up.clear": ("Remove all", "إزالة الكل"),
    "up.remove": ("Remove this video", "أزل هذا الفيديو"),
    "row.reading": ("Reading video…", "جارٍ قراءة الفيديو…"),
    "row.cant_read": ("Can't open this file", "تعذّر فتح هذا الملف"),
    "row.arrow": ("→", "←"),
    "up.size": ("How big should it be?", "ما الحجم الذي تريده؟"),
    "size.1080p": ("HD 1080p", "HD 1080p"),
    "size.1440p": ("2K 1440p", "2K 1440p"),
    "size.2160p": ("4K", "4K"),
    "size.2x": ("2× bigger", "أكبر مرتين"),
    "size.3x": ("3× bigger", "أكبر 3 مرات"),
    "size.4x": ("4× bigger", "أكبر 4 مرات"),
    "size.1080p.desc": (
        "Full HD. Looks sharp on phones, laptops and most TVs.",
        "دقة عالية كاملة. يبدو واضحًا على الهاتف واللابتوب ومعظم أجهزة التلفاز.",
    ),
    "size.1440p.desc": (
        "Even sharper. Great for big computer screens.",
        "أوضح من ذلك. ممتاز لشاشات الكمبيوتر الكبيرة.",
    ),
    "size.2160p.desc": (
        "The sharpest. Best for big TVs, but the file gets big too.",
        "الأوضح. الأفضل للتلفاز الكبير، لكن حجم الملف يكبر أيضًا.",
    ),
    "size.2x.desc": ("Twice as wide and twice as tall.", "ضعف العرض وضعف الطول."),
    "size.3x.desc": ("Three times as wide and three times as tall.", "ثلاثة أضعاف العرض وثلاثة أضعاف الطول."),
    "size.4x.desc": ("Four times as wide and four times as tall.", "أربعة أضعاف العرض وأربعة أضعاف الطول."),
    "up.mode": ("Fast or best quality?", "سريع أم أفضل جودة؟"),
    "mode.fast": ("Fast", "سريع"),
    "mode.fast.desc": (
        "Your NVIDIA graphics card makes the video sharper by itself. Usually done in minutes.",
        "بطاقة الرسومات NVIDIA في جهازك تجعل الفيديو أوضح بنفسها. عادةً ينتهي خلال دقائق.",
    ),
    "mode.fast.no_rtx": (
        "Needs an NVIDIA RTX graphics card. This PC has: {gpu}",
        "يحتاج إلى بطاقة رسومات NVIDIA RTX. جهازك فيه: {gpu}",
    ),
    "mode.fast.no_tool": (
        "Needs a free helper program first. Get it on the “My PC” page.",
        "يحتاج أولًا إلى برنامج مساعد مجاني. حمّله من صفحة «جهازي».",
    ),
    "mode.quality": ("Best quality", "أفضل جودة"),
    "mode.quality.desc": (
        "An AI redraws every picture in the video with more detail. Sharper, but much slower.",
        "ذكاء اصطناعي يعيد رسم كل صورة في الفيديو بتفاصيل أكثر. أوضح، لكنه أبطأ بكثير.",
    ),
    "mode.checking": ("Checking your graphics card…", "جارٍ فحص بطاقة الرسومات…"),
    "more.show": ("More options", "خيارات إضافية"),
    "more.hide": ("Fewer options", "خيارات أقل"),
    "more.model": ("AI style", "نوع الذكاء الاصطناعي"),
    "model.general": ("Everyday video (best choice)", "فيديو عادي (الخيار الأفضل)"),
    "model.max": ("Extra detail (very slow)", "تفاصيل إضافية (بطيء جدًا)"),
    "model.anime": ("Cartoons and anime", "رسوم متحركة وأنمي"),
    "more.vsr": ("Sharpening strength", "قوة التوضيح"),
    "more.vsr.help": ("4 is the strongest. Lower numbers are a little faster.", "‏4 هي الأقوى. الأرقام الأقل أسرع قليلًا."),
    "more.quality": ("File size", "حجم الملف"),
    "q.high": ("Bigger file, best picture", "ملف أكبر، صورة أفضل"),
    "q.balanced": ("Balanced (recommended)", "متوازن (مُستحسَن)"),
    "q.small": ("Smaller file", "ملف أصغر"),
    "more.codec": ("Video format", "صيغة الفيديو"),
    "codec.hevc": ("H.265: small files, plays almost everywhere", "H.265: ملفات صغيرة، يعمل في كل مكان تقريبًا"),
    "codec.h264": ("H.264: plays on every device, bigger files", "H.264: يعمل على كل الأجهزة، ملفات أكبر"),
    "codec.av1": ("AV1: smallest files, newer devices only", "AV1: أصغر الملفات، للأجهزة الحديثة فقط"),
    "more.container": ("File type", "نوع الملف"),
    "container.mp4": ("MP4 (works everywhere)", "MP4 (يعمل في كل مكان)"),
    "container.mkv": ("MKV (keeps every subtitle)", "MKV (يحتفظ بكل الترجمات)"),
    "more.folder": ("Save to", "احفظ في"),
    "folder.same": ("The same folder as each video", "نفس مجلد كل فيديو"),
    "folder.choose": ("Choose…", "اختر…"),
    "folder.reset": ("Use the video's folder", "استخدم مجلد الفيديو"),
    "folder.dialog": ("Where should new videos go?", "أين تُحفظ الفيديوهات الجديدة؟"),
    "sum.empty": ("Add a video to begin.", "أضف فيديو لتبدأ."),
    "sum.line": ("{count} · {size} · {time}", "{count} · {size} · {time}"),
    "sum.download": ("The first time, the AI is downloaded ({mb} MB).", "في المرة الأولى يُنزَّل الذكاء الاصطناعي ({mb} ميغابايت)."),
    "sum.guess": (
        "These are guesses. They get better each time you finish a video.",
        "هذه تقديرات، وتتحسّن كلما أنهيت فيديو.",
    ),
    "sum.added": ("Added to the list below. You can keep adding videos.", "أُضيفت إلى القائمة بالأسفل. يمكنك إضافة المزيد."),
    "up.start": ("Start", "ابدأ"),

    # --- jobs ---
    "jobs.title": ("Your list", "قائمتك"),
    "jobs.empty": ("Nothing here yet. Videos you start show up here.", "لا شيء هنا بعد. ستظهر هنا الفيديوهات التي تبدأها."),
    "jobs.clear": ("Clear finished", "مسح المنتهية"),
    "jobs.running": ("{n} working", "{n} قيد العمل"),
    "st.queued": ("Waiting for its turn", "ينتظر دوره"),
    "st.preparing": ("Getting ready…", "جارٍ التجهيز…"),
    "st.downloading": ("Downloading the AI (only the first time)…", "جارٍ تنزيل الذكاء الاصطناعي (مرة واحدة فقط)…"),
    "st.working": ("Working… {pct}", "جارٍ العمل… {pct}"),
    "st.left": ("{t} left", "متبقٍّ {t}"),
    "st.measuring": ("working out the time left", "جارٍ حساب الوقت المتبقي"),
    "st.done": ("Done in {t}", "اكتمل خلال {t}"),
    "st.cancelled": ("Stopped", "أُوقف"),
    "st.failed": ("Didn't work: {why}", "لم ينجح: {why}"),
    "btn.stop": ("Stop", "إيقاف"),
    "btn.show": ("Show file", "عرض الملف"),
    "btn.play": ("Play", "تشغيل"),
    "btn.details": ("Details", "التفاصيل"),
    "details.title": ("What happened", "ماذا حدث"),
    "details.copy": ("Copy", "نسخ"),
    "details.close": ("Close", "إغلاق"),
    "job.install": ("Installing {name}", "تثبيت {name}"),

    # --- friendly errors ---
    "err.already_big": ("This video is already that big. Pick a bigger size.", "هذا الفيديو بهذا الحجم أصلًا. اختر حجمًا أكبر."),
    "err.no_video": ("This file has no video in it.", "هذا الملف لا يحتوي على فيديو."),
    "err.file_missing": ("The video file is gone. Was it moved or deleted?", "ملف الفيديو غير موجود. هل نُقل أو حُذف؟"),
    "err.fast_needs_rtx": ("Fast needs an NVIDIA RTX card. Use Best quality.", "«سريع» يحتاج إلى بطاقة NVIDIA RTX. استخدم «أفضل جودة»."),
    "err.fast_needs_nvencc": ("Fast needs a helper program. Install it on “My PC”.", "«سريع» يحتاج إلى برنامج مساعد. ثبّته من «جهازي»."),
    "err.hdr_quality": ("This is an HDR video. Use Fast instead.", "هذا فيديو HDR. استخدم «سريع» بدلًا منه."),
    "err.hdr_faces": ("This is an HDR video. Fixing faces doesn't work on HDR yet.", "هذا فيديو HDR. تحسين الوجوه لا يعمل مع HDR بعد."),
    "err.no_audio": ("This video has no sound, so there is nothing to write down.", "هذا الفيديو بلا صوت، فلا يوجد شيء لكتابته."),
    "err.no_speech": ("No talking was found in this video.", "لم يُعثر على كلام في هذا الفيديو."),
    "err.same_file": ("The new video can't replace the original.", "لا يمكن أن يحل الفيديو الجديد محل الأصلي."),
    "err.bad_size": ("That size doesn't look right. Use numbers like 1920x1080.", "هذا الحجم غير صحيح. استخدم أرقامًا مثل 1920x1080."),
    "err.generic": ("something went wrong. Open Details to see why.", "حدث خطأ ما. افتح «التفاصيل» لمعرفة السبب."),

    # --- settings ---
    "set.title": ("Settings", "الإعدادات"),
    "set.subtitle": ("Change how VidForge looks and works.", "غيّر شكل VidForge وطريقة عمله."),
    "set.language": ("Language", "اللغة"),
    "lang.system": ("Same as Windows", "مثل لغة ويندوز"),
    "lang.en": ("English", "English"),
    "lang.ar": ("العربية", "العربية"),
    "set.restart_note": ("VidForge needs to restart to change the language.", "يحتاج VidForge إلى إعادة التشغيل لتغيير اللغة."),
    "set.restart": ("Restart now", "أعد التشغيل الآن"),
    "set.restart_busy": ("Wait until your videos finish, then restart.", "انتظر حتى تنتهي الفيديوهات، ثم أعد التشغيل."),
    "set.folder": ("Where new videos are saved", "مكان حفظ الفيديوهات الجديدة"),
    "set.gpu": ("Graphics card to use", "بطاقة الرسومات المستخدمة"),
    "set.gpu.help": ("Only matters if your PC has more than one.", "يهم فقط إذا كان في جهازك أكثر من بطاقة."),
    "set.estimates": ("Time and size guesses", "تقديرات الوقت والحجم"),
    "set.estimates.desc": (
        "VidForge learns from the videos you finish, so its guesses get better. You can make it start over.",
        "يتعلّم VidForge من الفيديوهات التي تنتهي، فتتحسّن تقديراته. يمكنك جعله يبدأ من جديد.",
    ),
    "set.reset": ("Start over", "ابدأ من جديد"),
    "set.reset_done": ("Done. Guesses start fresh.", "تم. التقديرات تبدأ من جديد."),
    "set.about": ("About", "حول البرنامج"),
    "set.about.text": (
        "VidForge {v}. Built from free, open-source parts. Your videos never leave your PC.",
        "VidForge {v}. مبني من أجزاء مجانية ومفتوحة المصدر. فيديوهاتك لا تغادر جهازك أبدًا.",
    ),
    "set.about.link": ("Project page on GitHub", "صفحة المشروع على GitHub"),

    # --- system ---
    "sys.title": ("My PC", "جهازي"),
    "sys.subtitle": ("What VidForge found on this computer.", "ما وجده VidForge في هذا الكمبيوتر."),
    "sys.gpu": ("Graphics card", "بطاقة الرسومات"),
    "sys.gpu.none": ("No graphics card found.", "لم يُعثر على بطاقة رسومات."),
    "sys.gpu.line": ("{name} · {vram} of video memory", "{name} · {vram} ذاكرة فيديو"),
    "sys.fast.ready": ("Fast mode can run on this PC.", "الوضع السريع يمكنه العمل على هذا الجهاز."),
    "sys.fast.no": ("Fast mode needs an NVIDIA RTX card. Best quality still works.", "الوضع السريع يحتاج إلى بطاقة NVIDIA RTX. «أفضل جودة» يعمل مع ذلك."),
    "sys.test": ("Check that Fast mode works", "تحقّق من أن الوضع السريع يعمل"),
    "sys.testing": ("Checking… this takes a few seconds.", "جارٍ التحقق… يستغرق بضع ثوانٍ."),
    "sys.test.ok": ("Yes, Fast mode works.", "نعم، الوضع السريع يعمل."),
    "sys.test.fail": ("It didn't work: {why}", "لم ينجح: {why}"),
    "sys.test.skip": ("Install the helper programs below first.", "ثبّت البرامج المساعدة في الأسفل أولًا."),
    "sys.tools": ("Helper programs", "البرامج المساعدة"),
    "sys.tools.desc": (
        "Free programs VidForge uses to open, change and save videos. They come from their official pages and are checked before use.",
        "برامج مجانية يستخدمها VidForge لفتح الفيديو وتعديله وحفظه. تأتي من صفحاتها الرسمية ويُتحقَّق منها قبل الاستخدام.",
    ),
    "tool.ffmpeg": ("Opens and saves videos. Needed for everything.", "يفتح الفيديوهات ويحفظها. ضروري لكل شيء."),
    "tool.nvencc": ("Runs Fast mode on NVIDIA cards.", "يشغّل الوضع السريع على بطاقات NVIDIA."),
    "tool.sevenzip": ("Unpacks downloaded programs.", "يفك ضغط البرامج المنزَّلة."),
    "tool.whisper": ("Turns speech into text. For a part coming soon.", "يحوّل الكلام إلى نص. لجزء قادم قريبًا."),
    "tool.llama": ("Translates text. For a part coming soon.", "يترجم النص. لجزء قادم قريبًا."),
    "tool.installed": ("Installed · {v}", "مثبّت · {v}"),
    "tool.found": ("Found on this PC · {v}", "موجود على الجهاز · {v}"),
    "tool.missing": ("Not installed", "غير مثبّت"),
    "tool.install": ("Install", "تثبيت"),
    "tool.reinstall": ("Reinstall", "إعادة التثبيت"),
    "sys.folder": ("VidForge keeps its programs and AI files here:", "يحفظ VidForge برامجه وملفات الذكاء الاصطناعي هنا:"),
    "sys.open": ("Open folder", "فتح المجلد"),

    # --- coming soon ---
    "faces.title": ("Fix blurry faces", "حسّن الوجوه غير الواضحة"),
    "faces.desc": (
        "Finds the faces in a video and makes them clear and sharp, as if someone carefully touched up every picture.",
        "يجد الوجوه في الفيديو ويجعلها واضحة وحادة، كأن أحدًا أصلح كل صورة بعناية.",
    ),
    "faces.lead": (
        "Pick a video and press Start. VidForge finds every face and makes it clear and sharp. Your original video stays exactly as it is.",
        "اختر فيديو واضغط «ابدأ». يجد VidForge كل وجه ويجعله واضحًا وحادًا. الفيديو الأصلي يبقى كما هو تمامًا.",
    ),
    "fx.how": ("How should faces be fixed?", "كيف نحسّن الوجوه؟"),
    "fx.gfpgan": ("Natural", "طبيعي"),
    "fx.gfpgan.desc": (
        "Makes faces clearer and sharper, and keeps them looking like the same person. Best for most videos.",
        "يجعل الوجوه أوضح وأحدّ، ويبقيها تشبه الشخص نفسه. الأفضل لمعظم الفيديوهات.",
    ),
    "fx.codeformer": ("Strong repair", "إصلاح قوي"),
    "fx.codeformer.desc": (
        "For very blurry or damaged faces. It fixes more, but may change small details of a face. Free for personal use only.",
        "للوجوه المشوشة جدًا أو التالفة. يصلح أكثر، لكنه قد يغيّر تفاصيل صغيرة في الوجه. مجاني للاستخدام الشخصي فقط.",
    ),
    "fx.amount": ("How much should faces change?", "إلى أي حد تتغير الوجوه؟"),
    "fx.amount.low": ("A little", "قليلًا"),
    "fx.amount.high": ("A lot", "كثيرًا"),
    "fx.amount.help": (
        "Move toward “A little” if faces look too smooth or too different.",
        "حرّك نحو «قليلًا» إذا بدت الوجوه ناعمة جدًا أو مختلفة جدًا.",
    ),
    "fx.fidelity": ("Stay close to the real face", "ابقَ قريبًا من الوجه الحقيقي"),
    "fx.fidelity.help": (
        "Only for Strong repair. Higher keeps more of the real face; lower repairs more.",
        "فقط مع «إصلاح قوي». الأعلى يحتفظ بالمزيد من الوجه الحقيقي، والأقل يصلح أكثر.",
    ),
    "fx.bigger": ("Make the video bigger too?", "هل تريد تكبير الفيديو أيضًا؟"),
    "size.keep": ("Keep the size", "نفس الحجم"),
    "size.keep.desc": ("Only the faces change. The video stays the same size.", "تتغير الوجوه فقط. يبقى الفيديو بالحجم نفسه."),
    "fx.bg": ("How to make it bigger", "طريقة التكبير"),
    "fx.bg.fast": ("Fast (RTX)", "سريع (RTX)"),
    "fx.bg.quality": ("Best quality", "أفضل جودة"),
    "fx.bg.help": (
        "Fast uses your NVIDIA card and takes minutes. Best quality uses AI on every picture and is much slower.",
        "«سريع» يستخدم بطاقة NVIDIA ويستغرق دقائق. «أفضل جودة» يستخدم الذكاء الاصطناعي على كل صورة وهو أبطأ بكثير.",
    ),
    "subs.lead": (
        "Pick a video and press Start. VidForge listens to what people say and writes it as subtitles, each line at the right moment. Your video is not changed.",
        "اختر فيديو واضغط «ابدأ». يستمع VidForge إلى ما يقوله الناس ويكتبه كترجمة على الشاشة، كل سطر في وقته المناسب. لا يتغير الفيديو الأصلي.",
    ),
    "sb.lang": ("What language do they speak?", "ما اللغة التي يتكلمون بها؟"),
    "sb.lang.auto": ("Find out by itself (recommended)", "اكتشفها تلقائيًا (مُستحسَن)"),
    "sb.lang.help": (
        "If it guesses wrong, pick the language here.",
        "إذا أخطأ في التخمين، اختر اللغة من هنا.",
    ),
    "sb.which": ("Which subtitles do you want?", "أي ترجمة تريد؟"),
    "sb.same": ("Same language", "باللغة نفسها"),
    "sb.same.desc": (
        "Writes down exactly what is said, in the language people speak.",
        "يكتب ما يُقال تمامًا، باللغة التي يتكلم بها الناس.",
    ),
    "sb.english": ("In English", "بالإنجليزية"),
    "sb.english.desc": (
        "Listens in any language and writes the subtitles in English.",
        "يستمع بأي لغة ويكتب الترجمة بالإنجليزية.",
    ),
    "sb.level": ("How carefully should it listen?", "بأي دقة يستمع؟"),
    "sb.level.quick": ("Quick", "سريع"),
    "sb.level.balanced": ("Balanced", "متوازن"),
    "sb.level.best": ("Most accurate", "الأدق"),
    "sb.level.quick.desc": (
        "Fastest. Good for clear speech in a quiet room.",
        "الأسرع. جيد للكلام الواضح في مكان هادئ.",
    ),
    "sb.level.balanced.desc": (
        "Hears well even with music or noise in the background. Best for most videos.",
        "يسمع جيدًا حتى مع موسيقى أو ضجيج في الخلفية. الأفضل لمعظم الفيديوهات.",
    ),
    "sb.level.best.desc": (
        "Makes the fewest mistakes, but takes a little longer.",
        "يرتكب أقل عدد من الأخطاء، لكنه يستغرق وقتًا أطول قليلًا.",
    ),
    "sb.inside": ("Also put the subtitles inside a copy of the video", "ضع الترجمة أيضًا داخل نسخة من الفيديو"),
    "sb.inside.help": (
        "You can then turn them on and off in your video player. A subtitle file (.srt) is always saved next to the video too.",
        "يمكنك بعدها تشغيلها وإيقافها من مشغّل الفيديو. يُحفظ أيضًا دائمًا ملف ترجمة (‎.srt) بجانب الفيديو.",
    ),
    "sb.track": ("Sound track", "المسار الصوتي"),
    "sb.track.n": ("Track {n}", "المسار {n}"),
    "sb.track.help": ("Only matters if the video has more than one sound track (for example two languages).", "يهم فقط إذا كان في الفيديو أكثر من مسار صوتي (مثل لغتين)."),
    "subs.title": ("Write down what people say", "اكتب ما يقوله الناس"),
    "subs.desc": (
        "Listens to the video and writes the words on the screen, each at the right moment.",
        "يستمع إلى الفيديو ويكتب الكلام على الشاشة، كل جملة في وقتها المناسب.",
    ),
    "dub.title": ("Translate and dub", "ترجمة ودبلجة"),
    "dub.desc": (
        "Turns the video into another language: new words on the screen and a new voice that sounds like the person speaking.",
        "يحوّل الفيديو إلى لغة أخرى: كلام جديد على الشاشة، وصوت جديد يشبه صوت المتكلم.",
    ),
    "soon.badge": ("Coming soon", "قريبًا"),
    "soon.note": ("This part isn't ready yet. It's next on the list.", "هذا الجزء ليس جاهزًا بعد. إنه التالي في القائمة."),

    # --- units ---
    "u.about": ("about {x}", "نحو {x}"),
    "u.lt_min": ("less than a minute", "أقل من دقيقة"),
    "u.kb": ("{n} KB", "{n} كيلوبايت"),
    "u.mb": ("{n} MB", "{n} ميغابايت"),
    "u.gb": ("{n} GB", "{n} غيغابايت"),
    "u.pct": ("{n}%", "{n}٪"),
    "u.res": ("{w} × {h}", "‎{w}‎ × ‎{h}‎"),
}

# Arabic nouns that change with the number: (one, two, few 3-10, many 11+)
_AR_PLURAL = {
    "sec": ("ثانية واحدة", "ثانيتان", "{n} ثوانٍ", "{n} ثانية"),
    "min": ("دقيقة واحدة", "دقيقتان", "{n} دقائق", "{n} دقيقة"),
    "hour": ("ساعة واحدة", "ساعتان", "{n} ساعات", "{n} ساعة"),
    "video": ("فيديو واحد", "فيديوهان", "{n} فيديوهات", "{n} فيديو"),
}
_EN_PLURAL = {
    "sec": ("{n} sec", "{n} sec"),
    "min": ("{n} min", "{n} min"),
    "hour": ("{n} hour", "{n} hours"),
    "video": ("1 video", "{n} videos"),
}


def set_language(lang: str) -> None:
    global _lang
    _lang = "ar" if lang == "ar" else "en"


def language() -> str:
    return _lang


def is_rtl() -> bool:
    return _lang == "ar"


def resolve(setting: str, system_lang: str) -> str:
    """'system' follows Windows: Arabic when the Windows language is Arabic, else English."""
    if setting in ("en", "ar"):
        return setting
    return "ar" if system_lang.lower().startswith("ar") else "en"


def tr(key: str, **kw) -> str:
    pair = S.get(key)
    if pair is None:
        return key
    text = pair[1] if _lang == "ar" else pair[0]
    return text.format(**kw) if kw else text


def plural(noun: str, n: int) -> str:
    if _lang == "ar":
        one, two, few, many = _AR_PLURAL[noun]
        form = one if n == 1 else two if n == 2 else few if 3 <= n % 100 <= 10 else many
        return form.format(n=n)
    one, many = _EN_PLURAL[noun]
    return (one if n == 1 else many).format(n=n)


def _join(a: str, b: str) -> str:
    return f"{a} و{b}" if _lang == "ar" else f"{a} {b}"


def duration(seconds: float, *, precise: bool = False) -> str:
    """'3 min', '1 hour 20 min'; precise adds seconds under 10 minutes (for remaining time)."""
    s = max(0, int(round(seconds)))
    if s < 60 and not precise:
        return tr("u.lt_min")
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return _join(plural("hour", h), plural("min", m)) if m else plural("hour", h)
    if precise and m < 10:
        if m == 0:
            return plural("sec", max(1, sec))
        return _join(plural("min", m), plural("sec", sec)) if sec else plural("min", m)
    if not precise and sec >= 30:
        m += 1
    return plural("min", max(1, m))


def size(n_bytes: float) -> str:
    if n_bytes >= 1e9:
        return tr("u.gb", n=f"{n_bytes / 1e9:.1f}".rstrip("0").rstrip("."))
    if n_bytes >= 1e6:
        v = n_bytes / 1e6
        return tr("u.mb", n=f"{v:.0f}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip("."))
    return tr("u.kb", n=f"{max(1, n_bytes / 1e3):.0f}")


def about(text: str) -> str:
    return tr("u.about", x=text)


def about_time(seconds: float) -> str:
    """'about 3 min', but just 'less than a minute' for short jobs."""
    return tr("u.lt_min") if seconds < 60 else about(duration(seconds))


def ltr(text: str) -> str:
    """Keep Latin text (versions, paths, times) in its own order inside Arabic sentences."""
    # Left-to-right marks: Qt honours them (it ignores the newer isolate characters).
    return f"‎{text}‎" if _lang == "ar" else text


def pct(fraction: float) -> str:
    return tr("u.pct", n=int(fraction * 100))


def videos(n: int) -> str:
    return plural("video", n)
