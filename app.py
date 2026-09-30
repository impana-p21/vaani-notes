"""Vaani Notes - local web UI (http://127.0.0.1:5000). Binds to localhost only."""
from __future__ import annotations

import os
import tempfile

from flask import Flask, Response, redirect, render_template_string, request, url_for

from vaani.backends import describe
from vaani.embed import load_embedder
from vaani.pipeline import Pipeline, fmt_ts, to_markdown
from vaani.store import NoteStore
from vaani.summarize import ExtractiveSummarizer, LocalLLMSummarizer

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Vaani Notes</title>
<style>
:root{--bg:#f4f6fb;--fg:#0b1437;--card:#fff;--ac:#3253dc}
@media(prefers-color-scheme:dark){:root{--bg:#0b1437;--fg:#eef1ff;--card:#141d4d}}
body{font:16px/1.5 Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--fg);max-width:880px;margin:0 auto;padding:24px}
.card{background:var(--card);border-radius:14px;padding:18px;margin:14px 0;box-shadow:0 2px 10px #0002}
button,input[type=submit]{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:15px;cursor:pointer}
input,select{padding:8px;border-radius:8px;border:1px solid #8886;background:transparent;color:inherit}
.badge{display:inline-block;background:#00c2a822;color:#00a58e;border-radius:99px;padding:2px 10px;font-size:13px}
a{color:var(--ac)} pre{white-space:pre-wrap}
</style>
<h1>🎙️ Vaani Notes <span class=badge>100% on-device</span></h1>
<p class=badge>{{hw}}</p>
{% block %}{% endblock %}
"""

HOME = """
<div class=card><form method=post action="/upload" enctype=multipart/form-data>
<b>New note</b><br><br>
<input type=file name=audio accept="audio/*,video/mp4" required>
<select name=language><option value="">Auto-detect</option><option value=en>English</option>
<option value=hi>हिन्दी</option><option value=kn>ಕನ್ನಡ</option><option value=ta>தமிழ்</option>
<option value=te>తెలుగు</option><option value=mr>मराठी</option></select>
<input type=submit value="Transcribe & summarise"></form></div>
<div class=card><form action="/search"><input name=q placeholder="Search across all notes by meaning…" size=44 required>
<input type=submit value=Search></form></div>
{% for n in notes %}<div class=card><a href="/note/{{n.id}}"><b>{{n.title}}</b></a>
<br><small>{{n.language}} · audio {{n.audio_seconds|round|int}}s · processed in {{n.process_seconds|round(1)}}s</small></div>
{% else %}<p>No notes yet. Upload a recording - it never leaves this PC.</p>{% endfor %}"""

NOTE = """<a href="/">← back</a><div class=card><h2>{{n.title}}</h2>
<h3>Summary</h3><p>{{n.summary.summary}}</p>
{% if n.summary.get('note') %}<small>{{n.summary.note}}</small>{% endif %}
{% if n.summary.decisions %}<h3>Decisions</h3><ul>{% for d in n.summary.decisions %}<li>{{d}}</li>{% endfor %}</ul>{% endif %}
{% if n.summary.action_items %}<h3>Action items</h3><ul>{% for a in n.summary.action_items %}<li>☐ {{a}}</li>{% endfor %}</ul>{% endif %}
<a href="/note/{{n.id}}/export.md">Export Markdown</a> ·
<form style=display:inline method=post action="/note/{{n.id}}/delete"><input type=submit value=Delete></form></div>
<div class=card><h3>Transcript</h3><pre>{{n.transcript}}</pre></div>"""

SEARCH = """<a href="/">← back</a><h2>Results for “{{q}}”</h2>
{% for r in results %}<div class=card><a href="/note/{{r.note_id}}"><b>{{r.title}}</b></a> @ {{ts(r.start)}}
<small>(match {{'%.2f'|format(r.score)}})</small><p>{{r.text}}</p></div>{% else %}<p>Nothing found.</p>{% endfor %}"""


def render(body, **kw):
    return render_template_string(PAGE.replace("{% block %}{% endblock %}", body), hw=hw_line(), **kw)


def hw_line():
    d = describe()
    emb = EMB.name if EMB else "n/a"
    return f"{d['machine']} · NPU: {'QNN active' if d['npu_available'] else 'not detected (CPU fallback)'} · embedder: {emb}"


def build_pipeline() -> Pipeline:
    global EMB
    prefer = os.environ.get("VAANI_BACKEND", "auto")
    EMB = load_embedder("models", prefer)
    extractive = ExtractiveSummarizer(EMB)
    if os.environ.get("VAANI_LLM", "on") == "off":
        summ = extractive
    else:
        summ = LocalLLMSummarizer(
            os.environ.get("VAANI_LLM_URL", "http://127.0.0.1:11434/v1"),
            os.environ.get("VAANI_LLM_MODEL", "llama3.2:3b"), fallback=extractive)
    from vaani.asr import WhisperTranscriber
    asr = WhisperTranscriber(os.environ.get("VAANI_WHISPER", "base"))
    return Pipeline(asr, EMB, summ, NoteStore(os.environ.get("VAANI_DB", "vaani.db")))


EMB = None


def create_app(pipeline: Pipeline | None = None) -> Flask:
    app = Flask(__name__)
    global EMB
    P = pipeline or build_pipeline()
    EMB = P.embedder

    @app.get("/")
    def home():
        return render(HOME, notes=P.store.list_notes())

    @app.post("/upload")
    def upload():
        f = request.files["audio"]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, os.path.basename(f.filename or "audio"))
            f.save(path)
            nid = P.process(path, os.path.splitext(f.filename or "Untitled")[0], request.form.get("language") or None)
        return redirect(url_for("note", nid=nid))

    @app.get("/note/<int:nid>")
    def note(nid):
        n = P.store.get_note(nid)
        return render(NOTE, n=n) if n else ("Not found", 404)

    @app.get("/note/<int:nid>/export.md")
    def export(nid):
        n = P.store.get_note(nid)
        if not n:
            return "Not found", 404
        return Response(to_markdown(n), mimetype="text/markdown",
                        headers={"Content-Disposition": f"attachment; filename=note_{nid}.md"})

    @app.post("/note/<int:nid>/delete")
    def delete(nid):
        P.store.delete_note(nid)
        return redirect("/")

    @app.get("/search")
    def search():
        q = request.args.get("q", "")
        return render(SEARCH, q=q, results=P.search(q), ts=fmt_ts)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
