"""audio -> transcript -> summary/action items -> searchable chunks."""
from __future__ import annotations

import os
import time

from .asr import Segment, Transcriber
from .embed import Embedder
from .store import NoteStore
from .summarize import Summarizer


def chunk_segments(segments: list[Segment], per_chunk: int = 4):
    out = []
    for i in range(0, len(segments), per_chunk):
        grp = segments[i:i + per_chunk]
        out.append((grp[0].start, " ".join(s.text for s in grp)))
    return out


def fmt_ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


class Pipeline:
    def __init__(self, asr: Transcriber, embedder: Embedder, summarizer: Summarizer, store: NoteStore):
        self.asr, self.embedder, self.summarizer, self.store = asr, embedder, summarizer, store

    def process(self, audio_path: str, title: str | None = None, language: str | None = None) -> int:
        t0 = time.time()
        segments, lang = self.asr.transcribe(audio_path, language)
        transcript = "\n".join(f"[{fmt_ts(s.start)}] {s.text}" for s in segments)
        plain = " ".join(s.text for s in segments)
        summary = self.summarizer.summarize(plain)
        chunks = chunk_segments(segments)
        vecs = self.embedder.encode([c[1] for c in chunks]) if chunks else []
        audio_len = segments[-1].end if segments else 0.0
        return self.store.add_note(
            title or os.path.splitext(os.path.basename(audio_path))[0], lang, transcript, summary,
            audio_len, time.time() - t0, [(c[0], c[1], v) for c, v in zip(chunks, vecs)])

    def search(self, query: str, k: int = 5):
        return self.store.search(self.embedder.encode([query])[0], k)


def to_markdown(note: dict) -> str:
    s = note["summary"]
    lines = [f"# {note['title']}", "", f"*Language: {note['language']}*", "", "## Summary", s.get("summary", "")]
    if s.get("decisions"):
        lines += ["", "## Decisions"] + [f"- {d}" for d in s["decisions"]]
    if s.get("action_items"):
        lines += ["", "## Action items"] + [f"- [ ] {a}" for a in s["action_items"]]
    lines += ["", "## Transcript", "", note["transcript"]]
    return "\n".join(lines)
