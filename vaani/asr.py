"""Speech-to-text (English, Hindi and 90+ other languages) - fully offline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    def transcribe(self, audio_path: str, language: str | None = None) -> tuple[list[Segment], str]: ...


class WhisperTranscriber:
    """Whisper via faster-whisper (CTranslate2, int8). Model downloads once, then works offline."""

    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str, language: str | None = None):
        segs, info = self.model.transcribe(audio_path, language=language, vad_filter=True)
        return [Segment(s.start, s.end, s.text.strip()) for s in segs], info.language
