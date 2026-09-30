"""Note generation: local LLM when available, extractive fallback otherwise."""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Protocol

import numpy as np

from .embed import Embedder

PROMPT = """You are a meeting/lecture note-taker. From the transcript below produce ONLY compact JSON:
{{"summary": "3-5 sentence summary", "decisions": ["..."], "action_items": ["owner - task - due if stated"]}}
Write in the same language as the transcript.

TRANSCRIPT:
{t}
"""

ACTION_RE = re.compile(
    r"\b(will|shall|need to|needs to|must|should|todo|to-do|action item|deadline|by (monday|tuesday|wednesday|thursday|friday|tomorrow|next week))\b",
    re.I,
)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


class Summarizer(Protocol):
    def summarize(self, transcript: str) -> dict: ...


class ExtractiveSummarizer:
    """Model-free fallback: embedding-centrality summary + regex action items."""

    def __init__(self, embedder: Embedder, k: int = 4):
        self.embedder, self.k = embedder, k

    def summarize(self, transcript: str) -> dict:
        sents = _sentences(transcript)
        if not sents:
            return {"summary": "", "decisions": [], "action_items": []}
        vecs = self.embedder.encode(sents)
        centroid = vecs.mean(0, keepdims=True)
        scores = (vecs @ centroid.T).ravel()
        top = sorted(np.argsort(-scores)[: self.k])  # keep original order
        return {
            "summary": " ".join(sents[i] for i in top),
            "decisions": [],
            "action_items": [s for s in sents if ACTION_RE.search(s)][:8],
        }


class LocalLLMSummarizer:
    """Talks to any local OpenAI-compatible server (Ollama, llama.cpp, LM Studio,
    Qualcomm Genie-based servers). Nothing leaves localhost."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434/v1", model: str = "llama3.2:3b",
                 fallback: Summarizer | None = None, timeout: int = 300):
        self.base_url, self.model, self.fallback, self.timeout = base_url.rstrip("/"), model, fallback, timeout

    def summarize(self, transcript: str) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT.format(t=transcript[:12000])}],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0))
            return {
                "summary": str(data.get("summary", "")),
                "decisions": [str(x) for x in data.get("decisions", [])],
                "action_items": [str(x) for x in data.get("action_items", [])],
            }
        except Exception as e:
            if self.fallback is None:
                raise
            out = self.fallback.summarize(transcript)
            out["note"] = f"LLM unavailable ({type(e).__name__}); extractive summary used."
            return out
