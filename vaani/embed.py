"""Sentence embeddings for semantic search across notes.

* OnnxEmbedder  - MiniLM-style ONNX model; runs on the Snapdragon NPU via QNN EP.
* HashingEmbedder - zero-dependency fallback (feature hashing) so the app works
  before any model is downloaded and in unit tests.
"""
from __future__ import annotations

import os
import re
import zlib
from typing import Protocol

import numpy as np

from .backends import pick_providers


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (fallback)."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = "hashing-fallback"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return _l2(out)


class OnnxEmbedder:
    """all-MiniLM-L6-v2 style encoder in ONNX (mean pooling + L2 norm)."""

    def __init__(self, model_path: str, tokenizer_path: str, max_len: int = 128, prefer: str = "auto"):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.providers = pick_providers(prefer)
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.active_provider = self.session.get_providers()[0]
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(length=max_len)  # static shape -> NPU friendly
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = 384
        self.name = f"onnx-minilm[{self.active_provider}]"

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]  # (B, T, H)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return _l2(pooled.astype(np.float32))


def load_embedder(model_dir: str = "models", prefer: str = "auto") -> Embedder:
    m = os.path.join(model_dir, "minilm.onnx")
    t = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(m) and os.path.exists(t):
        try:
            return OnnxEmbedder(m, t, prefer=prefer)
        except Exception as e:  # pragma: no cover
            print(f"[vaani] ONNX embedder failed ({e}); using fallback")
    return HashingEmbedder()
