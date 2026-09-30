"""Benchmark the embedding model on CPU vs Snapdragon NPU (QNN) and write results.json.

    python benchmark.py --runs 50
"""
import argparse
import json
import time

import numpy as np

from vaani.backends import describe
from vaani.embed import OnnxEmbedder

SAMPLE = ["We agreed to ship the beta on Friday and Priya will own the release notes."] * 8


def bench(prefer: str, runs: int):
    emb = OnnxEmbedder("models/minilm.onnx", "models/tokenizer.json", prefer=prefer)
    emb.encode(SAMPLE)  # warm-up (graph compile for NPU)
    lat = []
    for _ in range(runs):
        t = time.perf_counter()
        emb.encode(SAMPLE)
        lat.append((time.perf_counter() - t) * 1000)
    return {"provider": emb.active_provider, "median_ms_per_batch8": round(float(np.median(lat)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    out = {"hardware": describe(), "results": [bench("cpu", a.runs), bench("npu", a.runs)]}
    json.dump(out, open("results.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
