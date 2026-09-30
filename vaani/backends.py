"""Hardware backend selection for ONNX Runtime.

On a Snapdragon X-series PC with the ``onnxruntime-qnn`` wheel installed, the
``QNNExecutionProvider`` runs models on the Hexagon NPU (HTP backend). On any
other machine we transparently fall back to CPU so the project stays runnable.
"""
from __future__ import annotations

import platform
from typing import Any


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:  # onnxruntime not installed
        return []


def pick_providers(prefer: str = "auto") -> list[Any]:
    """Return an ONNX Runtime ``providers`` list.

    prefer: "auto" | "npu" | "cpu"
    """
    avail = available_providers()
    prefer = prefer.lower()
    if prefer in ("auto", "npu") and "QNNExecutionProvider" in avail:
        return [
            ("QNNExecutionProvider", {
                "backend_path": "QnnHtp.dll" if platform.system() == "Windows" else "libQnnHtp.so",
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1",
            }),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def describe() -> dict:
    avail = available_providers()
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "onnxruntime_providers": avail,
        "npu_available": "QNNExecutionProvider" in avail,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
