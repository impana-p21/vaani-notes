# Design notes

**Backend selection** (`vaani/backends.py`): asks ONNX Runtime which providers exist; if
`QNNExecutionProvider` is present (Snapdragon + `onnxruntime-qnn`), sessions are created with the HTP backend
(`QnnHtp.dll`, burst mode, fp16). Otherwise CPU. Same code path everywhere, so contributors without Snapdragon
hardware can still run and test.

**NPU-friendly shapes** (`vaani/embed.py`): tokenizer padding is fixed at 128 tokens so the graph has static shapes,
which the HTP backend requires for best results. If a given ONNX export doesn't compile for HTP, ONNX Runtime falls
back node-by-node to CPU; for best speed export/quantise the model with Qualcomm AI Hub.

**Graceful degradation**: no embedding model → hashing embedder; no LLM → extractive summary; no NPU → CPU.
The app never hard-fails because a component is missing.

**Privacy by construction**: only stdlib `urllib` call goes to a user-configured localhost LLM URL; Flask binds to
127.0.0.1; temp audio is deleted; no telemetry.
