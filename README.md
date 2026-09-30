# 🎙️ Vaani Notes

**Private, fully on-device meeting & lecture notes for Snapdragon® PCs.**
Record or upload audio → get a transcript, summary, decisions and action items → search everything by *meaning*.
Nothing ever leaves your laptop. Built for the *Snapdragon® AI Lab Build & Present Challenge*.

## Why
Students, freelancers, clinics, law and finance teams in India often can't upload conversations to a cloud
service (privacy, client confidentiality, patchy connectivity, data costs). Snapdragon X-series PCs have an NPU,
long battery life and strong CPUs - enough to run the whole pipeline locally.

## What it does
| Step | Component | Runs on |
|---|---|---|
| Speech → text (English, Hindi, Kannada, Tamil, Telugu, Marathi + 90 more) | Whisper (`faster-whisper`, int8) | CPU (Oryon) |
| Semantic search embeddings | all-MiniLM-L6-v2 (ONNX) via **ONNX Runtime QNN EP** | **Hexagon NPU** (auto CPU fallback) |
| Summary / decisions / action items | Any local OpenAI-compatible LLM (Ollama, llama.cpp, LM Studio, Genie server) - default `llama3.2:3b` | CPU / NPU / GPU |
| No LLM installed? | Embedding-centrality extractive summary + action-item detection | NPU/CPU |
| Storage & search | SQLite + cosine similarity, local file | Disk |

## Architecture
```
 audio ─▶ Whisper ─▶ transcript ─▶ LLM (localhost) ─▶ summary + action items ─┐
                        │                                                      ├─▶ SQLite (local)
                        └─▶ chunks ─▶ MiniLM ONNX on NPU (QNN EP) ─▶ vectors ──┘        │
                                                               semantic search ◀────────┘
                     Flask UI bound to 127.0.0.1 only · zero cloud calls after setup
```

## Quick start (Snapdragon X HP OmniBook, Windows 11 on Arm)
```powershell
git clone https://github.com/<your-username>/vaani-notes
cd vaani-notes
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_arm64.ps1
# optional, for LLM summaries:  install Ollama, then  ollama pull llama3.2:3b
python app.py                     # open http://127.0.0.1:5000
```
Any other machine (CPU fallback): `pip install -r requirements.txt && python scripts/download_models.py && python app.py`

Environment variables: `VAANI_BACKEND=auto|npu|cpu`, `VAANI_LLM=on|off`, `VAANI_LLM_URL`, `VAANI_LLM_MODEL`,
`VAANI_WHISPER=tiny|base|small`, `VAANI_DB`.

## Verify NPU use & benchmark
```powershell
python -m vaani.backends     # "npu_available": true means QNNExecutionProvider is present
python benchmark.py --runs 50   # CPU vs NPU latency -> results.json
```
The UI header also shows which provider is active.

## Tests
`python -m unittest discover -s tests -t .` - end-to-end pipeline (stubbed ASR), semantic search ranking,
deletion, LLM-offline fallback, and every web route.

## Privacy
Server binds to `127.0.0.1`. Audio is written to a temp folder and deleted after processing. Data lives in one
local SQLite file; delete a note in the UI to remove its transcript and vectors. Models download once, then the
app works in airplane mode.

## Roadmap
Live microphone capture · Whisper encoder on NPU via Qualcomm AI Hub · on-NPU LLM with Genie · speaker labels ·
PDF/Docx export · Windows tray app / MSIX installer.

## Credits
Whisper (OpenAI, MIT), faster-whisper, all-MiniLM-L6-v2 (Apache-2.0), ONNX Runtime, Qualcomm AI Hub, Flask.

MIT License.
