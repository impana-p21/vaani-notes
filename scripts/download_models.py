"""One-time model download (needs internet once). After this the app runs fully offline."""
import shutil

from huggingface_hub import hf_hub_download

REPO = "sentence-transformers/all-MiniLM-L6-v2"
shutil.copy(hf_hub_download(REPO, "onnx/model.onnx"), "models/minilm.onnx")
shutil.copy(hf_hub_download(REPO, "tokenizer.json"), "models/tokenizer.json")
print("Embedding model ready in ./models. Whisper downloads on first transcription.")
