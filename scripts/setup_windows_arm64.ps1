# Run in PowerShell on a Snapdragon X HP OmniBook (Windows 11 on Arm)
# Requires: Python 3.11+ (ARM64 build recommended)  ->  winget install Python.Python.3.12
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_models.py
python -m vaani.backends          # should print "npu_available": true
python -m unittest discover -s tests -t .
Write-Host "Setup complete. Start with:  python app.py   then open http://127.0.0.1:5000"
