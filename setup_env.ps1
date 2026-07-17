# One-time environment setup for the benchmark (native Windows).
# Run from the repo root:  .\setup_env.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# 1. Python 3.12 venv (PyTorch does not support 3.14 yet)
if (-not (Test-Path "$root\.venv")) {
    py -3.12 -m venv "$root\.venv"
    Write-Host "Created .venv with Python 3.12"
}
$py = "$root\.venv\Scripts\python.exe"

# 2. PyTorch with CUDA 12.4 wheels, then the rest
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -r "$root\requirements.txt"

# 3. Verify GPU
& $py -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "For gated models (Llama-2, Gemma): run  hf auth login  with your HF token after accepting the licenses on huggingface.co"
