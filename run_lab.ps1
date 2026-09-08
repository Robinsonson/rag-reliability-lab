$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing .venv312. Create it and install requirements first."
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m uvicorn lab_api:app --host 127.0.0.1 --port 8000
