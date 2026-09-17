param([switch]$Legacy)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
if (-not $Legacy) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker Compose is required for v2. See docs/UPGRADE_V2.md. Use -Legacy for the original local benchmark."
    }
    if (-not (Test-Path -LiteralPath '.env.v2')) {
        throw "Generate .env.v2 first: python -m policy_lab.manage setup-env"
    }
    & docker compose --env-file .env.v2 up --build -d
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose startup failed" }
    Write-Host "Open http://127.0.0.1:8000. First run: provision local models as described in docs/UPGRADE_V2.md."
    exit 0
}
$pythonPath = Join-Path $projectRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing .venv312. Create it and install requirements first."
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m uvicorn lab_api:app --host 127.0.0.1 --port 8000
