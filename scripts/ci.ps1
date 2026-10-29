# Local CI on Windows PowerShell: engine tests, API import smoke, web build.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $env:PYTHON) {
    $candidates = @(
        (Join-Path $Root ".venv312\Scripts\python.exe"),
        (Join-Path $Root ".venv\Scripts\python.exe")
    )
    $python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $python) { $python = "python" }
} else {
    $python = $env:PYTHON
}

Write-Host "[1/3] engine tests ($python)"
& $python -m pytest engine/tests -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] API import smoke"
$smokeData = Join-Path ([System.IO.Path]::GetTempPath()) ("neurokiln-ci-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $smokeData -Force | Out-Null
$previous = $env:NEUROKILN_DATA_DIR
$env:NEUROKILN_DATA_DIR = $smokeData
$code = @"
import sys
from pathlib import Path

sys.path.insert(0, str(Path('server').resolve()))
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    health = client.get('/api/health')
    assert health.status_code == 200, health.text
    assert health.json()['status'] == 'ok'
    assert client.get('/api/cache').status_code == 401
print('api smoke ok')
"@
try {
    $code | & $python -
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    if ($null -eq $previous) {
        Remove-Item Env:NEUROKILN_DATA_DIR -ErrorAction SilentlyContinue
    } else {
        $env:NEUROKILN_DATA_DIR = $previous
    }
}

Write-Host "[3/3] web build"
Push-Location (Join-Path $Root "web")
try {
    if (-not (Test-Path "node_modules")) { npm ci; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
    npm run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

Write-Host "ci passed"
