#!/usr/bin/env bash
# Local CI: engine tests, API import smoke, web build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv312/Scripts/python.exe" ]]; then
    PYTHON="$ROOT/.venv312/Scripts/python.exe"
  elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON="$ROOT/.venv/Scripts/python.exe"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

echo "[1/3] engine tests ($PYTHON)"
"$PYTHON" -m pytest engine/tests -q

echo "[2/3] API import smoke"
SMOKE_DATA="${NEUROKILN_DATA_DIR:-$(mktemp -d)}"
mkdir -p "$SMOKE_DATA"
NEUROKILN_DATA_DIR="$SMOKE_DATA" "$PYTHON" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("server").resolve()))
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"
    assert client.get("/api/cache").status_code == 401
print("api smoke ok")
PY

echo "[3/3] web build"
cd "$ROOT/web"
if [[ ! -d node_modules ]]; then
  npm ci
fi
npm run build

echo "ci passed"
