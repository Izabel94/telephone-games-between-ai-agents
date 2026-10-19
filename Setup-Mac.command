#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
echo
echo "=== Telephone workshop: setup (Mac) ==="
echo

finish() {
  echo
  if [ -z "${CI:-}" ]; then read -r -p "Press Enter to close this window."; fi
}
trap finish EXIT

PY=""
for candidate in python3.12 python3.13 python3.14 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
    "$candidate" -c "import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))" 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12, 3.13 or 3.14 was not found. Installing Python 3.12 into this folder..."
  export UV_PYTHON_INSTALL_DIR="$PWD/.python" UV_CACHE_DIR="$PWD/.uv/cache"
  if [ ! -x .uv/uv ]; then
    curl -LsSf https://github.com/astral-sh/uv/releases/download/0.12.19/uv-installer.sh | UV_UNMANAGED_INSTALL="$PWD/.uv" sh
  fi
  .uv/uv python install 3.12 --no-bin --no-registry
  PY=$(.uv/uv python find 3.12 --managed-python)
fi
echo "Using: $("$PY" --version)"

if [ ! -x .venv/bin/python ]; then
  echo
  echo "Creating the environment..."
  "$PY" -m venv .venv
fi
echo
echo "Installing packages. This can take a while the first time..."
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt

if [ ! -f .env ]; then
  if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo
    read -r -p "Paste your Gemini API key here (Cmd+V), then press Enter: " GEMINI_API_KEY
  fi
  if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "No key entered. Double-click Setup-Mac again when you have your key."
    exit 1
  fi
  printf 'GEMINI_API_KEY=%s\n' "$GEMINI_API_KEY" >.env
fi

echo
echo "Checking that the models answer..."
.venv/bin/python check_models.py
echo
echo "=== Setup done. Now double-click Start-Mac to open the notebook. ==="
