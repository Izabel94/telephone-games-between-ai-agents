#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/jupyter ]; then
  echo "Setup has not been run yet. Double-click Setup-Mac first."
  if [ -z "${CI:-}" ]; then read -r -p "Press Enter to close this window."; fi
  exit 1
fi
echo "Opening the notebook in your browser..."
echo "Keep this window open while you work. To stop, close the browser tab and this window."
exec .venv/bin/jupyter lab workshop.ipynb "$@"
