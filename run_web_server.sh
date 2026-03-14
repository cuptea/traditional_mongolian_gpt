#!/usr/bin/env bash
# Run the Traditional Mongolian Virtual Keyboard web app with autocomplete API.
# Usage: ./run_web_server.sh   (from project root)

set -e

# Project root: directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
export LOG_DIR
export LOG_FILE="${LOG_FILE:-web_server.log}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# Ensure font is available for the web app
FONT_SRC="$PROJECT_ROOT/assets/font/z52chimegtig.otf"
FONT_DST="$PROJECT_ROOT/web/assets/font/z52chimegtig.otf"
if [[ -f "$FONT_SRC" && ! -f "$FONT_DST" ]]; then
  echo "Copying font to web/assets/font/..."
  mkdir -p "$(dirname "$FONT_DST")"
  cp "$FONT_SRC" "$FONT_DST"
fi

# Prefer venv if present
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
  echo "Using venv: $PYTHON"
else
  PYTHON="$(command -v python3 || command -v python)"
  if [[ -z "$PYTHON" ]]; then
    echo "Error: python3 or python not found." >&2
    exit 1
  fi
  echo "Using: $PYTHON"
fi

# Install Flask if missing
if ! "$PYTHON" -c "import flask" 2>/dev/null; then
  echo "Installing Flask (web/requirements.txt)..."
  "$PYTHON" -m pip install -q -r "$PROJECT_ROOT/web/requirements.txt"
fi

# Keep NumPy compatible with the installed PyTorch build used by autocomplete.
if ! "$PYTHON" -c 'import numpy as np; import sys; sys.exit(0 if int(np.__version__.split(".")[0]) < 2 else 1)' 2>/dev/null; then
  echo "Installing compatible NumPy (<2) for PyTorch..."
  "$PYTHON" -m pip install -q "numpy<2"
fi

# Try default port first; use next if in use
for port in 5001 5002 5003; do
  if command -v lsof &>/dev/null; then
    if ! lsof -i ":$port" -sTCP:LISTEN -t &>/dev/null; then break; fi
  else
    break
  fi
done
export PORT="${PORT:-$port}"

echo ""
echo "Starting server at http://127.0.0.1:$PORT"
echo "Logs: $LOG_DIR/$LOG_FILE"
echo "Log level: $LOG_LEVEL"
echo "Open that URL in your browser. Press Ctrl+C to stop."
echo ""

exec "$PYTHON" "$PROJECT_ROOT/web/server.py"
