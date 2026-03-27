#!/usr/bin/env bash
# Create .venv and install: mongol_ml_autocomplete (pyproject.toml) + web (Flask, Pillow).
# If python3 is 3.14+, picks python3.13 / python3.12 / … so PyTorch can install (override: PYTHON=…).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

# PyTorch wheels usually lag the newest CPython (e.g. no torch for 3.14 yet).
# If PYTHON is unset, prefer `python3` when it is < 3.14, else first of 3.13 … 3.9.
if [[ -z "${PYTHON+x}" ]]; then
    if command -v python3 &>/dev/null && python3 -c 'import sys; sys.exit(0 if sys.version_info < (3, 14) else 1)' 2>/dev/null; then
        PYTHON=python3
    else
        PYTHON=
        for try in python3.13 python3.12 python3.11 python3.10 python3.9; do
            if command -v "$try" &>/dev/null && "$try" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)' 2>/dev/null; then
                PYTHON=$try
                echo "Note: python3 is 3.14+ (PyTorch has no wheel yet). Using $try for this venv."
                break
            fi
        done
        if [[ -z "$PYTHON" ]]; then
            echo "Error: PyTorch is not available for Python 3.14+ yet, and no python3.13 … python3.9 was found." >&2
            echo "Install e.g. Python 3.12 (brew install python@3.12), then run this script again." >&2
            exit 1
        fi
    fi
else
    if ! command -v "$PYTHON" &>/dev/null; then
        echo "Error: '$PYTHON' not found." >&2
        exit 1
    fi
    if "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)' 2>/dev/null; then
        echo "Error: $PYTHON is 3.14 or newer; pip cannot install PyTorch for it yet." >&2
        echo "Run without PYTHON=… to auto-pick python3.12, or set e.g. PYTHON=python3.12" >&2
        exit 1
    fi
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)' 2>/dev/null; then
    echo "Error: Python 3.7+ required ($PYTHON is too old)." >&2
    exit 1
fi

echo "=========================================="
echo "Virtual environment ($VENV_DIR)"
echo "=========================================="
echo "Using: $PYTHON ($("$PYTHON" --version))"

if [[ -d "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
    _venv_py=$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    _pick_py=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    if [[ "$_venv_py" != "$_pick_py" ]]; then
        echo "⚠ $VENV_DIR uses Python $_venv_py but this script selected $_pick_py (for PyTorch). Choose Y below to recreate the venv." >&2
    fi
fi

if [[ -d "$VENV_DIR" ]]; then
    echo "Existing environment: $VENV_DIR"
    if [[ -t 0 ]]; then
        read -r -p "Remove and recreate? [y/N] " reply
        if [[ "$reply" =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        fi
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet

if [[ -f requirements.txt ]]; then
    echo "Installing from requirements.txt ..."
    pip install -r requirements.txt --quiet
else
    echo "requirements.txt missing; installing package + web deps only."
    pip install -e . --quiet
    pip install -r web/requirements.txt 
fi

echo ""
echo "Done."
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Web app:   ./run_web_server.sh"
echo "  Optional:  pip install -e '.[dev]'  # pytest, coverage"
echo ""
