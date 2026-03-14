"""
Local server for the Traditional Mongolian Virtual Keyboard web app.
Serves static files and provides /api/suggest for autocomplete.

Run from the project root:
  python web/server.py

Then open http://127.0.0.1:5000/
"""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Optional

# Prefer the local src package so the web app uses current workspace code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, send_from_directory, send_file
from mongol_ml_autocomplete import font_utils

app = Flask(__name__)
WEB_DIR = Path(__file__).resolve().parent
FONT_DIR = PROJECT_ROOT / "assets" / "font"
KEYBOARD_LAYOUT_PATH = WEB_DIR / "keyboard-layout.json"
logger = logging.getLogger(__name__)

# Lazy-loaded autocomplete model
_autocomplete_model = None


def resolve_font_path(font_name: Optional[str]):
    """Resolve a requested project font name to a local path."""
    if not font_name:
        return font_utils.get_font_path(PROJECT_ROOT)

    candidate = FONT_DIR / Path(font_name).name
    if (
        candidate.exists()
        and candidate.is_file()
        and candidate.suffix.lower() in {".otf", ".ttf", ".woff", ".woff2"}
        and candidate.stem.lower().startswith("z52")
    ):
        return candidate
    return font_utils.get_font_path(PROJECT_ROOT)


def load_keyboard_layout():
    """Load the editable keyboard layout JSON from disk."""
    with KEYBOARD_LAYOUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_keyboard_layout(data):
    """Perform lightweight validation for the keyboard layout payload."""
    if not isinstance(data, dict):
        raise ValueError("Keyboard layout must be a JSON object.")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Keyboard layout must contain a rows array.")

    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each keyboard row must be an array.")
        for key in row:
            if not isinstance(key, dict):
                raise ValueError("Each keyboard key must be an object.")
            if not key.get("code") or not key.get("label"):
                raise ValueError("Each keyboard key must include code and label.")

    return data


def get_autocomplete():
    global _autocomplete_model
    if _autocomplete_model is None:
        from mongol_ml_autocomplete import MongolMLAutocomplete
        model_path = PROJECT_ROOT / "assets" / "model" / "zmodel.pt"
        mapping_path = PROJECT_ROOT / "assets" / "token" / "new_char_to_token.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")
        _autocomplete_model = MongolMLAutocomplete(
            path_custom_model=str(model_path),
            path_mappings=str(mapping_path),
            block_size=40,
            verbose=True,
            logger=logging.getLogger("mongol_ml_autocomplete"),
        )
        _autocomplete_model.initialize()
        logger.info("Autocomplete model initialized")
    return _autocomplete_model


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/fonts", methods=["GET"])
def list_fonts():
    """Return available font files from the project assets/font directory."""
    if not FONT_DIR.exists():
        logger.warning("Font directory not found: %s", FONT_DIR)
        return jsonify({"fonts": []})

    fonts = []
    for path in sorted(FONT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".otf", ".ttf", ".woff", ".woff2"}:
            continue

        # Only expose the Mongolian font family intended for the keyboard UI.
        # The folder also contains generic UI fonts (e.g. Segoe UI) that do not
        # render the Traditional Mongolian keyboard consistently.
        if not path.stem.lower().startswith("z52"):
            continue

        if path.is_file():
            fonts.append(
                {
                    "name": path.name,
                    "label": path.stem,
                    "url": f"/project-assets/font/{path.name}",
                    "format": path.suffix.lower().lstrip("."),
                }
            )
    return jsonify({"fonts": fonts})


@app.route("/api/suggest", methods=["POST"])
def suggest():
    """Accept JSON { \"text\": \"...\" }, return { \"completions\": [\"...\", ...] }."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text") or ""
        if not text:
            logger.info("Suggest request: empty input")
            return jsonify({"completions": []})
        model = get_autocomplete()
        completions = list(model.run_custom_model(text))
        logger.info(
            "Suggest request: input_length=%d completions=%d",
            len(text),
            len(completions),
        )
        return jsonify({"completions": completions})
    except FileNotFoundError as e:
        logger.exception("Suggest request failed: missing required file")
        return jsonify({"error": str(e), "completions": []}), 503
    except Exception as e:
        logger.exception("Suggest request failed")
        return jsonify({"error": str(e), "completions": []}), 500


@app.route("/api/export/pdf", methods=["POST"])
def export_pdf():
    """Accept JSON { "text": "...", "font_name": "..." } and return a PDF file."""
    try:
        from io import BytesIO

        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text") or ""
        font_name = data.get("font_name") or ""
        if not text:
            return jsonify({"error": "No text to export."}), 400

        font_path = resolve_font_path(font_name)
        pdf_bytes = font_utils.create_vertical_text_pdf_bytes(
            text=text,
            font_path=font_path,
            font_size=36,
        )
        if not pdf_bytes:
            logger.error("PDF export failed: renderer returned no output")
            return jsonify({"error": "Could not generate PDF."}), 500

        logger.info(
            "PDF export: input_length=%d font=%s",
            len(text),
            Path(font_path).name if font_path else "default",
        )
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="mongolian-text.pdf",
        )
    except Exception as e:
        logger.exception("PDF export failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/keyboard-layout", methods=["GET", "POST"])
def keyboard_layout():
    """Load or save the editable keyboard layout JSON."""
    try:
        if request.method == "GET":
            return jsonify(load_keyboard_layout())

        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Expected JSON body."}), 400

        layout = validate_keyboard_layout(data)
        KEYBOARD_LAYOUT_PATH.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Keyboard layout saved: rows=%d", len(layout.get("rows", [])))
        return jsonify({"ok": True})
    except FileNotFoundError:
        logger.exception("Keyboard layout file not found: %s", KEYBOARD_LAYOUT_PATH)
        return jsonify({"error": "Keyboard layout file not found."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Keyboard layout API failed")
        return jsonify({"error": str(e)}), 500


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(WEB_DIR, path)


@app.route("/project-assets/font/<path:filename>")
def project_font(filename):
    return send_from_directory(FONT_DIR, filename)


if __name__ == "__main__":
    import os
    log_level_name = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    log_level = getattr(logging, log_level_name, logging.DEBUG)
    log_dir = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("LOG_FILE", "web_server.log")

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Writing logs to %s", log_file)
    logger.info("Log level set to %s", log_level_name)
    # Run from project root so assets/ paths and mongol_ml_autocomplete resolve
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)
