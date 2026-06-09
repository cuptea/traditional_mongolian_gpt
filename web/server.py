"""
Gradio/FastAPI backend for the Traditional Mongolian Virtual Keyboard web app.

The module exposes two entry points:
  * app  - ASGI app for uvicorn/Render/Hugging Face Docker-style hosting.
  * demo - Gradio Blocks app for Hugging Face Gradio Spaces.

Run from the project root:
  python web/server.py

Then open http://127.0.0.1:5001/ for the existing keyboard UI or
http://127.0.0.1:5001/gradio for the Gradio interface.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Dict, List, Optional

# Prefer the local src package so the web app uses current workspace code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mongol_ml_autocomplete import font_utils

WEB_DIR = Path(__file__).resolve().parent
FONT_DIR = PROJECT_ROOT / "assets" / "font"
KEYBOARD_LAYOUT_PATH = WEB_DIR / "keyboard-layout.json"
logger = logging.getLogger(__name__)

# Lazy-loaded autocomplete model and bounded suggestion cache.
_autocomplete_model = None
_autocomplete_lock = Lock()
_suggest_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_suggest_cache_lock = Lock()


class SuggestRequest(BaseModel):
    text: str = ""


class ExportPdfRequest(BaseModel):
    text: str = ""
    font_name: str = ""
    font_size: int = 36


def configure_logging() -> None:
    """Configure root logging for local runs and hosted deployments."""
    log_level_name = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    log_level = getattr(logging, log_level_name, logging.DEBUG)
    log_dir = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("LOG_FILE", "web_server.log")

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
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


def get_cached_suggestions(text: str) -> Optional[List[str]]:
    with _suggest_cache_lock:
        value = _suggest_cache.get(text)
        if value is None:
            return None
        _suggest_cache.move_to_end(text)
        return list(value)


def set_cached_suggestions(text: str, completions: List[str]) -> None:
    cache_max = int(os.environ.get("AUTOCOMPLETE_CACHE_SIZE", "2000"))
    with _suggest_cache_lock:
        _suggest_cache[text] = tuple(completions)
        _suggest_cache.move_to_end(text)
        while len(_suggest_cache) > cache_max:
            _suggest_cache.popitem(last=False)


def resolve_font_path(font_name: Optional[str]) -> Path:
    """Resolve a requested project font name to a local path."""
    if not font_name:
        return Path(font_utils.get_font_path(PROJECT_ROOT))

    candidate = FONT_DIR / Path(font_name).name
    if (
        candidate.exists()
        and candidate.is_file()
        and candidate.suffix.lower() in {".otf", ".ttf", ".woff", ".woff2"}
        and candidate.stem.lower().startswith("z52")
    ):
        return candidate
    return Path(font_utils.get_font_path(PROJECT_ROOT))


def load_keyboard_layout() -> Dict[str, Any]:
    """Load the editable keyboard layout JSON from disk."""
    with KEYBOARD_LAYOUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_keyboard_layout(data: Any) -> Dict[str, Any]:
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
        with _autocomplete_lock:
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
                    verbose=False,
                    logger=logging.getLogger("mongol_ml_autocomplete"),
                )
                _autocomplete_model.initialize()
                logger.info("Autocomplete model initialized")
    return _autocomplete_model


def list_project_fonts() -> List[Dict[str, str]]:
    """Return available Traditional Mongolian font files."""
    if not FONT_DIR.exists():
        logger.warning("Font directory not found: %s", FONT_DIR)
        return []

    fonts: List[Dict[str, str]] = []
    for path in sorted(FONT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".otf", ".ttf", ".woff", ".woff2"}:
            continue
        if not path.stem.lower().startswith("z52"):
            continue
        fonts.append(
            {
                "name": path.name,
                "label": path.stem,
                "url": f"/project-assets/font/{path.name}",
                "format": path.suffix.lower().lstrip("."),
            }
        )
    return fonts


def suggest_text(text: str) -> List[str]:
    """Return model-backed autocomplete suggestions for API and Gradio callers."""
    if not text:
        logger.info("Suggest request: empty input")
        return []

    cached = get_cached_suggestions(text)
    if cached is not None:
        return cached

    model = get_autocomplete()
    completions = list(model.run_custom_model(text))
    set_cached_suggestions(text, completions)
    logger.info("Suggest request: input_length=%d completions=%d", len(text), len(completions))
    return completions


def create_pdf_bytes(text: str, font_name: str = "", font_size: int = 36) -> bytes:
    """Render text to vertical Traditional Mongolian PDF bytes."""
    if font_size < 16 or font_size > 96:
        font_size = 36
    if not text:
        raise ValueError("No text to export.")

    font_path = resolve_font_path(font_name)
    pdf_bytes = font_utils.create_vertical_text_pdf_bytes(
        text=text,
        font_path=font_path,
        font_size=font_size,
    )
    if not pdf_bytes:
        raise RuntimeError("Could not generate PDF.")

    logger.info("PDF export: input_length=%d font=%s", len(text), Path(font_path).name)
    return pdf_bytes


def gradio_suggest(text: str) -> str:
    """Format suggestions for the Gradio UI."""
    try:
        completions = suggest_text(text or "")
    except Exception as exc:
        logger.exception("Gradio suggest failed")
        return f"Error: {exc}"
    if not completions:
        return "No suggestions."
    return "\n".join(f"{index + 1}. {completion}" for index, completion in enumerate(completions))


def gradio_export_pdf(text: str, font_name: str, font_size: int) -> str:
    """Create a temporary PDF file for Gradio's file download component."""
    pdf_bytes = create_pdf_bytes(text or "", font_name or "", int(font_size or 36))
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".pdf",
        prefix="mongolian-text-",
        delete=False,
    )
    with handle:
        handle.write(pdf_bytes)
    return handle.name


def create_demo() -> gr.Blocks:
    """Create the Hugging Face Gradio interface."""
    font_choices = [font["name"] for font in list_project_fonts()]
    default_font = "z52chimegtig.otf" if "z52chimegtig.otf" in font_choices else (font_choices[0] if font_choices else "")

    with gr.Blocks(title="Traditional Mongolian GPT") as demo:
        gr.Markdown(
            "# Traditional Mongolian GPT\n"
            "Use the autocomplete model from a Hugging Face Gradio server. "
            "The full keyboard/editor remains available at `/`."
        )
        with gr.Row():
            text = gr.Textbox(
                label="Traditional Mongolian text",
                lines=8,
                placeholder="Type or paste Traditional Mongolian text…",
            )
            suggestions = gr.Textbox(label="Suggestions", lines=8, interactive=False)
        with gr.Row():
            suggest_button = gr.Button("Suggest", variant="primary")
            font_name = gr.Dropdown(
                choices=font_choices,
                value=default_font,
                label="PDF font",
                interactive=bool(font_choices),
            )
            font_size = gr.Slider(16, 96, value=36, step=1, label="PDF font size")
        with gr.Row():
            export_button = gr.Button("Export PDF")
            pdf_file = gr.File(label="Generated PDF")

        suggest_button.click(gradio_suggest, inputs=text, outputs=suggestions)
        text.submit(gradio_suggest, inputs=text, outputs=suggestions)
        export_button.click(gradio_export_pdf, inputs=[text, font_name, font_size], outputs=pdf_file)

    return demo


def create_app() -> FastAPI:
    """Create the ASGI app with Gradio mounted as the backend UI."""
    api = FastAPI(title="Traditional Mongolian GPT")

    @api.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @api.get("/api/fonts")
    async def fonts():
        return {"fonts": list_project_fonts()}

    @api.post("/api/suggest")
    async def suggest(payload: SuggestRequest):
        try:
            return {"completions": suggest_text(payload.text or "")}
        except FileNotFoundError as exc:
            logger.exception("Suggest request failed: missing required file")
            return JSONResponse({"error": str(exc), "completions": []}, status_code=503)
        except Exception as exc:
            logger.exception("Suggest request failed")
            return JSONResponse({"error": str(exc), "completions": []}, status_code=500)

    @api.post("/api/export/pdf")
    async def export_pdf(payload: ExportPdfRequest):
        try:
            pdf_bytes = create_pdf_bytes(payload.text or "", payload.font_name or "", payload.font_size)
            headers = {"Content-Disposition": 'attachment; filename="mongolian-text.pdf"'}
            return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.exception("PDF export failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @api.get("/api/keyboard-layout")
    async def get_keyboard_layout():
        try:
            return load_keyboard_layout()
        except FileNotFoundError as exc:
            logger.exception("Keyboard layout file not found: %s", KEYBOARD_LAYOUT_PATH)
            raise HTTPException(status_code=404, detail="Keyboard layout file not found.") from exc

    @api.post("/api/keyboard-layout")
    async def save_keyboard_layout(request: Request):
        try:
            data = await request.json()
            layout = validate_keyboard_layout(data)
            KEYBOARD_LAYOUT_PATH.write_text(
                json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Keyboard layout saved: rows=%d", len(layout.get("rows", [])))
            return {"ok": True}
        except json.JSONDecodeError:
            return JSONResponse({"error": "Expected JSON body."}, status_code=400)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            logger.exception("Keyboard layout API failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    api.mount("/project-assets/font", StaticFiles(directory=FONT_DIR), name="project_fonts")
    mounted_app = gr.mount_gradio_app(api, demo, path="/gradio")

    @mounted_app.get("/{static_path:path}", include_in_schema=False)
    async def static_web_file(static_path: str):
        requested = (WEB_DIR / static_path).resolve()
        try:
            requested.relative_to(WEB_DIR)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="File not found.") from exc
        if requested.is_file():
            return FileResponse(requested)
        raise HTTPException(status_code=404, detail="File not found.")

    return mounted_app


configure_logging()
demo = create_demo()
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "5001"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
