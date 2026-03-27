# Mongol ML Autocomplete

Traditional Mongolian text autocompletion using a local **TorchScript** model, plus a small **Flask** web keyboard (suggestions, PDF export). The installable package is `mongol_ml_autocomplete` under `src/`.

## What’s in the repo

| Part | Role |
|------|------|
| `src/mongol_ml_autocomplete/` | `MongolMLAutocomplete`, `font_utils` |
| `web/` | Static UI + `server.py` (`/api/suggest`, fonts, keyboard layout) |
| `assets/` | Fonts, `token/new_char_to_token.json`, and **your** `model/zmodel.pt` (not shipped in git) |

## Install

```bash
cd traditional_mongolian_gpt
./setup_venv.sh              # creates .venv and pip install -r requirements.txt
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

If `python3` is **3.14+**, `setup_venv.sh` automatically uses **python3.13**, **python3.12**, … if installed (PyTorch has no wheel for 3.14 yet). Override anytime: `PYTHON=python3.11 ./setup_venv.sh`.

Or manually: `python3 -m venv .venv`, activate, then `pip install -r requirements.txt`.

That installs the package in editable mode (`-e .`), **PyTorch** + **NumPy**, and **Flask** + **Pillow** for the web app. Optional tests tooling: `pip install -e ".[dev]"`.

**Python** ≥ 3.7.

## Web app

```bash
./run_web_server.sh
```

Open the printed URL (default port tries 5001–5003). More detail: [web/README.md](web/README.md).

You need `assets/model/zmodel.pt` and `assets/token/new_char_to_token.json` for autocomplete.

## Library usage

```python
from mongol_ml_autocomplete import MongolMLAutocomplete

ac = MongolMLAutocomplete(
    path_custom_model="assets/model/zmodel.pt",
    path_mappings="assets/token/new_char_to_token.json",
    block_size=40,
    number_of_sample_words=10,
    max_length_of_word=20,
)
ac.initialize()

completions = ac.run_custom_model("ᠠᠯ")  # returns a set[str]
```

The class defaults to `assets/machine_learning/zmodel.pt` and `assets/machine_learning/new_char_to_token.json`; the web server uses `assets/model/` and `assets/token/` instead—pass explicit paths so both match your files. `block_size` is clamped to the model’s limit. Optional: `verbose` / `logger`, or `load_model()` and `load_mappings()` separately.

**Embedding in a service:** create one `MongolMLAutocomplete`, call `initialize()` once at startup, reuse it per request. `run_custom_model()` uses the exact string you pass (whitespace is up to you). Completions may include a leading boundary token (space/newline/punctuation) so callers can see how generation ended.

## Font helpers (`font_utils`)

Resolve the project font and render PNG/PDF (PDF is used by the web app):

```python
from mongol_ml_autocomplete import font_utils

path = font_utils.get_font_path()  # or get_font_path("/path/to/repo/root")
font_utils.create_text_image(text="ᠠᠯᠭᠡᠨ", font_path=path, font_size=32, output_path="out.png")
font_utils.create_vertical_text_pdf_bytes(text="ᠠᠯᠭᠡᠨ", font_path=path)
```

**Pillow** is required for image/PDF paths. Matplotlib / IPython helpers (`setup_matplotlib_font`, `display_text_with_font`, …) need those packages installed separately.

## API (summary)

**`MongolMLAutocomplete`**

- **Args:** `path_custom_model`, `path_mappings`, `block_size`, `number_of_sample_words`, `max_length_of_word`, `verbose`, `logger`
- **`initialize()`** — load model + mappings
- **`run_custom_model(text: str) -> Set[str]`** — suggestions for context `text`

## Project layout

```
traditional_mongolian_gpt/
├── src/mongol_ml_autocomplete/
├── web/
├── assets/
├── pyproject.toml
├── requirements.txt
├── run_web_server.sh
└── setup_venv.sh
```

## Build a wheel/sdist

```bash
pip install build && python -m build
```

## License

MIT

## Acknowledgments

Based on an earlier Dart/Flutter Mongolian autocomplete approach.
