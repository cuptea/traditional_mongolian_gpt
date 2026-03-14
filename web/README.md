# Traditional Mongolian Virtual Keyboard (Web)

Standalone website version of the Jupyter virtual keyboard with **autocomplete**. Input Traditional Mongolian Unicode using the character set from `assets/token/new_char_to_token.json`, rendered with the project font.

## Setup

1. **Font**  
   Copy the Mongolian font into the web app so the browser can load it:

   ```bash
   cp assets/font/z52chimegtig.otf web/assets/font/
   ```

   If the font file has a different name, copy it to `web/assets/font/z52chimegtig.otf` (or update the `@font-face` `url()` in `web/styles.css` to match the filename).

2. **Run the app with autocomplete** (recommended)  
   From the **project root**, run the Flask server (serves the site and the autocomplete API):

   ```bash
   ./run_web_server.sh
   ```

   Or manually:

   ```bash
   pip install -r web/requirements.txt   # once: install Flask
   python web/server.py
   ```

   Then open: **http://127.0.0.1:5001/** (or the port printed by the script).

   The server loads the ML model on first “Suggest”; ensure `assets/model/zmodel.pt` and `assets/token/new_char_to_token.json` exist.

3. **Static-only (no autocomplete)**  
   To serve only the static files (keyboard, no Suggest API):

   From the project root: `python -m http.server 8000` → **http://localhost:8000/web/**  
   Or from `web`: `python -m http.server 8080` → **http://localhost:8080/**

## Features

- **Current input** – Vertical (top-down) preview using the same font and orientation as the notebook completion images.
- **Text area** – Type or paste; stays in sync with the preview.
- **Suggest** – Get autocomplete suggestions from the ML model (when using `python web/server.py`). Pick one to insert that completion plus a space.
- **Character grid** – Click to insert characters (space shown as “␣ space”).
- **Backspace / Clear / Copy** – Edit and copy the current text.

## Files

- `index.html` – Page structure and script/style links.
- `styles.css` – Layout, theme, font, vertical display, suggestions.
- `tokens.js` – Character list (from `new_char_to_token.json`, sorted by token ID).
- `app.js` – Keyboard, Suggest API call, suggestion buttons, sync, copy.
- `server.py` – Flask app: serves static files and `POST /api/suggest` for autocomplete.
- `requirements.txt` – Flask dependency for `server.py`.
- `assets/font/` – Put `z52chimegtig.otf` here (see Setup).
