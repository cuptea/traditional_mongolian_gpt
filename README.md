# Mongol ML Autocomplete

A Python library for Mongolian text autocompletion using PyTorch models. This library provides an easy-to-use interface for generating word completions based on trained PyTorch models.

## Features

- 🚀 Easy-to-use API for Mongolian text autocompletion
- 🧠 Powered by PyTorch models
- 📦 Installable as a Python package
- 🔌 Easy to embed in web apps, editors, and backend services
- 🧪 Comprehensive test suite
- 🔧 Configurable parameters for sampling and generation

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd traditional_mongolian_gpt

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.7
- PyTorch >= 1.9.0

### Model File

The trained TorchScript model file is not intended to be versioned in Git history. Place your local model at `assets/model/zmodel.pt`, or pass a custom path to `MongolMLAutocomplete(path_custom_model=...)`.

## Quick Start

```python
from mongol_ml_autocomplete import MongolMLAutocomplete

# Initialize the autocomplete instance
autocomplete = MongolMLAutocomplete(
    path_custom_model="assets/model/zmodel.pt",
    path_mappings="assets/token/new_char_to_token.json",
    block_size=40,
    number_of_sample_words=10,
    max_length_of_word=20,
)

# Load the model and mappings
autocomplete.initialize()

# Generate completions for the current text context
input_text = "ᠠᠯ"
completions = autocomplete.run_custom_model(input_text)

print(f"Generated completions: {completions}")
```

`block_size` is the requested context window. If the loaded TorchScript model supports a smaller maximum context, the library automatically clamps to the model's real limit instead of crashing.

## Usage

### Basic Usage

```python
from mongol_ml_autocomplete import MongolMLAutocomplete

# Create an instance with default parameters
autocomplete = MongolMLAutocomplete()

# Or customize parameters
autocomplete = MongolMLAutocomplete(
    path_custom_model="assets/model/zmodel.pt",
    path_mappings="assets/token/new_char_to_token.json",
    block_size=40,               # Requested maximum context length
    number_of_sample_words=10,   # Number of completions to sample
    max_length_of_word=20        # Maximum generated length per completion
)

# Initialize (loads model and mappings)
autocomplete.initialize()

# Generate completions
completions = autocomplete.run_custom_model("ᠠᠯ")
```

### Advanced Usage

```python
import logging

autocomplete = MongolMLAutocomplete(
    path_custom_model="assets/model/zmodel.pt",
    path_mappings="assets/token/new_char_to_token.json",
    verbose=True,
    logger=logging.getLogger("mongol_ml_autocomplete"),
)

# Load model and mappings separately if needed
autocomplete.load_model()
autocomplete.load_mappings()

# Generate completions with custom input
completions = autocomplete.run_custom_model("ᠠᠯᠭᠡᠨ")
```

## Using In Third-Party Applications

This library is designed to be embedded into other applications such as:

- web backends
- editor plugins
- desktop writing tools
- annotation tools
- custom keyboard or input-method prototypes

### Integration Checklist

1. Install the package and its runtime dependencies with `pip install -e .` or package it as part of your application.
2. Point `path_custom_model` to the exported TorchScript model file.
3. Point `path_mappings` to the character-to-token JSON file used by that model.
4. Create a single `MongolMLAutocomplete` instance and reuse it across requests instead of reloading the model each time.
5. Call `initialize()` once during app startup.
6. Pass the current text context to `run_custom_model()`.

### Minimal Service Wrapper

```python
from pathlib import Path

from mongol_ml_autocomplete import MongolMLAutocomplete

PROJECT_ROOT = Path("/path/to/traditional_mongolian_gpt")

autocomplete = MongolMLAutocomplete(
    path_custom_model=str(PROJECT_ROOT / "assets" / "model" / "zmodel.pt"),
    path_mappings=str(PROJECT_ROOT / "assets" / "token" / "new_char_to_token.json"),
    block_size=40,
    number_of_sample_words=10,
    max_length_of_word=20,
)
autocomplete.initialize()


def suggest(text: str):
    if not text:
        return []
    return sorted(autocomplete.run_custom_model(text))
```

### Example: Using It In A Web API

```python
from flask import Flask, jsonify, request

from mongol_ml_autocomplete import MongolMLAutocomplete

app = Flask(__name__)
autocomplete = MongolMLAutocomplete(
    path_custom_model="assets/model/zmodel.pt",
    path_mappings="assets/token/new_char_to_token.json",
)
autocomplete.initialize()


@app.post("/api/suggest")
def suggest():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text") or ""
    completions = sorted(autocomplete.run_custom_model(text)) if text else []
    return jsonify({"completions": completions})
```

### Behavior Notes For Integrators

- `run_custom_model()` returns a `set[str]` of completion strings.
- Returned completions may include the first generated boundary token, which helps callers distinguish whether generation ended with a space, newline, or punctuation boundary.
- Boundary generation stops at the first generated boundary token; repeated trailing boundary tokens are trimmed.
- The library uses the exact text context you pass in, so callers should decide whether to preserve or normalize whitespace before inference.
- If you request a `block_size` larger than the model supports, the library reduces it to the model's actual limit automatically.

### Font And Rendering Helpers For UI Clients

If your third-party application also needs to render Traditional Mongolian text using the bundled project font, you can use `font_utils`:

```python
from mongol_ml_autocomplete import font_utils

font_path = font_utils.get_font_path("/path/to/traditional_mongolian_gpt")

png_path = font_utils.create_text_image(
    text="ᠠᠯᠭᠡᠨ",
    font_path=font_path,
    font_size=32,
    output_path="preview.png",
)

pdf_bytes = font_utils.create_vertical_text_pdf_bytes(
    text="ᠠᠯᠭᠡᠨ",
    font_path=font_path,
)
```

## Font Utilities

The library includes utilities for working with the Mongolian font file (`assets/font/z52chimegtig.otf`):

```python
from mongol_ml_autocomplete import font_utils

# Get font path
font_path = font_utils.get_font_path()

# Setup matplotlib to use Mongolian font
font_utils.setup_matplotlib_font()

# Display text with proper font in Jupyter notebooks
font_utils.display_text_with_font("ᠠᠯᠭᠡᠨ", font_size=24)

# Create an image with Mongolian text
image_path = font_utils.create_text_image(
    text="ᠠᠯᠭᠡᠨ",
    font_size=32,
    output_path="output.png"
)
```

See `example_font.py` for more examples.

## API Reference

### `MongolMLAutocomplete`

Main class for Mongolian text autocompletion.

#### Parameters

- `path_custom_model` (str): Path to the PyTorch model file (.pt)
- `path_mappings` (str): Path to the character-to-token mapping JSON file
- `block_size` (int): Requested maximum context to look back when running model inference (default: 40, clamped to the model's actual limit if needed)
- `number_of_sample_words` (int): Maximum number of word completion attempts (default: 10)
- `max_length_of_word` (int): Maximum length of a generated word (default: 20)

#### Methods

- `initialize()`: Load both the model and mappings
- `load_model()`: Load the PyTorch model
- `load_mappings()`: Load character-to-token mappings from JSON file
- `run_custom_model(input_text: str) -> Set[str]`: Generate autocompletions for the input text

## Project Structure

```
traditional_mongolian_gpt/
├── src/
│   └── mongol_ml_autocomplete/
│       ├── __init__.py
│       └── autocomplete.py
├── tests/
│   ├── __init__.py
│   └── test_mongol_ml_autocomplete.py
├── assets/
│   ├── model/
│   │   └── zmodel.pt
│   └── token/
│       └── new_char_to_token.json
├── setup.py
├── pyproject.toml
└── README.md
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/mongol_ml_autocomplete
```

### Building the Package

```bash
# Build source distribution
python setup.py sdist

# Build wheel
python setup.py bdist_wheel
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

This library is based on the original Dart/Flutter implementation for Mongolian text autocompletion.
