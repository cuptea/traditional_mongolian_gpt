"""
Utility functions for working with Mongolian fonts.
"""
import base64
from pathlib import Path
from typing import Optional, Union, Set, List
from tempfile import NamedTemporaryFile
import unicodedata


def get_font_path(base_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Get the path to the Mongolian font file."""
    if base_path is None:
        current = Path.cwd()
        if current.name == "notebooks":
            base_path = current.parent
        elif current.name == "src":
            base_path = current.parent
        else:
            base_path = current
    else:
        base_path = Path(base_path)
    
    font_path = base_path / "assets" / "font" / "z52chimegtig.otf"
    if font_path.exists():
        return font_path
    
    # Try alternative locations
    for alt_path in [
        base_path.parent / "assets" / "font" / "z52chimegtig.otf",
        Path(__file__).parent.parent.parent / "assets" / "font" / "z52chimegtig.otf",
    ]:
        if alt_path.exists():
            return alt_path
    
    return None


def _load_font(font_path: Optional[Union[str, Path]], font_size: int):
    """Load font file or return default font."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None, "PIL/Pillow not installed"
    
    if font_path is None:
        font_path = get_font_path()
    
    if font_path is None or not Path(font_path).exists():
        return ImageFont.load_default(), None
    
    try:
        return ImageFont.truetype(str(font_path), font_size), None
    except OSError as e:
        return ImageFont.load_default(), f"Error loading font: {e}"
    except Exception as e:
        return ImageFont.load_default(), f"Unexpected error: {e}"


def _create_rotated_text_image(
    text: str,
    font,
    width: int,
    height: int,
    bg_color: str,
    text_color: str
):
    """Create an image with text, rotated 90° clockwise, centered."""
    from PIL import Image, ImageDraw
    
    # Create padded canvas for rotation
    padding = max(width, height) // 4
    canvas_width = width + 2 * padding
    canvas_height = height + 2 * padding
    
    img = Image.new('RGB', (canvas_width, canvas_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Center text on canvas
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (canvas_width - text_width) // 2
    y = (canvas_height - text_height) // 2
    
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Rotate 90° clockwise
    img = img.rotate(-90, expand=True)
    
    # Crop to desired size (dimensions swap after rotation)
    rotated_width, rotated_height = img.size
    crop_width = height
    crop_height = width
    left = (rotated_width - crop_width) // 2
    top = (rotated_height - crop_height) // 2
    
    return img.crop((left, top, left + crop_width, top + crop_height))


def create_text_image(
    text: str,
    font_path: Optional[Union[str, Path]] = None,
    font_size: int = 24,
    output_path: Optional[Union[str, Path]] = None,
    width: int = 800,
    height: int = 200,
    bg_color: str = "white",
    text_color: str = "black"
) -> Optional[Path]:
    """Create an image with Mongolian text, rotated 90° clockwise."""
    try:
        font, error = _load_font(font_path, font_size)
        if error:
            print(f"⚠ {error}. Using default font.")
        
        img = _create_rotated_text_image(text, font, width, height, bg_color, text_color)
        
        if output_path is None:
            output_file = NamedTemporaryFile(delete=False, suffix='.png')
            output_path = output_file.name
            output_file.close()
        else:
            output_path = Path(output_path)
        
        img.save(output_path)
        return Path(output_path)
        
    except ImportError:
        print("⚠ PIL/Pillow not installed. Install with: pip install Pillow")
        return None
    except Exception as e:
        print(f"⚠ Error creating image: {e}")
        return None


def get_text_image_base64(
    text: str,
    font_path: Optional[Union[str, Path]] = None,
    font_size: int = 36,
    width: int = 400,
    height: int = 120,
    bg_color: str = "white",
    text_color: str = "black",
) -> Optional[str]:
    """Return base64-encoded PNG of Mongolian text, rotated 90° clockwise (same as display_completions_as_images)."""
    try:
        from io import BytesIO
        font, error = _load_font(font_path, font_size)
        if error:
            return None
        # Use placeholder so empty input still renders a small image
        content = text if text.strip() else " "
        img = _create_rotated_text_image(content, font, width, height, bg_color, text_color)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def _is_wrap_boundary(ch: str) -> bool:
    """Return True when a character is a sensible column wrap point."""
    if not ch:
        return False
    if ch.isspace():
        return True
    category = unicodedata.category(ch)
    return category.startswith("P") or category.startswith("S")


def _measure_text_width(draw, text: str, font) -> float:
    """Measure text width, including trailing spaces when possible."""
    content = text if text else " "
    if hasattr(draw, "textlength"):
        return float(draw.textlength(content, font=font))
    bbox = draw.textbbox((0, 0), content, font=font)
    return float(max(1, bbox[2] - bbox[0]))


def _render_rotated_text_column(text: str, font, text_color: str, padding: int = 8):
    """Render a text column horizontally, then rotate it to top-down layout."""
    from math import ceil
    from PIL import Image, ImageDraw

    content = text if text else " "
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    text_width = max(1, int(ceil(_measure_text_width(probe_draw, content, font))))
    bbox = probe_draw.textbbox((0, 0), content if content.strip() else " ", font=font)
    text_height = max(1, bbox[3] - bbox[1])

    img = Image.new(
        "RGBA",
        (text_width + 2 * padding, text_height + 2 * padding),
        (255, 255, 255, 0),
    )
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding - bbox[1]), content, font=font, fill=text_color)
    return img.rotate(-90, expand=True)


def create_vertical_text_pdf_bytes(
    text: str,
    font_path: Optional[Union[str, Path]] = None,
    font_size: int = 36,
    page_width: int = 1240,
    page_height: int = 1754,
    margin: int = 72,
    column_gap: int = 20,
    bg_color: str = "white",
    text_color: str = "black",
) -> Optional[bytes]:
    """Render top-down Mongolian text into a multi-page PDF and return its bytes."""
    try:
        from io import BytesIO
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    try:
        font, error = _load_font(font_path, font_size)
        if error or font is None:
            return None

        content_height = max(1, page_height - 2 * margin)
        max_source_width = max(1, content_height - 16)
        blank_column_width = max(font_size, column_gap * 2)

        probe = Image.new("RGB", (1, 1), color=bg_color)
        probe_draw = ImageDraw.Draw(probe)

        columns: List[str] = []
        paragraphs = text.split("\n")
        for paragraph_index, paragraph in enumerate(paragraphs):
            remaining = paragraph
            if remaining == "":
                columns.append("")
            while remaining:
                best_end = 1
                last_boundary_end = 0
                for idx in range(1, len(remaining) + 1):
                    candidate = remaining[:idx]
                    width = _measure_text_width(probe_draw, candidate, font)
                    if width <= max_source_width:
                        best_end = idx
                        if _is_wrap_boundary(candidate[-1]):
                            last_boundary_end = idx
                    else:
                        break

                split_at = best_end
                if best_end < len(remaining) and last_boundary_end > 0:
                    split_at = last_boundary_end
                columns.append(remaining[:split_at])
                remaining = remaining[split_at:]

            if paragraph_index != len(paragraphs) - 1 and paragraph != "":
                columns.append("")

        pages = []
        page = Image.new("RGB", (page_width, page_height), color=bg_color)
        x = margin

        def flush_page():
            nonlocal page, x
            pages.append(page)
            page = Image.new("RGB", (page_width, page_height), color=bg_color)
            x = margin

        for column_text in columns:
            if column_text == "":
                needed_width = blank_column_width
                if x + needed_width > page_width - margin:
                    flush_page()
                x += needed_width
                continue

            column_img = _render_rotated_text_column(
                column_text,
                font=font,
                text_color=text_color,
            )
            if x + column_img.width > page_width - margin:
                flush_page()
            page.paste(column_img, (x, margin), column_img)
            x += column_img.width + column_gap

        if not pages or x > margin:
            pages.append(page)

        buffer = BytesIO()
        pages[0].save(
            buffer,
            format="PDF",
            resolution=150.0,
            save_all=True,
            append_images=pages[1:],
        )
        return buffer.getvalue()
    except Exception:
        return None


def display_text_as_image(
    text: str,
    font_path: Optional[Union[str, Path]] = None,
    font_size: int = 48,
    width: int = 600,
    height: int = 150,
    bg_color: str = "white",
    text_color: str = "black",
    display_inline: bool = True
):
    """Create and display Mongolian text as an image in Jupyter notebook."""
    try:
        from IPython.display import Image, display
        
        image_path = create_text_image(
            text, font_path, font_size, None, width, height, bg_color, text_color
        )
        
        if image_path and display_inline:
            display(Image(str(image_path)))
            return None
        return image_path
            
    except ImportError:
        print("⚠ IPython/Pillow not available. Install with: pip install ipython Pillow")
        return None
    except Exception as e:
        print(f"⚠ Error displaying image: {e}")
        return None


def display_completions_as_images(
    completions: Set[str],
    font_path: Optional[Union[str, Path]] = None,
    font_size: int = 36,
    images_per_row: int = 2,
    width: int = 400,
    height: int = 100
):
    """Display multiple completions as images in a grid layout, rotated 90° clockwise."""
    try:
        from IPython.display import HTML, display
        import base64
        from io import BytesIO
        
        font, error = _load_font(font_path, font_size)
        if error:
            print(f"⚠ {error}. Using default font.")
        
        image_htmls = []
        for completion in completions:
            img = _create_rotated_text_image(completion, font, width, height, 'white', 'black')
            
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            image_htmls.append(f'<img src="data:image/png;base64,{img_str}" style="margin: 5px;" />')
        
        # Create grid layout
        html_parts = ['<div style="display: flex; flex-wrap: wrap; justify-content: center;">']
        for i, img_html in enumerate(image_htmls):
            if i > 0 and i % images_per_row == 0:
                html_parts.append('</div><div style="display: flex; flex-wrap: wrap; justify-content: center;">')
            html_parts.append(img_html)
        html_parts.append('</div>')
        
        display(HTML(''.join(html_parts)))
        
    except ImportError:
        print("⚠ Required packages not available. Install with: pip install ipython Pillow")
    except Exception as e:
        print(f"⚠ Error displaying images: {e}")


def display_text_with_font(text: str, font_size: int = 24) -> None:
    """Display text in Jupyter notebook with Mongolian font styling."""
    try:
        from IPython.display import HTML, display
        
        font_path = get_font_path()
        if font_path and font_path.exists():
            html = f"""
            <style>
            @font-face {{
                font-family: 'MongolianFont';
                src: url('{font_path}') format('opentype');
            }}
            .mongolian-text {{
                font-family: 'MongolianFont', sans-serif;
                font-size: {font_size}px;
                direction: ltr;
                unicode-bidi: bidi-override;
            }}
            </style>
            <div class="mongolian-text">{text}</div>
            """
            display(HTML(html))
        else:
            print(text)
            print("⚠ Font file not found. Text displayed with default font.")
            
    except ImportError:
        print(text)
        print("⚠ IPython not available. Install with: pip install ipython")
    except Exception as e:
        print(text)
        print(f"⚠ Error displaying text: {e}")


def setup_matplotlib_font(base_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Set up matplotlib to use the Mongolian font."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        
        font_path = get_font_path(base_path)
        if font_path is None:
            print("⚠ Font file not found. Mongolian text may not display correctly.")
            return None
        
        font_prop = font_manager.FontProperties(fname=str(font_path))
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_prop.get_name()
        
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        
        print(f"✓ Mongolian font loaded: {font_name}")
        print(f"  Font path: {font_path}")
        return str(font_path)
        
    except ImportError:
        print("⚠ matplotlib not installed. Install with: pip install matplotlib")
        return None
    except Exception as e:
        print(f"⚠ Error setting up font: {e}")
        return None


def test_font_rendering(font_path: Optional[Union[str, Path]] = None, font_size: int = 48) -> bool:
    """Test that the font file can be loaded and used for rendering."""
    try:
        from PIL import Image, ImageDraw
        
        if font_path is None:
            font_path = get_font_path()
        
        if font_path is None or not Path(font_path).exists():
            print("✗ Font file not found")
            return False
        
        font, error = _load_font(font_path, font_size)
        if error:
            print(f"✗ {error}")
            return False
        
        # Test rendering
        test_text = "ᠠᠯ"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), test_text, font=font, fill='black')
        
        print(f"✓ Font file is working correctly!")
        print(f"  Font path: {font_path}")
        print(f"  Test text rendered: '{test_text}'")
        return True
            
    except ImportError:
        print("✗ PIL/Pillow not installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
