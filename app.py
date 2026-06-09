"""Hugging Face Gradio Spaces entry point."""
from web.server import demo

if __name__ == "__main__":
    demo.queue().launch()
