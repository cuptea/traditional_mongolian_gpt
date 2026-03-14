"""
Mongolian ML Autocomplete implementation.
"""
import json
import logging
import random
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
import torch
import torch.nn.functional as F
from . import font_utils


class MongolMLAutocomplete:
    """
    Mongolian ML Autocomplete class for generating word completions using a PyTorch model.
    """
    
    def __init__(
        self,
        path_custom_model: str = "assets/machine_learning/zmodel.pt",
        path_mappings: str = "assets/machine_learning/new_char_to_token.json",
        block_size: int = 40,
        number_of_sample_words: int = 10,
        max_length_of_word: int = 20,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the MongolMLAutocomplete class.
        
        Args:
            path_custom_model: Path to the PyTorch model file
            path_mappings: Path to the character to token mapping JSON file
            block_size: Maximum context to look back when running model inference
            number_of_sample_words: Max number of attempts for generating words
            max_length_of_word: Max length of a word
            verbose: Whether to emit detailed generation logs
            logger: Optional logger instance for backend/server integrations
        """
        self.path_custom_model = path_custom_model
        self.path_mappings = path_mappings
        self.requested_block_size = block_size
        self.block_size = block_size
        self.number_of_sample_words = number_of_sample_words
        self.max_length_of_word = max_length_of_word
        self.verbose = verbose
        
        self._custom_model: Optional[torch.nn.Module] = None
        self.char_to_token_mapping: Dict[str, int] = {}
        self.token_to_char_mapping: Dict[int, str] = {}
        self.rng = random.Random()
        self.logger = logger or logging.getLogger(__name__)

    def _log_info(self, message: str, *args) -> None:
        self.logger.info(message, *args)

    def _log_debug(self, message: str, *args) -> None:
        if self.verbose:
            self.logger.debug(message, *args)
    
    def initialize(self):
        """
        Initialize the model and mappings.
        """
        self._log_info("Loading autocomplete model")
        self.load_model()
        
        self._log_info("Loading character mappings")
        self.load_mappings()
        self._log_info(
            "Autocomplete ready: vocab_size=%d block_size=%d samples=%d max_length=%d",
            len(self.char_to_token_mapping),
            self.block_size,
            self.number_of_sample_words,
            self.max_length_of_word,
        )
    
    def load_model(self):
        """
        Load the PyTorch model.
        """
        try:
            self._custom_model = torch.jit.load(self.path_custom_model)
            self._custom_model.eval()
            model_block_size = self._infer_model_block_size()
            if model_block_size is not None and self.block_size > model_block_size:
                self.logger.warning(
                    "Configured block_size=%d exceeds model limit=%d; using model limit",
                    self.block_size,
                    model_block_size,
                )
                self.block_size = model_block_size
            self._log_debug(f"Loaded model from {self.path_custom_model}")
        except Exception as e:
            self.logger.exception("Failed to load model from %s", self.path_custom_model)
            raise

    def _infer_model_block_size(self) -> Optional[int]:
        """Infer the maximum context length baked into the TorchScript model."""
        if self._custom_model is None:
            return None

        try:
            model_code = str(self._custom_model.code)
        except Exception:
            return None

        match = re.search(r"if torch\.le\(t, (\d+)\):", model_code)
        if not match:
            return None

        try:
            return int(match.group(1))
        except ValueError:
            return None
    
    def load_mappings(self):
        """
        Load character to token mappings from JSON file.
        """
        try:
            with open(self.path_mappings, 'r', encoding='utf-8') as f:
                self.char_to_token_mapping = json.load(f)
            
            self.token_to_char_mapping = {
                v: k for k, v in self.char_to_token_mapping.items()
            }
            self._log_debug(
                "Loaded %d mapping entries from %s",
                len(self.char_to_token_mapping),
                self.path_mappings,
            )
        except FileNotFoundError:
            self.logger.exception("Mapping file not found: %s", self.path_mappings)
            raise
        except json.JSONDecodeError as e:
            self.logger.exception("Error parsing mapping JSON: %s", self.path_mappings)
            raise
    
    def tokens_to_text(self, tokens: List[int]) -> str:
        """
        Convert a sequence of token IDs to Mongolian text using the loaded mapping.
        
        Args:
            tokens: List of token IDs produced by the model.
        
        Returns:
            Decoded Mongolian string.
        """
        if not self.token_to_char_mapping:
            raise ValueError("Token mappings not loaded. Call initialize() or load_mappings() first.")
        
        return "".join(self.token_to_char_mapping.get(t, "") for t in tokens)

    def get_boundary_token_ids(self) -> Set[int]:
        """
        Return token ids that should terminate a completion.

        Boundary tokens are defined explicitly from the mapping pairs requested
        by the application:
        "\n": 0, " ": 1, "\u1801": 2, "\u1802": 3, "\u1803": 4,
        "\u1804": 5, "\u1805": 6, "\u1808": 7, "\u1809": 8,
        "\u1850": 9, "\u1851": 10, "\u1852": 11, "\u1853": 12,
        "\u1858": 13, "\u185b": 14
        """
        return set(range(15))

    def decode_generated_tokens(self, tokens: List[int]) -> str:
        """
        Decode generated tokens, including the terminating boundary token if one
        was generated before stopping. If batched generation padded a completed
        sequence by repeating its last token, trim at the first boundary token.
        """
        boundary_tokens = self.get_boundary_token_ids()
        trimmed_tokens: List[int] = []
        for token in tokens:
            trimmed_tokens.append(token)
            if token in boundary_tokens:
                break

        return "".join(
            self.token_to_char_mapping.get(token, "")
            for token in trimmed_tokens
        )
    
    def display_tokens_as_image(
        self,
        tokens: List[int],
        font_path: Optional[str] = None,
        font_size: int = 48,
        width: int = 400,
        height: int = 120,
        bg_color: str = "#e3f2fd",
        text_color: str = "#1976d2",
    ):
        """
        Render a sequence of token IDs as Mongolian text using the project font.
        
        This mirrors the logic used in the test notebook: tokens are decoded
        using the character mapping, then rendered with the Mongolian font as
        a rotated image in Jupyter (via font_utils.display_text_as_image).
        
        Args:
            tokens: List of token IDs to decode and render.
            font_path: Optional path to the .otf font file. If not provided,
                the default project font (z52chimegtig.otf) is used.
            font_size: Font size in pixels.
            width: Image width before rotation.
            height: Image height before rotation.
            bg_color: Background color.
            text_color: Text color.
        """
        text = self.tokens_to_text(tokens)
        if not text:
            self._log_info("No text decoded from tokens; nothing to render")
            return None
        
        if font_path is None:
            font_path = font_utils.get_font_path()
        
        return font_utils.display_text_as_image(
            text,
            font_path=font_path,
            font_size=font_size,
            width=width,
            height=height,
            bg_color=bg_color,
            text_color=text_color,
        )
    
    def _softmax(self, x: List[float]) -> List[float]:
        """Convert logits to probability distribution."""
        x_exp = [math.exp(xi) for xi in x]
        sum_exp = sum(x_exp)
        return [xi / sum_exp for xi in x_exp]
    
    def _sample(
        self,
        token_context: List[int],
        word_max_length: int,
        sample_number: int
    ) -> Set[str]:
        """
        Generate multiple words given tokenized context using top-k sampling.
        
        The token_context is a list of numbers, each of which maps to certain
        Mongolian basic character, e.g. 'ᠠ','ᠯ'. The mapping is defined in the
        char_to_token_mapping.
        
        Args:
            token_context: List of token IDs representing the context
            word_max_length: Maximum length of a word to generate
            sample_number: Number of words to sample
            
        Returns:
            Set of generated words
        """
        prediction: Set[str] = set()
        
        if len(token_context) == 0:
            return prediction
        
        # Truncate context to block_size if needed
        x = token_context.copy()
        if len(x) > self.block_size:
            x = x[-self.block_size:]
        
        # Clamp tokens to valid vocabulary range
        vocab_size = len(self.char_to_token_mapping)
        x = [max(0, min(token, vocab_size - 1)) for token in x]
        context_chars = [self.token_to_char_mapping.get(token, f'<{token}>') for token in x]
        context_str = ''.join(context_chars)
        self._log_debug(
            "Starting generation: context=%r context_length=%d block_size=%d sample_count=%d",
            context_str,
            len(x),
            self.block_size,
            sample_number,
        )
        
        boundary_tokens = self.get_boundary_token_ids()

        # Prepare initial tensor: (B, T) where B = sample_number, T = len(x)
        # Detect device from model if possible, otherwise use CPU
        device = 'cpu'
        try:
            if hasattr(self._custom_model, 'parameters'):
                device = next(self._custom_model.parameters()).device
        except:
            pass
        x_tensor = torch.tensor([x] * sample_number, dtype=torch.long, device=device)
        
        # Track which sequences are complete
        is_complete = [False] * sample_number
        
        # Generate until max_length or all sequences complete
        max_total_length = len(x) + word_max_length
        while x_tensor.size(1) < max_total_length and not all(is_complete):
            # Forward the model: it only accepts up to block_size tokens, so use a sliding window
            model_input = x_tensor[:, -self.block_size:]  # (B, min(T, block_size))
            with torch.no_grad():
                model_output = self._custom_model(model_input)

                # Handle different model output formats
                if isinstance(model_output, tuple):
                    logits, _ = model_output  # (B, T, vocab_size)
                else:
                    logits = model_output
                
                # Ensure logits is 3D: (B, T, vocab_size)
                if len(logits.shape) == 2:
                    # If (B, vocab_size), add sequence dimension
                    logits = logits.unsqueeze(1)  # (B, 1, vocab_size)
                elif len(logits.shape) == 1:
                    # If flattened, reshape
                    total_elements = logits.numel()
                    batch_size = x_tensor.size(0)
                    seq_len = x_tensor.size(1)
                    expected = batch_size * seq_len * vocab_size
                    if total_elements == expected:
                        logits = logits.reshape(batch_size, seq_len, vocab_size)
                    else:
                        # Fallback: assume it's for last position only
                        logits = logits.reshape(batch_size, 1, vocab_size)

                # Take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                logits = logits.clone()

                # Get the probabilities
                probs = F.softmax(logits, dim=-1)

                # Do top-k sampling of 50 (huggingface pipeline default)
                topk_probs, topk_indices = torch.topk(probs, min(50, vocab_size), dim=-1)

                # Select a token from the top-k probabilities for each sequence
                # Only sample for incomplete sequences
                xcol_list = []
                for k in range(sample_number):
                    if is_complete[k]:
                        # Keep the last token - ensure it's 2D (1, 1) to match xcol shape
                        last_token = x_tensor[k, -1].unsqueeze(0).unsqueeze(0)  # scalar -> (1, 1)
                        xcol_list.append(last_token)
                    else:
                        # Sample from top-k
                        ix = torch.multinomial(topk_probs[k:k+1], 1)  # (1, 1)
                        xcol = torch.gather(topk_indices[k:k+1], -1, ix)  # (1, 1)
                        xcol_list.append(xcol)
                        
                        token_id = xcol.item()
                        # Stop this sequence when whitespace boundary is generated.
                        # Punctuation/symbol tokens remain valid completion output.
                        if token_id in boundary_tokens:
                            is_complete[k] = True

                # Append to the sequence
                # All tensors in xcol_list should be (1, 1), so concatenating on dim=0 gives (B, 1)
                xcol_tensor = torch.cat(xcol_list, dim=0)  # (B, 1)
                x_tensor = torch.cat((x_tensor, xcol_tensor), dim=1)
        
        # Decode and collect predictions, keeping the terminating boundary token
        # so downstream clients can distinguish the actual stop character.
        for i in range(sample_number):
            # Get tokens excluding the original context
            tokens = x_tensor[i, len(x):].tolist()
            word_str = self.decode_generated_tokens(tokens)
            if word_str:
                prediction.add(word_str)

        self._log_debug(
            "Generation finished: context=%r completions=%d",
            context_str,
            len(prediction),
        )
        
        return prediction
    
    def run_custom_model(self, input_text: str) -> Set[str]:
        """
        Generate number of auto completed words given the context.
        
        Args:
            input_text: Input text string to generate completions for
            
        Returns:
            Set of auto-completed words
        """
        vocab_size = len(self.char_to_token_mapping)
        tokenized_context = [
            max(0, min(self.char_to_token_mapping.get(ch, 0), vocab_size - 1)) 
            for ch in input_text
        ]
        return self._sample(
            tokenized_context, self.max_length_of_word, self.number_of_sample_words
        )
