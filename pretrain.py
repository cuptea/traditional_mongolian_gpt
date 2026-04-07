# -*- coding: utf-8 -*-
import os

# Tokenizers otherwise may use multiprocessing; on macOS that can interact badly with
# abrupt exits (e.g. OOM kill) and trigger resource_tracker semaphore warnings at shutdown.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

#   0 Check and install libraries"""
import logging
import transformers
import torch

from pathlib import Path
from tokenizers import Tokenizer

from src.utils import (
    evaluation,
    load_model_state_from_training_checkpoint,
    sample_text,
    train,
)
from src.data.data_loader import DataLoaderLite, build_split_token_caches_from_file
from src.model.gpt import GPT, GPTConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("[1/9] Environment — transformers %s, torch %s", transformers.__version__, torch.__version__)

# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
logger.info("[1/9] Environment — using device: %s", device)

SAMPLE_PROMPT = "ᠳᠡᠯᠡᠬᠡᠶ ᠶᠢᠨ ᠪᠤᠳ᠋ᠳ᠋ᠾᠠ"
NUM_RETURN_SEQUENCES = 1
MAX_LENGTH = 10
DATA_PATH = Path("./data/data.txt")
TOKENIZER_PATH = Path("./artifacts/traditional_mongolian_bpe/tokenizer.json")
DATA_FRACTION = 1
TRAIN_DATA_PERCENTAGE = 0.9999
BATCH_SIZE = 64
NUM_EPOCHS = 5
CHECKPOINT_PATH = Path("./artifacts/checkpoints/pretrain_latest.pt")
TOKEN_CACHE_DIR = Path("./artifacts/token_cache")


def build_sample_tokens(prompt: str, tokenizer: Tokenizer, num_return_sequences: int):
    token_ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor(token_ids, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    return tokens


def peek_utf8_sample(path: Path, max_chars: int = 400) -> str:
    """Small disk read for logs — does not load the corpus."""
    with path.open("r", encoding="utf-8") as f:
        return f.read(max_chars)


def run_validation(model, valid_loader, device, label, step: str) -> float:
    logger.info("%s — running validation (%s)", step, label)
    loss = evaluation(model, valid_loader, device)
    logger.info("%s — validation loss (%s): %.6f", step, label, loss)
    return loss


"""#1 Prepare data"""
"""## 1.1 Load dataset from google drive"""
logger.info(
    "[2/9] Tokenizer + data path — tokenizer=%s, data=%s, fraction=%.4f%%",
    TOKENIZER_PATH,
    DATA_PATH,
    DATA_FRACTION * 100,
)
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
logger.info("[2/9] Tokenizer ready — corpus stays on disk until tokenization")

"""## 1.2 Build train / valid token caches from disk (low RAM)"""
logger.info("[3/9] Streaming train/valid token caches from %s", DATA_PATH)
logger.info("[3/9] Data file sample (first ~300 chars):\n%s", peek_utf8_sample(DATA_PATH, 300))
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
split_stats = build_split_token_caches_from_file(
    DATA_PATH,
    tokenizer,
    TOKEN_CACHE_DIR / "train_tokens.bin",
    TOKEN_CACHE_DIR / "valid_tokens.bin",
    data_fraction=DATA_FRACTION,
    train_data_percentage=TRAIN_DATA_PERCENTAGE,
    reuse_token_cache=True,
)
if split_stats is not None:
    n_kept, train_lines_n, valid_lines_n = split_stats
    logger.info(
        "[3/9] Stream done — kept %s lines → train %s, valid %s",
        n_kept,
        train_lines_n,
        valid_lines_n,
    )
else:
    logger.info("[3/9] Using existing token caches (train + valid .bin)")

"""# 2 Finetuning a Model on traditional Mongolian text data"""
"""## 2.1 Define the model (GPT2)"""
"""## 2.2 Define data loader"""

"""## 2.3 Create model with the custom tokenizer vocabulary"""
logger.info("[4/9] Building model (vocab_size=%s)", tokenizer.get_vocab_size())
model = GPT(GPTConfig(vocab_size=tokenizer.get_vocab_size()))
model.to(device)
logger.info("[4/9] Model on device: %s", device)
if CHECKPOINT_PATH.is_file():
    logger.info("[4/9] Checkpoint found at %s — loading model weights for pre-train validation", CHECKPOINT_PATH)
    load_model_state_from_training_checkpoint(CHECKPOINT_PATH, model, device)
    logger.info("[4/9] Optimizer + full training state will load again at the start of train()")
else:
    logger.info("[4/9] No checkpoint at %s — training from random init", CHECKPOINT_PATH)

train_loader = DataLoaderLite(
    B=BATCH_SIZE,
    T=64,
    token_cache_path=TOKEN_CACHE_DIR / "train_tokens.bin",
)
valid_loader = DataLoaderLite(
    B=BATCH_SIZE,
    T=64,
    token_cache_path=TOKEN_CACHE_DIR / "valid_tokens.bin",
)
sample_tokens = build_sample_tokens(SAMPLE_PROMPT, tokenizer, NUM_RETURN_SEQUENCES)

"""## 2.4 Evaluation on validation data before fine tuning"""
loss_main_before = run_validation(
    model, valid_loader, device, "before fine tuning", step="[5/9]"
)

"""## 2.7 Model fine tuning"""
logger.info(
    "[7/9] Fine-tuning — batch_size=%s, seq_len=64, epochs=%s",
    BATCH_SIZE,
    NUM_EPOCHS,
)
model, epoch_loss_main = train(
    train_loader,
    valid_loader,
    tokenizer,
    sample_tokens,
    model,
    NUM_EPOCHS,
    device,
    step="[7/9]",
    num_return_sequences=NUM_RETURN_SEQUENCES,
    max_length=MAX_LENGTH,
    checkpoint_path=CHECKPOINT_PATH,
)

logger.info(
    "[7/9] Fine-tuning finished — mean train loss this run (last %s epochs): %s",
    NUM_EPOCHS,
    epoch_loss_main[-NUM_EPOCHS:] if len(epoch_loss_main) >= NUM_EPOCHS else epoch_loss_main,
)
logger.info("[7/9] Full train-loss history (all resumed runs): %s", epoch_loss_main)
logger.info("[7/9] Latest checkpoint: %s", CHECKPOINT_PATH)



