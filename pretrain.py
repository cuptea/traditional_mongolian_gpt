# -*- coding: utf-8 -*-

#   0 Check and install libraries"""
import logging
import transformers
import torch

from pathlib import Path
from tokenizers import Tokenizer

from src.utils import evaluation, load_text_lines, sample_text, train
from src.data.data_loader import DataLoaderLite
from src.model.gpt import GPT, GPTConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("transformers version: %s", transformers.__version__)
logger.info("torch version: %s", torch.__version__)

# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
logger.info("using device: %s", device)

SAMPLE_PROMPT = "ᠳᠡᠯᠡᠬᠡᠶ ᠶᠢᠨ ᠪᠤᠳ᠋ᠳ᠋ᠾᠠ"
NUM_RETURN_SEQUENCES = 2
MAX_LENGTH = 300
DATA_PATH = Path("./data/data.txt")
TOKENIZER_PATH = Path("./artifacts/traditional_mongolian_bpe/tokenizer.json")
DATA_FRACTION = 0.00002


def build_sample_tokens(prompt: str, tokenizer: Tokenizer, num_return_sequences: int):
    token_ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor(token_ids, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    return tokens


def run_validation(model, valid_text, tokenizer, device, label):
    valid_loader = DataLoaderLite(B=8, T=64, text=valid_text, tokenizer=tokenizer)
    loss = evaluation(model, valid_loader, device)
    logger.info("validation loss %s: %.6f", label, loss)
    return loss

"""#1 Prepare data"""
"""## 1.1 Load dataset from google drive"""
logger.info("loading dataset from %s", DATA_PATH)
logger.info("loading tokenizer from %s", TOKENIZER_PATH)
logger.info("using %.2f%% of dataset for this run", DATA_FRACTION * 100)
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
lines = load_text_lines(DATA_PATH, DATA_FRACTION)

"""## 1.2 Create train and valid datasets"""
train_data_percentage = 0.99
train_line_count = int(len(lines) * train_data_percentage)
train_lines = lines[:train_line_count]
valid_lines = lines[train_line_count:]
train_text = "\n".join(train_lines).strip()
valid_text = "\n".join(valid_lines).strip()
train_chars = len(train_text)
valid_chars = len(valid_text)
logger.info(
    "training data: %s lines, %s characters",
    len(train_lines),
    train_chars,
)
logger.info("training data sample:\n%s", train_text[:300])
logger.info(
    "validation data: %s lines, %s characters",
    len(valid_lines),
    valid_chars,
)
logger.info("validation data sample:\n%s", valid_text[:300])



"""# 2 Finetuning a Model on traditional Mongolian text data"""
"""## 2.1 Define the model (GPT2)"""
"""## 2.2 Define data loader"""

"""## 2.3 Create model with the custom tokenizer vocabulary"""
model = GPT(GPTConfig(vocab_size=tokenizer.get_vocab_size()))
model.to(device)

"""## 2.4 Evaluation on validation data before fine tuning"""
loss_main_before = run_validation(model, valid_text, tokenizer, device, "before fine tuning")

"""##2.5 Sample text generation before fine tuning"""
sample_tokens = build_sample_tokens(SAMPLE_PROMPT, tokenizer, NUM_RETURN_SEQUENCES)
sample_text(model, NUM_RETURN_SEQUENCES, MAX_LENGTH, tokenizer, sample_tokens, device)

"""## 2.7 Model fine tuning"""
train_loader = DataLoaderLite(B=8, T=64, text=train_text, tokenizer=tokenizer)
model, epoch_loss_main = train(train_loader, model, 5, device)

"""## 2.8 Evaluation on validation data after fine tuning"""
loss_main_after = run_validation(model, valid_text, tokenizer, device, "after fine tuning")

"""##2.9 Sample text generation after fine tuning"""
sample_text(model, NUM_RETURN_SEQUENCES, MAX_LENGTH, tokenizer, sample_tokens, device)

