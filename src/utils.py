import logging
import re
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and re.fullmatch(r"[=_\- ]*", stripped) is not None


def load_text_lines(path: Path, data_fraction: float) -> List[str]:
    """Load approximately ``data_fraction`` of one text file as cleaned lines."""
    if not 0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be in the range (0, 1].")

    lines: List[str] = []
    total_chars = 0
    bytes_read = 0

    file_size = path.stat().st_size
    read_limit = file_size if data_fraction >= 1.0 else max(1, int(file_size * data_fraction))
    logger.info(
        "Loading text from %s using %.2f%% of the file (up to ~%s bytes)",
        path.name,
        data_fraction * 100,
        f"{read_limit:,}",
    )

    with path.open("r", encoding="utf-8") as f:
        for raw_line in tqdm(f, desc=f"Loading {path.name}", unit="lines"):
            bytes_read += len(raw_line.encode("utf-8"))
            if bytes_read > read_limit:
                break

            line = raw_line.rstrip("\n")
            if _is_separator_line(line):
                continue

            lines.append(line)
            total_chars += len(line) + 1

    logger.info(
        "Loaded %s lines from %s (~%s characters, %.2f%% of file)",
        len(lines),
        path.name,
        f"{total_chars:,}",
        min(bytes_read, read_limit) / file_size * 100,
    )
    return lines

    


def evaluation(model: nn.Module, valid_loader, device: str) -> float:
    # fine tune the model with BMW data
    model.eval()
    batch_loss = []

    epoch_number = 1
    for _ in tqdm(range(valid_loader.batch_per_epoch * epoch_number)):
        x, y = valid_loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        batch_loss.append(loss.item())

    return sum(batch_loss) / len(batch_loss)


def sample_text(
    model: nn.Module,
    num_return_sequences: int,
    max_length: int,
    tokenizer,
    tokens: torch.Tensor,
    device: str,
) -> None:
    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    x = tokens.to(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk = min(50, probs.size(-1))
            topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        logger.info("> %s", decoded)



def train(train_loader, model: nn.Module, epoch_number: int, device: str):
    # fine tune the model with BMW data
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    epoch_loss = []
    for _ in tqdm(range(epoch_number)):
        batch_loss = []
        for _ in range(train_loader.batch_per_epoch):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model, epoch_loss
