import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_train_losses: List[float],
    last_val_loss: Optional[float],
    epochs_completed: int,
    vocab_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch_train_losses": list(epoch_train_losses),
        "last_val_loss": last_val_loss,
        "epochs_completed": epochs_completed,
        "vocab_size": vocab_size,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info("saved training checkpoint to %s (epochs_completed=%s)", path, epochs_completed)


def load_model_state_from_training_checkpoint(path: Path, model: nn.Module, device: str) -> None:
    """Load only ``model`` weights (no optimizer). For pre-train validation when resuming."""
    try:
        bundle = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location=device)
    saved_vocab = bundle.get("vocab_size")
    if hasattr(model, "config") and saved_vocab is not None:
        if getattr(model.config, "vocab_size", None) != saved_vocab:
            raise ValueError(
                f"checkpoint vocab_size {saved_vocab} does not match model {model.config.vocab_size}"
            )
    model.load_state_dict(bundle["model"])
    logger.info("loaded model weights from %s (optimizer not restored here)", path)


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[List[float], Optional[float], int]:
    try:
        bundle = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location=device)
    if bundle.get("version") != _CHECKPOINT_VERSION:
        logger.warning(
            "checkpoint version mismatch (file=%s, expected=%s); loading anyway",
            bundle.get("version"),
            _CHECKPOINT_VERSION,
        )
    saved_vocab = bundle.get("vocab_size")
    if hasattr(model, "config") and saved_vocab is not None:
        if getattr(model.config, "vocab_size", None) != saved_vocab:
            raise ValueError(
                f"checkpoint vocab_size {saved_vocab} does not match model {model.config.vocab_size}"
            )
    model.load_state_dict(bundle["model"])
    optimizer.load_state_dict(bundle["optimizer"])
    epoch_loss = list(bundle.get("epoch_train_losses", []))
    prev_val = bundle.get("last_val_loss")
    start_epochs = int(bundle.get("epochs_completed", len(epoch_loss)))
    logger.info(
        "loaded training checkpoint from %s (epochs_completed=%s, history_len=%s)",
        path,
        start_epochs,
        len(epoch_loss),
    )
    return epoch_loss, prev_val, start_epochs


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
    if torch.cuda.is_available():
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



def train(
    train_loader,
    valid_loader,
    tokenizer,
    sample_tokens,
    model: nn.Module,
    epoch_number: int,
    device: str,
    step: str,
    num_return_sequences: int = 1,
    max_length: int = 20,
    checkpoint_path: Optional[Path] = None,
):
    # fine tune the model with BMW data
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lr = optimizer.param_groups[0]["lr"]

    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is None:
        raise AttributeError("model.config.vocab_size is required for checkpointing")

    epoch_loss: List[float] = []
    prev_val_loss: Optional[float] = None
    start_epochs = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        epoch_loss, prev_val_loss, start_epochs = load_training_checkpoint(
            checkpoint_path, model, optimizer, device
        )

    logger.info(
        "%s start training | run_epochs=%s | global_epoch_start=%s | "
        "train_batches/epoch=%s | val_batches/epoch=%s | AdamW lr=%s | sample: n=%s max_len=%s | "
        "checkpoint=%s",
        step,
        epoch_number,
        start_epochs,
        train_loader.batch_per_epoch,
        valid_loader.batch_per_epoch,
        lr,
        num_return_sequences,
        max_length,
        checkpoint_path if checkpoint_path is not None else None,
    )

    for epoch in tqdm(range(epoch_number), desc="epochs"):
        ep = epoch + 1
        global_ep = start_epochs + ep
        epoch_t0 = time.perf_counter()

        logger.info(
            "%s epoch run %s/%s (global epoch %s) — train phase (%s batches)",
            step,
            ep,
            epoch_number,
            global_ep,
            train_loader.batch_per_epoch,
        )

        batch_loss = []
        train_t0 = time.perf_counter()
        for _ in tqdm(
            range(train_loader.batch_per_epoch),
            desc="train batches",
            leave=False,
        ):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        train_secs = time.perf_counter() - train_t0
        mean_train_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_train_loss)

        model.eval()
        logger.info(
            "%s epoch run %s/%s (global %s) — validation (%s batches)",
            step,
            ep,
            epoch_number,
            global_ep,
            valid_loader.batch_per_epoch,
        )
        val_t0 = time.perf_counter()
        eval_loss = evaluation(model, valid_loader, device)
        val_secs = time.perf_counter() - val_t0
        logger.info(
            "%s epoch run %s/%s (global %s) — validation mean loss: %.6f (%.1fs)",
            step,
            ep,
            epoch_number,
            global_ep,
            eval_loss,
            val_secs,
        )

        logger.info(
            "%s epoch run %s/%s (global %s) — text generation (%s sequences)",
            step,
            ep,
            epoch_number,
            global_ep,
            num_return_sequences,
        )
        gen_t0 = time.perf_counter()
        sample_text(
            model,
            num_return_sequences,
            max_length,
            tokenizer,
            sample_tokens,
            device,
        )
        gen_secs = time.perf_counter() - gen_t0

        model.train()

        delta = ""
        if prev_val_loss is not None:
            delta = f" | val_change={eval_loss - prev_val_loss:+.6f}"
        prev_val_loss = eval_loss
        total_secs = time.perf_counter() - epoch_t0

        logger.info(
            "%s epoch run %s/%s (global %s) — summary | train_loss=%.6f | val_loss=%.6f%s | "
            "time: train %.1fs, val %.1fs, sample %.1fs, total %.1fs",
            step,
            ep,
            epoch_number,
            global_ep,
            mean_train_loss,
            eval_loss,
            delta,
            train_secs,
            val_secs,
            gen_secs,
            total_secs,
        )

    epochs_completed = start_epochs + epoch_number
    logger.info(
        "%s training finished | epochs_completed=%s | per-epoch train_loss (full history): %s",
        step,
        epochs_completed,
        epoch_loss,
    )
    if checkpoint_path is not None:
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch_loss,
            prev_val_loss,
            epochs_completed,
            vocab_size,
        )
    return model, epoch_loss
