import logging
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.utils import _is_separator_line


logger = logging.getLogger(__name__)

# Avoid a second full-file UTF-8 allocation when logging huge in-memory corpora.
_UTF8_LOG_CHAR_THRESHOLD = 8_000_000

# Little-endian uint32 raw stream on disk (one id per element).
_TOKEN_FILE_DTYPE = np.dtype("<u4")


def _iter_kept_raw_lines(path: Path, data_fraction: float) -> Iterator[str]:
    """Yield raw lines from disk (including trailing ``\\n``) matching ``load_text_lines`` filtering."""
    if not 0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be in the range (0, 1].")
    file_size = path.stat().st_size
    read_limit = file_size if data_fraction >= 1.0 else max(1, int(file_size * data_fraction))
    bytes_read = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            bytes_read += len(raw_line.encode("utf-8"))
            if bytes_read > read_limit:
                break
            line = raw_line.rstrip("\n")
            if _is_separator_line(line):
                continue
            yield raw_line


def _normalize_shard_line(raw_line: str, *, is_first: bool, is_last: bool) -> str:
    """Match ``\"\\n\".join(lines).strip()`` within one train or valid shard."""
    s = raw_line
    if is_first:
        s = s.lstrip()
    if is_last:
        s = s.rstrip()
    return s


def _append_tokens_for_text(f, tokenizer, text: str) -> int:
    if not text:
        return 0
    token_ids = tokenizer.encode(text).ids
    if not token_ids:
        return 0
    arr = np.asarray(token_ids, dtype=_TOKEN_FILE_DTYPE)
    f.write(arr.tobytes())
    return len(token_ids)


def _finalize_token_cache_file(tmp_path: Path, final_path: Path) -> None:
    if final_path.is_file():
        final_path.unlink()
    tmp_path.replace(final_path)


def build_split_token_caches_from_file(
    data_path: Path,
    tokenizer,
    train_cache_path: Path,
    valid_cache_path: Path,
    *,
    data_fraction: float,
    train_data_percentage: float,
    reuse_token_cache: bool = True,
) -> Optional[Tuple[int, int, int]]:
    """
    Two-pass stream from ``data_path`` into train/valid token files — O(1) RAM vs corpus size.

    Mirrors ``load_text_lines`` + ``\"\\n\".join(...).strip()`` per split, line-wise tokenization.

    Returns ``(kept_line_count, train_line_count, n_valid_lines)``, or ``None`` if caches were reused.
    """
    train_cache_path = Path(train_cache_path).resolve()
    valid_cache_path = Path(valid_cache_path).resolve()
    train_cache_path.parent.mkdir(parents=True, exist_ok=True)

    train_ok = train_cache_path.is_file() and train_cache_path.stat().st_size > 0
    valid_ok = valid_cache_path.is_file() and valid_cache_path.stat().st_size > 0
    if reuse_token_cache and train_ok and valid_ok:
        logger.info(
            "token cache pair present — skip disk stream (%s, %s)",
            train_cache_path,
            valid_cache_path,
        )
        return None

    logger.info(
        "streaming token caches from %s (fraction=%.4f%%, train_split=%.4f%%)",
        data_path,
        data_fraction * 100,
        train_data_percentage * 100,
    )

    n_kept = 0
    for _ in tqdm(
        _iter_kept_raw_lines(data_path, data_fraction),
        desc=f"Counting lines {data_path.name}",
        unit="lines",
    ):
        n_kept += 1

    train_line_count = int(n_kept * train_data_percentage)
    n_valid = n_kept - train_line_count
    logger.info(
        "kept lines=%s → train_lines=%s, valid_lines=%s",
        n_kept,
        train_line_count,
        n_valid,
    )

    train_tmp = train_cache_path.with_suffix(train_cache_path.suffix + ".tmp")
    valid_tmp = valid_cache_path.with_suffix(valid_cache_path.suffix + ".tmp")
    try:
        ti = vi = 0
        with open(train_tmp, "wb") as f_train, open(valid_tmp, "wb") as f_valid:
            for raw_line in tqdm(
                _iter_kept_raw_lines(data_path, data_fraction),
                desc=f"Tokenizing to disk {data_path.name}",
                unit="lines",
            ):
                if ti < train_line_count:
                    norm = _normalize_shard_line(
                        raw_line,
                        is_first=(ti == 0),
                        is_last=(ti == train_line_count - 1),
                    )
                    _append_tokens_for_text(f_train, tokenizer, norm)
                    ti += 1
                else:
                    norm = _normalize_shard_line(
                        raw_line,
                        is_first=(vi == 0),
                        is_last=(vi == n_valid - 1),
                    )
                    _append_tokens_for_text(f_valid, tokenizer, norm)
                    vi += 1
        _finalize_token_cache_file(train_tmp, train_cache_path)
        _finalize_token_cache_file(valid_tmp, valid_cache_path)
    except BaseException:
        train_tmp.unlink(missing_ok=True)
        valid_tmp.unlink(missing_ok=True)
        raise

    logger.info("wrote train cache %s, valid cache %s", train_cache_path, valid_cache_path)
    return (n_kept, train_line_count, n_valid)


def _open_token_mmap(path: Path) -> np.ndarray:
    nbytes = path.stat().st_size
    itemsize = _TOKEN_FILE_DTYPE.itemsize
    if nbytes == 0:
        # numpy.memmap rejects length 0; empty corpora still need a readable backing array
        return np.array([], dtype=_TOKEN_FILE_DTYPE)
    if nbytes % itemsize != 0:
        raise ValueError(
            f"token cache {path} has size {nbytes} bytes, not a multiple of {itemsize}"
        )
    return np.memmap(path, dtype=_TOKEN_FILE_DTYPE, mode="r")


def _stream_tokens_to_file(tokenizer, text: str, path: Path) -> int:
    """Write token ids to `path` without holding the full id list in memory (multi-line text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    parts = text.splitlines(keepends=True)
    n_written = 0
    try:
        with open(tmp_path, "wb") as f:
            if not parts:
                pass
            elif len(parts) == 1:
                logger.warning(
                    "corpus is a single line; tokenizing loads all ids into memory once "
                    "(use newline-separated text for fully streaming tokenization)"
                )
                with tqdm(
                    total=1, desc="Tokenizing corpus", unit="pass", dynamic_ncols=True
                ) as pbar:
                    n_written = _append_tokens_for_text(f, tokenizer, text)
                    pbar.update(1)
            else:
                for part in tqdm(
                    parts,
                    desc="Tokenizing corpus",
                    unit="lines",
                    dynamic_ncols=True,
                ):
                    n_written += _append_tokens_for_text(f, tokenizer, part)
        if path.is_file():
            path.unlink()
        tmp_path.replace(path)
    except BaseException:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise
    return n_written


class DataLoaderLite:
    def __init__(
        self,
        B,
        T,
        text: Optional[str] = None,
        tokenizer=None,
        token_cache_path=None,
        reuse_token_cache=True,
    ):
        self.B = B
        self.T = T
        self._mmap = None
        self.tokens = None

        cache_path = Path(token_cache_path).resolve() if token_cache_path else None

        if text is not None:
            if len(text) <= _UTF8_LOG_CHAR_THRESHOLD:
                utf8_nbytes = len(text.encode("utf-8"))
                logger.info(
                    "text corpus: %s characters, %s UTF-8 bytes, %s lines",
                    len(text),
                    utf8_nbytes,
                    text.count("\n") + (1 if text else 0),
                )
            else:
                logger.info(
                    "text corpus: %s characters, ~lines=%s (UTF-8 byte size omitted to save RAM)",
                    len(text),
                    text.count("\n") + (1 if text else 0),
                )
        elif cache_path is None:
            raise ValueError("DataLoaderLite: text is required when token_cache_path is None")

        if cache_path is not None:
            cache_nonempty = (
                cache_path.is_file() and cache_path.stat().st_size > 0
            )
            if reuse_token_cache and cache_nonempty:
                logger.info("token cache hit — mmap %s", cache_path)
                self._mmap = _open_token_mmap(cache_path)
            else:
                if text is None or tokenizer is None:
                    raise ValueError(
                        "DataLoaderLite: text and tokenizer are required to build token_cache_path "
                        f"{cache_path} (cache missing or reuse_token_cache=False)"
                    )
                if cache_path.is_file() and not cache_nonempty:
                    logger.info(
                        "token cache at %s is empty — rebuilding (stale or interrupted write)",
                        cache_path,
                    )
                logger.info("writing token cache — %s", cache_path)
                n_streamed = _stream_tokens_to_file(tokenizer, text, cache_path)
                logger.info("wrote %s token ids to disk", n_streamed)
                self._mmap = _open_token_mmap(cache_path)
            n_tokens = len(self._mmap)
            self.text = None
            self.tokenizer = None
        else:
            if text is None or tokenizer is None:
                raise ValueError("DataLoaderLite: text and tokenizer are required for in-memory mode")
            parts = text.splitlines(keepends=True)
            if not parts:
                token_ids = []
            elif len(parts) == 1:
                with tqdm(
                    total=1, desc="Tokenizing corpus", unit="pass", dynamic_ncols=True
                ) as pbar:
                    token_ids = tokenizer.encode(text).ids
                    pbar.update(1)
            else:
                token_ids = []
                for part in tqdm(
                    parts,
                    desc="Tokenizing corpus",
                    unit="lines",
                    dynamic_ncols=True,
                ):
                    token_ids.extend(tokenizer.encode(part).ids)
            self.tokens = torch.tensor(token_ids)
            n_tokens = len(self.tokens)
            self.text = text
            self.tokenizer = tokenizer

        if text is not None and n_tokens:
            chars_per_tok = len(text) / n_tokens
            logger.info(
                "loaded %s tokens (~%.3f chars/token)",
                n_tokens,
                chars_per_tok,
            )
        else:
            logger.info("loaded %s tokens (chars/token omitted — no in-memory text)", n_tokens)
        if cache_path is not None:
            file_bytes = cache_path.stat().st_size
            logger.info(
                "dataloader memory: token backing file ~%.2f MiB (%d bytes) mmap; "
                "Python text released after cache build",
                file_bytes / (1024**2),
                file_bytes,
            )
        else:
            tensor_bytes = self.tokens.numel() * self.tokens.element_size()
            text_bytes = sys.getsizeof(self.text)
            logger.info(
                "dataloader memory: token tensor %s ~%.2f MiB (%d bytes), text str ~%.2f KiB (%d bytes)",
                self.tokens.dtype,
                tensor_bytes / (1024**2),
                tensor_bytes,
                text_bytes / 1024,
                text_bytes,
            )
        self.batch_per_epoch = n_tokens // (B * T)
        logger.info("1 epoch = %s batches", self.batch_per_epoch)

        self.current_position = 0

    def _num_tokens(self) -> int:
        return len(self._mmap) if self._mmap is not None else len(self.tokens)

    def next_batch(self):
        B, T = self.B, self.T
        need = B * T + 1
        pos = self.current_position
        if self._mmap is not None:
            sl = self._mmap[pos : pos + need]
            chunk = np.asarray(sl, dtype=np.int64, order="C")
            buf = torch.from_numpy(chunk)
        else:
            buf = self.tokens[pos : pos + need]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + need > self._num_tokens():
            self.current_position = 0
        return x, y
