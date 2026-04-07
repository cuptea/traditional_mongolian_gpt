import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Little-endian uint32 raw stream on disk (one id per element).
_TOKEN_FILE_DTYPE = np.dtype("<u4")


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
                    token_ids = tokenizer.encode(text).ids
                    if token_ids:
                        arr = np.asarray(token_ids, dtype=_TOKEN_FILE_DTYPE)
                        f.write(arr.tobytes())
                        n_written = len(token_ids)
                    pbar.update(1)
            else:
                for part in tqdm(
                    parts,
                    desc="Tokenizing corpus",
                    unit="lines",
                    dynamic_ncols=True,
                ):
                    token_ids = tokenizer.encode(part).ids
                    if token_ids:
                        arr = np.asarray(token_ids, dtype=_TOKEN_FILE_DTYPE)
                        f.write(arr.tobytes())
                        n_written += len(token_ids)
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
        text,
        tokenizer,
        token_cache_path=None,
        reuse_token_cache=True,
    ):
        self.B = B
        self.T = T
        self._mmap = None
        self.tokens = None

        cache_path = Path(token_cache_path).resolve() if token_cache_path else None

        utf8_nbytes = len(text.encode("utf-8"))
        line_count = text.count("\n") + (1 if text else 0)
        logger.info(
            "text corpus: %s characters, %s UTF-8 bytes, %s lines",
            len(text),
            utf8_nbytes,
            line_count,
        )

        if cache_path is not None:
            cache_nonempty = (
                cache_path.is_file() and cache_path.stat().st_size > 0
            )
            if reuse_token_cache and cache_nonempty:
                logger.info("token cache hit — mmap %s", cache_path)
                self._mmap = _open_token_mmap(cache_path)
            else:
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

        chars_per_tok = len(text) / n_tokens if n_tokens else 0.0
        logger.info(
            "loaded %s tokens (~%.3f chars/token)",
            n_tokens,
            chars_per_tok,
        )
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
