import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]
DEFAULT_BPE_VOCAB_SIZE = 10_000


def _infer_vocab_size(
    tokenizer_path: Path,
    vocab_path: Path,
    default_vocab_size: int,
) -> int:
    if tokenizer_path.is_file():
        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            vocab_size = tokenizer.get_vocab_size()
            if vocab_size > 0:
                return vocab_size
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.warning("could not infer vocab size from %s: %s", tokenizer_path, exc)

    if vocab_path.is_file():
        try:
            with vocab_path.open("r", encoding="utf-8") as f:
                vocab = json.load(f)
            vocab_size = len(vocab)
            if vocab_size > 0:
                return vocab_size
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.warning("could not infer vocab size from %s: %s", vocab_path, exc)

    return default_vocab_size


def train_bpe_tokenizer_from_dataset(
    data_path: Path,
    tokenizer_dir: Path,
    *,
    vocab_size: int = DEFAULT_BPE_VOCAB_SIZE,
    special_tokens: Optional[list[str]] = None,
) -> Tokenizer:
    data_path = Path(data_path).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"dataset for tokenizer training not found: {data_path}")

    special_tokens = special_tokens or SPECIAL_TOKENS
    tokenizer = Tokenizer(models.BPE(unk_token=special_tokens[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    logger.info(
        "training ByteLevel BPE tokenizer from %s (target_vocab_size=%s)",
        data_path,
        vocab_size,
    )
    tokenizer.train([str(data_path)], trainer)

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer.model.save(str(tokenizer_dir))
    logger.info(
        "tokenizer artifacts ready at %s (%s, %s, %s)",
        tokenizer_dir,
        tokenizer_path.name,
        "vocab.json",
        "merges.txt",
    )
    return tokenizer


def ensure_bpe_tokenizer_artifacts(
    data_path: Path,
    tokenizer_dir: Path,
    *,
    default_vocab_size: int = DEFAULT_BPE_VOCAB_SIZE,
    special_tokens: Optional[list[str]] = None,
) -> Tuple[Tokenizer, bool]:
    tokenizer_dir = Path(tokenizer_dir).resolve()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    artifacts_present = (
        tokenizer_path.is_file() and vocab_path.is_file() and merges_path.is_file()
    )
    if artifacts_present:
        logger.info("tokenizer artifacts found — loading %s", tokenizer_path)
        return Tokenizer.from_file(str(tokenizer_path)), False

    missing = [
        path.name
        for path in (tokenizer_path, vocab_path, merges_path)
        if not path.is_file()
    ]
    vocab_size = _infer_vocab_size(tokenizer_path, vocab_path, default_vocab_size)
    logger.warning(
        "tokenizer artifacts missing (%s) — rebuilding from %s",
        ", ".join(missing),
        data_path,
    )
    tokenizer = train_bpe_tokenizer_from_dataset(
        data_path,
        tokenizer_dir,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    return tokenizer, True
