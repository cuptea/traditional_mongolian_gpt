from pathlib import Path

from src.data.tokenizer_utils import ensure_bpe_tokenizer_artifacts


def test_ensure_bpe_tokenizer_artifacts_trains_when_missing(tmp_path: Path):
    data_path = tmp_path / "data.txt"
    data_path.write_text(
        "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ\n"
        "ᠰᠠᠢᠨ ᠪᠠᠢᠢᠨ᠎ᠠ\n"
        "traditional mongolian dataset\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "artifacts" / "traditional_mongolian_bpe"

    tokenizer, rebuilt = ensure_bpe_tokenizer_artifacts(data_path, tokenizer_dir)

    assert rebuilt is True
    assert (tokenizer_dir / "tokenizer.json").is_file()
    assert (tokenizer_dir / "vocab.json").is_file()
    assert (tokenizer_dir / "merges.txt").is_file()
    assert tokenizer.get_vocab_size() > 4


def test_ensure_bpe_tokenizer_artifacts_reuses_existing(tmp_path: Path):
    data_path = tmp_path / "data.txt"
    data_path.write_text("ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ\n", encoding="utf-8")
    tokenizer_dir = tmp_path / "artifacts" / "traditional_mongolian_bpe"

    _, rebuilt_first = ensure_bpe_tokenizer_artifacts(data_path, tokenizer_dir)
    tokenizer, rebuilt_second = ensure_bpe_tokenizer_artifacts(data_path, tokenizer_dir)

    assert rebuilt_first is True
    assert rebuilt_second is False
    assert tokenizer.get_vocab_size() > 4
