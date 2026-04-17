"""Tests for training checkpoint save/load (src.utils)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.model.gpt import GPT, GPTConfig
from src.utils import (
    load_model_state_from_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _tiny_config() -> GPTConfig:
    return GPTConfig(vocab_size=64, n_layer=1, n_head=4, n_embd=32, block_size=32)


def test_checkpoint_roundtrip_restores_weights_and_optimizer_state():
    device = "cpu"
    cfg = _tiny_config()
    model = GPT(cfg)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.zeros(2, 16, dtype=torch.long)
    y = torch.zeros(2, 16, dtype=torch.long)
    opt.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    epoch_losses = [0.5, 0.4]
    last_val = 0.35
    epochs_done = 2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_training_checkpoint(
            path,
            model,
            opt,
            epoch_losses,
            last_val,
            epochs_done,
            cfg.vocab_size,
        )
        assert path.is_file()

        model_b = GPT(cfg)
        model_b.to(device)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
        loaded_losses, loaded_val, start_epochs = load_training_checkpoint(
            path, model_b, opt_b, device
        )

        assert loaded_losses == epoch_losses
        assert loaded_val == last_val
        assert start_epochs == epochs_done

        for p_a, p_b in zip(model.parameters(), model_b.parameters()):
            assert torch.allclose(p_a, p_b)

        model_c = GPT(cfg)
        model_c.to(device)
        load_model_state_from_training_checkpoint(path, model_c, device)
        for p_a, p_c in zip(model.parameters(), model_c.parameters()):
            assert torch.allclose(p_a, p_c)


def test_vocab_mismatch_raises():
    device = "cpu"
    cfg_a = GPTConfig(vocab_size=64, n_layer=1, n_head=4, n_embd=32, block_size=32)
    cfg_b = GPTConfig(vocab_size=99, n_layer=1, n_head=4, n_embd=32, block_size=32)
    model = GPT(cfg_a)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_training_checkpoint(path, model, opt, [], None, 0, cfg_a.vocab_size)

        wrong = GPT(cfg_b)
        wrong.to(device)
        try:
            load_model_state_from_training_checkpoint(path, wrong, device)
        except ValueError as e:
            assert "vocab_size" in str(e).lower()
        else:
            raise AssertionError("expected ValueError for vocab mismatch")
