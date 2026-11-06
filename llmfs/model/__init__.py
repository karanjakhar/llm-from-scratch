"""Models: GPT (Stage 06) and Llama-class (Stage 12). load_model reads either."""

from __future__ import annotations

import torch


def build_model(config: dict):
    if config.get("arch") == "llama":
        from .llama import Llama, LlamaConfig
        return Llama(LlamaConfig(**config))
    from .gpt import GPT, GPTConfig
    return GPT(GPTConfig(**config))


def load_model(path, device="cpu", **overrides):
    """Rebuild a model from any checkpoint saved by llmfs (trainer or save_checkpoint)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model({**ck["config"], **overrides})
    sd = {k.removeprefix("_orig_mod.").replace(".block.", "."): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model.to(device), ck
