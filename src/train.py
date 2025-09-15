"""
train.py
---------
Model-related utilities and lightweight wrappers extracted from the original
monolithic implementation.

Only one static-lint problem was detected (“unused # type: ignore”).  The whole
file remains identical to the source except that the offending comment has been
removed – **no functional change**.
"""
from __future__ import annotations

import contextlib
import os
from typing import Dict

import torch

try:
    # Heavy import – defer any device–specific initialisation until runtime
    from diffusers import DiTPipeline
except Exception:  # pragma: no cover – CPU-only CI workers might miss CUDA / diffusers
    DiTPipeline = None  # falls back to a sentinel; a RuntimeError is raised on use

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_AVAILABLE_MODELS = {
    "DiT-XL/2-256": "facebook/DiT-XL-2-256",
    "DiT-XL/2-512": "facebook/DiT-XL-2-512",
}


# -----------------------------------------------------------------------------
#  Public helpers
# -----------------------------------------------------------------------------

def load_dit(model_key: str, device: str | None = None):
    """Load a pre-trained DiT model from the 🤗 hub.

    Parameters
    ----------
    model_key : str
        A key from the `_AVAILABLE_MODELS` dictionary.
    device : str | None, optional
        Target device (defaults to the best available: CUDA → CPU).
    """
    device = device or _DEVICE

    if model_key not in _AVAILABLE_MODELS:
        raise ValueError(f"Unknown DiT variant: {model_key}")

    if DiTPipeline is None:
        raise RuntimeError(
            "diffusers.DiTPipeline is unavailable – install `diffusers[torch]` "
            "and ensure a compatible GPU driver is present."
        )

    token = os.getenv("HF_TOKEN")  # optional – avoids anonymous rate limiting
    pipe = DiTPipeline.from_pretrained(
        _AVAILABLE_MODELS[model_key],
        torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32,
        use_safetensors=True,
        token=token,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# -----------------------------------------------------------------------------
#  VRAM limiter (soft cap) – helps the experiments run on small GPUs
# -----------------------------------------------------------------------------

class VRAMLimiter(contextlib.ContextDecorator):
    """Temporarily cap *per-process* memory usage on CUDA.

    CPU-only environments simply no-op so that the entire suite can be executed
    inside CI pipelines without a GPU.
    """

    def __init__(self, max_gb: int):
        self.max_bytes = max_gb * 1024 ** 3
        self._old_fraction: float | None = None

    def __enter__(self):  # noqa: D401 – context-manager signature
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)

            # Guard: not every PyTorch build ships the fractional API (esp. wheels
            # on some distros); fall back to no-op when missing.
            if hasattr(torch.cuda, "get_per_process_memory_fraction") and hasattr(
                torch.cuda, "set_per_process_memory_fraction"
            ):
                self._old_fraction = torch.cuda.get_per_process_memory_fraction()
                torch.cuda.set_per_process_memory_fraction(
                    self.max_bytes / props.total_memory, 0
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401 – context-manager signature
        if (
            torch.cuda.is_available()
            and self._old_fraction is not None
            and hasattr(torch.cuda, "set_per_process_memory_fraction")
        ):
            torch.cuda.set_per_process_memory_fraction(self._old_fraction, 0)
        # propagate exceptions (if any)
        return False


# -----------------------------------------------------------------------------
#  Thin CREST wrapper – delays heavy import cost until actually used
# -----------------------------------------------------------------------------

class CRESTRuntime:
    """Lightweight wrapper around the local CREST implementation."""

    def __init__(self, variant: str, epsilon: float = float("inf")) -> None:
        try:
            from .crest_impl import DiT_CRESTRunner
        except ImportError as err:
            raise ImportError(
                "The CREST implementation could not be loaded. Check crest_impl.py"
            ) from err

        self.variant = variant
        self.epsilon = epsilon
        self.runner = DiT_CRESTRunner(variant=variant, epsilon=epsilon)

    @torch.no_grad()
    def generate(self, pipe, num_images: int, seed: int, **kwargs) -> Dict:
        torch.manual_seed(seed)
        return self.runner.generate(pipe, num_images=num_images, **kwargs)
