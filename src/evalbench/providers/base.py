"""The provider interface every model backend implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evalbench.datasets import Example


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class ProviderError(RuntimeError):
    """A failed model call. ``retryable`` marks transient failures (rate limits, 5xx, timeouts)."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, example: Example) -> Completion:
        """Return the model's answer to ``prompt``.

        ``example`` is passed for simulators and logging; real backends must
        send only ``prompt`` to the model.
        """
        ...


def count_tokens(text: str) -> int:
    """A whitespace token count: stable and dependency-free, not a real tokenizer."""
    return len(text.split())
