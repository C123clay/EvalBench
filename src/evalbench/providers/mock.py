"""A deterministic mock model, so the harness can be exercised without API keys.

The mock is a *simulator*: in ``oracle`` mode it may read the example's
reference and answer correctly with a chosen probability. That makes it a
calibration tool - run it at ``accuracy=0.7`` and the scores and confidence
intervals should recover roughly 0.7. Every decision is derived from a hash of
``(seed, model name, example id)``, so a run is reproducible across machines
and independent of thread scheduling or example order.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from evalbench.datasets import Example
from evalbench.providers.base import Completion, ProviderError, count_tokens

MODES = ("oracle", "echo", "constant", "rules")


def unit_interval(*parts: object) -> float:
    """Map arbitrary values to a stable float in [0, 1)."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def corrupt(answer: str, roll: float) -> str:
    """Produce a plausible wrong answer, choosing the kind of mistake from ``roll``."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if numbers and roll < 0.5:
        original = numbers[0]
        wrong = str(int(float(original)) + (1 if roll < 0.25 else -1))
        return answer.replace(original, wrong, 1)
    words = answer.split()
    if len(words) > 2 and roll < 0.75:
        return " ".join(words[: len(words) // 2])
    return "I am not sure."


@dataclass
class MockProvider:
    name: str = "mock"
    mode: str = "oracle"
    accuracy: float = 1.0
    seed: int = 0
    failure_rate: float = 0.0
    response: str = ""
    rules: list[tuple[re.Pattern[str], str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mock mode {self.mode!r}; choose from {', '.join(MODES)}")
        for label, value in (("accuracy", self.accuracy), ("failure_rate", self.failure_rate)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1, got {value}")

    @classmethod
    def from_rules_file(cls, path: str | Path, **kwargs) -> "MockProvider":
        """Load ``[{"pattern": "...", "response": "..."}, ...]`` for ``rules`` mode."""
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = [(re.compile(entry["pattern"], re.IGNORECASE), entry["response"]) for entry in entries]
        return cls(mode="rules", rules=rules, **kwargs)

    def complete(self, prompt: str, example: Example) -> Completion:
        if self.failure_rate and unit_interval(self.seed, self.name, example.id, "fail") < self.failure_rate:
            raise ProviderError(f"{self.name}: simulated transient failure", retryable=True)

        if self.mode == "echo":
            text = prompt
        elif self.mode == "constant":
            text = self.response
        elif self.mode == "rules":
            text = next((reply for pattern, reply in self.rules if pattern.search(prompt)), self.response)
        else:
            reference = example.references[0]
            roll = unit_interval(self.seed, self.name, example.id)
            if roll < self.accuracy:
                text = reference
            else:
                text = corrupt(reference, unit_interval(self.seed, self.name, example.id, "mistake"))

        latency = 20.0 + 80.0 * unit_interval(self.seed, self.name, example.id, "latency")
        return Completion(
            text=text,
            latency_ms=round(latency, 3),
            prompt_tokens=count_tokens(prompt),
            completion_tokens=count_tokens(text),
        )
