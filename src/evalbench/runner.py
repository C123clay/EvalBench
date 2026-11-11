"""Running every model over every example and collecting scored results.

Calls run concurrently on a thread pool, but results are always stored in
(model, dataset) order, so the same inputs produce byte-identical result files.
A failing call never aborts the run: transient errors are retried with backoff,
and a final failure is recorded on the item, scored zero, and counted.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from evalbench import __version__
from evalbench.datasets import Dataset, Example
from evalbench.providers.base import Provider, ProviderError
from evalbench.scorers import ScoreFn

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ItemResult:
    model: str
    example_id: str
    output: str
    score: float
    passed: bool
    detail: str
    tags: tuple[str, ...]
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    error: str | None = None


@dataclass
class RunResult:
    dataset: str
    dataset_fingerprint: str
    scorer: str
    models: list[str]
    items: list[ItemResult]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    evalbench_version: str = __version__

    def for_model(self, model: str) -> list[ItemResult]:
        return [item for item in self.items if item.model == model]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "evalbench_version": self.evalbench_version,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "scorer": self.scorer,
            "models": self.models,
            "items": [{**asdict(item), "tags": list(item.tags)} for item in self.items],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported result schema_version {version!r} (expected {SCHEMA_VERSION})")
        items = [ItemResult(**{**item, "tags": tuple(item.get("tags", ()))}) for item in data["items"]]
        return cls(
            dataset=data["dataset"],
            dataset_fingerprint=data["dataset_fingerprint"],
            scorer=data["scorer"],
            models=list(data["models"]),
            items=items,
            created_at=data["created_at"],
            evalbench_version=data.get("evalbench_version", "unknown"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run_evaluation(
    dataset: Dataset,
    providers: Sequence[Provider],
    scorer: ScoreFn,
    *,
    scorer_spec: str = "custom",
    max_workers: int = 4,
    retries: int = 2,
    backoff_seconds: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    on_item: Callable[[ItemResult], None] | None = None,
) -> RunResult:
    names = [provider.name for provider in providers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"model names must be unique; repeated: {', '.join(duplicates)} (set name=...)")
    if retries < 0 or max_workers < 1:
        raise ValueError("retries must be >= 0 and max_workers >= 1")

    jobs = [(provider, example) for provider in providers for example in dataset]

    def evaluate(job: tuple[Provider, Example]) -> ItemResult:
        provider, example = job
        result = _call_and_score(provider, example, scorer, retries, backoff_seconds, sleep)
        if on_item:
            on_item(result)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        items = list(pool.map(evaluate, jobs))  # map preserves submission order

    return RunResult(
        dataset=dataset.name,
        dataset_fingerprint=dataset.fingerprint,
        scorer=scorer_spec,
        models=names,
        items=items,
    )


def _call_and_score(
    provider: Provider,
    example: Example,
    scorer: ScoreFn,
    retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None],
) -> ItemResult:
    attempt = 0
    while True:
        attempt += 1
        try:
            completion = provider.complete(example.prompt, example)
            break
        except ProviderError as exc:
            if exc.retryable and attempt <= retries:
                sleep(backoff_seconds * 2 ** (attempt - 1))
                continue
            return _failed(provider, example, attempt, str(exc))
        except Exception as exc:  # a provider bug must not take down the whole run
            return _failed(provider, example, attempt, f"{type(exc).__name__}: {exc}")

    score = scorer(completion.text, example.references)
    return ItemResult(
        model=provider.name,
        example_id=example.id,
        output=completion.text,
        score=score.value,
        passed=score.passed,
        detail=score.detail,
        tags=example.tags,
        latency_ms=completion.latency_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        attempts=attempt,
    )


def _failed(provider: Provider, example: Example, attempts: int, message: str) -> ItemResult:
    return ItemResult(
        model=provider.name,
        example_id=example.id,
        output="",
        score=0.0,
        passed=False,
        detail="model call failed",
        tags=example.tags,
        attempts=attempts,
        error=message,
    )
