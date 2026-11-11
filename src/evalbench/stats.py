"""Summary statistics and paired model comparison.

Evaluation sets are usually small, so a raw difference in accuracy is easy to
over-read. Comparisons here are *paired*: two models are compared on the same
examples, which removes example difficulty from the noise. Uncertainty is
reported with a seeded percentile bootstrap (reproducible) and an exact
two-sided sign test on the examples where the models disagree.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from evalbench.runner import RunResult


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_mean_ci(
    values: Sequence[float], *, iterations: int = 2000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"))
    if n == 1 or len(set(values)) == 1:
        return Interval(values[0], values[0])
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iterations))
    alpha = 1 - confidence
    low_index = max(0, math.floor(alpha / 2 * iterations))
    high_index = min(iterations - 1, math.ceil((1 - alpha / 2) * iterations) - 1)
    return Interval(means[low_index], means[high_index])


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value for ``wins`` vs ``losses`` (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class ModelSummary:
    model: str
    n: int
    mean_score: float
    pass_rate: float
    pass_rate_ci: Interval
    errors: int
    mean_latency_ms: float
    total_tokens: int
    by_tag: dict[str, tuple[int, float]] = field(default_factory=dict)


def summarize(run: RunResult, *, iterations: int = 2000, seed: int = 0) -> list[ModelSummary]:
    summaries = []
    for model in run.models:
        items = run.for_model(model)
        passes = [1.0 if item.passed else 0.0 for item in items]
        tags: dict[str, list[float]] = {}
        for item, passed in zip(items, passes):
            for tag in item.tags:
                tags.setdefault(tag, []).append(passed)
        answered = [item.latency_ms for item in items if item.error is None]
        summaries.append(
            ModelSummary(
                model=model,
                n=len(items),
                mean_score=mean([item.score for item in items]),
                pass_rate=mean(passes),
                pass_rate_ci=bootstrap_mean_ci(passes, iterations=iterations, seed=seed),
                errors=sum(item.error is not None for item in items),
                mean_latency_ms=mean(answered) if answered else 0.0,
                total_tokens=sum(item.prompt_tokens + item.completion_tokens for item in items),
                by_tag={tag: (len(values), mean(values)) for tag, values in sorted(tags.items())},
            )
        )
    return summaries


@dataclass(frozen=True)
class Comparison:
    baseline: str
    candidate: str
    n_paired: int
    wins: int
    ties: int
    losses: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    delta_ci: Interval
    p_value: float
    verdict: str
    regressions: tuple[str, ...]
    unpaired: int

    @property
    def significant(self) -> bool:
        return self.verdict != "inconclusive"


def compare(
    run: RunResult, baseline: str, candidate: str, *, iterations: int = 2000, seed: int = 0
) -> Comparison:
    """Paired comparison of ``candidate`` against ``baseline`` on per-example scores.

    The verdict is ``improved`` or ``regressed`` only when the bootstrap
    interval for the mean score difference excludes zero; otherwise it is
    ``inconclusive``. ``regressions`` lists examples the baseline passed and
    the candidate failed - the ones worth reading first.
    """
    for model in (baseline, candidate):
        if model not in run.models:
            raise ValueError(f"model {model!r} is not in this result (models: {', '.join(run.models)})")
    if baseline == candidate:
        raise ValueError("baseline and candidate must be different models")

    base = {item.example_id: item for item in run.for_model(baseline)}
    cand = {item.example_id: item for item in run.for_model(candidate)}
    shared = [example_id for example_id in base if example_id in cand]
    unpaired = len(set(base) ^ set(cand))
    if not shared:
        raise ValueError("the two models share no examples to compare")

    deltas = [cand[i].score - base[i].score for i in shared]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ci = bootstrap_mean_ci(deltas, iterations=iterations, seed=seed)
    if ci.low > 0:
        verdict = "improved"
    elif ci.high < 0:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    return Comparison(
        baseline=baseline,
        candidate=candidate,
        n_paired=len(shared),
        wins=wins,
        ties=len(shared) - wins - losses,
        losses=losses,
        baseline_mean=mean([base[i].score for i in shared]),
        candidate_mean=mean([cand[i].score for i in shared]),
        mean_delta=mean(deltas),
        delta_ci=ci,
        p_value=sign_test(wins, losses),
        verdict=verdict,
        regressions=tuple(i for i in shared if base[i].passed and not cand[i].passed),
        unpaired=unpaired,
    )
