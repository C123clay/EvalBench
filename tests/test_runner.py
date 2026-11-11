import json
from dataclasses import dataclass, field

import pytest

from evalbench.datasets import Example
from evalbench.providers import MockProvider, ProviderError
from evalbench.providers.base import Completion
from evalbench.runner import RunResult, run_evaluation
from evalbench.scorers import build_scorer


@dataclass
class ScriptedProvider:
    """Fails a scripted number of times per example, then answers with the reference."""

    name: str
    failures: dict[str, list[Exception]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str, example: Example) -> Completion:
        self.calls.append(example.id)
        pending = self.failures.get(example.id)
        if pending:
            raise pending.pop(0)
        return Completion(example.references[0], 5.0, 1, 1)


def test_results_are_in_model_then_dataset_order_regardless_of_concurrency(small_dataset):
    providers = lambda: [MockProvider(name="a", accuracy=0.5, seed=1), MockProvider(name="b", accuracy=0.5, seed=2)]
    serial = run_evaluation(small_dataset, providers(), build_scorer("exact"), max_workers=1)
    parallel = run_evaluation(small_dataset, providers(), build_scorer("exact"), max_workers=8)

    expected = [(m, e.id) for m in ("a", "b") for e in small_dataset]
    assert [(item.model, item.example_id) for item in serial.items] == expected
    assert serial.items == parallel.items


def test_perfect_oracle_passes_everything(small_dataset):
    run = run_evaluation(small_dataset, [MockProvider(name="perfect")], build_scorer("exact"))
    assert all(item.passed and item.score == 1.0 for item in run.items)
    assert run.dataset_fingerprint == small_dataset.fingerprint


def test_transient_errors_are_retried_with_exponential_backoff(small_dataset):
    flaky = ProviderError("rate limited", retryable=True)
    provider = ScriptedProvider("p", failures={"a": [flaky, flaky]})
    delays = []

    run = run_evaluation(small_dataset, [provider], build_scorer("exact"), retries=2, backoff_seconds=0.5,
                         sleep=delays.append)

    item = run.items[0]
    assert (item.passed, item.attempts, item.error) == (True, 3, None)
    assert delays == [0.5, 1.0]


def test_exhausted_retries_are_recorded_not_raised(small_dataset):
    flaky = lambda: ProviderError("still rate limited", retryable=True)
    provider = ScriptedProvider("p", failures={"b": [flaky() for _ in range(5)]})

    run = run_evaluation(small_dataset, [provider], build_scorer("exact"), retries=1, sleep=lambda _: None)

    failed = next(item for item in run.items if item.example_id == "b")
    assert (failed.passed, failed.score, failed.attempts) == (False, 0.0, 2)
    assert failed.error == "still rate limited"
    assert sum(item.passed for item in run.items) == 3


def test_permanent_errors_and_provider_bugs_are_not_retried(small_dataset):
    provider = ScriptedProvider("p", failures={"a": [ProviderError("bad request")], "c": [KeyError("oops")]})
    run = run_evaluation(small_dataset, [provider], build_scorer("exact"), retries=3, sleep=lambda _: None)

    errors = {item.example_id: (item.attempts, item.error) for item in run.items if item.error}
    assert errors == {"a": (1, "bad request"), "c": (1, "KeyError: 'oops'")}
    assert provider.calls.count("a") == 1


def test_model_names_must_be_unique(small_dataset):
    with pytest.raises(ValueError, match="repeated: m"):
        run_evaluation(small_dataset, [MockProvider(name="m"), MockProvider(name="m")], build_scorer("exact"))


def test_on_item_callback_sees_every_item(small_dataset):
    seen = []
    run_evaluation(small_dataset, [MockProvider()], build_scorer("exact"), on_item=seen.append)
    assert sorted(item.example_id for item in seen) == ["a", "b", "c", "d"]


def test_results_round_trip_through_json(tmp_path, small_dataset):
    run = run_evaluation(small_dataset, [MockProvider(name="m", accuracy=0.5)], build_scorer("token_f1"),
                         scorer_spec="token_f1")
    path = tmp_path / "results.json"
    run.save(path)

    loaded = RunResult.load(path)
    assert loaded.items == run.items
    assert (loaded.scorer, loaded.models, loaded.dataset) == ("token_f1", ["m"], "small")


def test_unknown_schema_version_is_rejected(tmp_path, small_dataset):
    run = run_evaluation(small_dataset, [MockProvider()], build_scorer("exact"))
    data = run.to_dict() | {"schema_version": 99}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version 99"):
        RunResult.load(path)
