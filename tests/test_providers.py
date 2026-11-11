import json

import pytest

from evalbench.datasets import Example, build_dataset
from evalbench.providers import MockProvider, OpenAICompatibleProvider, ProviderError, SpecError, parse_provider
from evalbench.providers.mock import corrupt


def synthetic(n):
    return build_dataset(
        [{"id": f"q{i}", "prompt": f"question {i}", "reference": f"answer {i}"} for i in range(n)], name="synthetic"
    )


def answers(provider, dataset):
    return [provider.complete(example.prompt, example).text for example in dataset]


def test_mock_is_deterministic_across_instances_and_order():
    dataset = synthetic(50)
    first = answers(MockProvider(name="m", accuracy=0.5, seed=1), dataset)
    second = answers(MockProvider(name="m", accuracy=0.5, seed=1), dataset)
    reversed_order = answers(MockProvider(name="m", accuracy=0.5, seed=1), reversed(list(dataset)))
    assert first == second
    assert first == list(reversed(reversed_order))


def test_mock_seed_and_name_change_which_items_are_wrong():
    dataset = synthetic(200)
    base = answers(MockProvider(name="m", accuracy=0.5, seed=1), dataset)
    assert answers(MockProvider(name="m", accuracy=0.5, seed=2), dataset) != base
    assert answers(MockProvider(name="other", accuracy=0.5, seed=1), dataset) != base


@pytest.mark.parametrize("accuracy", [0.3, 0.7, 0.9])
def test_mock_oracle_accuracy_is_calibrated(accuracy):
    dataset = synthetic(4000)
    provider = MockProvider(accuracy=accuracy, seed=11)
    correct = sum(text == example.references[0] for text, example in zip(answers(provider, dataset), dataset))
    assert correct / len(dataset) == pytest.approx(accuracy, abs=0.03)


def test_mock_wrong_answers_never_equal_the_reference():
    dataset = synthetic(100)
    for text, example in zip(answers(MockProvider(accuracy=0.0), dataset), dataset):
        assert text != example.references[0]


def test_corrupt_produces_a_different_plausible_answer():
    assert corrupt("102", 0.1) == "103"
    assert corrupt("102", 0.3) == "101"
    assert corrupt("the Amazon River", 0.6) == "the"
    assert corrupt("Mars", 0.9) == "I am not sure."


def test_mock_modes_echo_constant_and_rules(tmp_path):
    example = Example(id="x", prompt="What is the capital of Japan?", references=("Tokyo",))
    assert MockProvider(mode="echo").complete(example.prompt, example).text == example.prompt
    assert MockProvider(mode="constant", response="Paris").complete(example.prompt, example).text == "Paris"

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps([{"pattern": "capital of japan", "response": "Tokyo"}]), encoding="utf-8")
    provider = MockProvider.from_rules_file(rules, response="unknown")
    assert provider.complete(example.prompt, example).text == "Tokyo"
    other = Example(id="y", prompt="Something else", references=("?",))
    assert provider.complete(other.prompt, other).text == "unknown"


def test_mock_reports_tokens_and_bounded_latency():
    example = Example(id="x", prompt="one two three", references=("a b",))
    completion = MockProvider().complete(example.prompt, example)
    assert (completion.prompt_tokens, completion.completion_tokens) == (3, 2)
    assert 20 <= completion.latency_ms <= 100


def test_mock_failure_rate_raises_retryable_errors():
    dataset = synthetic(300)
    provider = MockProvider(failure_rate=0.2, seed=4)
    failures = 0
    for example in dataset:
        try:
            provider.complete(example.prompt, example)
        except ProviderError as exc:
            assert exc.retryable
            failures += 1
    assert 30 <= failures <= 90


@pytest.mark.parametrize("kwargs", [{"mode": "psychic"}, {"accuracy": 1.5}, {"failure_rate": -0.1}])
def test_mock_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MockProvider(**kwargs)


def test_parse_provider_specs(examples_dir):
    provider = parse_provider("mock:oracle?accuracy=0.8&seed=3&name=candidate")
    assert isinstance(provider, MockProvider)
    assert (provider.name, provider.accuracy, provider.seed) == ("candidate", 0.8, 3)
    assert parse_provider("mock:echo").name == "mock-echo"
    rules = parse_provider(f"mock:rules?file={examples_dir / 'mock_rules.json'}")
    assert rules.mode == "rules" and rules.rules

    remote = parse_provider("openai:some-model?base_url=http://localhost:9/v1/&timeout=5")
    assert isinstance(remote, OpenAICompatibleProvider)
    assert (remote.name, remote.base_url, remote.timeout) == ("some-model", "http://localhost:9/v1", 5.0)


@pytest.mark.parametrize("spec, message", [
    ("oracle", "must look like"),
    ("mock:", "must look like"),
    ("cloud:x", "unknown provider kind"),
    ("mock:oracle?temperature=1", "unknown parameter"),
    ("mock:oracle?accuracy=lots", "not a valid float"),
    ("mock:oracle?accuracy=2", "between 0 and 1"),
    ("mock:telepathy", "unknown mock mode"),
    ("mock:rules", "needs file="),
    ("mock:rules?file=does-not-exist.json", "does-not-exist"),
    ("mock:oracle?accuracy", "malformed parameters"),
])
def test_parse_provider_rejects_bad_specs(spec, message):
    with pytest.raises(SpecError, match=message):
        parse_provider(spec)
