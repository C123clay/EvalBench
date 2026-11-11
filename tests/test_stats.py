import math
import random

import pytest

from evalbench.datasets import build_dataset
from evalbench.providers import MockProvider
from evalbench.runner import run_evaluation
from evalbench.scorers import build_scorer
from evalbench.stats import bootstrap_mean_ci, compare, sign_test, summarize


def dataset(n, tags=None):
    return build_dataset(
        [
            {"id": f"q{i}", "prompt": f"question {i}", "reference": f"answer {i}", "tags": tags(i) if tags else []}
            for i in range(n)
        ],
        name="synthetic",
    )


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    rng = random.Random(3)
    values = [rng.random() for _ in range(200)]
    first = bootstrap_mean_ci(values, seed=5)
    assert first == bootstrap_mean_ci(values, seed=5)
    assert first != bootstrap_mean_ci(values, seed=6)
    assert first.contains(sum(values) / len(values))
    assert first.high - first.low < 0.2


def test_bootstrap_interval_narrows_with_more_data():
    rng = random.Random(1)
    small = [float(rng.random() < 0.6) for _ in range(30)]
    large = [float(rng.random() < 0.6) for _ in range(3000)]
    width = lambda interval: interval.high - interval.low
    assert width(bootstrap_mean_ci(large)) < width(bootstrap_mean_ci(small)) / 4


def test_bootstrap_edge_cases():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0]) == bootstrap_mean_ci([1.0])
    assert math.isnan(bootstrap_mean_ci([]).low)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.0, 1.0], confidence=1.5)


@pytest.mark.parametrize(
    "wins, losses, expected",
    [(8, 0, 2 / 256), (0, 8, 2 / 256), (5, 5, 1.0), (0, 0, 1.0), (9, 1, 22 / 1024)],
)
def test_sign_test_exact_values(wins, losses, expected):
    assert sign_test(wins, losses) == pytest.approx(expected)


def test_summarize_reports_pass_rate_errors_and_tags():
    data = dataset(40, tags=lambda i: ["even"] if i % 2 == 0 else ["odd"])
    run = run_evaluation(data, [MockProvider(name="m", accuracy=0.75, seed=2)], build_scorer("exact"))
    (summary,) = summarize(run)

    passed = sum(item.passed for item in run.items)
    assert summary.n == 40
    assert summary.pass_rate == pytest.approx(passed / 40)
    assert summary.pass_rate_ci.contains(summary.pass_rate)
    assert summary.errors == 0
    assert set(summary.by_tag) == {"even", "odd"}
    assert sum(n for n, _ in summary.by_tag.values()) == 40


def test_clearly_better_candidate_is_reported_as_improved():
    run = run_evaluation(
        dataset(300),
        [MockProvider(name="base", accuracy=0.5, seed=1), MockProvider(name="cand", accuracy=0.9, seed=1)],
        build_scorer("exact"),
    )
    result = compare(run, "base", "cand")

    assert result.verdict == "improved"
    assert result.delta_ci.low > 0
    assert result.p_value < 0.001
    assert result.wins > result.losses
    assert result.n_paired == 300 and result.unpaired == 0


def test_sample_size_decides_whether_a_real_gap_is_detectable():
    # The README example: 65% vs 90% is inconclusive on 30 examples but clear on 300.
    models = lambda: [MockProvider(name="baseline", accuracy=0.65), MockProvider(name="candidate", accuracy=0.9)]
    large = run_evaluation(dataset(300), models(), build_scorer("exact"))
    assert compare(large, "baseline", "candidate").verdict == "improved"


def test_identical_models_are_inconclusive_with_all_ties():
    run = run_evaluation(
        dataset(20),
        [MockProvider(name="a", mode="constant", response="x"), MockProvider(name="b", mode="constant", response="x")],
        build_scorer("exact"),
    )
    result = compare(run, "a", "b")
    assert (result.verdict, result.ties, result.p_value, result.mean_delta) == ("inconclusive", 20, 1.0, 0.0)


def test_equally_accurate_models_are_usually_inconclusive():
    # Same accuracy, different mistakes: the difference is noise, so the CI should straddle zero.
    run = run_evaluation(
        dataset(300),
        [MockProvider(name="a", accuracy=0.7, seed=9), MockProvider(name="b", accuracy=0.7, seed=9)],
        build_scorer("exact"),
    )
    result = compare(run, "a", "b")
    assert result.verdict == "inconclusive"
    assert result.delta_ci.contains(0.0)


def test_regressions_list_examples_the_candidate_broke():
    run = run_evaluation(
        dataset(200),
        [MockProvider(name="good", accuracy=0.95, seed=4), MockProvider(name="bad", accuracy=0.4, seed=4)],
        build_scorer("exact"),
    )
    result = compare(run, "good", "bad")
    good = {item.example_id: item.passed for item in run.for_model("good")}
    bad = {item.example_id: item.passed for item in run.for_model("bad")}

    assert result.verdict == "regressed"
    assert set(result.regressions) == {i for i in good if good[i] and not bad[i]}


def test_compare_rejects_unknown_or_identical_models():
    run = run_evaluation(dataset(5), [MockProvider(name="a"), MockProvider(name="b")], build_scorer("exact"))
    with pytest.raises(ValueError, match="not in this result"):
        compare(run, "a", "zzz")
    with pytest.raises(ValueError, match="must be different"):
        compare(run, "a", "a")
