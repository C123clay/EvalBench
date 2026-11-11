import json

import pytest

from evalbench.cli import main


def run_cli(capsys, *argv):
    code = main(list(map(str, argv)))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_validate_prints_dataset_shape(capsys, examples_dir):
    code, out, _ = run_cli(capsys, "validate", examples_dir / "general_qa.jsonl")
    assert code == 0
    assert "30 examples" in out and "fingerprint" in out and "geography (6)" in out


def test_validate_reports_bad_data_with_exit_code_2(capsys, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "prompt": "P"}\n', encoding="utf-8")
    code, _, err = run_cli(capsys, "validate", bad)
    assert code == 2
    assert "line 1" in err and "'reference' must be" in err


def test_run_two_models_writes_json_and_markdown(capsys, tmp_path, examples_dir):
    results, report = tmp_path / "results.json", tmp_path / "report.md"
    code, out, _ = run_cli(
        capsys, "run", "--dataset", examples_dir / "general_qa.jsonl",
        "--model", "mock:oracle?accuracy=0.6&name=baseline",
        "--model", "mock:oracle?accuracy=0.9&name=candidate",
        "--scorer", "contains", "--output", results, "--markdown", report,
    )
    assert code == 0
    assert "baseline" in out and "candidate" in out and "Pass rate by tag" in out
    data = json.loads(results.read_text(encoding="utf-8"))
    assert data["models"] == ["baseline", "candidate"] and len(data["items"]) == 60
    assert report.read_text(encoding="utf-8").startswith("## Dataset general_qa")


def test_run_is_reproducible(capsys, tmp_path, examples_dir):
    outputs = []
    for name in ("one.json", "two.json"):
        run_cli(capsys, "run", "--dataset", examples_dir / "general_qa.jsonl",
                "--model", "mock:oracle?accuracy=0.7&seed=5", "--output", tmp_path / name, "--workers", "8")
        data = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        data.pop("created_at")
        outputs.append(data)
    assert outputs[0] == outputs[1]


def test_fail_under_gates_the_exit_code(capsys, examples_dir):
    args = ["run", "--dataset", examples_dir / "general_qa.jsonl", "--model", "mock:oracle?accuracy=0.5"]
    code, out, _ = run_cli(capsys, *args, "--fail-under", "0.9")
    assert code == 1 and "Quality gate failed" in out
    assert run_cli(capsys, *args, "--fail-under", "0.2")[0] == 0


def test_rules_mock_and_numeric_scorer(capsys, examples_dir):
    code, out, _ = run_cli(
        capsys, "run", "--dataset", examples_dir / "general_qa.jsonl",
        "--model", f"mock:rules?file={examples_dir / 'mock_rules.json'}&response=unknown",
        "--scorer", "contains", "--show-failures", "0",
    )
    assert code == 0 and "mock-rules" in out


def test_compare_reports_and_gates_regressions(capsys, tmp_path, examples_dir):
    results = tmp_path / "results.json"
    run_cli(capsys, "run", "--dataset", examples_dir / "general_qa.jsonl",
            "--model", "mock:oracle?accuracy=1.0&name=good", "--model", "mock:oracle?accuracy=0.3&name=bad",
            "--output", results)

    code, out, _ = run_cli(capsys, "compare", results, "--baseline", "bad", "--candidate", "good")
    assert code == 0 and "Verdict: IMPROVED" in out

    code, out, _ = run_cli(capsys, "compare", results, "--baseline", "good", "--candidate", "bad",
                           "--fail-on-regression", "--markdown", tmp_path / "cmp.md")
    assert code == 1 and "Verdict: REGRESSED" in out and "Regressions (passed before, fail now)" in out
    assert (tmp_path / "cmp.md").exists()


@pytest.mark.parametrize("argv, message", [
    (["--model", "nonsense"], "must look like"),
    (["--model", "mock:oracle", "--scorer", "vibes"], "unknown scorer"),
    (["--model", "mock:oracle", "--fail-under", "80"], "between 0 and 1"),
    (["--model", "mock:oracle?name=x", "--model", "mock:echo?name=x"], "must be unique"),
])
def test_run_input_errors_exit_2(capsys, examples_dir, argv, message):
    code, _, err = run_cli(capsys, "run", "--dataset", examples_dir / "general_qa.jsonl", *argv)
    assert code == 2 and message in err


def test_compare_unknown_model_exits_2(capsys, tmp_path, examples_dir):
    results = tmp_path / "r.json"
    run_cli(capsys, "run", "--dataset", examples_dir / "extraction.jsonl", "--model", "mock:oracle",
            "--scorer", "json", "--output", results)
    code, _, err = run_cli(capsys, "compare", results, "--baseline", "mock-oracle", "--candidate", "ghost")
    assert code == 2 and "not in this result" in err


def test_scorers_command_lists_every_scorer(capsys):
    code, out, _ = run_cli(capsys, "scorers")
    assert code == 0
    for name in ("exact", "contains", "token_f1", "numeric", "regex", "json"):
        assert name in out
