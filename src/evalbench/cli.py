"""Command-line interface.

Exit codes: 0 success, 1 a quality gate failed (``--fail-under`` or
``--fail-on-regression``), 2 invalid input or usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evalbench import __version__
from evalbench.datasets import DatasetError, load_dataset
from evalbench.providers import SpecError, parse_provider
from evalbench.report import render_comparison, render_summary
from evalbench.runner import RunResult, run_evaluation
from evalbench.scorers import SCORERS, ScorerError, build_scorer
from evalbench.stats import compare, summarize

EXIT_OK, EXIT_GATE_FAILED, EXIT_INPUT_ERROR = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalbench", description="Evaluate and compare AI model outputs against reference answers."
    )
    parser.add_argument("--version", action="version", version=f"evalbench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check a dataset file and print its shape")
    validate.add_argument("dataset", type=Path)

    run = commands.add_parser("run", help="run one or more models over a dataset and score them")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--model", dest="models", action="append", required=True, metavar="SPEC",
                     help="provider spec, e.g. mock:oracle?accuracy=0.8 (repeat to compare models)")
    run.add_argument("--scorer", default="exact", help="scorer spec, e.g. token_f1:threshold=0.6 (default: exact)")
    run.add_argument("--workers", type=int, default=4, help="concurrent model calls (default: 4)")
    run.add_argument("--retries", type=int, default=2, help="retries for transient failures (default: 2)")
    run.add_argument("--backoff", type=float, default=0.5, help="initial retry delay in seconds (default: 0.5)")
    run.add_argument("--output", type=Path, help="write full per-example results as JSON")
    run.add_argument("--markdown", type=Path, help="write the summary as a Markdown report")
    run.add_argument("--fail-under", type=float, metavar="RATE",
                     help="exit 1 if any model's pass rate is below RATE (0-1)")
    run.add_argument("--show-failures", type=int, default=3, metavar="N",
                     help="print up to N failing examples per model (default: 3)")
    run.add_argument("--seed", type=int, default=0, help="seed for bootstrap intervals (default: 0)")

    cmp = commands.add_parser("compare", help="paired comparison of two models in a results file")
    cmp.add_argument("results", type=Path)
    cmp.add_argument("--baseline", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--iterations", type=int, default=2000, help="bootstrap resamples (default: 2000)")
    cmp.add_argument("--seed", type=int, default=0)
    cmp.add_argument("--markdown", type=Path, help="also write the comparison as Markdown")
    cmp.add_argument("--fail-on-regression", action="store_true",
                     help="exit 1 if the candidate is significantly worse than the baseline")

    commands.add_parser("scorers", help="list available scorers")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        return {"validate": _validate, "run": _run, "compare": _compare, "scorers": _scorers}[args.command](args)
    except (DatasetError, SpecError, ScorerError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR


def _validate(args) -> int:
    dataset = load_dataset(args.dataset)
    references = sum(len(example.references) for example in dataset)
    print(f"{args.dataset}: {len(dataset)} examples, {references} references, fingerprint {dataset.fingerprint}")
    if dataset.tags:
        counts = {tag: sum(tag in example.tags for example in dataset) for tag in dataset.tags}
        print("tags: " + ", ".join(f"{tag} ({count})" for tag, count in counts.items()))
    return EXIT_OK


def _run(args) -> int:
    if args.fail_under is not None and not 0 <= args.fail_under <= 1:
        raise ValueError("--fail-under must be between 0 and 1")
    dataset = load_dataset(args.dataset)
    scorer = build_scorer(args.scorer)
    providers = [parse_provider(spec) for spec in args.models]

    run = run_evaluation(
        dataset, providers, scorer, scorer_spec=args.scorer,
        max_workers=args.workers, retries=args.retries, backoff_seconds=args.backoff,
    )
    summaries = summarize(run, seed=args.seed)
    print(render_summary(run, summaries), end="")

    if args.show_failures > 0:
        for model in run.models:
            failures = [item for item in run.for_model(model) if not item.passed][: args.show_failures]
            if failures:
                print(f"\nFirst failures for {model}:")
                for item in failures:
                    reason = item.error or item.detail
                    print(f"  [{item.example_id}] {item.output[:60]!r} -- {reason}")

    if args.output:
        run.save(args.output)
        print(f"\nResults written to {args.output}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_summary(run, summaries, markdown=True), encoding="utf-8")
        print(f"Markdown report written to {args.markdown}")

    if args.fail_under is not None:
        below = [s.model for s in summaries if s.pass_rate < args.fail_under]
        if below:
            print(f"\nQuality gate failed: pass rate below {args.fail_under:.0%} for {', '.join(below)}")
            return EXIT_GATE_FAILED
    return EXIT_OK


def _compare(args) -> int:
    run = RunResult.load(args.results)
    result = compare(run, args.baseline, args.candidate, iterations=args.iterations, seed=args.seed)
    print(render_comparison(result), end="")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_comparison(result, markdown=True), encoding="utf-8")
        print(f"Markdown comparison written to {args.markdown}")
    if args.fail_on_regression and result.verdict == "regressed":
        print("Regression gate failed: candidate is significantly worse than baseline")
        return EXIT_GATE_FAILED
    return EXIT_OK


def _scorers(args) -> int:
    width = max(map(len, SCORERS))
    for name, spec in SCORERS.items():
        params = f" [{', '.join(spec.params)}]" if spec.params else ""
        print(f"{name.ljust(width)}  {spec.description}{params}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
