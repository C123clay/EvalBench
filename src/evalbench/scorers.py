"""Scoring a model output against an example's reference answers.

Every scorer returns a :class:`Score` with a value in ``[0, 1]`` and a pass/fail
verdict. When an example has several references, the best-matching one wins.
Scorers are built from short specs such as ``"token_f1:threshold=0.6"`` so
they can be chosen on the command line and recorded in result files.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Score:
    value: float
    passed: bool
    detail: str = ""


ScoreFn = Callable[[str, Sequence[str]], Score]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and English articles, and collapse whitespace."""
    text = text.lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(output: str, references: Sequence[str]) -> Score:
    normalized = normalize(output)
    for reference in references:
        if normalized == normalize(reference):
            return Score(1.0, True, f"matched {reference!r}")
    return Score(0.0, False, "no reference matched after normalization")


def contains(output: str, references: Sequence[str]) -> Score:
    padded = f" {normalize(output)} "
    for reference in references:
        target = normalize(reference)
        if target and f" {target} " in padded:
            return Score(1.0, True, f"found {reference!r}")
    return Score(0.0, False, "no reference found in output")


def token_f1(output: str, references: Sequence[str], *, threshold: float = 0.5) -> Score:
    """Bag-of-tokens F1 between output and the closest reference."""
    predicted = normalize(output).split()
    best, best_reference = 0.0, references[0]
    for reference in references:
        expected = normalize(reference).split()
        if not predicted and not expected:
            value = 1.0
        elif not predicted or not expected:
            value = 0.0
        else:
            overlap = sum((Counter(predicted) & Counter(expected)).values())
            if overlap == 0:
                value = 0.0
            else:
                precision, recall = overlap / len(predicted), overlap / len(expected)
                value = 2 * precision * recall / (precision + recall)
        if value > best:
            best, best_reference = value, reference
    return Score(round(best, 6), best >= threshold, f"F1 {best:.3f} vs {best_reference!r}")


def extract_number(text: str) -> float | None:
    for match in _NUMBER.finditer(text):
        token = match.group(0)
        if any(char.isdigit() for char in token):
            try:
                return float(token.replace(",", ""))
            except ValueError:
                continue
    return None


def numeric(
    output: str, references: Sequence[str], *, abs_tol: float = 1e-9, rel_tol: float = 0.0
) -> Score:
    """Compare the first number in the output with each numeric reference."""
    value = extract_number(output)
    if value is None:
        return Score(0.0, False, "no number in output")
    for reference in references:
        expected = extract_number(reference)
        if expected is not None and math.isclose(value, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return Score(1.0, True, f"{value:g} matches {expected:g}")
    return Score(0.0, False, f"{value:g} matches no reference")


def regex(output: str, references: Sequence[str]) -> Score:
    """Treat each reference as a regular expression that must match somewhere in the output."""
    for reference in references:
        try:
            if re.search(reference, output, flags=re.IGNORECASE | re.MULTILINE):
                return Score(1.0, True, f"pattern {reference!r} matched")
        except re.error as exc:
            return Score(0.0, False, f"invalid pattern {reference!r}: {exc}")
    return Score(0.0, False, "no pattern matched")


def json_equal(output: str, references: Sequence[str]) -> Score:
    """Parse the output as JSON (tolerating a surrounding code fence) and compare structurally."""
    candidate = output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return Score(0.0, False, f"output is not JSON ({exc.msg})")
    for reference in references:
        try:
            if parsed == json.loads(reference):
                return Score(1.0, True, "structurally equal")
        except json.JSONDecodeError:
            continue
    return Score(0.0, False, "valid JSON but differs from every reference")


@dataclass(frozen=True)
class ScorerSpec:
    fn: Callable[..., Score]
    description: str
    params: dict[str, type]


SCORERS: dict[str, ScorerSpec] = {
    "exact": ScorerSpec(exact_match, "Normalized exact match against any reference.", {}),
    "contains": ScorerSpec(contains, "A reference appears in the output as whole words.", {}),
    "token_f1": ScorerSpec(token_f1, "Token-overlap F1; passes at or above `threshold`.", {"threshold": float}),
    "numeric": ScorerSpec(
        numeric, "First number in the output equals a reference within tolerance.",
        {"abs_tol": float, "rel_tol": float},
    ),
    "regex": ScorerSpec(regex, "Each reference is a case-insensitive regular expression.", {}),
    "json": ScorerSpec(json_equal, "Output parses as JSON equal to a reference.", {}),
}


class ScorerError(ValueError):
    pass


def build_scorer(spec: str) -> ScoreFn:
    """Turn ``"name"`` or ``"name:key=value,key=value"`` into a scoring function."""
    name, _, raw_params = spec.partition(":")
    name = name.strip()
    if name not in SCORERS:
        raise ScorerError(f"unknown scorer {name!r}; choose from {', '.join(SCORERS)}")
    entry = SCORERS[name]
    params: dict[str, object] = {}
    for item in filter(None, (part.strip() for part in raw_params.split(","))):
        key, sep, value = item.partition("=")
        if not sep or key not in entry.params:
            allowed = ", ".join(entry.params) or "none"
            raise ScorerError(f"scorer {name!r} does not accept {item!r} (parameters: {allowed})")
        try:
            params[key] = entry.params[key](value)
        except ValueError as exc:
            raise ScorerError(f"scorer {name!r}: {key} must be a number, got {value!r}") from exc

    def score(output: str, references: Sequence[str]) -> Score:
        return entry.fn(output, references, **params)

    return score
