import pytest

from evalbench.scorers import (
    ScorerError,
    build_scorer,
    contains,
    exact_match,
    extract_number,
    json_equal,
    normalize,
    numeric,
    regex,
    token_f1,
)


def test_normalize_drops_case_punctuation_and_articles():
    assert normalize("  The Indian  Ocean! ") == "indian ocean"
    assert normalize("An apple, a day.") == "apple day"


@pytest.mark.parametrize(
    "output, references, passed",
    [
        ("Tokyo", ["tokyo"], True),
        ("The Amazon River.", ["Amazon", "the Amazon River"], True),
        ("Tokyo, Japan", ["Tokyo"], False),
        ("", ["x"], False),
    ],
)
def test_exact_match(output, references, passed):
    assert exact_match(output, references).passed is passed


def test_contains_matches_whole_words_only():
    assert contains("The capital is Tokyo.", ["tokyo"]).passed
    assert not contains("concatenate", ["cat"]).passed


def test_token_f1_known_value():
    # normalized: output [cat, sat], reference [cat, sat, down] -> P=1, R=2/3, F1=0.8
    score = token_f1("The cat sat", ["cat sat down"])
    assert score.value == pytest.approx(0.8)
    assert score.passed


def test_token_f1_takes_best_reference_and_respects_threshold():
    assert token_f1("carbon dioxide", ["CO2", "carbon dioxide"]).value == 1.0
    assert not token_f1("blue whale", ["whale shark"], threshold=0.6).passed
    assert token_f1("blue whale", ["whale shark"], threshold=0.5).passed
    assert token_f1("", ["anything"]).value == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [("3,600 seconds", 3600.0), ("It is -4.5 degrees", -4.5), ("1e3", 1000.0), ("no digits", None), ("17 x 6 = 102", 17.0)],
)
def test_extract_number(text, expected):
    assert extract_number(text) == expected


def test_numeric_tolerances():
    assert numeric("There are 3,600 seconds.", ["3600"]).passed
    assert not numeric("3.14", ["3.14159"]).passed
    assert numeric("3.14", ["3.14159"], abs_tol=0.01).passed
    assert numeric("1010", ["1000"], rel_tol=0.02).passed
    assert numeric("none", ["1"]).detail == "no number in output"


def test_regex_scorer_and_invalid_pattern():
    assert regex("Answer: 42", [r"answer:\s*42"]).passed
    result = regex("x", ["("])
    assert not result.passed and "invalid pattern" in result.detail


def test_json_equal_ignores_key_order_and_code_fences():
    reference = '{"name": "Maria", "age": 34}'
    assert json_equal('```json\n{"age": 34, "name": "Maria"}\n```', [reference]).passed
    assert not json_equal('{"name": "Maria", "age": 35}', [reference]).passed
    assert "not JSON" in json_equal("name: Maria", [reference]).detail


def test_build_scorer_parses_parameters():
    scorer = build_scorer("numeric:abs_tol=0.5")
    assert scorer("10.4", ["10"]).passed
    assert not build_scorer("numeric")("10.4", ["10"]).passed
    assert build_scorer("token_f1:threshold=0.9")("cat sat", ["cat sat down"]).passed is False


@pytest.mark.parametrize("spec, message", [
    ("nope", "unknown scorer"),
    ("exact:threshold=1", "does not accept"),
    ("token_f1:threshold", "does not accept"),
    ("token_f1:threshold=high", "must be a number"),
])
def test_build_scorer_rejects_bad_specs(spec, message):
    with pytest.raises(ScorerError, match=message):
        build_scorer(spec)
