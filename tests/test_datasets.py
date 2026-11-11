import json

import pytest

from evalbench.datasets import DatasetError, build_dataset, load_dataset


def test_loads_jsonl_and_normalizes_reference_to_tuple(write_jsonl):
    path = write_jsonl([
        {"id": 1, "prompt": "Q1", "reference": "A1", "tags": ["x"]},
        {"id": "two", "prompt": "Q2", "references": ["A2", "a two"]},
    ])
    dataset = load_dataset(path)

    assert dataset.name == "data"
    assert [example.id for example in dataset] == ["1", "two"]
    assert dataset.examples[0].references == ("A1",)
    assert dataset.examples[1].references == ("A2", "a two")
    assert dataset.tags == ["x"]


def test_loads_json_document_with_examples_key(tmp_path):
    path = tmp_path / "set.json"
    path.write_text(json.dumps({"examples": [{"id": "q", "prompt": "P", "reference": "R"}]}), encoding="utf-8")
    assert len(load_dataset(path)) == 1


def test_blank_and_comment_lines_are_skipped(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('// a comment\n\n{"id": "a", "prompt": "P", "reference": "R"}\n', encoding="utf-8")
    assert len(load_dataset(path)) == 1


def test_invalid_json_reports_the_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "prompt": "P", "reference": "R"}\n{"id": "b", oops}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="line 2: invalid JSON"):
        load_dataset(path)


def test_duplicate_ids_are_rejected_with_both_locations(write_jsonl):
    path = write_jsonl([
        {"id": "a", "prompt": "P", "reference": "R"},
        {"id": "b", "prompt": "P", "reference": "R"},
        {"id": "a", "prompt": "P", "reference": "R"},
    ])
    with pytest.raises(DatasetError, match=r"line 3: duplicate id 'a' \(first seen at line 1\)"):
        load_dataset(path)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"prompt": "P", "reference": "R"}, "'id' must be"),
        ({"id": True, "prompt": "P", "reference": "R"}, "'id' must be"),
        ({"id": "a", "prompt": "  ", "reference": "R"}, "'prompt' must be"),
        ({"id": "a", "prompt": "P"}, "'reference' must be"),
        ({"id": "a", "prompt": "P", "reference": []}, "'reference' must be"),
        ({"id": "a", "prompt": "P", "reference": [1]}, "'reference' must be"),
        ({"id": "a", "prompt": "P", "reference": "R", "references": ["R"]}, "not both"),
        ({"id": "a", "prompt": "P", "reference": "R", "tags": "x"}, "'tags' must be"),
        ({"id": "a", "prompt": "P", "reference": "R", "metadata": []}, "'metadata' must be"),
        ({"id": "a", "prompt": "P", "reference": "R", "answer": "R"}, "unknown field"),
    ],
)
def test_invalid_examples_are_rejected(record, message):
    with pytest.raises(DatasetError, match=message):
        build_dataset([record], name="t")


def test_empty_and_missing_datasets_are_errors(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no examples"):
        load_dataset(empty)
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_fingerprint_is_stable_and_content_sensitive():
    record = {"id": "a", "prompt": "P", "reference": "R"}
    first = build_dataset([record], name="x").fingerprint
    assert build_dataset([dict(record)], name="other-name").fingerprint == first
    assert build_dataset([{**record, "reference": "S"}], name="x").fingerprint != first


def test_bundled_example_datasets_are_valid(examples_dir):
    qa = load_dataset(examples_dir / "general_qa.jsonl")
    extraction = load_dataset(examples_dir / "extraction.jsonl")
    assert len(qa) == 30
    assert {"arithmetic", "geography", "science"} <= set(qa.tags)
    assert len(extraction) == 6
