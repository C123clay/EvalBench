import json
from pathlib import Path

import pytest

from evalbench.datasets import build_dataset

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES_DIR


@pytest.fixture
def write_jsonl(tmp_path):
    def write(records, name="data.jsonl"):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    return write


@pytest.fixture
def small_dataset():
    return build_dataset(
        [
            {"id": "a", "prompt": "Capital of France?", "reference": "Paris", "tags": ["geo"]},
            {"id": "b", "prompt": "2 + 2?", "reference": "4", "tags": ["math"]},
            {"id": "c", "prompt": "Opposite of hot?", "reference": ["cold", "cool"], "tags": ["language"]},
            {"id": "d", "prompt": "3 * 5?", "reference": "15", "tags": ["math"]},
        ],
        name="small",
    )
