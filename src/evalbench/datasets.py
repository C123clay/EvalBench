"""Loading and validating evaluation datasets.

A dataset is a list of examples. Each example has a stable ``id``, a ``prompt``
sent to the model, and one or more acceptable ``reference`` answers. Datasets
are stored as JSON Lines (one example per line) or as a JSON document holding a
list of examples (optionally under an ``"examples"`` key).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


class DatasetError(ValueError):
    """Raised when a dataset file is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    references: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "references": list(self.references),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Dataset:
    name: str
    examples: tuple[Example, ...]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def tags(self) -> list[str]:
        return sorted({tag for example in self.examples for tag in example.tags})

    @property
    def fingerprint(self) -> str:
        """A content hash, so a result file can prove which data it was computed on."""
        canonical = json.dumps(
            [example.to_dict() for example in self.examples], sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"{path}: dataset file not found")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path}: not valid UTF-8 ({exc.reason})") from exc

    if path.suffix.lower() == ".jsonl":
        records = list(_parse_jsonl(text, path))
    else:
        records = _parse_json_document(text, path)
    return build_dataset(records, name=path.stem, source=str(path))


def build_dataset(
    records: Iterable[tuple[str, Any]] | Iterable[Any], *, name: str, source: str = "<memory>"
) -> Dataset:
    """Validate raw records into a Dataset.

    ``records`` may be plain dicts or ``(location, dict)`` pairs; the location
    (such as ``"line 4"``) is used in error messages.
    """
    examples: list[Example] = []
    seen: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        location, raw = record if isinstance(record, tuple) else (f"item {index}", record)
        where = f"{source}, {location}"
        example = _parse_example(raw, where)
        if example.id in seen:
            raise DatasetError(f"{where}: duplicate id {example.id!r} (first seen at {seen[example.id]})")
        seen[example.id] = location
        examples.append(example)
    if not examples:
        raise DatasetError(f"{source}: dataset contains no examples")
    return Dataset(name=name, examples=tuple(examples))


def _parse_jsonl(text: str, path: Path):
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            yield f"line {number}", json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}, line {number}: invalid JSON ({exc.msg})") from exc


def _parse_json_document(text: str, path: Path) -> list[tuple[str, Any]]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    if isinstance(document, dict) and "examples" in document:
        document = document["examples"]
    if not isinstance(document, list):
        raise DatasetError(f"{path}: expected a list of examples or an object with an 'examples' list")
    return [(f"item {index}", raw) for index, raw in enumerate(document, start=1)]


def _parse_example(raw: Any, where: str) -> Example:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: each example must be a JSON object")

    unknown = set(raw) - {"id", "prompt", "reference", "references", "tags", "metadata"}
    if unknown:
        raise DatasetError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")

    raw_id = raw.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or str(raw_id).strip() == "":
        raise DatasetError(f"{where}: 'id' must be a non-empty string or integer")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise DatasetError(f"{where}: 'prompt' must be a non-empty string")

    if "reference" in raw and "references" in raw:
        raise DatasetError(f"{where}: use either 'reference' or 'references', not both")
    references = raw.get("references", raw.get("reference"))
    if isinstance(references, str):
        references = [references]
    if (
        not isinstance(references, list)
        or not references
        or not all(isinstance(item, str) for item in references)
    ):
        raise DatasetError(f"{where}: 'reference' must be a string or a non-empty list of strings")

    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
        raise DatasetError(f"{where}: 'tags' must be a list of non-empty strings")

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"{where}: 'metadata' must be an object")

    return Example(
        id=str(raw_id),
        prompt=prompt,
        references=tuple(references),
        tags=tuple(dict.fromkeys(tags)),
        metadata=metadata,
    )
