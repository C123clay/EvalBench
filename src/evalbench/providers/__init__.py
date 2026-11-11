"""Model providers and the ``kind:target?key=value`` spec format used to select them.

Examples::

    mock:oracle?accuracy=0.8&seed=3&name=candidate
    mock:constant?response=Paris
    mock:rules?file=examples/mock_rules.json
    openai:gpt-4o-mini?base_url=http://localhost:8000/v1&api_key_env=MY_KEY
"""

from __future__ import annotations

from urllib.parse import parse_qsl

from evalbench.providers.base import Completion, Provider, ProviderError
from evalbench.providers.mock import MockProvider
from evalbench.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "Completion",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "SpecError",
    "parse_provider",
]


class SpecError(ValueError):
    pass


_MOCK_PARAMS = {"name": str, "accuracy": float, "seed": int, "failure_rate": float, "response": str, "file": str}
_OPENAI_PARAMS = {
    "name": str, "base_url": str, "api_key_env": str, "temperature": float,
    "max_tokens": int, "timeout": float, "system_prompt": str,
}


def parse_provider(spec: str) -> Provider:
    kind, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise SpecError(f"model spec {spec!r} must look like 'kind:target[?key=value&...]'")
    target, _, query = rest.partition("?")
    kind = kind.strip().lower()
    if kind == "mock":
        params = _typed_params(spec, query, _MOCK_PARAMS)
        params.setdefault("name", f"mock-{target}")
        rules_file = params.pop("file", None)
        try:
            if target == "rules":
                if not rules_file:
                    raise SpecError(f"{spec!r}: rules mode needs file=<path to rules JSON>")
                return MockProvider.from_rules_file(rules_file, **params)
            return MockProvider(mode=target, **params)
        except (ValueError, OSError, KeyError) as exc:
            if isinstance(exc, SpecError):
                raise
            raise SpecError(f"{spec!r}: {exc}") from exc
    if kind == "openai":
        params = _typed_params(spec, query, _OPENAI_PARAMS)
        return OpenAICompatibleProvider(model=target, **params)
    raise SpecError(f"unknown provider kind {kind!r} in {spec!r}; use 'mock' or 'openai'")


def _typed_params(spec: str, query: str, allowed: dict[str, type]) -> dict[str, object]:
    params: dict[str, object] = {}
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=bool(query))
    except ValueError as exc:
        raise SpecError(f"{spec!r}: malformed parameters ({exc})") from exc
    for key, value in pairs:
        if key not in allowed:
            raise SpecError(f"{spec!r}: unknown parameter {key!r} (allowed: {', '.join(allowed)})")
        try:
            params[key] = allowed[key](value)
        except ValueError as exc:
            raise SpecError(f"{spec!r}: {key}={value!r} is not a valid {allowed[key].__name__}") from exc
    return params
