"""A provider for any HTTP endpoint that speaks the OpenAI-style chat-completions protocol.

Many hosted and local model servers expose ``POST {base_url}/chat/completions``.
This adapter uses only the standard library. The API key is read from an
environment variable at call time and is never stored in result files.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from evalbench.datasets import Example
from evalbench.providers.base import Completion, ProviderError, count_tokens

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class OpenAICompatibleProvider:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 30.0
    system_prompt: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or self.model
        self.base_url = self.base_url.rstrip("/")

    def complete(self, prompt: str, example: Example) -> Completion:
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"{self.name}: HTTP {exc.code} from {self.base_url}", retryable=exc.code in RETRYABLE_STATUS
            ) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderError(f"{self.name}: request failed ({reason})", retryable=True) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name}: response was not JSON") from exc
        latency = (time.perf_counter() - started) * 1000

        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: response has no choices[0].message.content") from exc
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            latency_ms=round(latency, 3),
            prompt_tokens=int(usage.get("prompt_tokens", count_tokens(prompt))),
            completion_tokens=int(usage.get("completion_tokens", count_tokens(text))),
        )
