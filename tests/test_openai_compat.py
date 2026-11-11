"""The HTTP provider is tested against a local stub server, so no network or API key is needed."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evalbench.datasets import Example
from evalbench.providers import OpenAICompatibleProvider, ProviderError

EXAMPLE = Example(id="x", prompt="What is 2 + 2?", references=("4",))


@pytest.fixture
def stub_server():
    """Serve a scripted response; record each request's path, headers and JSON body."""
    state = {"status": 200, "body": None, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            state["requests"].append(
                {"path": self.path, "headers": dict(self.headers), "json": json.loads(self.rfile.read(length))}
            )
            body = state["body"]
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()
    server.server_close()


def chat_response(text, usage=None):
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage:
        body["usage"] = usage
    return body


def test_successful_call_sends_prompt_and_reads_usage(stub_server, monkeypatch):
    monkeypatch.setenv("EVALBENCH_TEST_KEY", "test-token")
    stub_server["body"] = chat_response("4", {"prompt_tokens": 12, "completion_tokens": 1})
    provider = OpenAICompatibleProvider(
        model="tiny", base_url=stub_server["url"], api_key_env="EVALBENCH_TEST_KEY", system_prompt="Be brief."
    )

    completion = provider.complete(EXAMPLE.prompt, EXAMPLE)

    assert completion.text == "4"
    assert (completion.prompt_tokens, completion.completion_tokens) == (12, 1)
    request = stub_server["requests"][0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-token"
    assert request["json"]["model"] == "tiny"
    assert request["json"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    assert set(request["json"]) == {"model", "messages", "temperature", "max_tokens"}  # no reference leaks


def test_no_authorization_header_without_a_key(stub_server, monkeypatch):
    monkeypatch.delenv("EVALBENCH_ABSENT_KEY", raising=False)
    stub_server["body"] = chat_response("ok")
    OpenAICompatibleProvider(model="m", base_url=stub_server["url"], api_key_env="EVALBENCH_ABSENT_KEY").complete(
        EXAMPLE.prompt, EXAMPLE
    )
    assert "Authorization" not in stub_server["requests"][0]["headers"]


def test_missing_usage_falls_back_to_word_counts(stub_server):
    stub_server["body"] = chat_response("four is the answer")
    completion = OpenAICompatibleProvider(model="m", base_url=stub_server["url"]).complete(EXAMPLE.prompt, EXAMPLE)
    assert (completion.prompt_tokens, completion.completion_tokens) == (5, 4)


@pytest.mark.parametrize("status, retryable", [(429, True), (503, True), (400, False), (401, False)])
def test_http_errors_are_classified(stub_server, status, retryable):
    stub_server["status"], stub_server["body"] = status, {"error": "nope"}
    with pytest.raises(ProviderError, match=f"HTTP {status}") as info:
        OpenAICompatibleProvider(model="m", base_url=stub_server["url"]).complete(EXAMPLE.prompt, EXAMPLE)
    assert info.value.retryable is retryable


@pytest.mark.parametrize("body, message", [(b"not json", "not JSON"), ({"choices": []}, "no choices")])
def test_malformed_responses_are_errors(stub_server, body, message):
    stub_server["body"] = body
    with pytest.raises(ProviderError, match=message) as info:
        OpenAICompatibleProvider(model="m", base_url=stub_server["url"]).complete(EXAMPLE.prompt, EXAMPLE)
    assert not info.value.retryable


def test_connection_failure_is_retryable():
    with socket.socket() as sock:  # find a port with nothing listening
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    provider = OpenAICompatibleProvider(model="m", base_url=f"http://127.0.0.1:{port}/v1", timeout=2)
    with pytest.raises(ProviderError, match="request failed") as info:
        provider.complete(EXAMPLE.prompt, EXAMPLE)
    assert info.value.retryable
