import json

import httpx
import pytest

from olist_insights.llm_client import (
    LLMError,
    LLMOutputError,
    OpenAICompatibleClient,
    ResponseCache,
    backoff_delay,
    make_client,
)

MESSAGES = [{"role": "user", "content": "Say hi in JSON"}]


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


class FakeServer:
    """Serveur HTTP simulé : renvoie les réponses prévues, dans l'ordre, et note les requêtes."""

    def __init__(self, responses: list[httpx.Response]):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def _client(server: FakeServer, cache=None, max_retries=3):
    sleeps: list[float] = []
    http = httpx.Client(base_url="https://fake.api/v1", transport=httpx.MockTransport(server),
                        headers={"Authorization": "Bearer test-key"})
    client = OpenAICompatibleClient("fake", "fake-model", "test-key", "https://fake.api/v1", cache=cache,
                                    min_interval_s=0, max_retries=max_retries, http=http, sleep=sleeps.append)
    return client, sleeps


def test_successful_call_sends_openai_format():
    server = FakeServer([_ok('{"hi": true}')])
    client, _ = _client(server)
    assert client.complete(MESSAGES) == '{"hi": true}'
    request = server.requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "fake-model" and body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}


def test_cache_avoids_second_call(tmp_path):
    cache = ResponseCache(tmp_path / "llm.duckdb")
    server = FakeServer([_ok('{"a": 1}')])
    client, _ = _client(server, cache=cache)
    assert client.complete(MESSAGES) == client.complete(MESSAGES) == '{"a": 1}'
    assert len(server.requests) == 1 and client.n_cache_hits == 1
    cache.close()


def test_retries_on_429_and_respects_retry_after():
    server = FakeServer([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(503), _ok("{}")])
    client, sleeps = _client(server)
    assert client.complete(MESSAGES) == "{}"
    assert len(server.requests) == 3
    assert sleeps[0] == 7.0                 # l'API a dit combien attendre : on obéit
    assert 1.0 <= sleeps[1] <= 4.0          # puis backoff exponentiel (2^1 s avec jitter)


def test_no_retry_on_auth_error():
    server = FakeServer([httpx.Response(401, text="invalid api key")])
    client, sleeps = _client(server)
    with pytest.raises(LLMError, match="401"):
        client.complete(MESSAGES)
    assert len(server.requests) == 1 and sleeps == []


def test_gives_up_after_max_retries():
    server = FakeServer([httpx.Response(429)] * 3)
    client, _ = _client(server, max_retries=2)
    with pytest.raises(LLMError, match="3 essais"):
        client.complete(MESSAGES)


def test_provider_json_rejection_raises_output_error():
    server = FakeServer([httpx.Response(400, json={"error": {"code": "json_validate_failed"}})])
    client, _ = _client(server)
    with pytest.raises(LLMOutputError):
        client.complete(MESSAGES)


def test_zero_quota_fails_fast():
    # compte sans accès API : 429 avec une limite de 0 requête/minute -> pas de nouvel essai
    server = FakeServer([httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0"})])
    client, sleeps = _client(server)
    with pytest.raises(LLMError, match="0 requête/minute"):
        client.complete(MESSAGES)
    assert len(server.requests) == 1 and sleeps == []


def test_backoff_grows_and_is_capped():
    assert 1.0 <= backoff_delay(0, None) <= 2.0
    assert 8.0 <= backoff_delay(3, None) <= 16.0
    assert backoff_delay(10, None) <= 60.0
    assert backoff_delay(0, "120") == 60.0  # Retry-After plafonné lui aussi


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(LLMError, match="MISTRAL_API_KEY"):
        make_client("mistral")


def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    client = make_client()
    assert (client.provider, client.model) == ("groq", "openai/gpt-oss-120b")


def test_unknown_provider(monkeypatch):
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        make_client("openai")


def test_provider_and_model_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    client = make_client("groq")
    assert (client.provider, client.model) == ("groq", "some-model")
