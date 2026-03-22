from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest

import sdg.commons.model as model_module
from sdg.commons.model import (
    LLM,
    Embedder,
    Reranker,
    load_clients,
    load_endpoints,
    resolve_client,
)
from sdg.commons.run_log import activate_run_log


def test_named_endpoints_support_multiple_models(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "SDG_ENDPOINT__OPENAI__BASE_URL=https://api.openai.com/v1",
                "SDG_ENDPOINT__OPENAI__API_KEY=test-openai-key",
                "SDG_ENDPOINT__OPENAI__DEFAULT_MODEL=gpt-4.1-mini",
                "SDG_ENDPOINT__LOCAL__BASE_URL=http://localhost:8000/v1",
                "SDG_ENDPOINT__LOCAL__API_KEY=dummy",
            ]
        )
    )

    endpoints = load_endpoints(env_path)
    assert endpoints["openai"].base_url == "https://api.openai.com/v1"
    assert endpoints["openai"].default_model == "gpt-4.1-mini"
    assert endpoints["openai"].max_concurrency == 4
    assert endpoints["local"].base_url == "http://localhost:8000/v1"

    models = load_clients(
        {
            "query_teacher": {"endpoint": "openai", "model": "gpt-4.1-mini"},
            "answer_teacher": {"endpoint": "openai", "model": "gpt-4.1"},
            "embedder": {"endpoint": "local", "type": "embedder", "model": "bge-small"},
            "reranker": {"endpoint": "local", "type": "reranker", "model": "bge-reranker-v2"},
        },
        env_path=env_path,
    )

    assert isinstance(models["query_teacher"], LLM)
    assert isinstance(models["answer_teacher"], LLM)
    assert isinstance(models["embedder"], Embedder)
    assert isinstance(models["reranker"], Reranker)

    assert models["query_teacher"].model == "gpt-4.1-mini"
    assert models["answer_teacher"].model == "gpt-4.1"
    assert models["embedder"].base_url == "http://localhost:8000/v1"


def test_legacy_openai_env_still_resolves(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_MODEL_NAME=gpt-4.1-mini",
            ]
        )
    )

    endpoints = load_endpoints(env_path)
    assert endpoints["openai"].default_model == "gpt-4.1-mini"

    client = resolve_client("openai", role="query_teacher", env_path=env_path)
    assert isinstance(client, LLM)
    assert client.model == "gpt-4.1-mini"


def test_sync_chat_retries_after_rate_limit() -> None:
    calls = {"count": 0}

    class RetryTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after-ms": "0"},
                    json={"error": "rate limited"},
                    request=request,
                )

            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        transport=RetryTransport(),
    )

    content = llm.chat([{"role": "user", "content": "hello"}], temperature=0.0)
    assert content == '{"target":"ok"}'
    assert calls["count"] == 2


def test_async_chat_retries_after_rate_limit() -> None:
    calls = {"count": 0}

    class RetryAsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after": "0"},
                    json={"error": "rate limited"},
                    request=request,
                )

            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        async_transport=RetryAsyncTransport(),
    )

    content = asyncio.run(llm.achat([{"role": "user", "content": "hello"}], temperature=0.0))
    assert content == '{"target":"ok"}'
    assert calls["count"] == 2


def test_async_chat_respects_shared_semaphore_limit() -> None:
    class ConcurrencyAsyncTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.current = 0
            self.max_seen = 0
            self.lock = asyncio.Lock()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            async with self.lock:
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)

            await asyncio.sleep(0.02)

            async with self.lock:
                self.current -= 1

            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    transport = ConcurrencyAsyncTransport()
    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_concurrency=1,
        async_transport=transport,
    )

    async def run_requests() -> None:
        await asyncio.gather(
            llm.achat([{"role": "user", "content": "one"}], temperature=0.0),
            llm.achat([{"role": "user", "content": "two"}], temperature=0.0),
        )

    asyncio.run(run_requests())
    assert transport.max_seen == 1


def test_sync_and_async_chat_share_one_concurrency_limit() -> None:
    class SharedState:
        def __init__(self) -> None:
            self.current = 0
            self.max_seen = 0
            self.lock = threading.Lock()

        def enter(self) -> None:
            with self.lock:
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)

        def leave(self) -> None:
            with self.lock:
                self.current -= 1

    class SyncTransport(httpx.BaseTransport):
        def __init__(self, state: SharedState) -> None:
            self.state = state

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            self.state.enter()
            time.sleep(0.05)
            self.state.leave()
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    class AsyncTransport(httpx.AsyncBaseTransport):
        def __init__(self, state: SharedState) -> None:
            self.state = state

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.state.enter()
            await asyncio.sleep(0.05)
            self.state.leave()
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    state = SharedState()
    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_concurrency=1,
        transport=SyncTransport(state),
        async_transport=AsyncTransport(state),
    )

    async def run_requests() -> None:
        await asyncio.gather(
            asyncio.to_thread(llm.chat, [{"role": "user", "content": "sync"}], temperature=0.0),
            llm.achat([{"role": "user", "content": "async"}], temperature=0.0),
        )

    asyncio.run(run_requests())
    assert state.max_seen == 1


def test_sync_chat_reuses_httpx_client(monkeypatch) -> None:
    created = {"count": 0}

    class ReusedClient:
        def __init__(self, **kwargs) -> None:
            created["count"] += 1

        def post(self, endpoint: str, json: dict[str, object], headers: dict[str, str]) -> httpx.Response:
            request = httpx.Request("POST", f"https://example.com{endpoint}")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    monkeypatch.setattr(model_module.httpx, "Client", ReusedClient)

    llm = LLM(model="test-model", base_url="https://example.com/v1")
    llm.chat([{"role": "user", "content": "one"}], temperature=0.0)
    llm.chat([{"role": "user", "content": "two"}], temperature=0.0)

    assert created["count"] == 1


def test_async_chat_reuses_httpx_client(monkeypatch) -> None:
    created = {"count": 0}

    class ReusedAsyncClient:
        def __init__(self, **kwargs) -> None:
            created["count"] += 1

        async def post(self, endpoint: str, json: dict[str, object], headers: dict[str, str]) -> httpx.Response:
            request = httpx.Request("POST", f"https://example.com{endpoint}")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    monkeypatch.setattr(model_module.httpx, "AsyncClient", ReusedAsyncClient)

    llm = LLM(model="test-model", base_url="https://example.com/v1")

    async def run_requests() -> None:
        await llm.achat([{"role": "user", "content": "one"}], temperature=0.0)
        await llm.achat([{"role": "user", "content": "two"}], temperature=0.0)

    asyncio.run(run_requests())
    assert created["count"] == 1


def test_model_logging_writes_metrics_and_events(tmp_path) -> None:
    calls = {"count": 0}

    class RetryTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after-ms": "0"},
                    json={"error": "rate limited"},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        transport=RetryTransport(),
    )

    with activate_run_log(run_dir):
        content = llm.chat([{"role": "user", "content": "hello"}], temperature=0.0)

    assert content == '{"target":"ok"}'

    metrics = json.loads((run_dir / "outputs" / "model_metrics.json").read_text())
    assert metrics["totals"]["requests_started"] == 1
    assert metrics["totals"]["requests_succeeded"] == 1
    assert metrics["totals"]["requests_failed"] == 0
    assert metrics["totals"]["request_retries"] == 1
    assert metrics["totals"]["rate_limits"] == 1
    assert metrics["targets"]["test-model /chat/completions"]["last_status_code"] == 200

    events = [
        json.loads(line)
        for line in (run_dir / "logs" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event_names = [row["event"] for row in events if row["component"] == "model"]
    assert "request_started" in event_names
    assert "request_retry" in event_names
    assert "request_finished" in event_names


def test_async_model_logging_records_cancelled_requests(tmp_path) -> None:
    class CancelledAsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise asyncio.CancelledError()

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        async_transport=CancelledAsyncTransport(),
    )

    async def run_request() -> None:
        with activate_run_log(run_dir):
            await llm.achat([{"role": "user", "content": "hello"}], temperature=0.0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_request())

    metrics = json.loads((run_dir / "outputs" / "model_metrics.json").read_text())
    assert metrics["totals"]["requests_started"] == 1
    assert metrics["totals"]["requests_succeeded"] == 0
    assert metrics["totals"]["requests_failed"] == 1

    events = [
        json.loads(line)
        for line in (run_dir / "logs" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event_names = [row["event"] for row in events if row["component"] == "model"]
    assert "request_started" in event_names
    assert "request_cancelled" in event_names
