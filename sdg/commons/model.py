from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import httpx
from dotenv import dotenv_values, load_dotenv

from sdg.commons.run_log import current_run_log, log_event, write_snapshot

ModelKind = Literal["llm", "embedder", "reranker"]


class EndpointBackedModelRef(TypedDict, total=False):
    endpoint: str
    type: ModelKind
    model: str
    base_url: str
    api_key: str
    api_key_env: str | None
    max_concurrency: int
    max_retries: int
    min_backoff_seconds: float
    max_backoff_seconds: float
    timeout_seconds: float


class DirectModelRef(TypedDict, total=False):
    type: ModelKind
    model: str
    base_url: str
    api_key: str
    api_key_env: str | None
    max_concurrency: int
    max_retries: int
    min_backoff_seconds: float
    max_backoff_seconds: float
    timeout_seconds: float


class ClientSpec(TypedDict):
    type: ModelKind
    model: str
    base_url: str
    api_key: str | None
    api_key_env: str | None
    max_concurrency: int
    max_retries: int
    min_backoff_seconds: float
    max_backoff_seconds: float
    timeout_seconds: float


ModelRef = str | EndpointBackedModelRef | DirectModelRef


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    default_model: str | None = None
    max_concurrency: int = 4
    max_retries: int = 4
    min_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    timeout_seconds: float = 120.0


class EndpointRuntime:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_retries: int,
        min_backoff_seconds: float,
        max_backoff_seconds: float,
        timeout_seconds: float,
    ):
        self.max_concurrency = max(max_concurrency, 1)
        self.max_retries = max_retries
        self.min_backoff_seconds = min_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.lock = threading.Lock()
        self.in_flight = 0
        self.next_allowed_at = 0.0

    def acquire_sync(self) -> None:
        while not self._try_acquire():
            time.sleep(0.01)

    async def acquire_async(self) -> None:
        while not self._try_acquire():
            await asyncio.sleep(0.01)

    def release(self) -> None:
        with self.lock:
            assert self.in_flight > 0, "Endpoint runtime released more times than acquired"
            self.in_flight -= 1

    def wait_sync(self) -> None:
        while True:
            delay = self._current_delay()
            if delay <= 0:
                return
            time.sleep(delay)

    async def wait_async(self) -> None:
        while True:
            delay = self._current_delay()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    def snapshot(self) -> tuple[int, float]:
        with self.lock:
            return self.in_flight, max(self.next_allowed_at - time.monotonic(), 0.0)

    def extend_backoff(self, delay: float) -> None:
        if delay <= 0:
            return
        with self.lock:
            self.next_allowed_at = max(self.next_allowed_at, time.monotonic() + delay)

    def observe_headers(self, headers: httpx.Headers) -> None:
        remaining = _remaining_requests(headers)
        if remaining is None or remaining > 0:
            return

        delay = _header_backoff_seconds(headers)
        if delay is None:
            return
        self.extend_backoff(delay)

    def retry_delay(self, headers: httpx.Headers, *, attempt: int) -> float:
        header_delay = _header_backoff_seconds(headers)
        if header_delay is not None:
            bounded = max(header_delay, self.min_backoff_seconds)
            return min(bounded, self.max_backoff_seconds)

        exponential = self.min_backoff_seconds * (2 ** attempt)
        return min(exponential, self.max_backoff_seconds)

    def _current_delay(self) -> float:
        with self.lock:
            return max(self.next_allowed_at - time.monotonic(), 0.0)

    def _try_acquire(self) -> bool:
        with self.lock:
            if self.in_flight >= self.max_concurrency:
                return False
            self.in_flight += 1
            return True


class EndpointClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        api_key: str | None = None,
        *,
        max_concurrency: int = 4,
        max_retries: int = 4,
        min_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.transport = transport
        self.async_transport = async_transport
        self.runtime = _get_runtime(
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            api_key=self.api_key,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            min_backoff_seconds=min_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            timeout_seconds=timeout_seconds,
        )
        self._sync_client: httpx.Client | None = None
        self._async_clients: dict[int, httpx.AsyncClient] = {}
        self._client_lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.api_key
        if token is None and self.api_key_env:
            token = os.environ.get(self.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(
            runtime=self.runtime,
            client=self._sync_http_client(),
            endpoint=endpoint,
            payload=payload,
            headers=self._headers(),
            request_label=self.model,
        )

    async def _apost_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await _apost_json(
            runtime=self.runtime,
            client=self._async_http_client(),
            endpoint=endpoint,
            payload=payload,
            headers=self._headers(),
            request_label=self.model,
        )

    def _sync_http_client(self) -> httpx.Client:
        client = self._sync_client
        if client is not None:
            return client

        with self._client_lock:
            client = self._sync_client
            if client is None:
                client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.runtime.timeout_seconds,
                    transport=self.transport,
                )
                self._sync_client = client
        return client

    def _async_http_client(self) -> httpx.AsyncClient:
        loop_id = id(asyncio.get_running_loop())
        with self._client_lock:
            client = self._async_clients.get(loop_id)
            if client is None:
                client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.runtime.timeout_seconds,
                    transport=self.async_transport,
                )
                self._async_clients[loop_id] = client
        return client


class LLM(EndpointClient):
    def complete(self, prompt: str, **gen: Any) -> str:
        payload = {"model": self.model, "prompt": prompt, **gen}
        data = self._post_json("/completions", payload)
        return data["choices"][0]["text"]

    async def acomplete(self, prompt: str, **gen: Any) -> str:
        payload = {"model": self.model, "prompt": prompt, **gen}
        data = await self._apost_json("/completions", payload)
        return data["choices"][0]["text"]

    def chat(self, messages: list[dict[str, Any]], **gen: Any) -> Any:
        payload = {"model": self.model, "messages": messages, **gen}
        data = self._post_json("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    async def achat(self, messages: list[dict[str, Any]], **gen: Any) -> Any:
        payload = {"model": self.model, "messages": messages, **gen}
        data = await self._apost_json("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]


class Embedder(EndpointClient):
    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        data = self._post_json("/embeddings", payload)
        return [row["embedding"] for row in data["data"]]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        data = await self._apost_json("/embeddings", payload)
        return [row["embedding"] for row in data["data"]]


class Reranker(EndpointClient):
    def score(self, query: str, docs: list[str]) -> list[float]:
        payload = {"model": self.model, "query": query, "documents": docs}
        data = self._post_json("/rerank", payload)
        if "results" in data:
            return [item.get("relevance_score", item.get("score")) for item in data["results"]]
        return [item["score"] for item in data["data"]]

    async def ascore(self, query: str, docs: list[str]) -> list[float]:
        payload = {"model": self.model, "query": query, "documents": docs}
        data = await self._apost_json("/rerank", payload)
        if "results" in data:
            return [item.get("relevance_score", item.get("score")) for item in data["results"]]
        return [item["score"] for item in data["data"]]


def load_endpoints(env_path: str | Path = ".env") -> dict[str, EndpointSpec]:
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file, override=False)

    file_values = dotenv_values(env_file) if env_file.exists() else {}
    values = {
        **{key: value for key, value in file_values.items() if value is not None},
        **{
            key: value
            for key, value in os.environ.items()
            if key.startswith(("SDG_ENDPOINT__", "OPENAI_"))
        },
    }

    endpoints: dict[str, EndpointSpec] = {}
    if values.get("OPENAI_BASE_URL") or values.get("OPENAI_API_KEY") or values.get("OPENAI_MODEL_NAME"):
        endpoints["openai"] = EndpointSpec(
            name="openai",
            base_url=values.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=values.get("OPENAI_API_KEY"),
            api_key_env="OPENAI_API_KEY",
            default_model=values.get("OPENAI_MODEL_NAME"),
        )

    raw_specs: dict[str, dict[str, str]] = {}
    for key, value in values.items():
        if not key.startswith("SDG_ENDPOINT__"):
            continue
        rest = key[len("SDG_ENDPOINT__") :]
        name, field = rest.split("__", 1)
        spec = raw_specs.setdefault(name.lower(), {})
        spec[field.lower()] = value

    for name, spec in raw_specs.items():
        base_url = spec.get("base_url")
        if not base_url:
            raise ValueError(f"Endpoint {name} is missing BASE_URL")
        endpoints[name] = EndpointSpec(
            name=name,
            base_url=base_url,
            api_key=spec.get("api_key"),
            api_key_env=spec.get("api_key_env"),
            default_model=spec.get("default_model"),
            max_concurrency=_int_value(spec.get("max_concurrency"), 4),
            max_retries=_int_value(spec.get("max_retries"), 4),
            min_backoff_seconds=_float_value(spec.get("min_backoff_seconds"), 1.0),
            max_backoff_seconds=_float_value(spec.get("max_backoff_seconds"), 60.0),
            timeout_seconds=_float_value(spec.get("timeout_seconds"), 120.0),
        )

    return endpoints


def load_clients(
    model_refs: dict[str, ModelRef],
    *,
    env_path: str | Path = ".env",
) -> dict[str, LLM | Embedder | Reranker]:
    endpoints = load_endpoints(env_path)
    return {
        role: resolve_client(ref, role=role, endpoints=endpoints)
        for role, ref in model_refs.items()
    }


def resolve_client(
    model_ref: ModelRef,
    *,
    role: str | None = None,
    endpoints: dict[str, EndpointSpec] | None = None,
    env_path: str | Path = ".env",
) -> LLM | Embedder | Reranker:
    resolved_endpoints = endpoints or load_endpoints(env_path)
    spec = _resolve_client_spec(model_ref, role=role, endpoints=resolved_endpoints)
    return build_client(spec)


def build_client(spec: ClientSpec) -> LLM | Embedder | Reranker:
    common = {
        "model": spec["model"],
        "base_url": spec["base_url"],
        "api_key_env": spec["api_key_env"],
        "api_key": spec["api_key"],
        "max_concurrency": spec["max_concurrency"],
        "max_retries": spec["max_retries"],
        "min_backoff_seconds": spec["min_backoff_seconds"],
        "max_backoff_seconds": spec["max_backoff_seconds"],
        "timeout_seconds": spec["timeout_seconds"],
    }

    match spec["type"]:
        case "llm":
            return LLM(**common)
        case "embedder":
            return Embedder(**common)
        case "reranker":
            return Reranker(**common)
        case _:
            raise AssertionError(f"Unsupported model type: {spec['type']}")


def infer_model_type(role: str | None) -> ModelKind:
    name = (role or "").lower()
    if "embed" in name:
        return "embedder"
    if "rerank" in name:
        return "reranker"
    return "llm"


def _resolve_client_spec(
    model_ref: ModelRef,
    *,
    role: str | None,
    endpoints: dict[str, EndpointSpec],
) -> ClientSpec:
    role_name = role or "unknown"

    if isinstance(model_ref, str):
        if model_ref not in endpoints:
            raise ValueError(f"Unknown endpoint alias: {model_ref}")
        endpoint = endpoints[model_ref]
        assert endpoint.default_model, f"Endpoint alias {model_ref} needs an explicit model"
        return {
            "type": infer_model_type(role),
            "model": endpoint.default_model,
            "base_url": endpoint.base_url,
            "api_key": endpoint.api_key,
            "api_key_env": endpoint.api_key_env,
            "max_concurrency": endpoint.max_concurrency,
            "max_retries": endpoint.max_retries,
            "min_backoff_seconds": endpoint.min_backoff_seconds,
            "max_backoff_seconds": endpoint.max_backoff_seconds,
            "timeout_seconds": endpoint.timeout_seconds,
        }

    ref = dict(model_ref)
    endpoint_name = ref.get("endpoint")
    endpoint = None
    if endpoint_name is not None:
        assert isinstance(endpoint_name, str) and endpoint_name, f"Model reference for role {role_name} has an invalid endpoint"
        endpoint = endpoints[endpoint_name]

    kind = ref.get("type", infer_model_type(role))
    assert kind in {"llm", "embedder", "reranker"}, f"Unsupported model type: {kind}"

    model = ref.get("model")
    if model is None and endpoint is not None:
        model = endpoint.default_model
    assert isinstance(model, str) and model, f"Model reference for role {role_name} is missing model"

    base_url = ref.get("base_url")
    if base_url is None and endpoint is not None:
        base_url = endpoint.base_url
    assert isinstance(base_url, str) and base_url, f"Model reference for role {role_name} is missing base_url"

    api_key = ref.get("api_key")
    if api_key is None and endpoint is not None:
        api_key = endpoint.api_key
    assert api_key is None or isinstance(api_key, str), f"Model reference for role {role_name} has an invalid api_key"

    api_key_env = ref.get("api_key_env")
    if api_key_env is None and endpoint is not None:
        api_key_env = endpoint.api_key_env
    if api_key_env is None:
        api_key_env = "OPENAI_API_KEY"
    assert api_key_env is None or isinstance(api_key_env, str), f"Model reference for role {role_name} has an invalid api_key_env"

    max_concurrency = int(ref.get("max_concurrency", endpoint.max_concurrency if endpoint else 4))
    max_retries = int(ref.get("max_retries", endpoint.max_retries if endpoint else 4))
    min_backoff_seconds = float(ref.get("min_backoff_seconds", endpoint.min_backoff_seconds if endpoint else 1.0))
    max_backoff_seconds = float(ref.get("max_backoff_seconds", endpoint.max_backoff_seconds if endpoint else 60.0))
    timeout_seconds = float(ref.get("timeout_seconds", endpoint.timeout_seconds if endpoint else 120.0))

    return {
        "type": kind,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "max_concurrency": max_concurrency,
        "max_retries": max_retries,
        "min_backoff_seconds": min_backoff_seconds,
        "max_backoff_seconds": max_backoff_seconds,
        "timeout_seconds": timeout_seconds,
    }


def _post_json(
    *,
    runtime: EndpointRuntime,
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_label: str,
) -> dict[str, Any]:
    request_id = _next_request_id()
    started_at = time.monotonic()
    _record_model_started(request_label, endpoint)
    _emit_model_event(
        "request_started",
        request_id=request_id,
        request_label=request_label,
        endpoint=endpoint,
        attempt=0,
        runtime=runtime,
    )
    for attempt in range(runtime.max_retries + 1):
        runtime.wait_sync()
        runtime.acquire_sync()
        try:
            runtime.wait_sync()
            response = client.post(endpoint, json=payload, headers=headers)
            if response.status_code == 429:
                delay = runtime.retry_delay(response.headers, attempt=attempt)
                runtime.extend_backoff(delay)
                _record_model_retry(request_label, endpoint, response.status_code)
                _emit_model_event(
                    "request_retry",
                    request_id=request_id,
                    request_label=request_label,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    delay_seconds=delay,
                    status_code=response.status_code,
                    headers=_response_headers(response.headers),
                    runtime=runtime,
                )
                if attempt == runtime.max_retries:
                    response.raise_for_status()
                continue
            response.raise_for_status()
            runtime.observe_headers(response.headers)
            duration_ms = _duration_ms(started_at)
            _record_model_finished(
                request_label,
                endpoint,
                status_code=response.status_code,
                duration_ms=duration_ms,
                succeeded=True,
            )
            _emit_model_event(
                "request_finished",
                request_id=request_id,
                request_label=request_label,
                endpoint=endpoint,
                attempt=attempt + 1,
                duration_ms=duration_ms,
                status_code=response.status_code,
                headers=_response_headers(response.headers),
                runtime=runtime,
            )
            return response.json()
        except Exception as error:
            if isinstance(error, httpx.HTTPStatusError):
                status_code = error.response.status_code
            else:
                status_code = None
            _record_model_finished(
                request_label,
                endpoint,
                status_code=status_code,
                duration_ms=_duration_ms(started_at),
                succeeded=False,
            )
            _emit_model_event(
                "request_failed",
                request_id=request_id,
                request_label=request_label,
                endpoint=endpoint,
                attempt=attempt + 1,
                duration_ms=_duration_ms(started_at),
                error_type=error.__class__.__name__,
                error_message=str(error),
                status_code=status_code,
                runtime=runtime,
            )
            raise
        finally:
            runtime.release()

    raise RuntimeError("Unreachable")


async def _apost_json(
    *,
    runtime: EndpointRuntime,
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_label: str,
) -> dict[str, Any]:
    request_id = _next_request_id()
    started_at = time.monotonic()
    _record_model_started(request_label, endpoint)
    _emit_model_event(
        "request_started",
        request_id=request_id,
        request_label=request_label,
        endpoint=endpoint,
        attempt=0,
        runtime=runtime,
    )
    for attempt in range(runtime.max_retries + 1):
        await runtime.wait_async()
        await runtime.acquire_async()
        try:
            await runtime.wait_async()
            response = await client.post(endpoint, json=payload, headers=headers)
            if response.status_code == 429:
                delay = runtime.retry_delay(response.headers, attempt=attempt)
                runtime.extend_backoff(delay)
                _record_model_retry(request_label, endpoint, response.status_code)
                _emit_model_event(
                    "request_retry",
                    request_id=request_id,
                    request_label=request_label,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    delay_seconds=delay,
                    status_code=response.status_code,
                    headers=_response_headers(response.headers),
                    runtime=runtime,
                )
                if attempt == runtime.max_retries:
                    response.raise_for_status()
                continue
            response.raise_for_status()
            runtime.observe_headers(response.headers)
            duration_ms = _duration_ms(started_at)
            _record_model_finished(
                request_label,
                endpoint,
                status_code=response.status_code,
                duration_ms=duration_ms,
                succeeded=True,
            )
            _emit_model_event(
                "request_finished",
                request_id=request_id,
                request_label=request_label,
                endpoint=endpoint,
                attempt=attempt + 1,
                duration_ms=duration_ms,
                status_code=response.status_code,
                headers=_response_headers(response.headers),
                runtime=runtime,
            )
            return response.json()
        except Exception as error:
            if isinstance(error, httpx.HTTPStatusError):
                status_code = error.response.status_code
            else:
                status_code = None
            _record_model_finished(
                request_label,
                endpoint,
                status_code=status_code,
                duration_ms=_duration_ms(started_at),
                succeeded=False,
            )
            _emit_model_event(
                "request_failed",
                request_id=request_id,
                request_label=request_label,
                endpoint=endpoint,
                attempt=attempt + 1,
                duration_ms=_duration_ms(started_at),
                error_type=error.__class__.__name__,
                error_message=str(error),
                status_code=status_code,
                runtime=runtime,
            )
            raise
        finally:
            runtime.release()

    raise RuntimeError("Unreachable")


_RUNTIMES: dict[tuple[Any, ...], EndpointRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def _get_runtime(
    *,
    base_url: str,
    api_key_env: str | None,
    api_key: str | None,
    max_concurrency: int,
    max_retries: int,
    min_backoff_seconds: float,
    max_backoff_seconds: float,
    timeout_seconds: float,
) -> EndpointRuntime:
    key = (
        base_url,
        api_key_env,
        api_key,
        max_concurrency,
        max_retries,
        min_backoff_seconds,
        max_backoff_seconds,
        timeout_seconds,
    )
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = EndpointRuntime(
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                min_backoff_seconds=min_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                timeout_seconds=timeout_seconds,
            )
            _RUNTIMES[key] = runtime
        return runtime


def _remaining_requests(headers: httpx.Headers) -> int | None:
    raw = _first_header(
        headers,
        [
            "x-ratelimit-remaining-requests",
            "ratelimit-remaining-requests",
            "x-ratelimit-remaining",
            "ratelimit-remaining",
        ],
    )
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _header_backoff_seconds(headers: httpx.Headers) -> float | None:
    retry_after_ms = _first_header(headers, ["retry-after-ms"])
    if retry_after_ms is not None:
        try:
            return max(float(retry_after_ms) / 1000.0, 0.0)
        except ValueError:
            pass

    retry_after = _first_header(headers, ["retry-after"])
    if retry_after is not None:
        parsed = _parse_retry_after(retry_after)
        if parsed is not None:
            return parsed

    for name in [
        "x-ratelimit-reset-requests",
        "ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "ratelimit-reset-tokens",
        "x-ratelimit-reset",
        "ratelimit-reset",
    ]:
        raw = _first_header(headers, [name])
        if raw is None:
            continue
        parsed = _parse_wait_value(raw)
        if parsed is not None:
            return parsed

    return None


def _parse_retry_after(value: str) -> float | None:
    text = value.strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return max((dt - datetime.now(UTC)).total_seconds(), 0.0)


def _parse_wait_value(value: str) -> float | None:
    text = value.strip().lower()
    if not text:
        return None

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        numeric = float(text)
        if numeric >= 1_000_000_000_000:
            return max((numeric / 1000.0) - time.time(), 0.0)
        if numeric >= 1_000_000_000:
            return max(numeric - time.time(), 0.0)
        return numeric

    matches = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text)
    if not matches:
        return _parse_retry_after(text)

    total = 0.0
    for amount, unit in matches:
        value = float(amount)
        if unit == "ms":
            total += value / 1000.0
        elif unit == "s":
            total += value
        elif unit == "m":
            total += value * 60.0
        elif unit == "h":
            total += value * 3600.0
    return total


def _first_header(headers: httpx.Headers, names: list[str]) -> str | None:
    for name in names:
        value = headers.get(name)
        if value is not None:
            return value
    return None


def _int_value(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _float_value(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


_REQUEST_DEBUG_LOCK = threading.Lock()
_REQUEST_DEBUG_COUNTER = 0
_MODEL_METRICS_LOCK = threading.Lock()
_MODEL_METRICS_SESSION: int | None = None
_MODEL_METRICS = {
    "totals": {
        "requests_started": 0,
        "requests_succeeded": 0,
        "requests_failed": 0,
        "request_retries": 0,
        "rate_limits": 0,
        "total_duration_ms": 0,
        "max_duration_ms": 0,
    },
    "targets": {},
}


def _next_request_id() -> int:
    global _REQUEST_DEBUG_COUNTER
    with _REQUEST_DEBUG_LOCK:
        _REQUEST_DEBUG_COUNTER += 1
        return _REQUEST_DEBUG_COUNTER


def _model_debug(message: str) -> None:
    if os.environ.get("SDG_MODEL_DEBUG") != "1":
        return
    print(f"[model] {message}", flush=True)


def _emit_model_event(
    event: str,
    *,
    request_id: int,
    request_label: str,
    endpoint: str,
    attempt: int,
    runtime: EndpointRuntime,
    **data: Any,
) -> None:
    in_flight, backoff = runtime.snapshot()
    payload = {
        "request_id": request_id,
        "model": request_label,
        "endpoint": endpoint,
        "attempt": attempt,
        "in_flight": in_flight,
        "backoff_seconds": round(backoff, 3),
        **data,
    }
    log_event("model", event, **payload)
    if os.environ.get("SDG_MODEL_DEBUG") == "1":
        ordered = " ".join(f"{key}={value}" for key, value in payload.items())
        _model_debug(f"event={event} {ordered}")


def _record_model_started(request_label: str, endpoint: str) -> None:
    with _MODEL_METRICS_LOCK:
        if not _ensure_model_metrics_session_locked():
            return
        _metrics_target(request_label, endpoint)["requests_started"] += 1
        _MODEL_METRICS["totals"]["requests_started"] += 1
        _write_model_metrics_snapshot_locked()


def _record_model_retry(request_label: str, endpoint: str, status_code: int | None) -> None:
    with _MODEL_METRICS_LOCK:
        if not _ensure_model_metrics_session_locked():
            return
        target = _metrics_target(request_label, endpoint)
        target["request_retries"] += 1
        _MODEL_METRICS["totals"]["request_retries"] += 1
        if status_code == 429:
            target["rate_limits"] += 1
            _MODEL_METRICS["totals"]["rate_limits"] += 1
        _write_model_metrics_snapshot_locked()


def _record_model_finished(
    request_label: str,
    endpoint: str,
    *,
    status_code: int | None,
    duration_ms: int,
    succeeded: bool,
) -> None:
    with _MODEL_METRICS_LOCK:
        if not _ensure_model_metrics_session_locked():
            return
        target = _metrics_target(request_label, endpoint)
        if succeeded:
            target["requests_succeeded"] += 1
            _MODEL_METRICS["totals"]["requests_succeeded"] += 1
        else:
            target["requests_failed"] += 1
            _MODEL_METRICS["totals"]["requests_failed"] += 1
        target["total_duration_ms"] += duration_ms
        target["max_duration_ms"] = max(target["max_duration_ms"], duration_ms)
        target["last_status_code"] = status_code
        _MODEL_METRICS["totals"]["total_duration_ms"] += duration_ms
        _MODEL_METRICS["totals"]["max_duration_ms"] = max(
            _MODEL_METRICS["totals"]["max_duration_ms"],
            duration_ms,
        )
        _write_model_metrics_snapshot_locked()


def _metrics_target(request_label: str, endpoint: str) -> dict[str, Any]:
    key = f"{request_label} {endpoint}"
    targets = _MODEL_METRICS["targets"]
    target = targets.get(key)
    if target is not None:
        return target
    target = {
        "model": request_label,
        "endpoint": endpoint,
        "requests_started": 0,
        "requests_succeeded": 0,
        "requests_failed": 0,
        "request_retries": 0,
        "rate_limits": 0,
        "total_duration_ms": 0,
        "max_duration_ms": 0,
        "last_status_code": None,
    }
    targets[key] = target
    return target


def _empty_model_metrics() -> dict[str, Any]:
    return {
        "totals": {
            "requests_started": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "request_retries": 0,
            "rate_limits": 0,
            "total_duration_ms": 0,
            "max_duration_ms": 0,
        },
        "targets": {},
    }


def _ensure_model_metrics_session_locked() -> bool:
    global _MODEL_METRICS, _MODEL_METRICS_SESSION
    logger = current_run_log()
    if logger is None:
        return False
    session = id(logger)
    if _MODEL_METRICS_SESSION == session:
        return True
    _MODEL_METRICS_SESSION = session
    _MODEL_METRICS = _empty_model_metrics()
    return True


def _write_model_metrics_snapshot_locked() -> None:
    totals = dict(_MODEL_METRICS["totals"])
    targets: dict[str, dict[str, Any]] = {}
    for key, target in _MODEL_METRICS["targets"].items():
        targets[key] = {
            **target,
            "avg_duration_ms": _average_duration(
                target["total_duration_ms"],
                target["requests_succeeded"] + target["requests_failed"],
            ),
        }
    snapshot = {
        "totals": {
            **totals,
            "avg_duration_ms": _average_duration(
                totals["total_duration_ms"],
                totals["requests_succeeded"] + totals["requests_failed"],
            ),
        },
        "targets": targets,
    }
    write_snapshot("model_metrics.json", snapshot, min_interval_seconds=1.0)


def _average_duration(total_duration_ms: int, completed_requests: int) -> float:
    if completed_requests == 0:
        return 0.0
    return round(total_duration_ms / completed_requests, 2)


def _duration_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _response_headers(headers: httpx.Headers) -> dict[str, str | None]:
    return {
        "retry_after": _first_header(headers, ["retry-after-ms", "retry-after"]),
        "remaining_requests": _first_header(
            headers,
            ["x-ratelimit-remaining-requests", "ratelimit-remaining-requests"],
        ),
        "reset_requests": _first_header(
            headers,
            ["x-ratelimit-reset-requests", "ratelimit-reset-requests"],
        ),
    }
