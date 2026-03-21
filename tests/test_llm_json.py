from __future__ import annotations

import asyncio

from sdg.packs.pleias_synth.llm_json import achat_json, chat_json, parse_json_response


def test_parse_json_response_handles_fenced_json() -> None:
    parsed = parse_json_response('```json\n{"target":"ok"}\n```')
    assert parsed == {"target": "ok"}


def test_parse_json_response_extracts_embedded_object() -> None:
    parsed = parse_json_response('Result:\n{"target":"ok"}\nThanks.')
    assert parsed == {"target": "ok"}


def test_parse_json_response_extracts_first_balanced_object() -> None:
    parsed = parse_json_response('noise {"target":"ok"}\n{"ignored":true}')
    assert parsed == {"target": "ok"}


def test_chat_json_repairs_malformed_response() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            self.calls += 1
            if self.calls == 1:
                return "not json"

            assert temperature == 0.0
            assert messages[0]["role"] == "system"
            assert "Repair malformed JSON" in messages[0]["content"]
            return '{"target":"ok"}'

    llm = FakeLLM()

    parsed = chat_json(llm, [{"role": "user", "content": "hello"}], temperature=0.4)

    assert parsed == {"target": "ok"}
    assert llm.calls == 2


def test_chat_json_repairs_non_object_response() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            self.calls += 1
            if self.calls == 1:
                return '["not","an object"]'

            assert temperature == 0.0
            return '{"target":"ok"}'

    llm = FakeLLM()

    parsed = chat_json(llm, [{"role": "user", "content": "hello"}], temperature=0.4)

    assert parsed == {"target": "ok"}
    assert llm.calls == 2


def test_chat_json_retries_repair_when_first_repair_is_still_invalid() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            self.calls += 1
            if self.calls == 1:
                return "not json"
            if self.calls == 2:
                return '["still","bad"]'
            return '{"target":"ok"}'

    llm = FakeLLM()

    parsed = chat_json(llm, [{"role": "user", "content": "hello"}], temperature=0.4)

    assert parsed == {"target": "ok"}
    assert llm.calls == 3


def test_achat_json_repairs_malformed_response() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def achat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            self.calls += 1
            if self.calls == 1:
                return "not json"

            assert temperature == 0.0
            assert messages[0]["role"] == "system"
            assert "Repair malformed JSON" in messages[0]["content"]
            return '{"target":"ok"}'

    llm = FakeLLM()

    parsed = asyncio.run(achat_json(llm, [{"role": "user", "content": "hello"}], temperature=0.4))

    assert parsed == {"target": "ok"}
    assert llm.calls == 2


def test_achat_json_retries_repair_when_first_repair_is_still_invalid() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def achat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            self.calls += 1
            if self.calls == 1:
                return "not json"
            if self.calls == 2:
                return '["still","bad"]'
            return '{"target":"ok"}'

    llm = FakeLLM()

    parsed = asyncio.run(achat_json(llm, [{"role": "user", "content": "hello"}], temperature=0.4))

    assert parsed == {"target": "ok"}
    assert llm.calls == 3
