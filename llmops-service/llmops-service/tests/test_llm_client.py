from __future__ import annotations

import asyncio

from app.llm.client import MockLLMClient, get_llm_client


def test_mock_client_returns_string():
    client = MockLLMClient()
    result = asyncio.run(client.generate("Hello world"))
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_client_includes_prompt_prefix():
    client = MockLLMClient()
    prompt = "Describe the retinal image findings"
    result = asyncio.run(client.generate(prompt))
    assert prompt[:50] in result


def test_mock_client_with_system_prompt():
    client = MockLLMClient()
    result = asyncio.run(client.generate("user prompt", system_prompt="system context"))
    assert isinstance(result, str)


def test_get_llm_client_returns_mock_for_unknown_provider():
    client = get_llm_client("unknown_provider")
    assert isinstance(client, MockLLMClient)


def test_get_llm_client_returns_mock_for_mock_provider():
    client = get_llm_client("mock")
    assert isinstance(client, MockLLMClient)


def test_get_llm_client_case_insensitive():
    client = get_llm_client("MOCK")
    assert isinstance(client, MockLLMClient)
