"""Free/subscription LLM providers added by the fork.

Covers opencode-go (Go sub incl. Grok + free models), xAI direct API and the
generic custom OpenAI-compatible endpoint. No network access in these tests.
"""

import pytest

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from src import ai, config
from src.ai import (
    OPENCODE_RESPONSES_MODELS,
    SUPPORTED_LLM_PROVIDERS,
    _build_transcript_model,
    _get_missing_llm_key_error,
    _split_llm_name,
)
from src.config import Config


def _clear_llm_env(monkeypatch):
    for key in (
        "LLM",
        "RERANK_LLM",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_FALLBACK_MODEL",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY",
        "OPENCODE_API_KEY",
        "OPENCODE_GO_API_KEY",
        "OPENCODE_BASE_URL",
        "OPENCODE_SESSION_ID",
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "CUSTOM_BASE_URL",
        "CUSTOM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_supported_providers_include_free_paths():
    assert {"opencode", "opencode-go", "xai", "custom"} <= SUPPORTED_LLM_PROVIDERS


def test_split_llm_name_new_providers():
    assert _split_llm_name("opencode-go:cat-jam-free") == (
        "opencode-go",
        "cat-jam-free",
    )
    assert _split_llm_name("xai:grok-4") == ("xai", "grok-4")
    assert _split_llm_name("custom:qwen3-8b") == ("custom", "qwen3-8b")


def test_missing_key_errors_point_at_right_env(monkeypatch):
    _clear_llm_env(monkeypatch)
    runtime_config = Config()

    assert "OPENCODE_API_KEY" in (
        _get_missing_llm_key_error("opencode-go:cat-jam-free", runtime_config) or ""
    )
    assert "XAI_API_KEY" in (
        _get_missing_llm_key_error("xai:grok-4", runtime_config) or ""
    )
    assert "CUSTOM_BASE_URL" in (
        _get_missing_llm_key_error("custom:mymodel", runtime_config) or ""
    )


def test_keys_clear_missing_errors(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-test")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:4000/v1")
    runtime_config = Config()

    assert _get_missing_llm_key_error("opencode-go:cat-jam-free", runtime_config) is None
    assert _get_missing_llm_key_error("xai:grok-4", runtime_config) is None
    assert _get_missing_llm_key_error("custom:mymodel", runtime_config) is None


def test_build_chat_model_for_free_opencode_model(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-test")
    model = _build_transcript_model(Config(), "opencode-go:cat-jam-free")

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "cat-jam-free"
    assert "opencode.ai" in model.base_url


def test_build_responses_model_for_grok_via_go(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test")
    assert "grok-4.7" in OPENCODE_RESPONSES_MODELS

    model = _build_transcript_model(Config(), "opencode-go:grok-4.7")

    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "grok-4.7"


def test_build_xai_model_uses_xai_endpoint(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    model = _build_transcript_model(Config(), "xai:grok-4")

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "grok-4"
    assert model.base_url.startswith("https://api.x.ai/v1")


def test_build_custom_model_uses_custom_endpoint(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:4000/v1")
    model = _build_transcript_model(Config(), "custom:qwen3-8b")

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "qwen3-8b"
    assert model.base_url.startswith("http://localhost:4000/v1")


def test_transcription_defaults_to_free_whisper(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("TRANSCRIPTION_PROVIDER", raising=False)

    assert Config().transcription_provider == "whisper"


def test_default_llm_prefers_opencode_go_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("LLM", raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test")

    assert Config().llm == "opencode-go:cat-jam-free"


def test_opencode_session_id_is_stable_but_overridable(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("OPENCODE_SESSION_ID", raising=False)
    assert Config().resolve_opencode_session_id() == Config().resolve_opencode_session_id()

    monkeypatch.setenv("OPENCODE_SESSION_ID", "my-session")
    assert Config().resolve_opencode_session_id() == "my-session"


def test_opencode_provider_sends_session_header(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-test")
    monkeypatch.setenv("OPENCODE_SESSION_ID", "my-session")
    model = _build_transcript_model(Config(), "opencode-go:cat-jam-free")

    headers = dict(model.client._client.headers)
    assert headers.get("x-opencode-session") == "my-session"


@pytest.mark.skipif(
    not hasattr(config, "DEFAULT_OPENCODE_BASE_URL"),
    reason="fork config constants missing",
)
def test_opencode_base_url_default():
    assert "opencode.ai" in config.DEFAULT_OPENCODE_BASE_URL
