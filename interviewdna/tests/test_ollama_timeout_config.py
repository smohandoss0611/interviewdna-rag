from llm.ollama_service import OllamaLLMService


def test_ollama_timeout_defaults_to_120_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_REQUEST_TIMEOUT", raising=False)
    svc = OllamaLLMService()
    assert svc.request_timeout == 120


def test_ollama_timeout_configurable_via_env_var(monkeypatch):
    """Regression test: this was previously hardcoded with no way to raise
    it without editing code -- a real problem for CPU-only inference under
    constrained resources that genuinely needs longer than 120s."""
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "300")
    svc = OllamaLLMService()
    assert svc.request_timeout == 300


def test_ollama_timeout_explicit_arg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "300")
    svc = OllamaLLMService(request_timeout=60)
    assert svc.request_timeout == 60
