from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from api.main import warmup_llm


@pytest.mark.asyncio
async def test_warmup_skipped_for_non_ollama_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch("llm.factory.get_llm_service") as mock_get_llm:
        await warmup_llm()
    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_calls_llm_invoke_for_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = "OK"

    with patch("llm.factory.get_llm_service", return_value=fake_llm), \
         patch("api.main.run_in_threadpool", new=AsyncMock(return_value="OK")) as mock_pool:
        await warmup_llm()

    mock_pool.assert_called_once()


@pytest.mark.asyncio
async def test_warmup_retries_then_gives_up_without_crashing(monkeypatch):
    """Regression-style test: even if Ollama is never reachable during
    startup (e.g. genuinely down, not just a startup race), the app must
    still finish starting -- warmup failing entirely should not crash
    the process or block it forever."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    fake_llm = MagicMock()

    with patch("llm.factory.get_llm_service", return_value=fake_llm), \
         patch("api.main.run_in_threadpool", new=AsyncMock(side_effect=ConnectionError("refused"))), \
         patch("asyncio.sleep", new=AsyncMock()):  # don't actually wait in the test
        await warmup_llm()  # must return normally, not raise


@pytest.mark.asyncio
async def test_warmup_succeeds_after_initial_failures(monkeypatch):
    """Matches the real observed production pattern: the first attempt(s)
    can fail/be slow while Ollama is starting or loading the model, and a
    later attempt succeeds once it's ready."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    fake_llm = MagicMock()
    call_count = {"n": 0}

    async def flaky_invoke(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("not ready yet")
        return "OK"

    with patch("llm.factory.get_llm_service", return_value=fake_llm), \
         patch("api.main.run_in_threadpool", new=flaky_invoke), \
         patch("asyncio.sleep", new=AsyncMock()):
        await warmup_llm()

    assert call_count["n"] == 3
