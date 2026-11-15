import pytest

from llm_lib import LLM, LLMError, Settings


def test_aliases_and_presets():
    assert Settings(provider="LM Studio").provider == "lmstudio"
    assert Settings(provider="claude").model == "claude-opus-5-5"
    assert Settings(provider="llama.cpp").resolved_base_url({}) == "http://127.0.0.1:8080/v1"
    with pytest.raises(ValueError, match="unknown LLM provider"):
        Settings(provider="gpt-lol")


def test_from_env_reads_each_apps_legacy_names():
    # Just-Finish-It
    jfi = Settings.from_env(
        env={
            "LLM_BACKEND": "lmstudio",
            "MODEL": "qwen",
            "TEMPERATURE": "0.7",
            "LLM_REQUEST_TIMEOUT": "120",
            "OPENAI_URL": "http://box:1234/v1",
        }
    )
    assert (jfi.provider, jfi.model, jfi.temperature, jfi.timeout) == ("lmstudio", "qwen", 0.7, 120.0)
    assert jfi.resolved_base_url({}) == "http://box:1234/v1"
    # STORY_* apps (Text-Speech-Voice, Stonks)
    story = Settings.from_env(
        env={
            "STORY_PROVIDER": "ollama",
            "STORY_MODEL": "m",
            "STORY_CONTEXT": "8192",
            "STORY_NO_THINK": "1",
            "STORY_LOAD_ARGS": "--gpu max",
        }
    )
    assert (story.provider, story.context, story.no_think, story.load_args) == ("ollama", 8192, True, ("--gpu", "max"))


def test_prefix_chain_falls_back_in_order():
    env = {
        "MODEL": "shared",
        "PLANNER_MODEL": "planner",
        "ARCHITECT_TEMPERATURE": "0.1",
        "TEMPERATURE": "0.9",
        "LLM_PROVIDER": "anthropic",
        "PLANNER_ANTHROPIC_API_KEY": "sk-planner",
        "ANTHROPIC_API_KEY": "sk-shared",
    }
    s = Settings.from_env(("ARCHITECT", "PLANNER"), env=env)
    assert (s.model, s.temperature, s.api_key) == ("planner", 0.1, "sk-planner")
    assert Settings.from_env(env=env).model == "shared"
    assert Settings.from_env(env=env).api_key == "sk-shared"


def test_env_file_is_under_the_process_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\nexport LLM_PROVIDER=ollama\nLLM_MODEL="from-file"\nLLM_CONTEXT=4096 # inline\n')
    s = Settings.from_env(env={"LLM_ENV_FILE": str(path), "LLM_MODEL": "from-env"})
    assert (s.provider, s.model, s.context) == ("ollama", "from-env", 4096)


def test_bad_number_names_the_variable():
    with pytest.raises(ValueError, match="PLANNER_TEMPERATURE"):
        Settings.from_env("PLANNER", env={"PLANNER_TEMPERATURE": "warm"})


def test_ollama_host_without_scheme():
    assert Settings(provider="ollama").resolved_base_url({"OLLAMA_HOST": "0.0.0.0:11434"}) == "http://0.0.0.0:11434"


def test_openai_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        LLM(Settings(provider="openai", model="gpt"))
