# Moving each app onto llm-lib

Existing `.env` files keep working: `Settings.from_env()` reads every app's current variable names.
Each section maps the app's own LLM code to what replaces it.

For each app:

1. Build the library and add it: `cd LLM-lib && uv run build`, then in the app
   `uv add /Volumes/apple_local/git/LLM-lib/dist/llm_lib-0.1.0.tar.gz`. Add the `[anthropic]` extra
   if the app talks to Claude, or use `uv add --editable /Volumes/apple_local/git/LLM-lib` while
   changing both at once.
2. Replace the calls per the table below, and delete the app's own `llm` module.
3. Run the app's tests. Its `.env` doesn't need to change.

## Just-Finish-It (`src/JFI/llm/`)

| Today | llm-lib |
|---|---|
| `base_llm_stream.phase_env(prefix, key)` / `env_prefixes` | `Settings.from_env(prefix)`, or `settings.EnvLookup(os.environ, prefix).get(key)` |
| `backend_select.make_llm_stream(prefix)` | `LLM(Settings.from_env(prefix))` |
| `OpenAICompatableStream`, `AnthropicStream` | `LLM.stream(messages, tools=tools)` |
| `anthropic_stream._translate_*` | `llm_lib.providers.anthropic.translate_messages` / `translate_tools` |
| `retry.is_retryable`, `format_error`, `interruptible_sleep` | `llm_lib.is_retryable`, `str(exc)` (HTML pages are already described), `llm_lib.interruptible_sleep(s, console.should_stop)` |
| `parallel.server_parallel` / `effective_parallel` | `llm_lib.servers.capacity.server_parallel(settings, window=CONTEXT_SIZE)` / `effective_parallel(PARALLEL_LLM, settings, window=...)` (also reads colibri's KV slots) |
| `lmstudio_control.LMStudioControl.models_unloaded()` | `LMStudio().models_unloaded()` |

JFI's default model is colibri's (`glm-5.3-flash-colibri`), so set `LLM_BACKEND=colibri` to get
colibri's handling. `FREQUENCY_PENALTY` from the shared `git/.env` is then left out (colibri answers
it with a 400). An unset `MODEL` resolves to the model colibri serves, and a full colibri queue (429)
is waited out instead of surfacing in the Retry/Stop menu.

What changes for callers: `print_agent_response` duck-types OpenAI chunks
(`chunk.choices[0].delta.content`). llm-lib yields `Event`s instead (`content` / `reasoning`, then
one `tool_calls` event with complete calls). The loop gets simpler, since it no longer has to stitch
together tool-call fragments by index. `LLM_BACKEND=ollama` now uses Ollama's native API, which
honours `num_ctx`. Anthropic no longer sends `temperature` to models that reject it (Opus 5.5 returns a 400).

## Text-Speech-Voice and TSV-kokoro (`src/text_speech_voice/llm/`)

| Today | llm-lib |
|---|---|
| `ProviderSettings(...)` | `Settings(...)` with the same fields: `max_output_tokens` → `max_tokens`, `load_context`, `load_args` (a tuple), `parallel`, `timeout`, `no_think` |
| `create_client(settings, context_tokens=n, log=log)` | `LLM(settings.with_(context=n), log=log)` |
| `client.complete(system, prompt, json_schema=...)` | same call, same behaviour (including the "retry without response_format" fallback) |
| `client.unload()` | same |
| `NoThinkClient`, `strip_reasoning`, `empty_reason`, `elapsed` | `Settings(no_think=True)`; reasoning is split off automatically; the log lines are built in |
| `openai_compat.lms_path`, `_ensure_loaded` | `llm_lib.servers.lmstudio.find_lms`, automatic with `load_context` |
| `--base-url` requirement for `openai-compatible` | same (raises if unset) |

TSV-kokoro is an older copy of the same module (it has no timeout handling), so both move the same way.

`STORY_PROVIDER=colibri` keeps its three-day timeout and still never resends after a 5xx (TSV had
`max_retries=0`). It now also resends after a 429 (queue full), which nothing had started generating
for. A server started with `COLI_API_KEY` needs that key in the app's environment, where TSV used to
send the placeholder `colibri`.

## Silly (`backend/adventures/llm.py`)

| Today | llm-lib |
|---|---|
| `stream(req, settings)` yielding `("content" \| "reasoning", text)` | `async for e in llm.astream(req.messages or None, prompt=req.prompt, stop=req.stop, ...)` → `(e.type, e.text)` |
| `_sampling(settings)` (temperature, top_p, top_k, min_p, repetition penalty, max_tokens) | `Settings` fields (same names; both repeat-penalty spellings are sent) |
| `_post` 5xx waits `(2, 5, 10)`, "Context size has been exceeded" not retried | built in (`retry_waits`, `LLMContextExceeded`) |
| `failover_settings` / `main_down_until` / `FAILOVER_RETRY_SECONDS` | `FailoverLLM(main, failover, primary_off=lambda: bool(paused_reason or unloaded))` |
| `list_models`, `list_model_details`, `loaded_context`, `load_model`, `unload_models` | `llm.list_models()`, `LMStudio(url).models()`, `.loaded_context(m)`, `.load(m, ctx, parallel=p)`, `.unload_all()` (run them with `asyncio.to_thread`) |
| `api_base_url = "mock"` | `Settings(provider="mock", extra={"delay": 0.05})` |

`ReplyParser` already splits out reasoning by the preset's own prefix/suffix (and the prefill), so
create Silly's client with `Settings(strip_think=False)`. That way inline `<think>` text reaches it untouched.

These stay in the app: `ask()`, prompt building, `ReplyParser`, `preset_set_for`, the video swap's
`in_flight` counter, and `tokens.py` (it uses the `lmstudio` SDK's tokenizer).

## Stonks and stock-portifolio-ui (`backend/src/portfolio_api/llm.py`)

| Today | llm-lib |
|---|---|
| `LLMSettings.load()` (`STORY_*`, `OLLAMA_URL`, `LMSTUDIO_URL`, `LLM_ENV_FILE`) | `Settings.from_env()` reads all of these |
| `chat(system, user, s, cancel)` | `llm.complete(system, user, cancel=cancel, temperature=0.3)` |
| `chat_stream(messages, s, cancel, temperature, tools)` + `ToolCalls` | `llm.stream(messages, cancel=cancel, temperature=t, tools=tools)` (`tool_calls` event) |
| `_ollama_stream` / `_lmstudio_stream` / `_openai_messages` | built in, and history is in OpenAI shape for both providers |
| `strip_reasoning`, the streaming `<think>` hider | built in (`strip_think=True`) |
| `available(s)` | `llm.available()` |
| `_lmstudio_ensure_loaded` | `Settings(load_context=...)` — note: set `STORY_LOAD_CONTEXT` explicitly, it no longer defaults to `STORY_CONTEXT` |
| `LLMUnavailable`, `LLMCancelled` | `llm_lib.LLMUnavailable`, `llm_lib.LLMCancelled` |

These stay in the app: the prompts (`SYSTEM`, `SENTIMENT_SYSTEM`, `CHAT_SYSTEM`), `facts`,
`deep_facts`, `parse_call` and `classify`. `classify` can use `llm_lib.extract_json` for its JSON parsing.

## Apps that talk to colibri

Use `provider="colibri"` (or `LLM_PROVIDER=colibri`), not `openai-compatible`: it sends the right
`model`, leaves out penalties, waits out a full queue and maps `thinking`/`effort`. For System One
decisions or image generation, see `llm_lib.servers.colibri` in the README.

## Not covered

- **The colibri repo itself** is the inference *server*: it serves these APIs rather than calling them.
- **SillyTavern-foriOS** is a vendored fork of upstream SillyTavern (JavaScript), best left in sync with upstream.
- **The Next.js `api/llm` routes** in Stonks just proxy to the Python backend.
