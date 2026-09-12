# Codex with Kimi through AgentGateway

Use a Chat Completions-only custom backend to make AgentGateway translate
Codex's Responses requests and streaming responses. Declaring the backend as
`OpenAI` advertises native Responses support and therefore selects passthrough.

For an `AgentgatewayModel`, keep the existing model match, base URL and backend
credentials, and use:

```yaml
provider: Custom
custom:
  formats:
    - type: Completions
```

For standalone AgentGateway, the equivalent provider is:

```yaml
provider:
  custom:
    formats:
      - type: completions
```

## Codex configuration

Merge the settings in `config.toml` into the user-level `~/.codex/config.toml`.
Set `base_url` to the reachable Gateway URL ending in `/v1`, set
`model_catalog_json` to this directory's absolute `models.json` path, and make
`AGENTGATEWAY_API_KEY` available to the Codex process. Use a Gateway client key;
the upstream provider credential stays in Gateway.

The supplied model catalog enables Codex's native shell and free-form
`apply_patch` tools. It also avoids Codex attempting to parse the standard
OpenAI `/v1/models` response as its richer model catalog. The 65,536-token
context is a conservative client limit, not the provider's advertised maximum.

Restart the desktop app and start a new local task after changing its default
provider. Existing tasks retain their provider. To test separately from the
app's defaults, use `~/.codex/kimi.config.toml` and `codex --profile kimi`.
Profiles are separate files in current Codex, not `[profiles.kimi]` tables.

The settings use HTTP/SSE and client-managed conversation history. Provider-side
Responses state (`previous_response_id`, conversation IDs, saved prompts),
hosted web search and other hosted tools are not emulated. The adapter rejects
unsupported tool definitions instead of silently dropping them. Native Codex
shell tools continue to work.

Custom tools become functions with an `{ "input": "raw tool input" }` argument.
Namespaces and tool names are restored on the response, and `call_id` is
preserved for replay. Grammars are included in tool instructions; a Chat
Completions backend does not provide Responses CFG enforcement. Tool arguments
are buffered until complete so escaped text and parallel calls can be decoded
correctly. Text streams incrementally. The response buffer limit also bounds
the accumulated translated output. Plain reasoning content is preserved in
Responses reasoning items for subsequent tool-call turns.

## Validation

```sh
cargo test -p agent-llm --lib
cargo test -p agentgateway --lib codex_responses_custom_tools_round_trip
cargo build -p agentgateway-app --bin agentgateway
python3 tools/codex-responses-smoke.py --codex /path/to/codex
```

The smoke test launches the built Gateway, a deterministic loopback Chat
Completions backend, and a real Codex CLI. It verifies that Codex executes a
custom `apply_patch` call, writes a temporary file, replays the matching tool
result and assistant commentary, reads the file with its shell tool, and
completes the turn. It requires no API credentials and makes no
requests to a paid model. All test processes and temporary files are cleaned up.

Also verified with Codex CLI 0.153.4 and the real `moonshotai/Kimi-K3` backend:
Codex created a temporary file with custom `apply_patch`, read it with its native
shell tool, replayed the tool results, and completed the turn. This exercised a
locally built Gateway; it did not deploy the patch or change the desktop app's
default provider.

OpenAI references: [custom providers and profiles](https://learn.chatgpt.com/docs/config-file/config-advanced),
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
