# Configurable timeouts proposal

Task: `t_c033682a`
Workspace: `/home/yozakura/Projects/hermes-agent`

Scope of this document:

- Inventory runtime timeout values that are newly relevant to the local branch work, especially OpenViking prefetch/context injection and Telegram split streaming.
- Propose where they should live: `config.yaml` vs `context_control.yaml`.
- No code changes were made for this analysis.

## Placement rule

Use `config.yaml` for durable operational/runtime behavior:

- network request timeouts
- background worker drain/join timeouts
- gateway stream finalization timeouts
- platform connect/disconnect timeouts
- user-facing behavioral defaults that are not secrets

Use `context_control.yaml` only for live context-injection policy that the agent may tune during a conversation:

- OpenViking/context injection enabled/disabled
- retrieval position
- retrieval quality thresholds and item counts

Rationale: `agent/context_control.py` is intentionally re-read at turn boundaries so an agent can adjust context-injection behavior live. Timeouts are operational stability settings and should not be mixed with retrieval quality knobs unless they specifically govern recall/injection behavior.

## High-priority hardcoded timeouts from local branch

### 1. OpenViking HTTP API timeout

Current code:

- File: `plugins/memory/openviking/__init__.py`
- Constant: `_TIMEOUT = 30.0`
- Uses:
  - `_VikingClient.get(... timeout=_TIMEOUT ...)`
  - `_VikingClient.post(... timeout=_TIMEOUT ...)`
  - `_VikingClient.upload_temp_file(... timeout=_TIMEOUT ...)`

Proposal:

```yaml
# config.yaml
memory:
  openviking:
    request_timeout_seconds: 30.0
```

Default: `30.0`.

Why `config.yaml`: this is network I/O behavior for a provider, not context-ranking policy. It affects every OpenViking operation, including sync, resource upload, and search.

Implementation sketch:

- Extend the OpenViking provider config load path to read `memory.openviking.request_timeout_seconds`.
- Thread the value into `_VikingClient` instead of using module global `_TIMEOUT`.
- Keep env vars only for credentials/endpoint/tenant identity; do not add a new public `HERMES_*` env var for this behavioral setting.

### 2. OpenViking health-check timeout

Current code:

- File: `plugins/memory/openviking/__init__.py`
- Code: `self._httpx.get(self._url("/health"), headers=self._headers(), timeout=3.0)`

Proposal:

```yaml
memory:
  openviking:
    health_timeout_seconds: 3.0
```

Default: `3.0`.

Why `config.yaml`: operational connectivity probe timeout.

Note: this could be derived from `request_timeout_seconds` with a cap, but explicit is clearer because health checks should usually fail faster than full search/upload calls.

### 3. OpenViking prefetch join timeout

Current code:

- File: `plugins/memory/openviking/__init__.py`
- Constant: `_PREFETCH_JOIN_TIMEOUT = 3.0`
- Use: `thread.join(timeout=_PREFETCH_JOIN_TIMEOUT)` in `OpenVikingMemoryProvider.prefetch()`.

Proposal:

```yaml
memory:
  openviking:
    prefetch_join_timeout_seconds: 3.0
```

Default: `3.0`.

Why `config.yaml`: this is an execution/drain bound for a background worker handoff. It is related to recall quality, but its role is runtime safety: how long a turn may wait for the background prefetch thread before falling back to synchronous fetch.

### 4. OpenViking sync fallback/join timeouts

Current code:

- File: `plugins/memory/openviking/__init__.py`
- Observed hardcoded joins:
  - `self._sync_thread.join(timeout=5.0)` around sync/drain paths.
  - `self._sync_thread.join(timeout=10.0)` around session-end/commit style paths.
  - `t.join(timeout=5.0)` for cleanup/helper thread joins.

Proposal:

```yaml
memory:
  openviking:
    sync_join_timeout_seconds: 5.0
    session_commit_timeout_seconds: 10.0
    shutdown_join_timeout_seconds: 5.0
```

Defaults:

- `sync_join_timeout_seconds: 5.0`
- `session_commit_timeout_seconds: 10.0`
- `shutdown_join_timeout_seconds: 5.0`

Why `config.yaml`: these bound provider shutdown/session durability behavior. They are not context ranking controls.

Implementation caution:

- Preserve bounded shutdown. Do not allow `0` or negative to mean infinite wait unless the dashboard/config docs make that explicit; otherwise a wedged OpenViking server can hang the agent/gateway.
- Coerce invalid values back to defaults and log a warning.

### 5. Telegram/gateway stream-consumer drain timeout

Current code:

- File: `gateway/run.py`
- Constant: `_STREAM_CONSUMER_DRAIN_TIMEOUT_SECS = 30.0`
- Uses: three `await asyncio.wait_for(stream_task, timeout=_STREAM_CONSUMER_DRAIN_TIMEOUT_SECS)` sites.

Proposal:

```yaml
gateway:
  streaming:
    drain_timeout_seconds: 30.0
```

Default: `30.0`.

Why `config.yaml`: gateway streaming behavior already has a config home: `GatewayConfig.streaming` / `StreamingConfig` in `gateway/config.py`. This timeout belongs next to existing streaming keys:

- `enabled`
- `transport`
- `edit_interval`
- `buffer_threshold`
- `cursor`
- `fresh_final_after_seconds`

Implementation sketch:

- Add `drain_timeout_seconds: float = 30.0` to `StreamingConfig`.
- Include it in `to_dict()` and `from_dict()` with `_coerce_float`.
- In `gateway/run.py`, read from `self.config.streaming.drain_timeout_seconds` at each stream drain site, with a default fallback of `30.0`.
- Enforce a sane lower bound. Suggested: values below `1.0` fall back to `30.0` or clamp to `1.0` with a warning. Very small values recreate the duplicate-send race.

### 6. Gateway inbound timestamp prefix gap

Current code:

- File: `gateway/run.py`
- Symbol named in task: `_INBOUND_TIMESTAMP_PREFIX_MIN_GAP_SECS`
- Purpose: avoid adding timestamp prefixes too frequently to inbound gateway messages.

Proposal:

```yaml
gateway:
  inbound_timestamp_prefix:
    min_gap_seconds: 60.0  # use current constant as default when implementing
```

Default: use the current `_INBOUND_TIMESTAMP_PREFIX_MIN_GAP_SECS` value from `gateway/run.py` when implementing.

Why `config.yaml`: this is gateway message formatting/session-context behavior, not retrieval policy.

Implementation sketch:

- Add a small nested config object or plain scalar under `GatewayConfig`.
- Avoid placing this under `streaming` because it applies to inbound messages even when streaming is disabled.
- If dashboard support is added later, expose it as an advanced gateway setting.

## Existing config-adjacent values that should stay in `context_control.yaml`

Current code:

- File: `agent/context_control.py`
- Defaults:
  - `enabled: True`
  - `position: before`
  - `min_score: 0.55`
  - `top_k: 5`
  - `max_items: 5`

Current file format supports both top-level and nested OpenViking keys:

```yaml
openviking:
  enabled: true
  position: before
  min_score: 0.55
  top_k: 5
  max_items: 5
```

Recommendation: keep these in `context_control.yaml` because they are live retrieval/injection policy and are deliberately reloadable during turns.

Optional schema cleanup:

```yaml
# context_control.yaml
openviking:
  enabled: true
  position: before      # before | after | system
  min_score: 0.55
  top_k: 5
  max_items: 5
```

Do not add OpenViking HTTP, thread join, or gateway stream timeouts here.

## Lower-priority timeout values found during scan

These are runtime timeouts, but they are outside the immediate local-branch cleanup or already have a config mechanism. They should not block this refactor plan.

### Already configurable or tool-local

- `tools/mcp_tool.py`
  - `_DEFAULT_TOOL_TIMEOUT = 120`
  - `_DEFAULT_CONNECT_TIMEOUT = 60`
  - Config already supports `timeout` and `connect_timeout` per MCP server.
  - Recommendation: leave as-is.

- `tools/transcription_tools.py`
  - `DEFAULT_COMMAND_STT_TIMEOUT_SECONDS = 300`
  - Provider config already reads `timeout` / `timeout_seconds`.
  - Recommendation: leave as-is.

- `tools/x_search_tool.py`
  - `DEFAULT_X_SEARCH_TIMEOUT_SECONDS = 180`
  - Reads `timeout_seconds` from config.
  - Recommendation: leave as-is.

- `tools/code_execution_tool.py`
  - `DEFAULT_TIMEOUT = 300`
  - Tool-specific sandbox/RPC timeouts, many internal cleanup waits.
  - Recommendation: do not mix with gateway/context-control timeout refactor.

- `tools/checkpoint_manager.py`
  - `_GIT_TIMEOUT` reads `HERMES_CHECKPOINT_TIMEOUT` today.
  - This is tool/checkpoint-specific and outside this branch's feature scope.
  - If revisited later, migrate to config because it is non-secret behavior, but do not scope-creep here.

### Plugin-specific generation polling

- `plugins/image_gen/krea/__init__.py`
  - `_POLL_TIMEOUT_SECONDS = 180.0`
  - request/poll timeouts `30`.
- `plugins/image_gen/xai/__init__.py`
  - request timeout `120`.
- `plugins/video_gen/xai/__init__.py`
  - `DEFAULT_TIMEOUT_SECONDS = 240`
  - request/poll timeouts `60`/`30`.

Recommendation: leave for plugin-specific config follow-up. These are not part of OpenViking/gateway local branch work.

### Secret provider / model catalog / account usage

- `agent/secret_sources/bitwarden.py`
  - `_BWS_DOWNLOAD_TIMEOUT = 60`
  - `_BWS_RUN_TIMEOUT = 30`
  - cache TTL defaults.
- `agent/models_dev.py`
  - `requests.get(MODELS_DEV_URL, timeout=15)`.
- `agent/account_usage.py`
  - `httpx.Client(timeout=15.0)` and `timeout=10.0`.

Recommendation: leave as operational defaults unless a user-facing settings request appears. These are unrelated to current local branch features.

## Proposed `config.yaml` schema

Recommended consolidated schema for this branch:

```yaml
gateway:
  streaming:
    # Existing keys:
    enabled: false
    transport: auto
    edit_interval: 1.0
    buffer_threshold: 80
    cursor: "▌"
    fresh_final_after_seconds: 60.0

    # New:
    drain_timeout_seconds: 30.0

  inbound_timestamp_prefix:
    min_gap_seconds: 60.0  # use current code default when implementing

memory:
  openviking:
    request_timeout_seconds: 30.0
    health_timeout_seconds: 3.0
    prefetch_join_timeout_seconds: 3.0
    sync_join_timeout_seconds: 5.0
    session_commit_timeout_seconds: 10.0
    shutdown_join_timeout_seconds: 5.0
```

Keep context injection quality controls separate:

```yaml
# context_control.yaml
openviking:
  enabled: true
  position: before
  min_score: 0.55
  top_k: 5
  max_items: 5
```

## Implementation checklist for a future coding task

1. Add schema fields with defaults:
   - `GatewayConfig.streaming.drain_timeout_seconds` in `gateway/config.py`.
   - `GatewayConfig.inbound_timestamp_prefix.min_gap_seconds` or equivalent scalar in `gateway/config.py`.
   - OpenViking provider timeout settings in its plugin config load path.
2. Replace hardcoded constants/calls:
   - `_STREAM_CONSUMER_DRAIN_TIMEOUT_SECS` -> config value.
   - `_INBOUND_TIMESTAMP_PREFIX_MIN_GAP_SECS` -> config value.
   - `_TIMEOUT`, health `3.0`, `_PREFETCH_JOIN_TIMEOUT`, sync/session/shutdown joins in OpenViking -> provider settings.
3. Keep defaults identical to current behavior.
4. Add coercion/validation:
   - numeric values only
   - positive finite values only
   - warn and fall back on invalid values
5. Add tests:
   - default config preserves current constants.
   - custom `gateway.streaming.drain_timeout_seconds` is used at all stream drain sites.
   - low/invalid drain timeout cannot recreate the duplicate-send race silently.
   - custom OpenViking timeouts are threaded into `_VikingClient` and join paths.
6. Do not introduce public `HERMES_*` env vars for these non-secret settings.

## Verification performed for this inventory

Commands/files checked:

- `rg -n "timeout\s*=\s*[0-9]|_TIMEOUT|TIMEOUT|wait_for\(|join\(timeout=|sleep\([0-9]|settimeout\(|read_timeout|connect_timeout|DRAIN_TIMEOUT|MIN_GAP|interval\s*=\s*[0-9]|poll_interval|rate_limit|cooldown|ttl|expires|stale" agent gateway plugins tools scripts -g '*.py' -S`
- `agent/context_control.py`
- `gateway/config.py` (`StreamingConfig`, `GatewayConfig`)
- `plugins/memory/openviking/__init__.py`
- `gateway/run.py` stream drain references via ripgrep
