# Squash plan for `local` branch

Task: `t_c033682a`
Workspace: `/home/yozakura/Projects/hermes-agent`

Scope checked:

- `git rev-list --count local --not main` => 169 commits.
- `git log --reverse --format='%h %s' local --not main --grep='^local:'` => 21 local feature commits.
- Top five unsquashed local follow-up fixes are:
  - `8e20c9e fix(memory): split prefetch to separate executor to avoid head-of-line blocking`
  - `a25f970 fix(gateway): prevent duplicate TG message delivery on image turns + streaming split`
  - `b3743ad fix(memory): keep OpenViking prefetch alive after sync failures`
  - `1b72249 fix: make OpenViking prefetch cache miss synchronous`
  - `3246907 fix(gateway): extend stream consumer drain timeout to prevent duplicate TG sends`

Important: this is a plan only. Do not run the rebase from this file without a final human review.

## Base feature sequence

Keep the 21 `local:` feature patches in their current relative order unless review finds a dependency issue:

1. `2109690 local: vision-anthropic`
2. `5f3ee58 local: gemini-cli-aux`
3. `687cbce local: prompt-overrides`
4. `12cbc98 local: memory-context-header-override`
5. `0a2a12d local: context-window-override`
6. `bfbcc3f local: compaction-role-labels`
7. `fc28406 local: post-compaction-message-loss`
8. `28d30f5 local: session-db-flush-cursor`
9. `41e8d7f local: gateway-db-persistence-fallback`
10. `d209937 local: telegram-hide-reasoning`
11. `a5df4f6 local: telegram-split-replies`
12. `1d6c762 local: telegram-stream-split-replies`
13. `c7a0629 local: stop-retry-message-loss`
14. `ed6c387 local: background-review-file-tools`
15. `bd6b27e local: nixos-path-fill`
16. `dfea998 local: fallback-custom-api-mode`
17. `f18d3d8 local: context-control`
18. `4a16bb8 local: compaction-preamble-override`
19. `4465186 local: telegram-split-reply-first-only`
20. `e324fbd local: openviking-prefetch-quality`
21. `f6ff319 local: timestamp-prefix`

## Recommended squash/fixup mapping

### `e324fbd local: openviking-prefetch-quality`

Squash/fixup these commits into this feature, or place them immediately after it as fixups during review:

- `8e20c9e fix(memory): split prefetch to separate executor to avoid head-of-line blocking`
  - Files: `agent/memory_manager.py`, `plugins/memory/openviking/__init__.py`, `tests/agent/test_memory_async_sync.py`.
  - Reason: this fixes the OpenViking prefetch quality path by preventing `queue_prefetch_all()` from being head-of-line blocked behind `sync_all()` on the same single-worker executor. It also removes duplicated `## OpenViking Context` in the provider and improves prefetch logging.
  - Note: `agent/memory_manager.py` is generic external-memory infrastructure. If reviewers want to keep generic infrastructure separate, this can become a small standalone feature commit named like `local: memory-prefetch-executor-lane`; otherwise it belongs with the OV prefetch feature because it was discovered and required there.

- `b3743ad fix(memory): keep OpenViking prefetch alive after sync failures`
  - Files: `agent/memory_manager.py`, `plugins/memory/openviking/__init__.py`, `run_agent.py`, tests.
  - Reason: continues the same OV prefetch reliability work: content block normalization, query/sync robustness, debug logging, and sync lifecycle guardrails.

- `1b72249 fix: make OpenViking prefetch cache miss synchronous`
  - Files: `agent/conversation_loop.py`, `agent/memory_manager.py`, `agent/turn_context.py`, `plugins/memory/openviking/__init__.py`, tests.
  - Reason: fixes the same user-visible feature by making `prefetch()` synchronously fetch on cache miss or stuck background prefetch, instead of injecting empty recall.
  - Cross-feature note: part of this uses `agent.context_control.load_settings()` and therefore depends on `f18d3d8 local: context-control`. Keep `f18d3d8` before `e324fbd` if squashing this entirely into `e324fbd`, or split the context-control loading hunk into the context-control feature and the OpenViking fetch logic into `e324fbd`.

### `f18d3d8 local: context-control`

Recommended content:

- Keep the base `agent/context_control.py` loader, `ContextControlSettings`, and conversation-loop injection wiring here.
- Consider pulling the context-control-dependent parts of `1b72249` here only if the final rebase gets conflicts: debug logging around `Memory context injection check` and use of `load_settings()` is conceptually context-control-adjacent, while `_fetch_prefetch_context()` remains OpenViking-specific.

### `1d6c762 local: telegram-stream-split-replies`

Squash/fixup these commits into this feature:

- The streaming-split half of `a25f970 fix(gateway): prevent duplicate TG message delivery on image turns + streaming split`
  - Files: `gateway/stream_consumer.py`, `tests/gateway/test_stream_consumer.py`.
  - Reason: delimiter-split streaming delivered final-answer bubbles but did not set `final_response_sent`, so the normal final-send path re-split and sent the whole logical answer again. This is directly part of stream split reply behavior.
  - Keep the fallback-mode exception: do not mark the final response sent when `_fallback_final_send` is active, because a delivered prefix may still need a missing tail.

- `3246907 fix(gateway): extend stream consumer drain timeout to prevent duplicate TG sends`
  - Files: `gateway/run.py`, `tests/gateway/test_stream_consumer.py`.
  - Reason: the old hardcoded 5s wait cancelled the stream consumer while Telegram split bubbles were still serially flushing; `final_response_sent` then stayed false and gateway fallback duplicated the response. This is the same stream split reply bug class.
  - If timeouts are made configurable before rebase, replace the hardcoded `_STREAM_CONSUMER_DRAIN_TIMEOUT_SECS = 30.0` with a config read in this feature commit instead of preserving the constant.

### `f6ff319 local: timestamp-prefix`

Squash/fixup the DB-prefix half of `a25f970` here:

- Files: `gateway/run.py`, `tests/test_lazy_session_regressions.py`.
- Reason: `_count_gateway_current_turn_db_prefix()` compared raw DB content to expected multipart content. Gateway-persisted image turns can include volatile `[Image attached at: ...]` path lines that are absent from expected content; without normalization the prefix length falls back to 0 and the gateway mirrors the whole current turn again. This is current-turn prefix normalization and belongs with timestamp/prefix work.

## Special review: should `a25f970` be dropped?

Recommendation: do **not** drop `a25f970`; split and squash it into the relevant features.

Why it still has independent value after `3246907`:

1. `3246907` fixes a timing/race root cause: the stream consumer was cancelled after 5s while split Telegram sends were still flushing. Extending the drain timeout to 30s gives the consumer time to set final-send state.
2. `a25f970` fixes a state/normalization bug class independent of that race:
   - Image-turn DB prefix normalization in `gateway/run.py` is unrelated to drain timing and is not replaced by `3246907`.
   - Delimiter-split `final_response_sent` marking in `gateway/stream_consumer.py` is still the correct invariant once split bubbles have been delivered. The longer drain timeout merely makes it more likely this invariant can be set before gateway fallback runs.
3. Current working tree still contains both sets of logic:
   - `_message_content_for_db_compare()` strips volatile `[Image attached at: ...]` lines and normalizes whitespace.
   - `_split_replies_delivered_segments` sets `_final_response_sent`/`_final_content_delivered` when appropriate.
   - `_STREAM_CONSUMER_DRAIN_TIMEOUT_SECS = 30.0` is separately used at three `asyncio.wait_for(stream_task, ...)` drain sites.

So `a25f970` is not redundant; it should be split across `f6ff319` and `1d6c762`.

## Commits with no observed local follow-up fix

Keep these feature commits as standalone review units unless a later manual rebase reveals conflict-only fixups:

- `2109690 local: vision-anthropic`
- `5f3ee58 local: gemini-cli-aux`
- `687cbce local: prompt-overrides`
- `12cbc98 local: memory-context-header-override`
- `0a2a12d local: context-window-override`
- `bfbcc3f local: compaction-role-labels`
- `fc28406 local: post-compaction-message-loss`
- `28d30f5 local: session-db-flush-cursor`
- `41e8d7f local: gateway-db-persistence-fallback`
- `d209937 local: telegram-hide-reasoning`
- `a5df4f6 local: telegram-split-replies`
- `c7a0629 local: stop-retry-message-loss`
- `ed6c387 local: background-review-file-tools`
- `bd6b27e local: nixos-path-fill`
- `dfea998 local: fallback-custom-api-mode`
- `4a16bb8 local: compaction-preamble-override`
- `4465186 local: telegram-split-reply-first-only`

## Suggested interactive rebase shape, high level

Do not copy/paste this blindly; it is intentionally descriptive rather than a literal todo file because some commits need splitting.

1. Keep the 21 `local:` commits in reverse-log order listed above.
2. Split `a25f970` into two logical fixups:
   - DB prefix/image normalization -> fixup into `f6ff319 local: timestamp-prefix`.
   - stream split `final_response_sent` invariant -> fixup into `1d6c762 local: telegram-stream-split-replies`.
3. Fixup `3246907` into `1d6c762 local: telegram-stream-split-replies`.
4. Fixup `8e20c9e`, `b3743ad`, and `1b72249` into `e324fbd local: openviking-prefetch-quality`, with the dependency note that `f18d3d8 local: context-control` must remain before any OpenViking code that imports it.
5. Leave upstream/salvage/merge commits outside this local cleanup unless the reviewer wants to rebase the entire branch onto current `main`; this task only planned the local feature/fix squash, not a full upstream history rewrite.

## Verification performed for this plan

Commands run:

- `git status --short --branch`
- `git rev-list --count local --not main`
- `git log --oneline local --not main`
- `git log --reverse --format='%h %s' local --not main --grep='^local:'`
- `git show --name-status --stat` for the 21 local features and five top follow-up fixes.
- `git show --patch` for `8e20c9e`, `a25f970`, `b3743ad`, `1b72249`, and `3246907`.
- `rg` over gateway stream/final-send symbols to confirm `a25f970` and `3246907` address different invariants.
