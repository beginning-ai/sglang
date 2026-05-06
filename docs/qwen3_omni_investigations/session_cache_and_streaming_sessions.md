# Session Cache And Streaming Sessions

## Summary

Closed voice sessions previously retained Thinker radix/KV state and
caused later bs=30 sequential runs to stall. That cleanup is fixed. Issue
#108 migrated Qwen3 Omni audio sessions toward upstream SGLang
streaming-session/session-aware-cache semantics for the append-only happy
path.

## Decision

Use upstream streaming sessions for Thinker KV and anchor retained Talker
user-prefix KV to the same append-only audio session lifecycle through
`Session.component_kv`. Rollback/non-latest parent behavior should
release the slot and rebuild from the selected safe turn, not complicate
the hot path.

The Beginnings production client and local stub have already been updated
to the Qwen3-Omni `AudioTurn` gRPC API. `streaming_audio.proto` now only
carries scheduler `Abort` and `GetLoads`; production voice turns use
`AudioTurn`, `CloseAudioSession`, and `HeartbeatAudioSession` on the
Qwen3-Omni service. The local `chat_stub` implements append-latest parent
semantics: omitted `parent_rid` appends to the latest turn, while explicit
missing or stale parents fail loudly.

## Closed-Session Radix Cleanup

Bug signature:

- `audio_sessions=0`
- `stored_audio_turns=0`
- `active_audio_turns=0`
- Thinker KV and radix protected tokens still allocated

The accepted fix:

- keep radix enabled during live sessions
- count active audio turns in scheduler idleness
- flush cache only when the last audio session has closed and the
  scheduler is fully idle

Accepted validation:

- five Beginnings fixtures
- bs=30
- realtime input
- `chelsie`, Beginnings prompt, 64/256 caps
- radix enabled
- 150/150 ok, 150/150 realtime
- copied-back Parakeet 150/150
- Thinker/radix KV returned to zero after fixture close

## Session/Cache Overhead Probe

Temporary broad probes measured radix/session bookkeeping before #108.
The probes were removed after measurement; the main branch keeps only
small runtime snapshot metrics.

30-minute bs=30 AMD probe:

- 2458/2458 turns ok
- 2422/2458 strict realtime
- mean/max RTF after EOI 0.555 / 1.318
- max first-audio after EOI 1.011s
- `match_prefix`: 18.3s
- `cache_unfinished`: 39.8s
- `cache_finished`: 6.3s
- total observed cache-ish cost: ~64.4s / 1800s wall, about 3.6%

Cost grows with prefix length:

| event | 1024-2047 mean | 2048-4095 mean | 4096-8191 mean |
|---|---:|---:|---:|
| `match_prefix` | 0.24ms | 1.04ms | 3.05ms |
| `cache_unfinished` | 0.59ms | 2.29ms | 6.58ms |
| `cache_finished` | 0.39ms | 1.49ms | 4.04ms |

Conclusion: cache overhead was not the largest component, but it had the
wrong architecture shape for long sessions and justified #108 for cleaner
ownership and less fork diff.

## #108 Migration Result

- Qwen3 Omni now requires `--enable-streaming-session` and fails fast at
  startup without it.
- Upstream `StreamingSession` owns Thinker KV for the append-only happy
  path.
- Chunked audio prefill uses the streaming-session slot.
- Audio turns append to the latest committed streaming-session request;
  explicit `parent_rid` is accepted only if it names that latest request.
- Retained Talker user-prefix KV is stored on `Session.component_kv`.
- Response-local Talker KV is still released at turn end.
- Historical `AudioTurnState` maps, per-turn radix locks, cached text
  responses, `system_prompt`, and `last_activity` were removed from the
  audio session layer.

Remaining constraint:

- Rare safety rollback still fails fast on non-latest parents. Add an
  explicit streaming-slot release plus full-prefill rebuild only if
  product needs rollback on this hot path.

## Post-Migration Validation

Correctness:

- `$qwen3-omni-voice-validate --multi-turn --no-sync --no-restart --label streaming-session-cleanup-v18b`
- all 10 copied-back Parakeet fixtures passed
- report: `/tmp/qwen3_voice_validate/streaming-session-cleanup-v18b/report.json`

Clean no-debug bs=30 30-minute soak:

- artifact: `/tmp/qwen3_session_streaming_post108_nodebug_1800s.json`
- server log: `/tmp/qwen3_session_streaming_post108_nodebug_server.log`
- 2487/2487 turns ok
- 30/30 sessions closed
- 0 failed turns, 0 close failures
- 2470/2487 strict realtime
- 2485/2487 audio-stream realtime
- mean/max RTF after EOI: 0.515 / 1.195
- mean/max audio-stream RTF: 0.460 / 1.008
- p50/p95/p99/max first audio after EOI: 0.219s / 0.426s / 0.617s / 1.046s
- strict misses: 17, with 12 at turn index >= 60

Debug-snapshot bs=30 30-minute lifecycle run:

- artifact: `/tmp/qwen3_session_streaming_post108_1800s.json`
- server log: `/tmp/qwen3_session_streaming_post108_server.log`
- 2471/2471 turns ok
- 30/30 sessions closed
- final snapshot had `streaming_session.held_tokens=0`,
  `streaming_session.sessions=0`, `talker_user_prefix_cache.stored_tokens=0`,
  `talker_user_prefix_cache.stored_turns=0`, and Thinker KV fully free

## Constraints Kept For Future Work

- Keep full logical `origin_input_ids` for correctness/debuggability.
- Fail fast on non-latest parent unless explicitly entering rollback
  rebuild.
- Close standard streaming session and Qwen3 `AudioSessionState`
  together.
- Verify Thinker KV, streaming held tokens, and radix protected tokens
  reach zero after final active turn drains.
- Preserve Talker semantic rule: keep only committed user-prefix KV
  across turns; free assistant/bootstrap/generated-codec KV after each
  turn.

## References

- Thinking Machines interaction-models blog:
  https://thinkingmachines.ai/blog/interaction-models/
- SGLang streaming session PR:
  https://github.com/sgl-project/sglang/pull/19171
- vLLM streaming/realtime blog:
  https://vllm.ai/blog/streaming-realtime
- GitHub issue #108:
  https://github.com/beginning-ai/beginnings/issues/108
