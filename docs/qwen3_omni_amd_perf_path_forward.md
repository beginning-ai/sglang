# Qwen3 Omni AMD Performance Path Forward

## Goal

Maximize Qwen3 Omni realtime voice concurrency on one AMD MI300X.

The production target is AMD MI300X. Nvidia RTX PRO 6000 is only for fast
correctness iteration and reference checks. The engineering target remains
`rtf_after_eoi <= 1.0` for every admitted active voice user after startup
prewarm, with explicit capacity rejection above the admitted cap instead of
hidden queues, stream hangs, or degraded audio. For launch, the measured
bs=50 tail rate below is accepted: rare p100 misses are allowed while the
team continues post-launch capacity and tail hardening.

Concurrency targets:

1. **bs=50, 30-minute sessions:** accepted launch cap with rollback
   enabled. The 2-hour rolling soak below had rare strict misses, but no
   failures, no rollback failures, and low enough miss frequency to ship.
2. **bs=30, 30-minute sessions:** strict-realtime baseline remains locked
   and is the fallback launch cap if a later deployment change invalidates
   bs=50.
3. **bs=100:** later production-economics stretch target.

`bs=N` means N concurrently active voice users, not a turn cap. The normal
launch/expansion gate runs each user until the 30-minute wall-clock session
window ends; the harness then lets the final in-flight turn finish before
scoring. Extended stability soaks use the same realtime targets with
`--duration-s 7200 --session-duration-s 1800`: each concurrency slot rotates
fresh sessions so the whole soak lasts 2 hours while each individual chat
session is still capped at 30 minutes.

The launch workload is production cadence: `turn_min_s=8`,
`turn_max_s=25`, `chunk_ms=1000`, `initial_jitter_s=60`, `chelsie`, and
the Beginnings system prompt. Shorter stress probes are useful after
launch lock, but they are not launch evidence.

## Current Launch State

The launch candidate is:

- checkpoint:
  `/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1`
- HF repo: `beginning-ai/lucy-1-FP8`, revision
  `7aaecb8d422964d4ace0f98ff73d980945a5796a`
- FP8 Thinker plus FP8 Talker attention projections
- BF16 Talker MoE, CodePredictor, Code2Wav, codec/audio bridges,
  embeddings, router gates, shared experts, and output heads
- upstream streaming sessions with Qwen3 Omni component KV ownership
- overlap scheduling enabled
- CodePredictor argmax enabled after constrained human A/B approval
- Talker decode, Thinker prefill, and cached-prefix Talker prefill capture
  enabled
- automatic Qwen3 Omni gRPC startup prewarm required before health reports
  `SERVING`, including derived launch-capacity batch shapes, Code2Wav
  runtime shapes, and observed late-session AITER BF16 GEMM shapes
- safety rollback implemented as a rare-path full-prefix rebuild/prefill

The removed Qwen3 Omni launch-prewarm flags are stale compatibility
surface. Qwen3 Omni gRPC startup always runs launch prewarm and derives its
batch sizes from `--qwen3-omni-max-concurrent-audio-turns`.

## Current Metrics

Update this section after any roadmap item that changes Qwen3 Omni runtime
behavior. Move detailed historical runs to the sidecar docs.

**bs=30 strict fallback proof:** locked after the upstream rebase and
post-cleanup crash fix.
Artifact:
`/tmp/qwen3_bs30_crashfix_relock_20260519T230758Z/stdout.json`.
Telemetry:
`/tmp/qwen3_bs30_crashfix_relock_20260519T230758Z/telemetry.jsonl`,
`/tmp/qwen3_bs30_crashfix_relock_20260519T230758Z/telemetry.png`.
Server log: `/tmp/qwen3_bs30_crashfix_relock_server.log`.
Result: 2615/2615 strict realtime, 2615/2615 audio-stream realtime,
0 failures, 0 close failures, 46/46 rollback success, mean/max RTF
0.329/0.755, mean/max audio-stream RTF 0.296/0.675, max first audio
1.185s, max completion tail 0.113s, max rollback rebuild 1.114s.
Telemetry: avg/max GPU busy 80.0%/99.0%, avg/max HBM activity
6.7%/14.0%, avg/max VRAM used 148.7/167.6 GiB. During the soak window,
max observed KV use was Thinker 202,295/399,999 tokens, Talker
104,252/213,333 tokens, CodePredictor 800/800 slots, minimum Thinker KV
available was 197,704 tokens, and max streaming-session held tokens was
98,250.

Post-rebase validation notes: copied-back voice validation passed all five
fixtures after the cleanup crash fix:
`/tmp/qwen3_voice_validate/bs30-crashfix-token-ids/report.json`. The
failed first post-cleanup soak exposed a scheduler crash in the
multimodal-prefix cache truncation path (`token_ids` renamed upstream to
`token_ids_to_match`); the one-line fix above is covered by the full relock
soak with rollback traffic.

**bs=50 accepted launch cap:** validated on the launch stack after the
upstream rebase, post-cleanup crash fix, and 2-hour rolling stability soak.
The strict engineering target is not fully met at p100, but the observed
miss frequency is accepted for launch.

Latest 2-hour rolling production-cadence soak with rollback traffic:
`/tmp/qwen3_2h_soak_20260520T133125Z/stdout.json`.
Server log: `/tmp/qwen3_2h_soak_20260520T133125Z/server.log`.
Telemetry:
`/tmp/qwen3_2h_soak_20260520T133125Z/gpu_telemetry.jsonl`,
`/tmp/qwen3_2h_soak_20260520T133125Z/kv_telemetry.jsonl`,
`/tmp/qwen3_2h_soak_20260520T133125Z/telemetry.svg`.
Result: 16,658/16,658 turns ok, 0 failures, 0 close failures,
317/317 rollback success. Strict realtime missed 10/16,658 turns;
audio-stream realtime missed 3/16,658 turns. Mean/max RTF after EOI
0.463/1.203, mean/max audio-stream RTF 0.423/1.125, max first audio
1.606s, max completion tail 1.429s, max rollback rebuild 2.045s.
Telemetry: avg/max GPU busy 91.5%/98.0%, avg/max HBM activity
9.6%/16.0%, max VRAM used 191.6 GiB. During the soak window, max
observed KV use was Thinker 399,996/399,999 tokens, Talker
166,529/213,333 tokens, CodePredictor 800/800 slots, max
streaming-session held tokens was 174,789, and max active sessions was 50.

The launch decision accepts this result because the tail misses are rare
enough for first launch, while the system remained stable under 2 hours of
50-slot rolling sessions and rollback traffic. Treat additional p100 tail
reduction and KV-headroom work as post-launch hardening unless a later
deployment change worsens this envelope.

Latest fresh-start 30-minute production-cadence soak:
`/tmp/qwen3_bs50_postcleanup_20260519T234538Z/stdout.json`.
Server log: `/tmp/qwen3_bs50_postcleanup_server.log`.
Telemetry:
`/tmp/qwen3_bs50_postcleanup_20260519T234538Z/telemetry.jsonl`,
`/tmp/qwen3_bs50_postcleanup_20260519T234538Z/telemetry.png`.
Result: 4215/4215 strict realtime, 4215/4215 audio-stream realtime,
0 failures, 0 close failures, 73/73 rollback success, mean/max RTF
0.457/0.896, mean/max audio-stream RTF 0.416/0.807, max first audio
0.876s, max completion tail 0.486s, max rollback rebuild 1.050s.
Telemetry: avg/max GPU busy 88.8%/97.0%, avg/max HBM activity
9.3%/16.0%, avg/max VRAM used 152.8/182.7 GiB. During the soak window,
max observed KV use was Thinker 330,772/399,999 tokens, Talker
169,584/213,333 tokens, CodePredictor 800/800 slots, minimum Thinker KV
available was 69,227 tokens, and max streaming-session held tokens was
171,261.

The earlier projection warmup fixed the previous 867ms Talker prefill prep
spike. The post-cleanup run above no longer has strict realtime or
audio-stream misses, so the active bs=50 work shifts from tail debugging to
completion audit, diff minimization, and keeping this gate green across
future rebases.

Previous pre-projection full soak:
`/tmp/qwen3_bs50_final_probe_30min/stdout.json`, strict realtime 4239/4242,
audio-stream realtime 4241/4242, max first audio 1.322s, max RTF 1.303.
That run's concrete root cause was sampled Talker-prefill
`prepare_inputs_ms=867.14`; keep it as the reason the projection warmup
exists, not as the current blocker.

Copied-back voice validation passed all five fixtures after the projection
warmup code changes:
`/tmp/qwen3_voice_validate/talker-projection-prewarm/report.json`.

Historical bs=50 artifacts and rejected probes live in the sidecars. The
most important decisions remain: use 1s client chunks for capacity
claims, keep the 20ms global audio batching window, keep steady 29-frame
Code2Wav graph replay, keep Talker-prefill capture, and do not retry
smaller Talker-prefill chunking or generic grace/yield windows without
new causal evidence.

Interpretation:

- bs=50 is the accepted launch cap on the rebased stack, with the explicit
  caveat that rare p100 strict/audio-stream realtime misses are accepted
  for launch.
- bs=30 remains the fully strict fallback launch cap unless a later code,
  checkpoint, flag, harness, or deployment change invalidates the proof.
- Code2Wav is not the current first-audio blocker. Oldest-ready scheduling
  fixed the stale ready-state starvation shape, and the worst latest
  first-audio row had Code2Wav queue/launch time in the normal range.
- Startup prewarm must include parented continuation traffic at the target
  capacity. Raising the continuation prewarm default to bs=50 fixed the
  early turn-1 first-audio tails seen in the previous 300s smoke.
- The remaining p100 work is late-session EOI-to-first-codec latency for
  long-prefix turns. The latest full soak no longer shows a huge prep/model
  spike; root-cause the first-codec chain before more generic throughput
  tuning.
- A bounded small-M output GEMM prewarm probe (`M=1..256`,
  `(N=128,K=2048)`) did not prove transfer in a fresh bs=50 300s smoke:
  strict/audio realtime stayed green, but max first audio was 0.971s.
  After removing that broad range and restarting from automatic prewarm
  only, the repeat 300s smoke improved to max first audio 0.621s. The top
  rows were not caused by the small-M output first-hit class. Keep the
  narrower observed warm shapes and only revisit broad output-shape
  warming with new causal evidence.
- Rejected paths stay rejected unless new evidence invalidates them:
  10ms global audio deadline, smaller Talker-prefill chunking,
  PrefillDelayer as a launch flag, off-thread single-worker Code2Wav,
  full Talker MoE FP8, FP8 KV, and Talker multi-shared-expert fusion.

## Active Roadmap For Goal Runs

1. **Rebase first.** Fetch latest `main@upstream`, rebase this stack, fix
   conflicts in the smallest possible way, and run quick copied-back voice
   validation before new optimization work. If the rebase changes launch
   behavior or serving flags, rerun the bs=50 launch gate before trusting
   the accepted launch cap.

2. **Shrink the Qwen3 Omni fork diff.** The current full fork diff is large
   enough to hide bugs and slow rebases. Before adding more performance
   code, audit `jj diff -r wt:: --stat` and simplify the whole fork: core
   Qwen3 Omni model code, scheduler/audio-lane logic, graph/capture paths,
   component-KV handling, custom kernels, docs/tests/clients, helper code,
   and every branch-local CLI/env knob that is not strictly required for the
   launch stack. Prefer derived values from top-level required settings,
   fail-fast behavior, and upstream SGLang APIs. Preserve measured
   performance and audio correctness; validate after cleanup.

3. **Preserve the bs=50 launch cap.** After any change that can affect
   launch behavior, run copied-back voice validation first. If the change
   affects server launch, scheduler behavior, cache/session lifecycle,
   rollback, or model execution, re-run the full startup-prewarmed bs=50
   production-cadence soak before claiming the accepted launch envelope.
   Use bs=30 as the strict fallback proof if bs=50 regresses.

4. **Measure bs=50 only from a fresh startup-prewarmed server.** Use the
   accepted launch checkpoint/flags, startup prewarm only, runtime snapshots,
   GPU telemetry, and server phase timers. Report strict realtime,
   audio-stream realtime, first-audio latency, completion tail, rollback
   success, KV/radix usage, GPU/HBM telemetry, and phase tables. Do not run a
   manual warm traffic pass before a capacity claim.

5. **Post-launch tail hardening.** Focus on late-session long-prefix turns,
   first-codec emission, and completion-tail pickup. The accepted launch
   stack has rare p100 misses, so investigate cumulative scheduling latency
   across CodePredictor, Talker prefill/decode, tiny cached-prefix Thinker
   work, and Code2Wav completion before broad throughput tuning.

6. **Exploit MI300X HBM bandwidth where the evidence supports it.** Current
   HBM activity is low even when GPU busy is high, so look for launch-bound
   or underbatched paths that can trade more parallel memory work for fewer
   serial tiny launches: larger safe batches, component work grouping,
   shape-stable graph replay, and memory-parallel kernels. Do not pursue
   generic kernel work unless profiler/timers show latency or throughput
   headroom at the actual bs=50 shapes.

7. **Investigate regressions before rejecting an approach.** On a roadblock
   or regression, do one root-cause pass from the obvious angle, then a
   second pass through a different lens, such as logs vs profiler, isolated
   microbench vs end-to-end soak, CPU queueing vs GPU event time, or shape
   coverage vs runtime contention. Reject an idea only after both passes fail
   or expose a hard correctness/performance blocker.

8. **Review, simplify, and commit as you go.** Keep each completion unit
   small: research the issue/docs/code, implement minimal code, validate,
   review for bugs and unjustified defensive programming, simplify the diff,
   `jj commit`, and update the relevant issue/doc.

## Durable Rules

1. **Do not derive `Req.talker_prefill_len` from
   `len(req.talker_kv_cache_locs)`.** `talker_prefill_len` is fixed when
   Talker prefill runs. `talker_kv_cache_locs` grows during Talker decode.

2. **Run `$qwen3-omni-voice-validate` after audio code changes.** Text and
   RTF can stay green while Talker/CodePredictor/Code2Wav audio breaks.
   Use `--multi-turn` for session/state-flow, Talker decode, cache,
   scheduler lifecycle, or rollback changes.

3. **Preserve Qwen3 Omni semantics.** Reordering and overlap are allowed
   only when they preserve the model dependency graph, output ordering,
   hidden-state reuse, cache reuse, rollback behavior, and audio timing
   contract.

4. **Human listening approval is required for non-semantic-preserving
   optimizations.** This includes sampling changes, codec-token
   distribution changes, new quantization scopes, Code2Wav semantic/replay
   changes, cadence changes, cache behavior changes, and any change that
   can alter generated audio rather than only equivalent execution.

5. **Keep the fork rebasing-friendly.** Prefer upstream SGLang mechanisms
   and Qwen3-local changes. Touch core SGLang only when correctness or
   performance requires it, and keep those changes self-contained.

6. **Rollback is rare-path correctness, not hot-path performance.** A
   safety block may rebuild/prefill the safe branch and take multiple
   seconds. Do not optimize rollback at the expense of realtime chat.

## Validation

Correctness uses Lucy-1, the Beginnings system prompt, and `chelsie`.
Performance is judged on AMD MI300X.

For copied-back audio correctness:

```sh
cd ~/repos/qwen3-voice-validate
uv run python ~/repos/sglang-async-amd/test/manual/qwen3_omni_voice_validate.py
```

Use multi-turn validation for session/cache/scheduler/rollback changes:

```sh
cd ~/repos/qwen3-voice-validate
uv run python ~/repos/sglang-async-amd/test/manual/qwen3_omni_voice_validate.py --multi-turn
```

The validation script launches the default candidate with streaming
sessions, overlap scheduling, AITER attention, `--mem-fraction-static
0.76`, `--max-total-tokens 400000`,
`--qwen3-omni-max-concurrent-audio-turns 50`, CodePredictor argmax,
Talker decode/Thinker prefill/Talker prefill capture, steady 29-frame
Code2Wav graph replay, and current Code2Wav Triton flags.

For launch-capacity evidence, start from a fresh Qwen3 Omni gRPC server and
let the automatic startup prewarm finish. Do not run an extra hidden warm
smoke. The server launch shape includes:

```sh
python3 -m sglang.launch_server \
  --grpc-mode \
  --enable-streaming-session \
  --attention-backend aiter \
  --mem-fraction-static 0.76 \
  --max-total-tokens 400000 \
  --qwen3-omni-max-concurrent-audio-turns 50 \
  ...
```

Manual prewarm is only for debugging an already running server:

```sh
PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/manual \
  /opt/venv/bin/python /sgl-workspace/sglang/test/manual/qwen3_omni_audio_prewarm.py \
  --host 127.0.0.1 --port 41012 \
  --batch-sizes 1,2,4,8,16,32,50 \
  --chunk-ms 1000 \
  --speaker chelsie --max-new-tokens 64 --talker-max-new-tokens 256
```

For profiling and soak runs, use:

- built-in Qwen3-Omni phase logs and 5s runtime snapshots
- `test/manual/qwen3_omni_soak_telemetry.py monitor`
- `qwen3_omni_audio_soak_test.py --telemetry-jsonl "$OUT/kv_telemetry.jsonl"
  --runtime-snapshot-log /tmp/qwen3_2h_soak_server.log` to record
  main/component KV token usage and KV capacity from the scheduler snapshots
- server phase timer aggregation by `(phase, batch_size)`

For the 2-hour rolling bs=50 stability soak, keep the same fixtures,
rollback settings, realtime scoring, and launch-prewarmed server, but run:

```sh
PROMPT_FILE="$OUT/beginnings_system_prompt.txt"
PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/manual \
  /opt/venv/bin/python - <<'PY' > "$PROMPT_FILE"
from qwen3_omni_audio_prewarm import DEFAULT_SYSTEM_PROMPT
print(DEFAULT_SYSTEM_PROMPT, end="")
PY

PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/manual \
  /opt/venv/bin/python /sgl-workspace/sglang/test/manual/qwen3_omni_audio_soak_test.py \
  --host 127.0.0.1 --port 41012 \
  --sessions 50 \
  --duration-s 7200 \
  --session-duration-s 1800 \
  --turn-min-s 8 --turn-max-s 25 --initial-jitter-s 60 \
  --chunk-ms 1000 --realtime-input \
  --speaker chelsie \
  --system-prompt-file "$PROMPT_FILE" \
  --fixture greeting:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-greeting-01_turn-0.wav \
  --fixture question:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-question-01_turn-0.wav \
  --fixture emotional:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-emotional-01_turn-0.wav \
  --fixture excited:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-excited-01_turn-0.wav \
  --fixture accent:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-accent-01_turn-0.wav \
  --save-output-dir "$OUT" \
  --telemetry-jsonl "$OUT/kv_telemetry.jsonl" \
  --runtime-snapshot-log /tmp/qwen3_2h_soak_server.log \
  --rollback-rate 0.02 \
  --rollback-min-completed-turns 3 \
  --rollback-max-rewind-turns 2
```

Report at least:

- ok / failed / close failures
- strict realtime and audio-stream realtime counts
- mean/max `rtf_after_eoi`
- mean/max audio-stream RTF
- max first audio after EOI
- max complete tail after last audio
- rollback attempts/failures and max rebuild elapsed
- KV/radix usage over time
- GPU utilization, HBM activity, and VRAM over time

## Historical Investigation Index

Details live under `docs/qwen3_omni_investigations/`. Keep this file to
current goals, current metrics, validation rules, and one-line decisions.

When a topic changes direction:

- update the active roadmap here
- move detailed runs, logs, and rejected trials to a sidecar
- update exactly one index row with the latest decision
- remove or rewrite stale text instead of leaving contradictions

- [Audio correctness and validation](qwen3_omni_investigations/audio_correctness_and_validation.md):
  Parakeet copied-back STT is mandatory; text/RTF gates alone miss
  broken-audio regressions.
- [Startup prewarm and launch readiness](qwen3_omni_investigations/startup_prewarm_and_launch_readiness.md):
  Qwen3 Omni gRPC startup always runs production audio prewarm before
  health reports `SERVING`; no compatibility prewarm flag remains.
- [Streaming cadence and first-audio latency](qwen3_omni_investigations/streaming_cadence_and_latency.md):
  Current production/capacity claims use 1s client chunks; the old 100ms
  default is now a stress/profiling variant, not the launch target.
- [Session cache and streaming sessions](qwen3_omni_investigations/session_cache_and_streaming_sessions.md):
  #108 and the Beginnings client/stub migration are complete; rollback is
  rare full-prefix rebuild/prefill through the gRPC API.
- [FP8 quantization and overlap](qwen3_omni_investigations/fp8_quantization_and_overlap.md):
  FP8 Thinker plus Talker attention and overlap scheduling are accepted
  for launch; full Talker MoE FP8 is rejected.
- [Long-lived scheduler stability](qwen3_omni_investigations/long_lived_scheduler_stability.md):
  Accepted scheduler fixes preserve audio phase order; more
  scheduler-delay tuning needs a fresh profile.
- [Thinker/Talker capture and prefix cache](qwen3_omni_investigations/thinker_talker_capture_and_prefix_cache.md):
  Keep accepted capture paths; CodePredictor argmax and cached-prefix
  Talker prefill capture are approved.
- [Code2Wav replay and kernels](qwen3_omni_investigations/code2wav_replay_and_kernels.md):
  Preserve 25-frame replay semantics; optimize measured replay hot shapes
  and prove end-to-end transfer.
- [Profiling and bottlenecks](qwen3_omni_investigations/profiling_and_bottlenecks.md):
  The async architecture works; remaining work is long-history and
  p99/p100 tail reduction, especially for bs=50.
- [Post-launch capacity ideas](qwen3_omni_investigations/post_launch_capacity_ideas.md):
  SpecForge/speculative decoding may be high-EV later, but it adds
  training and semantic validation risk and is not part of the current
  bs=50 launch-stack tail work.

## Non-Goals

- Do not optimize Nvidia throughput.
- bs=50 is the accepted launch cap. Keep the rare-tail caveat visible in
  deployment docs until post-launch hardening makes the strict gate green
  over the 2-hour rolling soak.
- Do not enable broad graph bucket coverage; capture observed audio-lane
  shapes only.
- Do not tune all AITER GEMMs without a fresh profile proving they are hot.
- Do not retry generic scheduler delay/window tweaks without evidence.
- Do not pursue SpecForge/speculative decoding before launch. Track it in
  the post-launch capacity sidecar.
- Do not pursue fully stateful Code2Wav unless explicitly approved as a
  model-semantics change.
- Do not put safety-stack co-residency on the critical path for Qwen3 Omni
  launch capacity; use a sidecar if needed.
