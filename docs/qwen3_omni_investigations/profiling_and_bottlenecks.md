# Profiling And Bottlenecks

## Summary

The async architecture is broadly working: component work overlaps enough
that summed phase time can fit inside soak wall time. Remaining misses are
long-history/tail-latency issues, not a simple "audio lane is serialized"
failure.

## Decision

After #108, the next large slice should target the remaining late-turn
strict-realtime misses, not stale broad bottlenecks. Generic KV telemetry
is now collected during soaks; use it to separate long-prefix/KV pressure,
phase compute, and scheduler observation delay before changing execution
paths.

Overlap scheduling is accepted for the FP8 Thinker launch stack. It
passed copied-back multi-turn Parakeet validation, cleared the historical
bs=30 30-minute gate, and improved bs=50 versus the corrected no-overlap
FP8 baseline. No-overlap FP8 is now historical comparison material only,
not an active roadmap item.

Recent rejected launch-shape trials are historical guardrails: 80ms
partial-prefill deadline, 5-chunk partial prefill, immediate Code2Wav
completion, 5s realtime input chunks, naive idle Code2Wav coalescing, and
lower static KV reservation did not improve the full bs=50 launch gate.

## Latest Relevant Soaks

Cached-prefix Talker-prefill capture:

- change: capture/replay cached-prefix Talker prefill with reconstructed
  AITER prefill metadata, gated to small cached-prefix production shapes
- copied-back voice validation on the final soak server passed all five
  fixtures: `/tmp/qwen3_voice_validate/talker-prefill-capture-soak-server/report.json`
- bs=50 / 300s smoke: `/tmp/qwen3_talker_prefill_capture_perf2_smoke_300s/stdout.json`
  - 713/713 ok
  - 711/713 strict realtime
  - 713/713 audio-stream realtime
  - 14/14 rollback success
  - mean/max RTF 0.509/1.031
- bs=50 / 1800s full soak:
  `/tmp/qwen3_talker_prefill_capture_soak_30min/stdout.json`
  - 4122/4122 ok, 0 close failures
  - 4117/4122 strict realtime
  - 4119/4122 audio-stream realtime
  - 73/73 rollback success
  - mean/max completion RTF 0.548/1.142
  - mean/max audio-stream RTF 0.500/1.078
  - max first-audio-after-EOI 0.980s
  - max completion tail 0.522s
- previous full baseline before this change:
  `/tmp/qwen3_bs50_flush_priority_30min_20260519T0309Z/stdout.json`
  was 4054/4075 strict, mean/max RTF 0.608/1.273
- decision: accept the capture path as the best bs=50 direction so far.
  It removes most steady Talker-prefill fixed cost and leaves only p100
  tail work.
- caveat: an eager-then-captured hidden checker was removed. It is not a
  valid cached-prefix equivalence test because the eager pass mutates the
  same component KV slots before graph replay. Any future hidden checker
  must use independent component KV slots.
- remaining misses in the full soak:
  - audio_stream>1: 3
  - first_audio>800ms: 1
  - fixed_overhead: 1
  - no completion-tail strict misses
- remaining top causes:
  - Code2Wav forward-enqueue spikes at line 169880 (`bs=3`,
    `wall_ms=782.44`, `max_forward_enqueue_ms=782.12`) and line 318296
    (`bs=3`, `wall_ms=870.40`, `max_forward_enqueue_ms=870.09`)
    in the raw container log
  - Talker-prefill new-shape/capture spikes around 120-270 ms despite a
    warmed p50 around 7-9 ms
- artifacts:
  `/tmp/qwen3_talker_prefill_capture_soak_30min_server.log`,
  `/tmp/qwen3_talker_prefill_capture_soak_30min/miss_report.md`,
  `/tmp/qwen3_talker_prefill_capture_soak_30min_telemetry.jsonl`,
  `/tmp/qwen3_talker_prefill_capture_soak_30min/telemetry.svg`

Clean long-bucket / prefix-cache / Thinker-capture baseline:

- bs=30 / 1800s randomized long-lived sessions
- 2259/2259 turns ok
- 0 generation failures
- 0 close failures
- completion realtime: 1451/2259
- audio-stream realtime: 1678/2259
- mean/max completion RTF 1.016/4.517
- mean/max audio-stream RTF 0.834/3.543
- max first-audio-after-EOI 11.835s
- failure worsened with turn index
- phase elapsed led by Code2Wav and Talker prefill:
  - Code2Wav 766s
  - Talker prefill 536s
  - CodePredictor 196s
  - Talker decode 151s

Later kernel/checkpoint context:

- SnakeBeta + audio-only mRoPE 30-minute soak improved but still missed:
  2406/2406 ok, 2098/2406 completion realtime, mean/max completion RTF
  0.680/2.386
- Round 2 transconv soak moved mean RTF in the right direction but did
  not reliably improve the strict gate; turn0/cold-shape workflow
  dominated one comparison.

Session/cache overhead probe:

- 2458/2458 ok
- 2422/2458 strict realtime
- mean/max RTF 0.555/1.318
- cache-ish overhead about 64.4s over 30 minutes
- useful for #108, but not the largest raw component

Generic-KV telemetry and Code2Wav timing split:

- bs=30 / 1800s randomized long-lived sessions
- 2482/2482 ok, 0 close failures
- strict realtime: 2475/2482
- audio-stream realtime: 2478/2482
- mean/max completion RTF 0.513/1.254
- mean/max audio-stream RTF 0.462/1.166
- max first-audio-after-EOI 1.302s
- GPU utilization mean/p95/max 89.1%/97.0%/99.0%
- VRAM used mean/p95/max 163.6/173.7/175.0 GiB
- main/Thinker KV used mean/p95/max 120315/191471/199360 slots
- streaming-session held tokens mean/p95/max 104943/171553/191671
- Talker component KV used mean/p95/max 52739/98093/103250 slots
- strict misses were late/long-prefix: prompt tokens 4070-6090, with
  5/7 misses at turn index >= 73
- Code2Wav steady enqueue/GPU time was bounded after warmup:
  `enqueue_wall_ms` p50/p95/p99 12.38/13.26/18.86 and
  `gpu_elapsed_ms` p50/p95/p99 12.38/13.92/19.01
- Code2Wav completion observation still had pending tails:
  `pending_elapsed_ms` p50/p95/p99/max 0.53/37.69/53.70/380.99
- Some rare true enqueue/GPU spikes remain, but repeated huge
  `complete_async_code2wav.wall_ms` rows are usually scheduler/stream
  observation delay, not unbounded Code2Wav input size
- Root cause for much of the observation delay: the selector could log
  `reason=code2wav_async_complete`, but the runner would still drain
  another audio phase first when the audio phase queue was non-empty. In
  the pre-fix soak, 4089/29781 `code2wav_async_complete` decisions were
  followed by Talker/CodePredictor work before Code2Wav completion was
  harvested.

Completion-priority validation:

- bs=30 / 300s warm smoke passed: 430/430 strict realtime, max RTF 0.906,
  max first-audio 0.551s
- bs=30 / 1800s randomized long-lived sessions still missed:
  2458/2458 ok, 2440/2458 strict realtime, 2451/2458 audio-stream
  realtime, mean/max completion RTF 0.532/1.233
- first-audio max was 0.941s; 2 turns exceeded the 800 ms target
- Code2Wav completion priority inversion was eliminated:
  0/27347 full-soak `code2wav_async_complete` decisions were followed by
  another audio phase before completion
- Code2Wav completion observation improved materially:
  `pending_elapsed_ms` p50/p95/p99/max 0.51/18.92/29.42/65.24 ms
- Misses remain late/long-prefix dominated and mostly have near-zero
  final completion tail, so the next work should target long-prefix
  Talker prefill / fixed latency rather than more Code2Wav completion
  polling
- Voice-only resource peaks: GPU utilization 99%, HBM activity 22%, VRAM
  175.1/191.7 GiB, main/Thinker KV 199976 slots, Talker KV 101989 slots

Post-rebase bs=50 Code2Wav/Thinker overlap check:

- accepted scheduler change: pending non-urgent Code2Wav work no longer
  blocks Thinker prefill; first PCM, frame-lag-limit work, deadline-missed
  work, and stale pending work still block Thinker to protect audio timing
- copied-back voice validation passed all five fixtures
- bs=50 / 300s smoke passed 701/701 strict realtime and 701/701
  audio-stream realtime, max RTF 0.968, max first-audio 0.807s, 14/14
  rollback success
- bs=50 / 1800s full soak remained short of the expansion gate:
  4060/4060 ok, 4031/4060 strict realtime, 4053/4060 audio-stream
  realtime, 0 close failures, 73/73 rollback success, mean/max RTF
  0.613/1.327, max first-audio 1.152s, max completion tail 0.536s
- miss classes were fixed-overhead and late/long-prefix dominated:
  fixed_overhead 15, audio_stream>1 7, first_audio>800ms 6,
  complete_tail>100ms 1
- phase totals in the full soak: Talker prefill 156.5s, Code2Wav 487.5s,
  CodePredictor 291.3s, Talker decode 226.6s. Talker prefill batch cost
  was nearly flat, around 39ms for bs=1 and bs=2, but most first-PCM
  prefills naturally arrived as bs=1.
- artifacts:
  `/tmp/qwen3_bs50_overlap_gate_smoke_client_20260518T144733Z/stdout.json`,
  `/tmp/qwen3_bs50_overlap_gate_30min_20260518T145509Z/stdout.json`

Talker multi-shared-expert fusion trial:

- hypothesis: Talker `shared_expert_intermediate_size=768` can be split
  into two `moe_intermediate_size=384` fused shared experts, avoiding a
  separate shared MLP branch inside the Qwen2-style Talker MoE block
- copied-back voice validation passed all 5 fixtures
- first bs=50 / 300s smoke regressed: 697/697 ok, 687/697 strict,
  mean/max RTF 0.615/1.288, max first-audio 1.233s
- warm repeat mostly recovered: 708/708 ok, 707/708 strict, 708/708
  audio-stream realtime, mean/max RTF 0.569/1.033, max first-audio
  1.015s, rollback 31/31
- phase timing did not show the needed Talker-prefill win:
  `talker_prefill bs=1` p50/p95/max wall 36.15/39.38/47.66 ms and
  model p50/p95/max 36.07/39.37/47.57 ms, effectively unchanged from
  the prior ~36.5 ms GPU timing
- decision: reject for now and do not carry the loader/model complexity.
  Likely root cause is that increasing MoE top-k from 6 to 8 with fused
  shared experts cancels the removed shared MLP for the tiny Talker
  prefill shapes.
- artifacts:
  `/tmp/qwen3_voice_validate/run/report.json`,
  `/tmp/qwen3_bs50_shared_fusion_300s_stdout.json`,
  `/tmp/qwen3_bs50_shared_fusion_300s_r2_stdout.json`

Talker-prefill per-layer timing:

- temporary env-gated CUDA-event instrumentation was used for diagnosis
  only, then removed; clean code was restored on AMD and sanity
  validation passed at `/tmp/qwen3_voice_validate/restore-clean/report.json`
- artifact: `/tmp/qwen3_talker_layer_profile_local/server.log`
- diagnostic workload: bs=20 / 60s, 69/69 strict realtime
- warm `bs=1`, `total_tokens p50=49` split over 20 layers:
  - attention prep p50 1.16 ms
  - attention p50 16.00 ms
  - MLP prep p50 1.09 ms
  - MoE/MLP p50 16.25 ms
  - postprocess p50 0.24 ms
- higher batch sizes were similar: attention stayed around 16 ms and
  MoE/MLP around 17 ms. One cold MLP outlier hit 3438 ms and was not
  representative of warmed rows.
- decision: Talker prefill is not dominated by setup or one shared-expert
  submodule. Prefer whole-model capture/fast-path with hidden-state
  equivalence, or semantic batching that produces larger real batches,
  over isolated shared-MLP rewrites.

Text-token Thinker tiny-prefix capture extension:

- change: text-only cached-prefix extends pre-embed their tiny suffix and
  reuse the existing Thinker prefill graph path, matching the audio-input
  capture shape without changing model semantics
- copied-back voice validation passed all five fixtures
- bs=50 / 300s smoke was noisy: first run 703/705 strict realtime,
  repeat 704/704 strict realtime with mean/max RTF 0.531/0.867 and max
  first-audio 0.584s
- bs=50 / 1800s full soak improved the strict miss count but still missed
  the expansion gate: 4087/4087 ok, 4072/4087 strict realtime,
  4079/4087 audio-stream realtime, 0 close failures, 74/74 rollback
  success, mean/max RTF 0.600/1.347, max first-audio 1.230s, max
  completion tail 0.860s
- full-soak misses were still late/long-prefix dominated; the loud new
  p100 events were true wall spikes, including one `code2wav` row at
  831.81 ms for bs=5 and one `talker_prefill` row at 774.19 ms for bs=1
- artifacts:
  `/tmp/qwen3_bs50_text_prefill_capture_smoke_20260518T170506Z/stdout.json`,
  `/tmp/qwen3_bs50_text_prefill_capture_smoke2_20260518T171251Z/stdout.json`,
  `/tmp/qwen3_bs50_text_prefill_capture_30min_20260518T172012Z/stdout.json`

Current bs=50 tail snapshot after rejected/offloaded paths were reverted:

- artifact: `/tmp/qwen3_bs50_launch_debug_30min_20260518T190100Z/stdout.json`
- miss report: `/tmp/qwen3_bs50_launch_debug_30min_20260518T190100Z/miss_report.md`
- telemetry: `/tmp/qwen3_bs50_launch_debug_30min_20260518T190100Z/telemetry.jsonl`
- aggregate: 4064/4064 ok turns, 0 close failures, 4050/4064 strict
  realtime, 4061/4064 audio-stream realtime, 73/73 rollback success,
  mean/max RTF 0.603/1.112, max first audio 1.377s, max completion tail
  0.374s
- telemetry: avg/p95/max GPU busy 92.6%/96.0%/99.0%, avg/p95/max HBM
  activity 8.4%/12.0%/15.0%, avg/p95/max VRAM used
  149.3/170.0/173.1 GiB
- miss classes: fixed_overhead 8, first_audio>800ms 3,
  audio_stream>1 3, complete_tail>100ms 0
- misses were late/long-history skewed: median/max turn index 72/84,
  median/max prompt tokens 5752/6852
- one Code2Wav `launch_async_code2wav` row at 751.62 ms affected several
  nearby audio-stream misses; completion happened promptly after launch,
  so this was a true forward/enqueue wall spike rather than a
  completion-polling tail
- first-audio misses had several Talker-prefill rows after EOI, typically
  38-46 ms each. The hot path remains model-forward dominated, not Python
  queue observation.

Talker chunked-prefill gap:

- follow-up log review found the largest Talker-prefill rows were
  `sample_codec=False`, `has_prefix=False`, 2.5k-3.2k token full-prefix or
  rebuild-style rows. They ran as one custom Qwen3 Omni Talker
  component-KV prefill instead of SGLang's normal chunked-prefill path.
- this looked like a high-EV architecture oversight because rare-path
  rollback can take seconds and a full-prefix prefill should not delay live
  speech for other sessions.
- first implementation note: a 512-token Qwen3-local Talker-prefill chunk
  path passed copied-back voice validation for all 10 single-turn and
  multi-turn fixtures, but regressed bs=50 tail performance:
  `/tmp/qwen3_talker_chunk_bs50_30min_20260518T205514Z/stdout.json`
  produced 4067/4096 strict realtime and 4085/4096 audio-stream realtime,
  versus the current baseline
  `/tmp/qwen3_bs50_launch_debug_30min_20260518T190100Z/stdout.json`
  at 4050/4064 strict realtime and 4061/4064 audio-stream realtime with
  better p100. The chunked run had max RTF 1.242, max audio-stream RTF
  1.142, max first audio 1.341s, max tail 0.454s, and max rollback rebuild
  2.265s.
- root cause pass 1: the old 2.5k-3.2k `sample_codec=False` rows were
  already about 70-80 ms total with about 35 ms model time. The 512-token
  prototype made every chunk pay roughly the same 33-35 ms Talker forward,
  plus prep, so long prefills became 6-7 full forwards. That is why a
  superficially reasonable chunk split can regress realtime performance.
- root cause pass 2: miss reports showed the chunked run still missed in
  audio-stream/fixed-overhead rows with repeated Talker-prefill rows around
  40-70 ms. Chunk continuations often drained as bs=1, so the scheduler
  paid extra launches without enough useful interleaving.
- root cause pass 3: an adaptive 2048-token/live-pressure policy passed
  copied-back voice validation but did not activate in the full bs=50
  30-minute run
  `/tmp/qwen3_pressure_chunk_bs50_30min_20260518T224331Z/stdout.json`.
  That run saw 60 long non-codec Talker-prefill rows, 24 above 2048
  tokens, but `chunk_allowed=0` and `chunked=0`: the long rows did not
  coincide with queued live-audio pressure under the scheduler's view. The
  run missed 18/4071 strict turns and 7/4071 audio-stream turns, with mean
  / max RTF 0.603 / 1.175, mean / max audio-stream RTF 0.545 / 1.051, max
  first audio 1.268s, max completion tail 0.518s, and max rollback rebuild
  1.902s. The prototype was removed from code after this evidence.
- current decision: do not pursue rare non-codec Talker chunking next.
  Keep it as a post-evidence option only if a future miss report shows
  long `sample_codec=False` rows actually coinciding with live-audio
  pressure. The next higher-EV Talker-prefill work is committed
  `sample_codec=True` rows: they remain mostly bs=1 and cost about
  37-46 ms each in miss windows after EOI or during audio streaming.
- follow-up debug smoke
  `/tmp/qwen3_debug_talker_prefill_300s_local/stdout.json` passed
  704/705 strict realtime and 705/705 audio-stream realtime. It captured
  Talker-prefill substages with `SGLANG_QWEN3_OMNI_DEBUG_AUDIO=1`:
  ordinary cached-prefix committed rows were mostly bs=1
  (`662/707` Talker-prefill drains). For bs=1 committed rows with a
  prefix, p50/p95/p99/max total was `36.49/37.62/46.19/47.65 ms`.
  Substage p50 was prep `1.66 ms`, alloc `0.02 ms`, metadata `0.61 ms`,
  model `33.40 ms`, sampling `0.68 ms`, and dependent CodePredictor
  setup `0.12 ms`.
- follow-up GPU-timed smoke
  `/tmp/qwen3_debug_talker_prefill_gpu_120s_local/stdout.json` passed
  296/296 strict realtime. With `SGLANG_QWEN3_OMNI_PROFILE_GPU_TIMING=1`,
  Talker-prefill wall/model/GPU times matched: bs=1 p50 was
  `36.72/36.56/36.58 ms`, so this is real GPU model-forward time, not
  hidden Python or queue observation overhead. Batching has strong
  throughput leverage (`bs=12` p50 GPU about `51.57 ms`), but previous
  first-PCM grace and nonconsecutive batching attempts regressed end to
  end. Therefore the next acceptable Talker-prefill paths are hidden-state
  equivalence before any capture/model-fast-path retry, or a batching
  change that avoids the already-documented scheduler regressions.
- follow-up lane-decision parsing of
  `/tmp/qwen3_bs50_launch_debug_30min_20260518T190100Z/server.log` showed
  why simple queue reordering is unlikely to help. Every Talker-prefill
  scheduler decision used `phase_depends_on_thinker_lane`. Out of 3959
  Talker-prefill drains, 3789 were bs=1, 164 were bs=2, 5 were bs=3, and
  1 was bs=5. The pending audio queue depth at Talker-prefill selection
  was almost always 1; only 3 decisions had depth above 1, and max depth
  was 2. That means there are not hidden adjacent compatible prefill rows
  waiting in the queue. A future batching improvement would have to create
  a deliberate yield window, not just pull later work forward, and must
  explain why it differs from the rejected first-PCM grace/nonconsecutive
  batching trials.
- 2026-05-19 early Talker-prefill capture probes were rejected. The
  first probe reused the CodePredictor-style AITER `audio_prefill` metadata
  for uniform no-prefix Talker prefill; copied-back validation failed
  during launch prewarm on the first bs=1, seq_len=42 row with hidden-state
  mismatch (`/tmp/qwen3_validate_talker_prefill_capture_metadata_probe_server.log`,
  `max_abs=17.5`). A second probe attempted to reconstruct the normal
  Talker prefill metadata with stable buffers and compared it before graph
  replay; it still failed eager-vs-candidate hidden-state equivalence
  (`/tmp/qwen3_validate_talker_prefill_capture_clone_probe_server.log`,
  `max_abs=14.25`). Decision from those probes: do not swap in
  CodePredictor metadata or compare eager/captured results after mutating
  the same component KV slots. The accepted cached-prefix capture path is
  documented at the top of this file and validated by copied-back Parakeet
  checks plus bs=50 smoke/soak results.

## Bottleneck Interpretation

- CodePredictor and Talker decode have already received graph/capture
  work and are closer to graph-replay floor than kernel floor.
- Talker prefill is the top current architecture gap for bs=50, but not
  because rare non-codec chunking is immediately useful. The urgent issue
  is ordinary committed Talker prefill: many bs=1 rows around 37-46 ms show
  up in miss windows, and debug GPU timing shows the Talker model forward
  itself accounts for almost all of that time. Metadata/setup cleanup is
  not the high-EV path for this row class. The immediate audit should
  cover all non-Thinker components, not only Talker prefill, then
  prioritize fixes by measured bs=50 p99/p100 impact.
- Code2Wav replay remains a large explicit component, but many local
  kernel micro-wins did not transfer to server-level bs=30.
- Hidden normal Thinker tiny-prefix extends were a previous load source;
  #108 should reduce repeated radix/session overhead and should be
  measured before more bespoke work.
- Tail misses can be environment/cold-shape/workflow driven; always
  inspect turn index, first-audio, output duration, and phase rows before
  declaring a new bottleneck.
- A `complete_async_code2wav.wall_ms` spike is not by itself proof that
  Code2Wav compute spiked. Check the split fields:
  `pending_elapsed_ms`, `gpu_elapsed_ms`, and `enqueue_wall_ms`.
- If Code2Wav completion tails remain high after the runner priority fix,
  inspect true `gpu_elapsed_ms` / `enqueue_wall_ms` spikes separately from
  scheduler-observation delay.
- The current bs=50 blocker is now less about average throughput and more
  about rare p100 spikes in Code2Wav and Talker prefill. Do not smooth
  them away with mean phase totals; inspect the single offending rows and
  correlate them with GPU telemetry, cache pressure, and first-use/cold
  shape behavior.

Code2Wav flush-vs-live priority follow-up:

- previous failing bs=50 full run had two true `launch_async_code2wav`
  enqueue stalls, 751.6 ms and 841.2 ms, both inside
  `max_forward_enqueue_ms`; this was not completion-pickup delay
- the 841 ms row was a one-frame `flush=True` finished-turn Code2Wav row
  that ran before a dependent CodePredictor batch, causing that
  CodePredictor row to wait 848 ms
- accepted scheduler rule: finished-turn flush audio is tail work; when
  urgent non-flush Code2Wav or frame-producing audio work is waiting,
  drain non-flush Code2Wav first
- copied-back voice validation after the change:
  `/tmp/qwen3_voice_validate/code2wav_flush_live_priority/report.json`,
  10/10 single + multi-turn fixtures passed
- bs=50 300s smoke after the change:
  `/tmp/qwen3_code2wav_flush_priority_smoke_300s/stdout.json`,
  704/704 strict realtime, 704/704 audio-stream realtime, mean/max RTF
  0.554/0.941, max first audio 0.832s, max completion tail 0.355s,
  rollback 14/14
- smoke server-log parse: Code2Wav launch p99 was about 19 ms, max
  22.93 ms, and no `flush=True` launch exceeded 50 ms
- full bs=50 / 1800s soak after the change:
  `/tmp/qwen3_bs50_flush_priority_30min_20260519T0309Z/stdout.json`
  remained short of the expansion gate: 4075/4075 ok, 4054/4075 strict
  realtime, 4069/4075 audio-stream realtime, 73/73 rollback success,
  mean/max RTF 0.608/1.273, mean/max audio-stream RTF 0.549/1.079,
  max first audio 1.293s, max completion tail 0.604s
- telemetry for that run: avg/p95/max GPU busy 91.5/96.0/98.0%, avg/p95/max
  HBM activity 8.3/12.0/14.0%, avg/p95/max VRAM 150.3/176.0/178.3 GiB
- full-run Code2Wav launch parse: 34,518 launches, wall p99 about
  19.4 ms, only 7 launches with queue_wait >=20 ms, and only 2 >=100 ms.
  One non-flush row reported queue_wait 743 ms with wall 20.6 ms; treat
  this as a stale ready-state/scheduler-observation investigation item
  unless it recurs as a causal miss root.
- runtime histograms confirmed Talker prefill remains the structural
  mismatch: 3,781/3,973 Talker-prefill drains were bs=1, while Talker
  decode and CodePredictor naturally batched into the high single digits
  and beyond. Smaller Talker-prefill chunks are not the next lever; the
  next useful work is either semantics-preserving yield/batching that
  avoids the rejected grace-window failures, or an actual fixed-cost
  reduction for the committed Talker-prefill model forward.
- one-Thinker-pass Talker-prefill yield was tested and rejected:
  copied-back single + multi-turn voice validation passed, but the first
  bs=50 300s smoke had zero `talker_prefill_yield_once` scheduler
  decisions and missed 2/701 strict realtime turns. A second pass narrowed
  `last_batch` prefill detection so the guard could open, but it still
  produced zero yield decisions and regressed the smoke to 691/711 strict
  realtime, 709/711 audio-stream realtime, max RTF 1.298. The regression
  came from changing other audio/Code2Wav scheduling behavior, not from
  useful Talker-prefill batching. Do not retry this shape without a new
  causal proof that it will actually create committed Talker-prefill
  batches.
- upstream SGLang `PrefillDelayer` was tested as a low-diff candidate for
  creating better prefill batching, then rejected for launch flags. A
  same-seed bs=50 / 300s smoke with
  `--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.20
  --prefill-delayer-max-delay-ms 20` improved strict count versus the
  local default smoke (`715/719` vs `705/717`) and modestly improved mean
  RTF, but Talker prefill still did not batch materially: 611/653
  Talker-prefill rows were bs=1, and every Talker-prefill decision still
  used `phase_depends_on_thinker_lane`.
- full bs=50 / 1800s prefill-delayer run
  `/tmp/qwen3_prefill_delayer_bs50_30min_20260519/stdout.json` remained
  short of the expansion gate: 4129/4129 ok, 4105/4129 strict realtime,
  4117/4129 audio-stream realtime, 0 close failures, 66/66 rollback
  success, mean/max RTF 0.604/1.279, mean/max audio-stream RTF
  0.545/1.188, max first audio 1.144s, max completion tail 0.804s.
  Telemetry was high GPU-busy but low HBM activity:
  mean/max GPU busy 85.8/97.0%, mean/max HBM activity 8.0/14.0%, max VRAM
  177.2 GiB.
- prefill-delayer full-run miss classes were `audio_stream>1` 12,
  fixed-overhead 8, `first_audio>800ms` 4, and `complete_tail>100ms` 1.
  Talker-prefill hist stayed structurally unchanged: 3749/3966 rows were
  bs=1, only 217 rows were bs>=2, and the only scheduler reason was
  `phase_depends_on_thinker_lane`. CodePredictor, Talker decode, and
  Code2Wav continued to batch normally. Decision: do not use upstream
  `PrefillDelayer` as a bs=50 launch flag. It may still be useful only if
  the Qwen3 Omni audio-phase scheduler is changed so Talker-prefill work
  can actually accumulate compatible batches without repeating the
  rejected grace/yield regressions.
- Talker-prefill shape parse for the same full run: committed
  `sample_codec=True` rows were 3919 items / 4250 request rows; item
  batch-size top counts were bs=1:3690, bs=2:208, bs=3:10, bs=12:5.
  Request-row suffix lengths had p50/p95/max `49/65/65`, with top lengths
  36, 65, 49, 42, and 59. Cached prefix was present for 3852/3919
  committed items. Non-codec rows were rare but long: 66 items, all bs=1,
  p50/p95/max `1587/2608/3076`. This means the hot committed path is a
  tiny cached-prefix extend, but not merely the fixed 9-token assistant
  bootstrap; a whole-small-extend fast path has more upside than only
  splitting the assistant bootstrap.
- Matching shape rows against drain rows showed bs=1 committed
  Talker-prefill wall time is essentially length-independent across the
  production suffix range: p50 was 37.52 ms at length 20, 37.35 ms at 25,
  37.52 ms at 29, 37.38 ms at 36, 37.40 ms at 42, 37.55 ms at 49,
  37.63 ms at 59, and 37.74 ms at 65. This rules out smaller audio chunks
  as the next high-EV fix for committed prefill; the problem is fixed
  per-forward Talker cost.

Code2Wav runtime-stream prewarm follow-up:

- root cause under test: model startup already prewarmed Code2Wav shapes
  on the default stream, while live Code2Wav runs on a dedicated async
  stream. A full prefill-delayer run had one non-flush Code2Wav
  `launch_async_code2wav` row with `max_forward_enqueue_ms=769.77`, so the
  next startup-hardening change was to prewarm the same production
  `(bs,dframes)` shape grid on the dedicated runtime stream before first
  traffic.
- implementation: `Qwen3OmniMoeCode2Wav.prewarm_runtime_shapes(...)`
  centralizes the shape grid and is called both by model prewarm and by
  the first runtime Code2Wav stream creation.
- copied-back voice validation passed all five fixtures:
  `/tmp/qwen3_voice_validate/code2wav-stream-prewarm/report.json`.
- clean bs=50 / 300s smoke:
  `/tmp/qwen3_code2wav_stream_prewarm_clean_smoke_300s/stdout.json`
  reported 711/711 ok, 704/711 strict realtime, 711/711 audio-stream
  realtime, 12/12 rollback success, mean/max RTF 0.602/1.078, max first
  audio 0.833s, and max completion tail 0.286s.
- runtime-stream prewarm cost in the clean server log was 12.44s after
  the normal Code2Wav model prewarm. This is acceptable startup cost
  because Qwen3 Omni startup prewarm runs before health reports `SERVING`.
- the clean smoke had zero Code2Wav forward-enqueue spikes above 50 ms.
  Remaining strict misses were all fixed-overhead misses, with miss
  windows dominated by Talker-prefill rows around 42-58 ms. Decision:
  keep runtime-stream Code2Wav prewarm as startup/tail hardening, but do
  not treat it as the bs=50 unlock. The next high-EV target remains
  Talker-prefill fixed-cost reduction or a semantics-preserving batching
  policy that avoids the rejected grace/yield regressions.

Code2Wav oldest-ready scheduling and latest bs=50 p100 shape:

- root cause fixed: Code2Wav ready states were grouped by request/group
  insertion order while `max_pending_groups=1` allowed only one async
  Code2Wav group per scheduler pass. An older ready state in a later
  group could wait behind newer groups. Work groups now run oldest-ready
  first, and items inside each group are sorted by `ready_at`.
- copied-back voice validation passed all five fixtures:
  `/tmp/qwen3_voice_validate/code2wav-oldest-ready/report.json`.
- first smoke after the fix:
  `/tmp/qwen3_code2wav_oldest_ready_smoke_300s/stdout.json`,
  726/726 strict realtime, 726/726 audio-stream realtime, max first audio
  0.909s. The top rows were early parented continuation turns, so startup
  prewarm coverage, not Code2Wav ordering, was still incomplete.
- repeat smoke after expanding continuation prewarm to bs=50:
  `/tmp/qwen3_code2wav_oldest_ready_bs50prewarm_smoke_300s/stdout.json`,
  726/726 strict realtime, 726/726 audio-stream realtime, max first audio
  0.675s.
- full bs=50 / 1800s after both changes:
  `/tmp/qwen3_code2wav_oldest_ready_bs50prewarm_soak_30min/stdout.json`,
  4227/4227 strict realtime, 4227/4227 audio-stream realtime, mean/max
  RTF 0.461/0.969, mean/max audio-stream RTF 0.419/0.845, max first audio
  1.385s, max completion tail 0.196s, 75/75 rollback success.
- diagnosis: Code2Wav queueing was not causal for the latest p100
  first-audio outlier. The worst turn's first Code2Wav row had
  queue_wait_ms=0.21 and max_forward_enqueue_ms=13.40. The remaining
  p100 gate is upstream of first Code2Wav, in late-session
  EOI-to-first-codec scheduling/model work with long cached prefixes.
- first-hit clue: the same p100 cluster logged fresh AITER BF16 output
  GEMMs at `M=154` and `M=74` with `(N=128,K=2048)`. Those are not Talker
  prefill GEMMs and were not covered by the observed Talker-prefill warm
  list.
- bounded small-M output prewarm probe: warming every `(M,128,2048)` for
  `M=1..256` passed copied-back voice validation, but the fresh bs=50
  300s smoke `/tmp/qwen3_small_m_output_prewarm_smoke_300s/stdout.json`
  did not prove transfer. It stayed 728/728 strict and audio-stream
  realtime, but max first audio was 0.971s versus 0.675s in the prior
  bs=50-continuation-prewarm smoke. Root-cause pass 1: the top
  first-audio rows (`s47-t0011`, `s22-t0011`, `s00-t0003`) had normal
  Code2Wav queue/enqueue behavior and were not caused by fresh
  `(M<257,128,2048)` AITER output first hits. Root-cause pass 2: the
  remaining AITER churn in that run was mostly long output rows and
  Talker-prefill triplets such as `(M,1536,1024)`, `(M,1024,768)`, and
  `(M,128,1024)`, so the broad output-only range was not a complete
  structural fix. Decision: reject the broad `M=1..256` output prewarm for
  now, keep only narrower observed output warm shapes, and revisit broad
  output-shape warming only with a new causal p100 row.
- validation after removing the broad range: copied-back voice validation
  passed all five fixtures at
  `/tmp/qwen3_voice_validate/remove-broad-small-m-prewarm/report.json`.
  A fresh-start bs=50 / 300s smoke from automatic startup prewarm only,
  `/tmp/qwen3_remove_broad_small_m_smoke_300s/stdout.json`, passed
  730/730 strict realtime and 730/730 audio-stream realtime, with 14/14
  rollback success, mean/max RTF `0.417/0.817`, mean/max audio-stream RTF
  `0.380/0.700`, max first audio `0.621s`, and max completion tail
  `0.134s`. This is better than both the broad-prewarm smoke and the
  previous continuation-prewarm smoke for first-audio p100, so the broad
  output range stays removed.

## Profiling Runbook

Use server phase timers as the primary signal:

```sh
rg "Qwen3-Omni audio phase drained|Qwen3-Omni scheduler action selected" \
  /tmp/qwen3_omni_server.log
```

Aggregate warm-window rows by `(phase, batch_size)`:

- call count
- mean/p50/p95/max `wall_ms`
- mean/max `queue_wait_ms`
- mean/max `deadline_miss_ms`
- phase share

Report Talker prefill separately. Exclude cold outliers whose `wall_ms`
is more than twice the median for the same `(phase, batch_size)`.

The AMD HTTP sidecar `/start_profile` and `/stop_profile` routes are not
currently reliable in the container because the installed
`smg-grpc-servicer` lacks those routes. Full-process `rocprof` can
support a hypothesis, but startup/H2D pollution means it should not be
the primary bottleneck signal.

## Artifact Examples

- `/tmp/qwen3_current_30min.json`
- `/tmp/qwen3_current_30min_server.log`
- `/tmp/qwen3_snakebeta_mrope_30min.json`
- `/tmp/qwen3_soak_round2_s3_s4_30min.json`
- `/tmp/qwen3_session_cache_probe2_30min.json`
- `/tmp/qwen3_generic_kv_soak/qwen3_generic_kv_soak_1800s.json`
- `/tmp/qwen3_generic_kv_soak/qwen3_generic_kv_soak_1800s_telemetry.jsonl`
- `/tmp/qwen3_generic_kv_soak/qwen3_generic_kv_soak_1800s_telemetry.svg`
- `/tmp/qwen3_generic_kv_soak/qwen3_generic_kv_telemetry_server.log`
- `/tmp/qwen3_code2wav_priority_soak/qwen3_code2wav_priority_soak_1800s.json`
- `/tmp/qwen3_code2wav_priority_soak/qwen3_code2wav_priority_soak_1800s_telemetry.jsonl`
- `/tmp/qwen3_code2wav_priority_soak/qwen3_code2wav_priority_soak_1800s_telemetry.svg`
- `/tmp/qwen3_code2wav_priority_soak/qwen3_code2wav_priority_perf_server.log`
