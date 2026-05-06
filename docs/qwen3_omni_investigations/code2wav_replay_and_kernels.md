# Code2Wav Replay And Kernels

## Summary

Code2Wav remains an important measured component, but the obvious
"streaming state" shortcut is blocked by model semantics. The current
safe scope is optimizing the existing 25-frame replay semantics, not
changing Code2Wav into a stateful absolute-position decoder.

## Decision

Preserve upstream `chunked_decode` semantics unless explicitly approved as
a model change. Future Code2Wav optimization must target measured replay
hot shapes and prove server transfer after isolated wins.

Do not treat every `complete_async_code2wav.wall_ms` tail as Code2Wav
compute. The May 14 generic-KV telemetry soak split async Code2Wav timing
into launch/enqueue, GPU-event elapsed, and scheduler-observed pending
elapsed. Steady enqueue/GPU time is bounded around 12-14 ms at p50-p95,
while many large completion rows are delayed scheduler observation on the
Code2Wav stream. Rare true enqueue/GPU spikes still exist, but they are
not explained by unbounded input size.

The main scheduler-observation bug was a priority inversion: the lane
selector could choose `code2wav_async_complete`, then the runner drained
Talker/CodePredictor work first if the audio phase queue was non-empty.
That made completed Code2Wav work sit in the pending list. Fix scheduler
ordering before drawing more kernel conclusions from async completion
latency.

After the completion-priority fix, the full-soak runner check showed
0/27347 `code2wav_async_complete` decisions followed by another audio
phase before completion. Code2Wav `pending_elapsed_ms` improved to
p50/p95/p99/max 0.51/18.92/29.42/65.24 ms. The remaining bs=30
30-minute misses are therefore not primarily Code2Wav completion-polling
tails.

The May 14 bs=30 priority soak also exposed a separate startup/cold-shape
problem. Code2Wav prewarm was enabled, but it warmed only 4-frame and
29-frame windows. Production traffic still hit cold one-request windows
at 8, 12, 16, 20, 24, and 25-28 frames, with launch rows such as
`bs=1/min_window_frames=28/wall_ms=844.79`, 27/822.22, 26/800.16,
25/767.50, 24/727.14, 20/623.09, and 16/494.17. Those rows had almost no
queue wait, so they were real warmup/JIT/MIOpen shape misses rather than
bounded Code2Wav compute. The committed prewarm path should cover the
observed rolling and flush window shapes; if a production prewarm shape
fails, startup should fail loudly.

Post-fix smoke: bs=30 / 300s on AMD with the expanded 133-shape prewarm
produced 428/428 strict realtime turns, max RTF 0.908, and max first
audio 0.624s. Code2Wav `launch_async_code2wav` was bounded at
p95/p99/max 13.11/19.15/24.11 ms; the old 0.5-0.8s launch rows were not
present. The remaining largest Code2Wav rows were completion observation
tails such as 176.01 ms pending with 12.98 ms GPU elapsed, not unbounded
Code2Wav computation. Artifact:
`/tmp/qwen3_prewarm_coverage_smoke_300s/result.json` in the AMD
container; server log:
`/tmp/qwen3_validate_code2wav-prewarm-coverage_server.log`.

Full-soak result: bs=30 / 1800s with the expanded prewarm still missed
strict realtime on 23/2463 turns and first-audio on 6 turns. The broad
missing-shape startup issue is fixed, but not every long-tail Code2Wav
launch is gone: one late row hit
`launch_async_code2wav bs=3 min_window_frames=24 wall_ms=805.51` even
though `(bs=3, frames=24)` is in the prewarm set. That points away from
simple shape coverage and toward a rarer runtime effect such as ROCm/MIOpen
runtime behavior, allocator/cache pressure, or shared-VF noise. Telemetry
around the spike showed high sampled GPU utilization and no obvious VRAM
limit. Do not treat more prewarm-shape enumeration as proven high EV until
this specific prewarmed-shape spike is root-caused.

The same full soak still shows most Code2Wav completion tails as scheduler
observation rather than GPU work: e.g. 200.46 ms pending with 14.39 ms GPU
elapsed. Artifacts in the AMD container:
`/tmp/qwen3_prewarm_coverage_soak_1800s/result.json`,
`/tmp/qwen3_prewarm_coverage_soak_1800s/telemetry.jsonl`, and
`/tmp/qwen3_prewarm_coverage_soak_1800s/telemetry.png`.

The prewarmed-shape launch spike was then root-caused with optional stage
timing. Repeated bs=30 / 300s diagnostic smokes reproduced the long row on
a prewarmed `(bs=5, frames=29)` Code2Wav input. It was not list
construction, padding, H2D materialization, decoder replay, or cat/clamp:
`pre_enqueue` stayed about 0.01 ms, H2D about 0.16-0.17 ms, and decoder
replay about 7 ms. The long row was Code2Wav encode, specifically the
pre-transformer: encode was 384-389 ms, embedding+mean was 0.10 ms, and
pre-transformer was 384-389 ms. This points the remaining rare Code2Wav
launch tail at the pre-transformer runtime path or contention, not the
mid-decoder convolution replay path. Diagnostic artifacts:
`/tmp/qwen3_code2wav_pre_enqueue_timing_smoke_300s`,
`/tmp/qwen3_code2wav_encode_timing_smoke_300s`,
`/tmp/qwen3_code2wav_pre_enqueue_timing_server.log`, and
`/tmp/qwen3_code2wav_encode_timing_server.log`.

The same instrumentation also exposed one host-side work-group-build
outlier in a normal 300s smoke: Code2Wav enqueue/GPU stayed normal, but
`work_group_build_ms` hit 380 ms. A small simplification removed the extra
copy from immutable ready-state tuples into mutable work-item lists. After
that change, a normal bs=30 / 300s smoke produced 430/430 ok turns,
427/430 strict realtime, 430/430 audio-stream realtime, and Code2Wav
launch p95/p99/max 13.25/19.69/22.20 ms with `work_group_build_ms`
p95/p99/max 0.01/0.02/0.04 ms. Treat this as a promising smoke result,
not a full long-soak proof. Artifact:
`/tmp/qwen3_code2wav_tuple_workitem_smoke_300s/result.json`; server log:
`/tmp/qwen3_validate_code2wav-workitem-tuple_server.log`.

The next 1800s confirmation run did not reproduce the host-build tail,
but it did uncover that the benchmark must distinguish cold startup from
steady-state capacity. Immediately after a server restart, the run
admitted 30 sessions and all 30 first turns missed strict realtime. The
first Talker prefill took 3696.57 ms and the first CodePredictor item
waited 6957.65 ms. Turn>=1 was much closer: 2447/2457 strict realtime
and 2452/2457 audio-stream realtime. A warmed no-restart bs=30 / 300s
smoke on the same server had 30/30 turn-0 strict realtime, 420/422 total
strict realtime, 421/422 audio-stream realtime, and max first audio
1.149s. This is not a Code2Wav replay issue; future milestone soaks need
explicit unmeasured audio warmup or a broader startup prewarm path before
the measurement window. Artifacts:
`/tmp/qwen3_code2wav_tuple_soak_1800s/result.json`,
`/tmp/qwen3_code2wav_tuple_soak_1800s/telemetry.jsonl`,
`/tmp/qwen3_code2wav_tuple_soak_1800s/telemetry.svg`, and
`/tmp/qwen3_code2wav_tuple_warm_smoke_300s/result.json`.

Fresh-server warmed confirmation: restart, run an unmeasured 60s bs=30
audio warmup, then measure bs=30 / 1800s. This produced 2511/2511 ok
turns, 2498/2511 strict realtime, 2510/2511 audio-stream realtime,
mean/max RTF 0.518/1.145, max first audio 0.819s, and one audio-stream
miss. Turn-0 was clean at 30/30 strict realtime with max first audio
0.386s. The remaining 13 strict misses are steady-state tail/fixed
latency, not cold startup and not the removed Code2Wav work-item copy.
Artifacts:
`/tmp/qwen3_code2wav_tuple_warmed_soak_1800s/result.json`,
`/tmp/qwen3_code2wav_tuple_warmed_soak_1800s/telemetry.jsonl`, and
`/tmp/qwen3_code2wav_tuple_warmed_soak_1800s/telemetry.svg`.

Mixed Talker-attention FP8 / bs=50 stage-timing diagnostic:

- artifact: `/tmp/qwen3_stage_timing_smoke_300s/stdout.json`
- telemetry: `/tmp/qwen3_stage_timing_smoke_300s/telemetry.jsonl`
- plot: `/tmp/qwen3_stage_timing_smoke_300s/telemetry.png`
- server log: `/tmp/qwen3_stage_timing_server.log`
- setup: warmed server, `SGLANG_QWEN3_OMNI_CODE2WAV_STAGE_TIMING=1`,
  mixed FP8 Thinker+Talker-attention checkpoint, 50 long-lived sessions,
  300s duration, 60s arrival ramp, normal 100 ms client audio chunks
- aggregate: 701/701 ok turns, 687/701 strict realtime, 699/701
  audio-stream realtime, mean/max RTF 0.637/1.193, max first audio
  0.914s, max completion tail 0.445s
- measured-window Code2Wav launch rows did not reproduce the previous
  0.8s `forward_enqueue_ms` spike: `launch_async_code2wav` p95/p99/max
  wall was 12.83/16.96/22.18 ms, queue p99/max was 1.18/4.19 ms,
  pre-transformer p99/max was 8.38/11.86 ms, and decoder p99/max was
  11.67/14.74 ms
- measured-window CodePredictor queue waits were larger than Code2Wav
  queue waits: p99/max was 19.78/27.63 ms, with wall p99/max
  7.04/7.59 ms
- strict misses in this 300s diagnostic were mild and mostly bounded by
  sustained audio-stream RTF or first-audio fixed latency, not a large
  Code2Wav completion tail

This diagnostic is not a 30-minute launch proof, but it weakens the
current "Code2Wav forward is the remaining huge tail" hypothesis. If a
future full bs=50 soak reproduces the 0.8s Code2Wav launch row with
stage timing enabled, use the `max_code2wav_*_ms` fields to classify it.
If it does not reproduce, move the next optimization target to
CodePredictor queue/cadence, Talker prefill fixed latency, or broader
scheduler bubbles based on miss attribution.

Mixed Talker-attention FP8 / bs=50 30-minute stage-timing confirmation:

- artifact: `/tmp/qwen3_stage_timing_bs50_30min/stdout.json`
- telemetry: `/tmp/qwen3_stage_timing_bs50_30min/telemetry.jsonl`
- plot: `/tmp/qwen3_stage_timing_bs50_30min/telemetry.png`
- server log: `/tmp/qwen3_stage_timing_server.log`
- aggregate: 3990/3990 ok turns, 0 close failures, 3741/3990 strict
  realtime, 3882/3990 audio-stream realtime, mean/max RTF 0.712/1.712,
  max first audio 1.517s, max completion tail 0.479s
- measured-window phase totals:
  - Code2Wav: 60444 rows, 433.6s total, p95/p99/max wall
    13.02/17.05/918.03 ms
  - CodePredictor: 41928 rows, 244.5s total, p95/p99/max wall
    6.65/7.09/8.06 ms, queue p99/max 20.02/29.79 ms
  - Talker decode: 38196 rows, 192.5s total, p95/p99/max wall
    6.56/7.47/251.89 ms
  - Talker prefill: 3775 rows, 156.0s total, p95/p99/max wall
    47.75/53.28/67.24 ms
- the 30-minute run reproduced two true Code2Wav forward/enqueue spikes
  with near-zero queue wait and immediate completion:
  - line 184608: bs=2, frames=8, window 20-29,
    `wall_ms=794.39`, `max_forward_enqueue_ms=794.13`,
    `max_code2wav_pre_transformer_ms=786.43`, decoder 7.53 ms
  - line 338851: bs=3, frames=12, window 29,
    `wall_ms=918.03`, `max_forward_enqueue_ms=917.76`,
    `max_code2wav_pre_transformer_ms=910.15`, decoder 7.45 ms
- this localizes the rare huge Code2Wav tail to the pre-transformer
  enqueue/runtime path, not H2D, work-group build, decoder replay,
  cat/clamp, event record, or completion observation
- because the stage timing is CPU wall/enqueue timing, the next pass
  still needs to distinguish a genuinely slow pre-transformer kernel from
  HIP/PyTorch runtime blocking, stream backpressure, or external GPU
  contention at the first blocking operation inside the pre-transformer.
  The next diagnostic should use the detailed pre-transformer timing
  fields emitted under `SGLANG_QWEN3_OMNI_CODE2WAV_STAGE_TIMING=1`,
  including max per-layer qkv, rotary, SDPA, output projection, MLP,
  RMSNorm, and residual timings.

Code2Wav contiguous-layout and full outer-batch prewarm follow-up:

- commit `83f67ecf` changed `hidden.permute(0, 2, 1)` to
  `.contiguous()` before decoder replay and expanded Code2Wav prewarm to
  every outer batch size from 1 through the configured max concurrent
  audio turns
- copied-back Parakeet gate passed:
  `/tmp/qwen3_voice_validate/code2wav-contiguous-prewarm/report.json`
- startup prewarm at bs=50 covered 421 `(bs, dframes)` combos in 38.67s
- cold 300s traffic was still polluted by lazy Talker decode graph
  captures and first-use Talker prefill/CodePredictor work, so it is not
  steady-state evidence
- warmed 300s no-restart traffic improved to 685/693 strict realtime and
  692/693 audio-stream realtime; Code2Wav launch p95/p99/max was
  13.34/19.69/22.53ms with no Talker graph captures after start
- warmed 30-minute no-restart traffic improved versus the prior
  stage-timed full soak but did not pass: 3984/3984 ok,
  3845/3984 strict realtime, 3950/3984 audio-stream realtime, mean/max
  RTF 0.711/1.367, first-audio p99/max 0.680/1.484s
- the full warmed run still had two rare p100 stalls:
  - Talker prefill `model_ms=808.89ms` at batch 2, total tokens 101
  - Code2Wav pre-transformer layer 0 MLP
    `max_code2wav_pre_transformer_max_layer_mlp_ms=863.52ms` at
    batch 2, window 29
- interpretation: contiguous layout and fuller prewarm remove the obvious
  missing-shape/stride issue from the steady path, but the remaining
  launch blocker is a rare warmed GPU/runtime stall that can hit multiple
  small model components. Do not narrow future work to Code2Wav decoder
  replay unless a new profile proves decoder replay is the culprit again.

## Semantic Blocker

Current/upstream Code2Wav `chunked_decode` replays the 25-frame context
window and resets transformer positions to `0..seq_len-1` on every chunk.
Once the window slides, the same codec frame appears at different
relative rotary positions in later chunks. Persistent transformer KV or
decoder activations would reuse activations with the wrong positions
unless the shifted context is replayed.

Therefore:

- exact incremental transformer KV is blocked
- exact incremental decoder activation state is blocked
- a true stateful Code2Wav decoder is a model-semantics change, not a
  drop-in optimization

## Current Hot Areas

Long 29-frame replay:

- dense kernel-7 causal Conv1d groups dominate much of the replay path
- important shapes include C=96/L=55125 and C=192/L=18376 in late decoder
  blocks
- mid-decoder C=384/L=4595 and C=768/L=920 were not improved by the
  existing exact-shape Triton family

First 4-frame window:

- first-window latency is not dominated by late dense Conv1d
- prior profile showed bs1 encode around 4.9 ms and pre-transformer
  attention rows around 0.46-0.64 ms each
- first-audio work should target encode/pre-transformer attention or a
  measured split-decoder design, not only long-window Conv1d

Async completion tail:

- latest soak `enqueue_wall_ms` p50/p95/p99 was 12.38/13.26/18.86 ms
- latest soak `gpu_elapsed_ms` p50/p95/p99 was 12.38/13.92/19.01 ms
- latest soak `pending_elapsed_ms` p50/p95/p99/max was
  0.53/37.69/53.70/380.99 ms
- after the completion-priority fix, latest soak `pending_elapsed_ms`
  p50/p95/p99/max was 0.51/18.92/29.42/65.24 ms
- before optimizing Code2Wav kernels for a tail row, verify whether that
  row is a true enqueue/GPU spike or only delayed completion observation

## Accepted Baseline

- shared embedding + `pre_transformer`
- split convolutional decoder at max batch 8
- `CODE2WAV_DECODER_MAX_BATCH=8`
- steady 29-frame split-decoder graph replay is accepted as tail
  hardening. Copied-back voice validation passed all five fixtures at
  `/tmp/qwen3_voice_validate/code2wav-capture-all5/report.json`. The
  repeat bs=50 / 300s smoke passed 722/722 strict realtime and 722/722
  audio-stream realtime with mean/max RTF 0.425/0.813, max first audio
  0.663s, max completion tail 0.104s, and smoke-window Code2Wav forward
  enqueue p50/p95/p99/max 0.23/12.66/13.28/21.65 ms. Artifact:
  `/tmp/qwen3_code2wav_capture_smoke2_300s/stdout.json`.
- full bs=50 / 1800s with graph replay still missed the expansion gate:
  4174/4188 strict realtime, 4182/4188 audio-stream realtime, mean/max
  RTF 0.486/1.443, max first audio 2.165s, max completion tail 0.269s.
  The old Code2Wav 0.8-0.9s forward-enqueue spike did not recur:
  soak-window Code2Wav forward-enqueue p50/p95/p99/max was
  0.24/12.72/13.60/55.77 ms and pending elapsed was
  3.61/8.48/13.84/60.09 ms. The miss cluster coincided with live AITER
  lazy-load messages for `M=50` and `M=71` GEMM shapes plus Talker-prefill
  p100 rows. Artifact:
  `/tmp/qwen3_code2wav_capture_soak_30min/stdout.json`; miss report:
  `/tmp/qwen3_code2wav_capture_soak_30min/miss_report.md`.
- latest long-tail recheck: 150/150 ok, zero tails, max output 9.577s,
  Parakeet 150/150

## Rejected Or Insufficient Paths

- full-prefix/stateful Code2Wav streaming prototype: local prefix tests
  passed but server-level bs=30 regressed
- `torch.compile` Code2Wav: killed after minutes of Inductor autotune
- broad Code2Wav whole-decoder HIP graph as a primary lever: worked in
  isolation but only saved a few ms and previous server graph attempts
  regressed. This rejection does not apply to the accepted steady
  29-frame split-decoder graph replay above.
- max batch 30, four streams, same-work graph, per-request GPU frame
  buffer: regressed warm bs=30
- MIOpen immediate / cudnn benchmark knobs: no useful win
- Conv1d-as-height-1-Conv2d rewrite: no material MI300X win
- built-in Conv1d padding: isolated replay improved but realistic bs=30
  tail regressed
- dense Conv1d shift-GEMM: slower than MIOpen on representative shapes
- fused SnakeBeta: isolated timing improved but server smoke regressed
- ConvTranspose phase-split GEMM and direct Triton transconv: isolated
  or replay wins did not transfer to server
- short-window no-split decoder: replay looked attractive but server
  tail regressed
- frame-lag budget 16->8 or 16->24: both regressed
- split-decoder bs=16: rejected twice; keep max batch 8
- first chunk 4 plus steady chunk 8; max group 6/4; left context 25->16:
  all regressed warm bs=30 or first-audio
- background-thread Code2Wav launch/enqueue: copied-back voice validation
  passed, but bs=50 300s smoke regressed to 682/702 strict realtime and
  690/702 audio-stream realtime. Two root-cause passes showed why:
  moving launch off the scheduler thread hid the 12-800 ms enqueue wall,
  but the single-pending-Code2Wav invariant then let later Code2Wav-ready
  requests queue behind the in-flight launch, with submit-row queue waits
  up to 750 ms and `submit_elapsed_ms` max 846 ms. Do not retry this
  single-worker design; any future off-thread Code2Wav work needs a new
  scheduler design that preserves bounded Code2Wav-ready queue wait.

## Kernel Work Requirements

Every Code2Wav kernel effort needs:

1. isolated benchmark on exact hot shape
2. numerical equivalence against eager
3. copied-back Parakeet audio validation
4. bs=30 300s smoke
5. full 30-minute soak before accepting a launch claim

Do not accept isolated kernel wins without server transfer.

## Artifacts

- `/tmp/qwen3_code2wav_shapeexact_long_shape_profile.json`
- `/tmp/qwen3_code2wav_firstchunk_shapeexact_profile.json`
- `/tmp/qwen3_code2wav_pad_profile_summary.json`
- `/tmp/qwen3_code2wav_builtinpad_r30.json`
- `/tmp/qwen3_code2wav_transconv_direct_sweep.json`
- `/tmp/qwen3_code2wav_miopen_baseline_r30.json`
- `/tmp/qwen3_code2wav_miopen_find_{DEFAULT,FAST,HYBRID,NORMAL}.json`
- `/tmp/qwen3_generic_kv_soak/qwen3_generic_kv_soak_1800s.json`
- `/tmp/qwen3_generic_kv_soak/qwen3_generic_kv_telemetry_server.log`
- `/tmp/qwen3_code2wav_priority_soak/qwen3_code2wav_priority_soak_1800s.json`
- `/tmp/qwen3_code2wav_priority_soak/qwen3_code2wav_priority_perf_server.log`
