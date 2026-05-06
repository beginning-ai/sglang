# Profiling And Bottlenecks

This sidecar keeps only the investigation results that matter for operating
or extending the launch stack. The active launch state and commands live in
`docs/qwen3_omni_amd_perf_path_forward.md`.

## Current Bottleneck Picture

The async architecture is working: phase work overlaps enough that summed
component time fits inside wall time. The remaining launch risk is p100
latency and long-history/KV pressure, not a serialized audio lane.

The most useful recent warm profile is:
`/tmp/qwen3_bs50_warm_only_profile_20260521T115301Z/stdout.log`.

Warm-only phase summary:

- Talker decode: 64.6s total, 5.45ms mean, `apply_ms` 0.12ms mean,
  `codec_id_materialize_ms` 2.35ms mean
- Code2Wav: 37.2s total, 2.72ms mean, `event_wait_ms` 4.33ms mean
- Talker prefill: 10.2s total, 15.5ms mean
- GPU busy mean/p95/max: 83.0%/98.0%/99.0%
- HBM activity mean/p95/max: 8.4%/13.0%/14.0%
- max KV: Thinker 106,489/399,999, Talker 26,634/213,333,
  CodePredictor 800/800

The latest accepted async codec-id staging run is:
`/tmp/qwen3_bs50_codec_id_event_query_repeat_20260521T125630Z/stdout.log`.

It passed 675/675 strict realtime and audio-stream realtime with 14/14
rollback success, mean/max RTF 0.352/0.564, max first audio 0.507s, and
max completion tail 0.059s. Talker decode improved from 64.6s/5.45ms mean
to 52.8s/4.39ms mean. The old `codec_id_materialize_ms` almost disappeared
(2.35ms mean to 0.05ms), with a smaller readiness wait showing in
`apply_ms` (0.12ms to 1.44ms mean).

The final launch-lock soak is:
`/tmp/qwen3_bs50_launch_lock_20260521T135836Z/stdout.log`.

It passed 4,201/4,201 strict realtime and audio-stream realtime with 88/88
rollback success. Mean/max RTF was 0.390/0.937, max first audio after EOI was
1.420s, and max completion tail was 0.244s. The longest measured phase rows
were isolated outliers; p99 Talker decode was 6.60ms, p99 Code2Wav was
13.59ms, and p99 Talker prefill was 67.1ms.

## Accepted Findings

- Overlap scheduling is accepted for the FP8 Thinker launch stack.
- Streaming-session migration is accepted; it reduced fork-specific cache
  management and supports rollback rebuilds through full-prefix prefill.
- Code2Wav steady 29-frame replay and short first/flush windows are
  graph-captured. See `code2wav_replay_and_kernels.md`.
- CodePredictor now runs inline with Talker phases; future profiles should
  treat Talker decode as including CodePredictor work.
- Pinned CPU staging for codec frames is accepted. Raw Python lists on the
  Talker side were a real launch-shape cost.
- Current-stream async staging for Talker codec ids is accepted. It avoids
  blocking before inline CodePredictor while preserving captured-graph output
  ordering.
- Non-final realtime audio prefill uses a 200ms bucket. EOI/final chunks
  bypass this bucket to preserve speech-to-speech latency.
- `apply_ms` is included in phase logs; old phase tables that exclude apply
  time can be misleading.

## Rejected Findings

Avoid repeating these without a new hypothesis:

- Separate copy-stream codec-id staging failed copied-back voice validation on
  every fixture. It likely raced captured Talker graph output readiness.
- Batch-level codec-id materialization passed validation but did not improve
  Talker `apply_ms`; per-frame event query/scalar conversion is not the main
  remaining apply cost.
- Reusable pinned codec-id buffers hid the wait after phase logging and
  regressed end-to-end RTF.
- Raising Talker decode cap/capture from bs=32 to bs=50 created larger rows
  but regressed RTF/first-audio with no phase-wall win.
- Removing the pending-Code2Wav block before Thinker prefill regressed
  first-audio/RTF while Code2Wav event wait stayed flat.
- 300ms/400ms stream-prefill buckets widened prefill batches but worsened
  first-audio and RTF tails.
- Raw GPU codec-frame handoff moved cost from Talker into Code2Wav and did
  not improve end-to-end metrics.
- NumPy Code2Wav window packing, Code2Wav async cap=4, side-stream
  CodePredictor overlap, 30ms audio batching window, first-PCM scheduler
  relaxation, repetition-mask history trim, fresh Talker requeue, and simple
  two-frame Talker burst did not transfer to better bs=50 metrics.

## Profiling Rules

Use clean warm windows only:

1. Start/restart the launch stack.
2. Wait for startup prewarm/readiness.
3. Truncate the server log.
4. Run the smoke/soak.
5. Parse phase rows from the measured window only.

Primary signals:

- strict/audio-stream realtime pass counts
- mean/max RTF and first-audio latency
- completion tail and rollback rebuild time
- phase rows by `(phase, batch_size)`
- `queue_wait_ms`, `deadline_miss_ms`, `wall_ms`, `apply_ms`
- GPU busy, HBM activity, VRAM
- Thinker/Talker/CodePredictor KV usage and capacity
- streaming-session held tokens

Do not rely on full-process `rocprof` alone for ranking launch work; startup
and H2D pollution can dominate. Use it only to support a phase-timer
hypothesis.
