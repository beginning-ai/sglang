# Streaming Cadence And First-Audio Latency

## Summary

The product path is realtime input with paced client chunks, internal
streaming prefill, and EOI-triggered finalization. Full-batch audio input
is not representative for production performance decisions.

## Decision

Use 1s client chunks for launch and capacity claims. The previous 100ms
client-chunk default is now a stress/profiling variant, not the production
gate. Smaller 50-frame internal chunks and naive bs=1 audio boot warmup
remain rejected.

## Accepted Work

- `StreamingAudioBuffer.add_samples()` emits every complete 100-frame
  internal chunk from an incoming client message.
- The harness supports true paced larger chunks: e.g. send a 5s chunk at
  t=5s, not at t=0.
- Async feature extraction offload moved audio feature extraction to
  `asyncio.to_thread()`, avoiding event-loop serialization at EOI.

## Important Measurements

Historical 100 ms client chunk cadence after buffer fix:

| batch | ok | realtime | mean/max RTF | first-audio p95/max |
|---:|---:|---:|---:|---:|
| 8 | 40/40 | 40/40 | 0.324 / 0.378 | 0.459s / 0.552s |
| 16 | 80/80 | 80/80 | 0.452 / 0.581 | 0.777s / 0.915s |
| 30 | 150/150 | 148/150 | 0.727 / 1.072 | 1.551s / 1.849s |

Larger paced chunks were worse at bs=30:

| client chunk | realtime | mean/max RTF | first-audio p95/max |
|---:|---:|---:|---:|
| 1s | 108/150 | 0.920 / 1.430 | 1.856s / 2.325s |
| 2s | 94/150 | 0.935 / 1.444 | 2.415s / 2.913s |
| 5s | 73/150 | 1.100 / 1.645 | 4.253s / 4.675s |

Async feature extraction offload accepted result:

- warm bs=30 five-fixture aggregate: 150/150 ok, all realtime,
  mean/max RTF 0.596/0.865
- first-audio p50/p95/max 0.497/0.545/0.545s
- scheduler first PCM p50/p95/max 420/491/491 ms
- copied-back Parakeet 150/150 pass

Current launch-stack 1s client chunk evidence:

- artifact: `/tmp/qwen3_chunk1000_smoke_300s/stdout.json`
- full-soak artifact: `/tmp/qwen3_chunk1000_full_soak_30min/stdout.json`
- setup: current FP8 Thinker + FP8 Talker-attention launch checkpoint,
  streaming sessions, overlap scheduling, Code2Wav/Talker/Thinker capture,
  startup prewarm, bs=50, 300s, rollback rate 0.02
- 300s smoke result: 724/724 strict realtime, 724/724 audio-stream realtime,
  14/14 rollback success, mean/max RTF 0.414/0.875, mean/max
  audio-stream RTF 0.379/0.806, max first audio 0.605s, max completion
  tail 0.321s, max input chunks 4
- 30-minute soak result: 4193/4198 strict realtime, 4198/4198
  audio-stream realtime, 73/73 rollback success, mean/max RTF
  0.468/1.142, mean/max audio-stream RTF 0.425/0.949, max first audio
  0.994s, max completion tail 0.311s, max input chunks 4
- decision: 1s chunks are the production/capacity cadence. The full bs=50
  run is not locked yet, but the remaining misses are fixed-overhead
  strict misses with audio-stream fully green. The latest 100ms-chunk full
  soak is still useful stress evidence because it showed long-prefix tiny
  streaming-prefill rows, but it is not the production capacity gate.

Latest bs=50 continuation-prewarmed 1s client chunk evidence:

- smoke artifact:
  `/tmp/qwen3_remove_broad_small_m_smoke_300s/stdout.json`
- full-soak artifact:
  `/tmp/qwen3_code2wav_oldest_ready_bs50prewarm_soak_30min/stdout.json`
- setup: current FP8 Thinker + FP8 Talker-attention launch checkpoint,
  streaming sessions, overlap scheduling, Code2Wav/Talker/Thinker capture,
  startup prewarm with 50 continuation sessions, bs=50, 1s chunks,
  rollback rate 0.02
- 300s smoke result after removing the rejected broad small-M output
  prewarm: 730/730 strict realtime, 730/730 audio-stream realtime,
  mean/max RTF 0.417/0.817, mean/max audio-stream RTF 0.380/0.700, max
  first audio 0.621s
- 30-minute soak result: 4227/4227 strict realtime, 4227/4227
  audio-stream realtime, mean/max RTF 0.461/0.969, mean/max audio-stream
  RTF 0.419/0.845, max first audio 1.385s, max completion tail 0.196s
- decision: throughput is now green for bs=50 production cadence, but the
  first-audio p100 gate is still open. The top first-audio rows are late
  long-prefix turns, not Code2Wav completion tails.

## Rejected Paths

- Naive single-turn audio boot warmup: warmed the wrong bs=1/silence/tiny
  shapes and regressed the next bs=30 100 ms cadence run to 119/150
  realtime.
- 50-frame internal chunks padded to 100-frame feature width: avoided
  mixed-shape crashes but doubled prefill cadence without reducing model
  feature width; warm bs=30 question regressed to mean/max RTF
  1.305/1.514 and max first-audio 4.256s.
- 100ms client chunks as the canonical capacity gate: superseded by the
  current plan's 1s production cadence after the 1s bs=50 smoke passed
  with substantially lower RTF and first-audio latency on the current
  launch stack.

## Artifacts

- `/tmp/qwen3_streaming_cadence_current_after_buffer`
- `/tmp/qwen3_streaming_cadence_multichunk`
- `/tmp/qwen3_async_feature_extract_*`
- `/tmp/qwen3_voice_validate/async-feature-extract/report.json`
