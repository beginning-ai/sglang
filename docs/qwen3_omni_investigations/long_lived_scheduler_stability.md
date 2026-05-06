# Long-Lived Scheduler Stability

## Summary

Several long-lived-session failures were scheduler/lifecycle issues, not
model-quality or kernel issues. The accepted fixes preserve the semantic
ordering of Qwen3 Omni audio phases while preventing frame-lag stalls and
head-of-line blocking.

## Decision

Keep the accepted scheduler invariants. Do not return to generic
scheduler delay/window tuning unless a fresh profile proves the batching
pattern changed.

## Accepted Fixes

- Cap same-request frame-producing bursts to one Code2Wav chunk.
- Launch ready Code2Wav before exceeding frame-lag budget.
- Block on already-pending Code2Wav when necessary.
- Promote urgent independent audio work ahead of non-urgent
  head-of-line work.
- Materialize missing `Qwen3OmniCode2WavReady` state during ready-req
  scanning when `should_run_code2wav_phase(req)` is true.
- Drain all currently ready Code2Wav requests after applying
  Talker/CodePredictor outputs, instead of only requests touched by the
  just-drained audio item.
- Cap committed Talker-prefill batching by the same per-request frame
  budget used by Talker decode and CodePredictor. A committed Talker
  prefill can create a dependent CodePredictor row, so batching multiple
  same-request prefills past the remaining frame budget can create an
  oversized CodePredictor item before Code2Wav is ready to run.

## Accepted Validation Anchors

2-hour rolling bs=50 launch soak with rollback traffic:

- artifact:
  `/tmp/qwen3_2h_soak_20260520T133125Z/stdout.json`
- telemetry:
  `/tmp/qwen3_2h_soak_20260520T133125Z/gpu_telemetry.jsonl`,
  `/tmp/qwen3_2h_soak_20260520T133125Z/kv_telemetry.jsonl`,
  `/tmp/qwen3_2h_soak_20260520T133125Z/telemetry.svg`
- 16,658/16,658 turns ok, 0 failures, 0 close failures
- 317/317 rollback rebuilds successful
- strict realtime missed 10/16,658 turns; audio-stream realtime missed
  3/16,658 turns
- mean/max RTF after EOI 0.463/1.203
- mean/max audio-stream RTF 0.423/1.125
- max first audio 1.606s; max completion tail 1.429s
- max observed KV use: Thinker 399,996/399,999 tokens, Talker
  166,529/213,333 tokens, CodePredictor 800/800 slots
- accepted for launch at bs=50 because the rare tail misses are tolerable
  for first launch; keep p100 tail and KV-headroom work as post-launch
  hardening

Urgent audio promotion:

- multi-turn voice validation passed 10/10
- 2-session / 45s parent-rid smoke passed 12/12 turns ok and realtime
- 0 close failures
- mean/max RTF 0.353/0.482
- max first-audio 0.552s

Code2Wav ready-state materialization:

- multi-turn Parakeet passed 10/10
- default-off bs=30 / 300s completed 418/418 ok
- 0 close failures
- no prior frame-lag crash

Global ready Code2Wav drain:

- multi-turn voice validation passed 10/10
- warm bs=30 / 300s improved but did not pass strict realtime:
  408/408 ok, 360/408 realtime, mean/max RTF 0.732/2.081

Committed Talker-prefill frame-budget cap:

- root cause: long-lived sessions could queue several committed
  Talker-prefill items for one request while Thinker advanced; batching
  them together produced a dependent CodePredictor item with more rows
  than the remaining per-request Code2Wav chunk budget
- failure mode: full 1800s soak crashed with
  `Qwen3-Omni frame-producing audio phase reached frame lag budget without
  pending Code2Wav work`
- minimal-diff validation passed `$qwen3-omni-voice-validate --multi-turn
  --label prefill-frame-budget-minimal`
- warm bs=30 / 300s passed 431/431 strict realtime, 0 failures, 0 close
  failures, max RTF 0.934, max first-audio 0.708s
- bs=30 / 1800s no longer crashed and had 2467/2467 ok, 0 close
  failures, but remained short of launch at 2456/2467 strict realtime

## Rejected Scheduler Paths

- Per-request blocking of queued audio phases: fragmented audio work into
  small decode/CodePredictor batches and failed to complete the bs=30
  question probe before kill.
- Non-first-PCM Talker-prefill batching with same-request Thinker work:
  voice validation passed, but bs=50 / 300s smoke still missed 3/707
  strict turns and did not materially batch the first-PCM Talker prefills
  that matter for first-audio pressure.
- 5 ms first-PCM Talker-prefill grace: created larger Talker-prefill
  batches but regressed bs=50 / 300s to 685/694 strict realtime and
  increased queue wait sharply, so the delay cost outweighed the batch
  size win.
- 20 ms streaming Talker-prefill batch delay: one run had ~10s first
  audio, warm run still missed target.
- Aligned streaming prefill tick: correctness passed but warm bs=30
  regressed.
- Output-aware deferral of non-final streaming prefills: correctness
  passed but starved incremental prefill during output and regressed
  warm bs=30 / 300s.
- First-PCM audio-lane priority: violated streaming-prefill /
  final-prefill ordering and crashed with unknown ChatML role token.
- 10 ms Code2Wav-only scheduler delay: did not create useful batching and
  regressed first-audio/RTF.
- Letting non-urgent ready-to-launch Code2Wav wait behind Thinker prefill:
  copied-back voice validation passed, but bs=50 / 300s regressed to
  682/698 strict realtime with 16 strict misses. A second pass showed no
  material Code2Wav batching win and more small CodePredictor/TalkerDecode
  shapes; only the narrower non-urgent pending-Code2Wav overlap is kept.
- Nonconsecutive independent Talker-prefill batching: local focused tests
  and copied-back voice validation passed, but bs=50 / 300s regressed
  from the accepted 701/701 strict realtime smoke to 695/696 strict
  realtime. First-pass miss analysis showed one fixed-overhead miss, not
  an audio-stream or completion-tail issue. Second-pass phase comparison
  showed Talker-prefill batch histograms and p95/p99 were effectively
  unchanged, so the real queue rarely had later compatible Talker-prefill
  work to pull forward.
- Upstream `PrefillDelayer` as a pure flag change: same-seed 300s smoke
  improved modestly, but the full bs=50 30-minute run still missed the
  expansion gate and doubled audio-stream misses versus the accepted
  baseline. Root cause was architectural: the delayer acts on normal
  SGLang prefill admission, while Qwen3 Omni Talker-prefill rows are
  drained from the audio-phase queue and stayed gated by
  `phase_depends_on_thinker_lane`.
- Lowering `AUDIO_BATCH_DELAY_MS` from 20ms to 10ms: copied-back voice
  validation passed, and the bs=50 / 300s production-cadence smoke passed
  723/723 strict realtime, but the fresh bs=50 / 30-minute soak regressed
  to 4190/4194 strict realtime and 4192/4194 audio-stream realtime.
  First-pass evidence: the miss shape moved from the accepted baseline's
  late strict-only fixed overhead to early first-audio/audio-stream
  misses, with max first-audio 1.363s and max audio-stream RTF 1.013.
  Second-pass evidence: miss windows showed Talker-prefill p95/max around
  109.84/120.48ms, so the smaller batching window exposed or amplified
  early Talker-prefill p100 instead of safely removing fixed overhead.
  Keep the accepted 20ms window unless a new miss report changes this
  causal picture.

## Practical Rule

If a future scheduler change improves one phase, verify it does not
fragment downstream Code2Wav grouping or violate the Talker/CodePredictor
serial dependency. Scheduler wins must transfer to bs=30 smoke, not just
phase micro-timers.
