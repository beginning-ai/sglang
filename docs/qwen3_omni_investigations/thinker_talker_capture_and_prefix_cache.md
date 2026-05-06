# Thinker/Talker Capture And Prefix Cache

## Summary

Capture and prefix-cache work delivered real gains, but not every model
component is safe to graph as a small wrapper. CodePredictor capture,
Thinker tiny-extend model-forward capture, Talker decode capture, and
cached-prefix Talker prefill capture helped. CodePredictor argmax is
enabled in the current launch/perf path, and is now a safe accepted
audio-quality optimization: the fair
constrained-text A/B package at
`/tmp/qwen3_code_predictor_argmax_ab_constrained/review` passed Parakeet
in both lanes, then human review preferred argmax 6-0 with 3 ties and 1
both-bad pair. Early Talker-prefill capture probes corrupted audio or
failed invalid hidden checks, but the accepted cached-prefix replay path
passed copied-back Parakeet validation and improved the bs=50 30-minute
soak.

## Decision

Keep accepted capture paths. Do not retry Talker prefill capture as a
CodePredictor-metadata shim or same-KV eager-vs-captured comparison.
Future hidden-equivalence checks for cached-prefix Talker prefill must use
independent component KV slots before perf soaks.

## Accepted Work

CodePredictor:

- capture passed AMD correctness at bs=2/4 with copied-back Parakeet
- argmax is accepted for launch after constrained-text human A/B review
  in Beginnings issue #124: argmax 6, stochastic 0, tie 3, both bad 1
- wall dropped to roughly 5.4-5.6 ms

Thinker tiny-extend:

- Qwen3-local replay captures only `thinker.model(...)` for tiny
  cached-prefix extend rows
- text-token cached-prefix extends now pre-embed the tiny suffix and
  enter that same replay path; this is semantically equivalent to the
  model embedding tokens internally, but avoids leaving text-only
  late-session extends on the eager path
- logits/sampling, prompt accept-hidden assembly, and Talker-prefill
  payload construction stay eager
- local Qwen3 control-token suppression prevents `<think>` / `</think>`
  or other control tokens from poisoning parent-rid history
- warm 300s bs=30 long-lived smoke passed on clean-server repeat:
  424/424 ok, all completion/audio-stream realtime, mean/max completion
  RTF 0.535/0.938, max first-audio 0.980s
- bs=50 validation after the text-token capture extension was mixed but
  positive overall: copied-back voice validation passed all 5 fixtures;
  one bs=50 / 300s smoke missed 2/705 strict turns, the repeat passed
  704/704 strict realtime, and the 30-minute bs=50 soak improved from
  4031/4060 to 4072/4087 strict realtime while still missing the gate
  because of rare p100 Code2Wav/Talker-prefill spikes

Talker decode:

- mixed-sequence capture buckets by max Talker sequence length while
  preserving large scheduler batches
- long-bucket fix filled padded `kv_indices` tails with a valid KV slot,
  matching the Thinker prefill graph pattern
- validation passed 10/10 after fixes
- Talker decode is no longer the dominant long-history bottleneck

Cached-prefix Talker prefill:

- accepted path captures/replays the production cached-prefix AITER
  prefill metadata shape with static dummy input ids
- copied-back voice validation passed on the soak server:
  `/tmp/qwen3_voice_validate/talker-prefill-capture-soak-server/report.json`
- bs=50 / 1800s improved from 4054/4075 strict, mean/max RTF
  0.608/1.273 to 4117/4122 strict, mean/max RTF 0.548/1.142
- steady cached-prefix `sample_codec=True` rows dropped from about 37 ms
  p50 to roughly 7-9 ms p50
- remaining work is p99/new-shape/capture tail, not the steady path

Talker user-prefix cache:

- accepted correctness path retains only semantic user-prefix Talker KV
  across `parent_rid` turns
- assistant bootstrap and generated codec KV remain response-local
- multi-turn voice validation passed 10/10
- prefix reuse engaged in long-lived runs, but it did not finish the
  bs=30 target

## Rejected Work

Rejected early Talker prefill capture probes:

- first version crashed because Qwen/MoE layers read
  `forward_batch.input_ids.shape` even when `input_embeds` were supplied
- static dummy input ids avoided the crash but corrupted audio
- multi-turn validation failed 7/10 with blank or badly mismatched
  Parakeet transcripts despite non-silent WAVs
- restored baseline passed 10/10
- later same-KV eager-vs-captured hidden checker was invalid for
  cached-prefix replay because the eager pass mutated the same component KV
  slots before graph replay

Uniform Talker-decode sequence-length batching:

- reduced Talker decode phase time
- fragmented downstream Code2Wav into mostly batch-1 work
- regressed later-turn RTF badly

Upstream piecewise CUDA/HIP graph for Qwen3 streaming prefill:

- captured tiny cached streaming-prefill rows
- crashed under real voice validation from runtime recompilation over
  dynamic cached-prefix audio shapes
- should not be reapplied wholesale

Talker multi-shared-expert fusion:

- the Talker shared expert can be represented as two fused shared experts
  at the weight/semantics level
- copied-back voice validation passed, but bs=50 smoke/phase timing did
  not move Talker-prefill p50 off ~36 ms
- rejected for now because it adds Qwen2 MoE loader/model complexity
  without a measured launch-shape win

## Artifacts

- `/tmp/qwen3_voice_validate/thinker-prefill-capture-local-mask-v2/report.json`
- `/tmp/qwen3_voice_validate/run/report.json` after text-token capture
  extension
- `/tmp/qwen3_bs50_text_prefill_capture_smoke_20260518T170506Z/stdout.json`
- `/tmp/qwen3_bs50_text_prefill_capture_smoke2_20260518T171251Z/stdout.json`
- `/tmp/qwen3_bs50_text_prefill_capture_30min_20260518T172012Z/stdout.json`
- `/tmp/qwen3_soak_stream_metric_5min_warm2.json`
- `/tmp/qwen3_voice_validate/talker-decode-long-buckets-kvfill/report.json`
- `/tmp/qwen3_soak_long_buckets_kvfill_20min.json`
- `/tmp/qwen3_voice_validate/talker-user-prefix-cache-postcommit/report.json`
- `/tmp/qwen3_voice_validate/talker-prefill-capture-v2/report.json`
- `/tmp/qwen3_voice_validate/post-talker-prefill-capture-revert/report.json`
- `/tmp/qwen3_voice_validate/talker-prefill-capture-soak-server/report.json`
- `/tmp/qwen3_talker_prefill_capture_soak_30min/stdout.json`
- `/tmp/qwen3_bs50_shared_fusion_300s_r2_stdout.json`
