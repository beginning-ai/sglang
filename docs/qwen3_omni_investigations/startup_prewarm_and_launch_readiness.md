# Startup Prewarm And Launch Readiness

## Summary

Qwen3 Omni gRPC startup must pay launch-shape warmup before health reports
`SERVING`. A manual warm run after startup is not part of the launch
contract; if a manual warm run changes capacity, startup prewarm coverage
is incomplete.

## Decision

Run the production audio prewarm automatically for Qwen3 Omni gRPC
servers. The prewarm is required for readiness and must fail startup if it
fails. It hydrates fixtures from Hugging Face, uses the baked Beginnings
system prompt, derives its batch ladder from launch capacity, uses
production-shaped 1s chunks, and exercises rollback rebuild.

The removed Qwen3 Omni launch-prewarm compatibility flags are not part of
the launch surface. Qwen3 Omni gRPC startup always runs the prewarm and
derives the batch ladder from `--qwen3-omni-max-concurrent-audio-turns`.

## Launch Config Anchors

- checkpoint:
  `/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1`
- HF repo: `beginning-ai/lucy-1-FP8`, revision
  `7aaecb8d422964d4ace0f98ff73d980945a5796a`
- `--grpc-mode`
- `--enable-streaming-session`
- `--attention-backend aiter`
- `--mem-fraction-static 0.76`
- `--max-total-tokens 400000`
- `--qwen3-omni-max-concurrent-audio-turns 50`
- automatic prewarm batch sizes: `1,2,4,8,16,32,50`

## Historical Evidence

Initial #118 validation on the launch checkpoint passed:
`/tmp/qwen3_validate_startup-prewarm-readiness_prewarm.json` reported
zero failed prewarm phases over `bs=1,2,4,8,16,30` plus rollback rebuilds.
The first cold `bs=1` prewarm request paid the expected startup tail, then
copied-back voice validation passed:
`/tmp/qwen3_voice_validate/startup-prewarm-readiness/report.json`.

After switching to production-shaped 1s chunks and adding
`--max-total-tokens 400000`, the capped server passed a bs=50 / 300s
production-cadence smoke with rollback enabled:
`/tmp/qwen3_bs50_max_tokens_400k_300s_20260517T112210Z/stdout.json`
reported 710/710 strict realtime, 710/710 audio-stream realtime, 0
failures, 0 close failures, 8/8 rollback success, mean/max RTF
0.537/0.912, max first audio 0.936s, and telemetry max VRAM 139.66 GiB.

The self-contained prewarm validation passed after removing manual prompt
and fixture requirements. With `HF_TOKEN` available inside the AMD
container, `test/manual/qwen3_omni_audio_prewarm.py` hydrated the five
launch WAV fixtures from
`beginning-ai/beginnings-assets/test_audio/qwen3_omni_prewarm/v0/`, wrote
the baked Beginnings prompt into the output directory, and reported zero
failed phases in `/tmp/qwen3_self_contained_prewarm_final.json`.

After expanding default coverage to `bs=1,2,4,8,16,32,50`, validation
again passed with zero request, close, or rollback failures:
`/tmp/qwen3_prewarm_bs50_default_validation/qwen3_prewarm_bs50_default_validation.json`.
The bs=50 prewarm phase had max first-audio 1.632s and max RTF 1.115;
this is acceptable because startup has not reported healthy yet.

Code2Wav runtime-stream prewarm was added after a full bs=50 run showed a
rare non-flush Code2Wav launch/enqueue wall spike on the dedicated async
stream. Model startup prewarm already covered the production Code2Wav
shape grid on the default stream, but live traffic uses the runtime
Code2Wav stream, so startup now warms the same grid on that stream too.

Validation:

- copied-back voice validation:
  `/tmp/qwen3_voice_validate/code2wav-stream-prewarm/report.json`,
  5/5 fixtures passed
- clean server log:
  `/tmp/qwen3_code2wav_stream_prewarm_clean_server.log`
- normal Code2Wav model prewarm: 421 specs, 31.95s
- runtime Code2Wav stream prewarm: 421 specs, 12.44s
- bs=50 / 300s smoke:
  `/tmp/qwen3_code2wav_stream_prewarm_clean_smoke_300s/stdout.json`,
  704/711 strict realtime and 711/711 audio-stream realtime

The smoke had zero Code2Wav forward-enqueue spikes above 50 ms, so this is
accepted as startup/tail hardening. It does not replace the bs=50 roadmap:
remaining strict misses were fixed-overhead Talker-prefill misses.

Continuation prewarm was expanded from 8 to 50 sessions after a fresh
bs=50 / 300s smoke exposed turn-1 first-audio tails on parented
multi-turn requests:

- before expansion:
  `/tmp/qwen3_code2wav_oldest_ready_smoke_300s/stdout.json`
- result: 726/726 strict realtime, 726/726 audio-stream realtime,
  mean/max RTF 0.425/0.912, max first audio 0.909s
- the top rows were early continuation turns, which means startup prewarm
  had warmed single-turn bs=50 shapes but not enough parented continuation
  traffic at the target capacity

After changing the default continuation prewarm to 50 sessions,
copied-back voice validation passed all five fixtures:
`/tmp/qwen3_voice_validate/bs50-continuation-prewarm/report.json`.
The repeat bs=50 / 300s smoke passed with max first audio under the
conversation target:

- artifact:
  `/tmp/qwen3_code2wav_oldest_ready_bs50prewarm_smoke_300s/stdout.json`
- result: 726/726 strict realtime, 726/726 audio-stream realtime,
  14/14 rollback success, mean/max RTF 0.420/0.889, mean/max audio-stream
  RTF 0.381/0.826, max first audio 0.675s, max completion tail 0.116s

After rejecting the broad `M=1..256` output GEMM prewarm probe and
restarting from automatic startup prewarm only, the follow-up smoke
improved again:

- copied-back voice validation:
  `/tmp/qwen3_voice_validate/remove-broad-small-m-prewarm/report.json`
- artifact:
  `/tmp/qwen3_remove_broad_small_m_smoke_300s/stdout.json`
- result: 730/730 strict realtime, 730/730 audio-stream realtime,
  14/14 rollback success, mean/max RTF 0.417/0.817, mean/max audio-stream
  RTF 0.380/0.700, max first audio 0.621s, max completion tail 0.134s

Decision: startup prewarm must cover parented continuation traffic at the
capacity currently being claimed. If a fresh-start smoke needs a manual
warm run to avoid early first-audio tails, startup prewarm coverage is
incomplete.

A later full bs=50 30-minute soak found a different first-hit class:
sampled Talker prefill spent `prepare_inputs_ms=867.14` while its Talker
model body was only `0.57ms`. The missing warmup was the pre-model Talker
projection MLP first GEMM, `(M,1024,2048)`, for observed cached-prefix
continuation lengths. Startup prewarm now warms that family alongside the
existing Talker model/body GEMMs.

Validation:

- copied-back voice validation:
  `/tmp/qwen3_voice_validate/talker-projection-prewarm/report.json`,
  5/5 fixtures passed
- clean bs=50 / 300s smoke:
  `/tmp/qwen3_projection_prewarm_clean_smoke_300s/stdout.json`
- result: 729/729 strict realtime, 729/729 audio-stream realtime,
  14/14 rollback success, mean/max RTF 0.414/0.842, mean/max
  audio-stream RTF 0.376/0.715, max first audio 0.727s, max completion
  tail 0.643s

Decision: prewarm Talker prefill prep projections, not only the captured
Talker model body.

Full soak after projection warmup:

- artifact: `/tmp/qwen3_projection_prewarm_full_soak_30min/stdout.json`
- server log: `/tmp/qwen3_projection_prewarm_full_soak_server.log`
- telemetry:
  `/tmp/qwen3_projection_prewarm_full_soak_telemetry.jsonl`,
  `/tmp/qwen3_projection_prewarm_full_soak_30min/telemetry.png`
- result: 4209/4210 strict realtime, 4210/4210 audio-stream realtime,
  74/74 rollback success, mean/max RTF 0.462/1.053, mean/max audio-stream
  RTF 0.420/0.836, max first audio 0.940s, max completion tail 0.205s

Decision: the projection warmup fixed the concrete first-hit prep spike.
The remaining bs=50 miss is a cumulative late-session first-codec latency
chain, not another giant warmup hole.
