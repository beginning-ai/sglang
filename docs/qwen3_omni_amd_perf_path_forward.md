# Qwen3 Omni AMD Performance Path Forward

## Goal

Maximize Qwen3 Omni realtime voice concurrency on one AMD MI300X.

Launch target: bs=50 concurrent voice sessions, each capped at 30 minutes,
with rollback enabled. `bs=N` means N active voice users, not a turn cap.
The harness runs each user until the session wall-clock limit and lets the
last in-flight turn finish before scoring.

Engineering target: `rtf_after_eoi <= 1.0` and audio-stream realtime for
every admitted turn after startup prewarm. The current launch decision accepts
rare p100 misses in the 2-hour rolling soak, but production telemetry should
keep the cap visible and reduce it if live traffic is worse than the accepted
soak.

## Launch Stack

- checkpoint: `/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1`
- HF repo: `beginning-ai/lucy-1-FP8`, revision `7aaecb8d422964d4ace0f98ff73d980945a5796a`
- FP8 Thinker plus FP8 Talker attention projections
- BF16 Talker MoE, CodePredictor, Code2Wav, embeddings, router/shared experts,
  output heads, and codec/audio bridges
- upstream streaming sessions with Qwen3 Omni component KV ownership
- overlap scheduling enabled
- CodePredictor argmax enabled after constrained human A/B approval in #124
- Talker decode, Thinker prefill, cached-prefix Talker prefill, and steady plus
  short-window Code2Wav graph capture enabled
- automatic Qwen3 Omni gRPC startup prewarm before health reports `SERVING`
- safety rollback implemented as rare-path full-prefix rebuild/prefill

Canonical AMD launch command:

```sh
/opt/venv/bin/python -m sglang.launch_server \
  --model-path /mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1 \
  --mem-fraction-static 0.76 \
  --max-total-tokens 400000 \
  --qwen3-omni-max-concurrent-audio-turns 50 \
  --grpc-mode \
  --enable-streaming-session \
  --attention-backend aiter \
  --host 127.0.0.1 \
  --port 41012
```

## Current Metrics

Update this section after any launch-stack runtime change.

**Final bs=50 30-minute launch-lock soak:**
`/tmp/qwen3_bs50_launch_lock_20260521T135836Z/stdout.log`

- 4,201/4,201 strict realtime and audio-stream realtime
- failures/close failures: 0/0
- rollback 88/88
- mean/max RTF after EOI: 0.390/0.937
- mean/max audio-stream RTF: 0.351/0.700
- max first audio after EOI: 1.420s
- max completion tail: 0.244s
- max rollback rebuild: 1.470s
- telemetry avg/p95/max GPU busy: 89.7%/97.0%/99.0%
- telemetry avg/p95/max HBM activity: 9.7%/14.0%/16.0%
- avg/p95/max VRAM: 156.3/166.4/170.0 GiB
- max KV: Thinker 330,795/399,999, Talker 168,589/213,333,
  CodePredictor 800/800
- max streaming-session held tokens: 152,481

Phase totals from the same window:

- Talker decode: 302.0s total, 4.56ms mean, 5.83ms p95, 6.60ms p99,
  max batch size 18
- Code2Wav: 232.9s total, 2.79ms mean, 8.36ms p95, 13.59ms p99,
  max batch size 11
- Talker prefill: 63.1s total, 15.2ms mean, 17.3ms p95, 67.1ms p99,
  max batch size 4

**Latest short bs=50 optimization smoke:**
`/tmp/qwen3_bs50_codec_id_event_query_repeat_20260521T125630Z/stdout.log`

- 675/675 strict realtime and audio-stream realtime
- rollback 14/14
- mean/max RTF after EOI: 0.352/0.564
- mean/max audio-stream RTF: 0.319/0.483
- max first audio after EOI: 0.507s
- max completion tail: 0.059s
- Talker decode: 52.8s total, 4.39ms mean
- Code2Wav: 38.8s total, 2.69ms mean
- Talker prefill: 10.1s total, 15.0ms mean

**Accepted bs=50 launch soak:**
`/tmp/qwen3_2h_soak_20260520T133125Z/stdout.json`

- 16,658/16,658 turns ok
- rollback 317/317
- strict realtime misses: 10/16,658
- audio-stream misses: 3/16,658
- mean/max RTF after EOI: 0.463/1.203
- mean/max audio-stream RTF: 0.423/1.125
- max first audio after EOI: 1.606s
- max completion tail: 1.429s
- max rollback rebuild: 2.045s
- telemetry avg/max GPU busy: 91.5%/98.0%
- telemetry avg/max HBM activity: 9.6%/16.0%
- max VRAM: 191.6 GiB
- max KV: Thinker 399,996/399,999, Talker 166,529/213,333,
  CodePredictor 800/800
- max streaming-session held tokens: 174,789

**bs=30 strict fallback proof:**
`/tmp/qwen3_bs30_crashfix_relock_20260519T230758Z/stdout.json`

- 2615/2615 strict realtime and audio-stream realtime
- rollback 46/46
- mean/max RTF after EOI: 0.329/0.755
- max first audio after EOI: 1.185s
- max completion tail: 0.113s

## Latest Accepted Runtime Changes

- Small-window Code2Wav capture:
  `/tmp/qwen3_bs50_code2wav_smallbucket_capture_20260520T225550Z/stdout.json`
- Inline CodePredictor with Talker phases:
  `/tmp/qwen3_bs50_inline_codepredictor_20260520T231308Z/stdout.json`
- Dead Talker decode state cleanup:
  `/tmp/qwen3_bs50_dead_decode_state_cleanup_20260521T003606Z`
- Pinned Code2Wav windows:
  `/tmp/qwen3_bs50_code2wav_pinned_windows_20260521T025722Z/stdout.log`
- 200ms bucketed non-final realtime audio prefill:
  `/tmp/qwen3_bs50_stream_prefill_bucket_200ms_20260521T090107Z/stdout.log`
- Apply-inclusive audio phase timing:
  `/tmp/qwen3_bs50_apply_timing_20260521T103332Z/stdout.log`
- Current-stream async Talker codec-id staging:
  `/tmp/qwen3_bs50_codec_id_event_query_repeat_20260521T125630Z/stdout.log`

## Rejected Paths To Avoid

- Moving current codec-id staging to a separate CUDA/HIP copy stream: failed
  copied-back voice validation on all fixtures.
- Batch-level codec-id materialization: voice validation passed, but bs=50
  smokes did not improve Talker-decode `apply_ms`.
- Reusable pinned-buffer codec-id staging: moved the wait after phase logging
  and regressed end-to-end RTF.
- Raising Talker-decode capture/cap to bs=50: created larger rows but
  regressed RTF/first-audio with no wall-time gain.
- Removing the pending-Code2Wav block before Thinker prefill: regressed
  first-audio/RTF while Code2Wav event wait stayed flat.
- 300ms/400ms realtime prefill buckets: widened prefill batches but worsened
  first-audio and RTF tails. Keep 200ms.
- Raw GPU codec-frame handoff, NumPy Code2Wav packing, Code2Wav async cap=4,
  side-stream CodePredictor overlap, 30ms audio batching window, fresh
  Talker-decode requeue, simple Talker burst, and broad Code2Wav whole-decoder
  graph as launch work.

For details, see:

- `docs/qwen3_omni_investigations/profiling_and_bottlenecks.md`
- `docs/qwen3_omni_investigations/code2wav_replay_and_kernels.md`

## Validation

Run copied-back voice validation after any Qwen3 Omni audio-path code change:

```sh
cd /home/thomas/repos/qwen3-voice-validate
uv run python /home/thomas/repos/sglang-async-amd/test/manual/qwen3_omni_voice_validate.py
```

The output WAV directory is printed by the validator. Do not accept a change
that touches Talker, CodePredictor, Code2Wav, audio phase scheduling, or
Qwen3 Omni request state without this gate.

## Final bs=50 Soak

Use this command for the final 30-minute launch-lock soak after the server is
started with the launch command above and reports ready:

```sh
OUT=/tmp/qwen3_bs50_launch_lock_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
LOG=/tmp/qwen3_launch_lock_server.log
: > "$LOG"

PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/manual \
  /opt/venv/bin/python - <<'PY' > "$OUT/beginnings_system_prompt.txt"
from qwen3_omni_audio_prewarm import DEFAULT_SYSTEM_PROMPT
print(DEFAULT_SYSTEM_PROMPT, end="")
PY

PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/manual \
  /opt/venv/bin/python /sgl-workspace/sglang/test/manual/qwen3_omni_soak_telemetry.py monitor \
  --output-jsonl "$OUT/gpu_telemetry.jsonl" --duration-s 1830 --interval-s 5 &
MON_PID=$!

PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/manual \
  /opt/venv/bin/python /sgl-workspace/sglang/test/manual/qwen3_omni_audio_soak_test.py \
  --host 127.0.0.1 --port 41012 \
  --sessions 50 \
  --duration-s 1800 \
  --session-duration-s 1800 \
  --turn-min-s 8 --turn-max-s 25 --initial-jitter-s 60 \
  --chunk-ms 1000 --realtime-input \
  --speaker chelsie \
  --system-prompt-file "$OUT/beginnings_system_prompt.txt" \
  --fixture greeting:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-greeting-01_turn-0.wav \
  --fixture question:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-question-01_turn-0.wav \
  --fixture emotional:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-emotional-01_turn-0.wav \
  --fixture excited:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-excited-01_turn-0.wav \
  --fixture accent:/tmp/qwen3_omni_prewarm_assets/test_audio/qwen3_omni_prewarm/v0/am-accent-01_turn-0.wav \
  --save-output-dir "$OUT" \
  --telemetry-jsonl "$OUT/kv_telemetry.jsonl" \
  --runtime-snapshot-log "$LOG" \
  --rollback-rate 0.02 \
  --rollback-min-completed-turns 3 \
  --rollback-max-rewind-turns 2 | tee "$OUT/stdout.log"
STATUS=${PIPESTATUS[0]}
wait "$MON_PID" || true
exit "$STATUS"
```

Report:

- strict/audio-stream realtime pass counts
- failure and rollback counts
- mean/max RTF after EOI
- mean/max audio-stream RTF
- max first audio after EOI
- max completion tail
- max rollback rebuild elapsed
- GPU busy, HBM activity, VRAM
- Thinker/Talker/CodePredictor KV max used/capacity
- max streaming-session held tokens

## Deployment Notes

- Deploy with backend `MAX_CONCURRENT_SESSIONS=50`.
- SGLang should use `--qwen3-omni-max-concurrent-audio-turns 50`,
  `--max-total-tokens 400000`, and `--mem-fraction-static 0.76`.
- Rollback rebuild can take multiple seconds; do not optimize rollback at the
  expense of realtime happy-path chat.
- Main Thinker KV reached near capacity in the 2-hour rolling soak. Further
  capacity work should measure KV pressure before raising concurrency.
