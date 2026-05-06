# FP8 Quantization And Overlap

## Summary

Offline FP8 Thinker plus FP8 Talker attention is accepted as the launch
foundation for MI300X. The checkpoint is uploaded as
`beginning-ai/lucy-1-FP8` at revision
`7aaecb8d422964d4ace0f98ff73d980945a5796a` and served from
`/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1`.
Overlap scheduling is accepted for the mixed-FP8 launch stack.

## Decisions

- Use offline SGLang FP8 safetensors, not online startup quantization, for
  the launch path.
- Quantize the same Thinker module set as the GPTQModel recipe; keep
  `lm_head` and router gates matching `*.mlp.gate` out of FP8.
- Full Talker FP8 is rejected for launch. It is audio-correct after the
  loader fix and uses hardware AITER FP8 kernels, but launch-shape
  throughput regressed.
- Talker-attention-only FP8 is accepted for the launch checkpoint.
- FP8 KV cache is rejected for the current voice launch stack. It uses
  real AITER FP8-KV kernels on MI300X, but repeated copied-back
  multi-turn voice validation failed on the excited turn with audio that
  diverged from the Thinker text.
- Keep overlap scheduling enabled for the launch stack; no-overlap FP8 is
  historical baseline material only.

## Key Evidence

- SGLang detects the FP8 checkpoint as `quant=fp8`.
- AITER kernels use `torch.float8_e4m3fnuz`.
- Disassembly showed `v_mfma_f32_16x16x32_fp8_fp8`, proving MI300X
  hardware FP8 execution rather than a BF16 fallback.
- Copied-back multi-turn Parakeet validation passed for FP8 Thinker and
  for the mixed FP8 Thinker + FP8 Talker-attention checkpoint.
- The corrected full Talker FP8 loader fixed the earlier garbled-audio
  result by passing `quant_config` into Talker. The corrected full Talker
  FP8 checkpoint then passed listening/Parakeet gates, but regressed
  launch-shape performance, likely due to tiny Talker MoE expert shapes
  and dynamic activation quantization overhead.
- `--kv-cache-dtype fp8_e4m3` confirmed hardware-backed FP8 KV execution:
  startup compiled AITER paged-attention with `kv_dtype=uint8_t` and
  `fp8_kv_dtype=fp8_e4m3`. It is still not acceptable for launch voice
  quality. Two warm copied-back validations against the same FP8-KV
  server failed `multi_t4/excited`: first run heard "Okay Spale..." for
  Thinker text "Okay, spill. I'm listening." (`stt_distance=0.333`), and
  retry produced extra nonsensical speech (`stt_distance=0.714`).

## Historical Artifacts

- FP8 Thinker checkpoint:
  `/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-sglang-v1`
- mixed Talker-attention checkpoint:
  `/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1`
  and HF repo `beginning-ai/lucy-1-FP8` revision
  `7aaecb8d422964d4ace0f98ff73d980945a5796a`
- full Talker FP8 checkpoint:
  `/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-sglang-v1`
- full Talker FP8 validation:
  `/tmp/qwen3_voice_validate/fp8-thinker-talker-fixed-full/report.json`
- Talker-attention FP8 validation:
  `/tmp/qwen3_voice_validate/fp8-thinker-talker-attn/report.json`
- accepted overlap+FP8 validation:
  `/tmp/qwen3_voice_validate/overlap-fp8-final-duration-aware/report.json`
- accepted overlap+FP8 bs=50 baseline:
  `/tmp/qwen3_partial_prefill_i4_bs50_30min/result.json`
- rejected FP8-KV validation:
  `/tmp/qwen3_voice_validate/fp8_kv_voice/report.json`,
  `/tmp/qwen3_voice_validate/fp8_kv_voice_retry/report.json`
