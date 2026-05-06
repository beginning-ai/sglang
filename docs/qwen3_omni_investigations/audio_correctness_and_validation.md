# Audio Correctness And Validation

## Summary

Text correctness and RTF are not enough for Qwen3 Omni. The Thinker can
return correct text while Talker decode, CodePredictor, or Code2Wav
produce speech-like but semantically broken audio. Copied-back Parakeet
STT is therefore a required gate after audio code changes.

## Decision

Run `$qwen3-omni-voice-validate` after every Qwen3 Omni audio
code-touching commit. Use `--multi-turn` for session/state-flow, Talker
decode, cache, scheduler lifecycle, or Talker-prefix work. If it fails,
stop stacking commits and bisect or abandon the bad change.

## Key Findings

- A 2026-05 simplification loop stacked many commits while text and RTF
  gates stayed green; user listening later found audio that began
  coherent and degraded into incoherent speech.
- The root bug class included deriving `talker_prefill_len` from the
  growing `talker_kv_cache_locs` list, which drifted Talker rotary
  positions.
- The correct invariant is that `talker_prefill_len` is the fixed prefix
  length established at Talker-prefill time, while `talker_kv_cache_locs`
  grows per generated frame.
- Parakeet should compare the copied-back WAV transcript against the
  Thinker's actual `full_text`, not a frozen reference string, to avoid
  false failures from legitimate sampling variation.

## Current Gate

```sh
cd ~/repos/qwen3-voice-validate
uv run python ~/repos/sglang-async-amd/test/manual/qwen3_omni_voice_validate.py --multi-turn
```

Pass criterion:

- audio duration and RMS sanity pass
- normalized Parakeet word distance <= 0.30 versus `full_text`
- all requested fixtures pass

## Historical Anchors

- Latest long-tail recheck with split-decoder baseline: 150/150 ok,
  zero length-cap tails, max output 9.577s, copied-back Parakeet 150/150
  with max normalized distance 0.12.
- If length-cap audio tails recur, compare exact prompt, input audio,
  speaker, token caps, and sampling mode against vanilla Transformers or
  the old coupled SGLang fork before treating it as a local pipeline bug.
