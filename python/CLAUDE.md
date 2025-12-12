# Project Context

## About This Project

SGLang is a fast serving framework for large language models and vision language models written in Python.

## Coding Practice

* Write clean and readable code.
* Avoid unnecessary defensive programming.
* Write comment ONLY when the logic is unavoidably complex.
* Prefer refactoring or removing existing code over adding new code.
* After implementing an approach as part of a plan and you discovered it doesn't work, pause and ask the user before reverting changes.
* Use print instead of logging.x for logging.

## Continuity across conversation compaction

To maintain continuity across compaction, you should always:
* Write down the context, implementation steps and expected result for each task you work on into the active plan
* Write down any finding into the plan. Annotate it with CONFIRMED or UNCONFIRMED depending on whether the finding is verified.

## Debugging

YOU MUST NOT VIBE DEBUG.
Always start off by adding logging to suspected problematic areas. Compared the results against known good logs from either SGLang before the recent changes or compared to the transformer's implementation.
When a suspected issue is disproved, first remove the logs before adding logs for another potential problem.
When an issue is confirmed or disproved, write it down into the plan before making any other changes.

## About My Task

I am adding Qwen3 Omni support into SGLang. Qwen3-Omni is the natively end-to-end multilingual omni-modal foundation models. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech.

I am only implementing streaming text/audio input and streaming text/audio output. No need to consider image/video input.

Current status:
- [x] Thinker (already implemented by sglang upstream)
- [x] Talker
- [x] Code Predictor
- [x] Code2wav
- [x] Batch size > 1
- [x] Streaming chunked prefill
- [x] gRPC bidirectional streaming endpoint for real-time audio I/O

## Architecture Overview

### Component Summary

| Component | Purpose | KV Heads | Parameters |
|-----------|---------|----------|------------|
| **Thinker** | Text generation (main LLM) | 4 | 30B-A3B MoE |
| **Talker** | Codec token generation | 2 | 3B-A0.3B MoE |
| **Code Predictor** | Multi-token prediction (15 residual codes) | Isolated | 80M dense |
| **Code2Wav** | Codec frames → waveform | 16 | 200M ConvNet |

### Submodules

* Audio Encoder `thinker.audio_tower` (Audio transformer, 650M parameters) - attention-encoder-decoder model
* Vision Encoder `thinker.visual` - Ignore for my task
* Thinker `thinker.model` (MoE Transformer, 30B-A3B parameters) - almost the same architecture as Qwen3-30B-A3B
* Talker `talker` (MoE Transformer, 3B-A0.3B parameters) - scaled down version of Qwen3 MoE but with Qwen2 sparse MoE decoder layers
* Multi-token Prediction (MTP) `talker.code_predictor` (Dense Transformer, 80M parameters) - fixed-step autoregressive dense transformer. Once talker generates a token, MTP predicts the remaining 16-1=15 tokens for the current frame
* Code2wav `code2wav` (ConvNet, 200M parameters) - multi-codebook codec decoder that only attends to the left context. Decodes tokens from talker and MTP into waveforms

### Audio Encoder Architecture

The audio encoder uses windowed bidirectional attention via `cu_seqlens`. Each audio chunk is processed independently with its own attention window.

**Chunked processing:**
- `feature_lens = [100, 57]` creates two attention windows via `cu_seqlens`
- Each chunk's output tokens are concatenated
- Supports true streaming audio encoding

**Token output formula** (`_get_feat_extract_output_lengths`):
```python
feat_lengths = (input_lengths - 1) // 2 + 1
output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
```

## Streaming Design

### Streaming Audio Input

**Goal:** Minimize latency from "user finishes speaking" to "first output token".

**gRPC Flow:**
```
Client                          gRPC Server                    Scheduler
  │                                 │                              │
  ├─ StreamingAudioConfig ─────────►│                              │
  │  (text prompt)                  ├─ StreamingAudioStartReqInput ►│
  │                                 │                              ├─► Text prefix prefill
  │                                 │                              │   (KV cached)
  ├─ audio_chunk (PCM16) ──────────►│                              │
  │  (buffered until 1 sec)         ├─ StreamingAudioChunkReqInput ►│
  │                                 │  (when 100 mel frames ready)  ├─► Audio tokens added
  │                                 │                              │   (not prefilled yet)
  ├─ end_of_input ─────────────────►│                              │
  │                                 ├─ StreamingAudioEndReqInput ──►│
  │                                 │  (with final mel chunk)       ├─► Final prefill
  │◄─ text_delta ───────────────────┤◄─────────────────────────────┤   (all audio + suffix)
  │◄─ audio_chunk (PCM16) ──────────┤◄─────────────────────────────┤
  │◄─ StreamingAudioComplete ───────┤                              │
```

**Latency Optimization:**
- Text prefix KV is cached during first prefill (while user is still speaking)
- Final prefill reuses cached text KV (saves ~25% compute)
- Audio encoder uses windowed attention per chunk via `cu_seqlens`

**Note:** Audio is not incrementally prefilled - all audio tokens wait for `end_of_input` before the final prefill runs. The buffering happens in `StreamingAudioBuffer` which accumulates 1 second of audio (100 mel frames) before sending to scheduler.

### Why Talker is delayed 1 step behind Thinker

**The Problem:** TTS requires "look-ahead" - at decode step N, talker needs token N+1's embedding to generate proper prosody/coarticulation.

**The Mismatch:**
```
Prefill:
  - Thinker forward → logits
  - If talker runs now, it would use argmax(logits) as "first response token"
  - But scheduler samples with temperature → different token
  - Result: Talker generated audio for wrong text!
```

**The Solution:** Delay talker by 1 step so it always uses actual sampled tokens:
```
Thinker Prefill  → token_0 sampled    | Talker: wait (set talker_needs_prefill=True)
Thinker Decode 0 → token_1 sampled    | Talker: prefill with token_1 as look-ahead
Thinker Decode 1 → token_2 sampled    | Talker: decode 0, use token_2 as look-ahead
Thinker Decode 2 → token_3 sampled    | Talker: decode 1, use token_3 as look-ahead
...
Thinker EOS      →                    | Talker: continues with tts_pad_embed
                                       | Talker codec_eos → Done
```

### Talker Before Code Predictor Design

The talker runs FIRST to get hidden states, then samples codec, then runs code predictor with current hidden.

```python
# In talker.forward() decode path:

# 1. Build input from PREVIOUS frame's codes
prev_codec_embed = talker.embed_tokens(prev_codec_id)  # [1, hidden]
prev_residual_embeds = [code_predictor.embed[i](code) for i, code in enumerate(prev_residual_codes)]
input_embeds = sum([prev_codec_embed] + prev_residual_embeds) + trailing_text_hidden

# 2. Run talker model FIRST
hidden_states = talker.model(input_embeds=input_embeds)  # [1, hidden]
logits = talker.codec_head(hidden_states)

# 3. Sample codec token (inside model, not scheduler)
codec_token = _sample_codec_token(logits)  # int

# 4. Run code predictor with CURRENT hidden + sampled codec
codec_embed = talker.embed_tokens(codec_token)
predictor_input = stack([hidden_states, codec_embed])  # [2, hidden]
residual_codes = code_predictor.generate(predictor_input)  # [1, 15]

# 5. Return complete frame
codec_frame = [codec_token] + residual_codes.tolist()  # 16 codes
```

**Key insight:** By running talker first, we use current hidden directly for code predictor - no need to store past_hidden tensor between steps.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREFILL PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  input_ids ──► Thinker ──► thinker_logits ──► Scheduler samples token_0  │
│               (prefill)                                                     │
│                                                                             │
│  Talker: SKIP (set talker_needs_prefill=True, store tts_pad_embed)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIRST DECODE (Talker Prefill)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  token_0 ──► Thinker ──► thinker_logits ──► Scheduler samples token_1    │
│             (decode)                                                        │
│                                                                             │
│  origin_input_ids + token_1 ──► Talker prefill ──► hidden, logits         │
│  (build ChatML structure)                             ↓                    │
│                                          Sample codec_0 (inside model)      │
│                                          Code predictor → residual_codes_0 │
│                                          Return codec_frame_0 (16 codes)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        SUBSEQUENT DECODES                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  token_N ──► Thinker ──► thinker_logits ──► Scheduler samples token_{N+1} │
│             (decode)                                                         │
│                                                                              │
│  prev_sampled_token ──► text_projection ──► trailing_text_hidden           │
│  (token_N, for look-ahead)                   [1, hidden]                     │
│                                                                              │
│  prev_codec_id ────────┐                                                     │
│  prev_residual_codes ──┼──► sum(16 embeds) + trailing_text_hidden           │
│                        │                       ↓                            │
│                        └──────────────► Talker decode ──► hidden, logits   │
│                                                              ↓              │
│                                              Sample codec_N (inside model)   │
│                                              Code predictor → residual_codes│
│                                              Return codec_frame_N (16 codes) │
│                                                                              │
│  codec_frame_N ──► req.talker_output_codes                                  │
│  Store: prev_codec_id = codec_frame[0], prev_residual_codes = codec_frame[1:]│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           CODE2WAV (Streaming)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  When enough frames accumulated (chunk_size=10):                            │
│                                                                             │
│  talker_output_codes ──► Code2Wav ──► PCM16 waveform                      │
│  [num_frames, 16]        (ConvNet)    [num_samples]                         │
│                                                                             │
│  Uses left_context (25 frames) for continuity, crops context from output    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### gRPC Streaming Audio Endpoint

The `StreamingAudio` RPC provides bidirectional streaming for real-time audio-to-audio inference with multi-turn support.

**Single-Turn Protocol:**
```
Client → Server:
  1. StreamingAudioConfig (session_id, system_prompt, speaker, sampling_params)
  2. audio_chunk (PCM16 bytes, 16kHz mono) × N
  3. end_of_input = true

Server → Client:
  1. text_delta (generated text tokens) × N
  2. audio_chunk (PCM16 bytes, 24kHz mono) × N
  3. StreamingAudioComplete (full_text, turn_rid)
```

**Available Speakers:**
| Voice | Speaker ID |
|-------|------------|
| chelsie | 2301 |
| ethan | 2302 (default) |
| aiden | 2303 |

**Multi-Turn Protocol:**
```python
# Turn 1: Start new session
config = StreamingAudioConfig(
    session_id="conv_123",
    system_prompt="You are Qwen-Omni...",
    speaker="chelsie",  # Optional: set voice (default: ethan)
)
# ... send audio chunks ...
# Receive response with turn_rid

# Turn 2: Continue session (speaker persists from turn 1)
config = StreamingAudioConfig(
    session_id="conv_123",
    parent_rid=turn1_rid,  # Links to previous turn
)
# ... send new audio chunks ...
```

**Multi-Turn ChatML Format:**
```
Turn 1:
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
<|audio_start|>{audio_tokens_turn1}<|audio_end|><|im_end|>
<|im_start|>assistant
{text_response_turn1}<|im_end|>

Turn 2 (extends Turn 1):
... (cached: system + user1 audio + assistant1 response) ...
<|im_start|>user
<|audio_start|>{audio_tokens_turn2}<|audio_end|><|im_end|>
<|im_start|>assistant
```

**Key insight**: Assistant audio responses are NOT in token history - only TEXT. The talker regenerates speech based on text context each turn. KV cache is locked during session and reused across turns.

**Key Components:**

1. **StreamingAudioBuffer** (`srt/audio/streaming_buffer.py`)
   - Buffers PCM16 audio until 100 mel frames (1 second) available
   - Uses WhisperFeatureExtractor for mel spectrogram computation
   - 100 mel frames → 13 audio tokens after CNN compression

2. **IO Structs** (`srt/managers/io_struct.py`)
   - `StreamingAudioStartReqInput`: Initiates session with session_id, parent_rid, system_prompt
   - `StreamingAudioChunkReqInput`: Sends mel features to scheduler
   - `StreamingAudioEndReqInput`: Signals audio input complete

3. **Session State** (`srt/managers/scheduler.py`)
   - `AudioSessionState`: Tracks session_id, system_prompt, turns
   - `AudioTurnState`: Stores input_ids, output_ids, text_response per turn
   - `cleanup_stale_audio_sessions()`: Auto-cleanup after 5 min inactivity

4. **Audio Output Streaming** (`srt/managers/scheduler_output_processor_mixin.py`)
   - `maybe_decode_code2wav_chunk()`: Decodes when 10 frames accumulated
   - `flush_code2wav_chunk()`: Decodes remaining frames at end
   - `_store_audio_turn_state()`: Stores turn state for multi-turn

**Audio Parameters:**
| Parameter | Input | Output |
|-----------|-------|--------|
| Sample rate | 16kHz | 24kHz |
| Format | PCM16 mono | PCM16 mono |
| Chunk size | 1 sec (100 mel frames) | 10 codec frames (~800ms) |

### Tensor Shapes Reference

| Tensor | Shape | Description |
|--------|-------|-------------|
| **Thinker** |||
| `input_ids` | `[seq_len]` prefill, `[batch]` decode | Input token IDs |
| `positions` | `[seq_len]` or `[batch]` | Position indices |
| `thinker_logits` | `[batch, vocab_size]` | Text token logits |
| **Talker** |||
| `input_embeds` | `[seq_len, hidden]` prefill, `[1, hidden]` decode | Built from codec + text embeds |
| `trailing_text_hidden` | `[1, hidden]` | Projected text embedding for look-ahead |
| `codec_frame` | `List[int]` (16 elements) | Complete frame: [codec, 15 residuals] |
| **Code Predictor** |||
| `input_embeds` | `[2, hidden]` | Stack of (current_hidden, codec_embed) |
| `residual_codes` | `[1, 15]` | Generated residual codes |
| **Code2Wav** |||
| `codes` | `[1, 16, num_frames]` | All codec codes |
| `waveform` | `[1, 1, num_samples]` | Output audio |

### State Stored in Req (Between Steps)

| State | Type | Purpose |
|-------|------|---------|
| `prev_sampled_thinker_token` | `int` | For computing trailing_text_hidden (look-ahead) |
| `talker_needs_prefill` | `bool` | Signals talker prefill pending at first decode |
| `prev_residual_codes` | `List[int]` | 15 residual codes from previous frame (for decode input) |
| `tts_pad_embed` | `Tensor [hidden]` | Used when thinker is done (EOS) |
| `talker_codec_ids` | `List[int]` | First codec token per step |
| `talker_output_codes` | `List[List[int]]` | All 16 codes per frame |
| `talker_kv_cache_locs` | `Tensor` | KV cache locations for switching |
| `talker_step` | `int` | Current talker decode step |
| `talker_prefill_len` | `int` | Length of talker prefill (for decode positions) |
| `talker_pcm16_chunks` | `List[bytes]` | Accumulated PCM16 audio chunks for streaming |
| `talker_last_decoded_frame` | `int` | Last codec frame index decoded to PCM |
| `talker_pcm16_chunk_cap` | `int` | Max chunks to buffer (default 1000) |

### KV Cache Switching

Thinker and Talker share the same `forward_batch` but have independent KV caches:

```python
# Before talker forward:
save thinker's req_to_token, seq_lens, out_cache_loc
switch req_to_token to talker's KV locations
reinitialize attention metadata

# Run talker forward

# After talker forward:
restore thinker's req_to_token, seq_lens, out_cache_loc
reinitialize attention metadata
```

Code predictor has its own isolated KV cache (separate model, not sharing forward_batch).

## Implementation Notes

### Key Functions in qwen3_omni_moe.py

| Function | Purpose |
|----------|---------|
| `_forward_prefill()` | Runs thinker prefill only, sets `talker_needs_prefill=True` |
| `_forward_decode()` | Main decode loop with 1-step delay logic |
| `_run_talker_prefill_at_decode()` | Runs talker prefill at first decode step, returns first codec_frame |
| `_forward_talker_only()` | Continues talker after thinker EOS |
| `_prepare_talker_prefill_1step()` | Builds talker prefill input from ChatML structure |
| `Qwen3OmniMoeTalker.forward()` | Talker forward: runs model, samples codec, runs code predictor, returns codec_frame |
| `_sample_codec_token()` | Custom codec sampling with temperature=0.9, top_k=50 |

### Scheduler Integration

State flows through these files:
1. **schedule_batch.py** - `Req` class stores per-request talker state
2. **scheduler_output_processor_mixin.py** - `_update_qwen3_omni_state()` extracts state from model output
3. **forward_batch_info.py** - `_init_qwen3_omni_states()` packs state into `model_specific_states`

### Talker Sampling

Talker uses custom sampling inside the model (not SGLang's scheduler sampler):
- Temperature: 0.9
- Top-k: 50
- Top-p: 1.0
- Suppress tokens: Last 1024 vocab IDs except `codec_eos_token_id`

See `_sample_codec_token()` in qwen3_omni_moe.py.

### Common Pitfalls

1. **Token ID ranges**: Thinker uses text vocab, Talker uses codec vocab. Don't mix embeddings.
2. **tts_pad_embed**: Must be computed by embedding with THINKER then projecting through `talker.text_projection`.
3. **origin_input_ids**: Needed at first decode for building talker prefill. Must be passed via `model_specific_states`.
4. **KV cache restoration**: Always restore thinker's forward_batch state after talker forward.

### Known Limitations

#### CUDA Graphs (NOT SUPPORTED)

CUDA graphs are **not currently supported** for Qwen3-Omni. Use `--disable-cuda-graph` when starting the server.

**Why CUDA graphs don't work:**

The Code Predictor (`talker.code_predictor`) runs a 15-step autoregressive loop to generate residual codes. Each step dynamically allocates KV cache:
```python
for step in range(15):
    loc = allocator.alloc(...)  # Dynamic allocation - incompatible with CUDA graphs
```

CUDA graphs require fixed memory addresses and control flow. Dynamic allocation returns different addresses each call, breaking graph replay.

**Future Plan for CUDA Graph Support:**

To enable CUDA graphs, the code predictor needs further refactoring:

1. **Batch the code predictor** - ✅ DONE. `generate_batched()` runs all batch items in a single set of forward passes (1 prefill + 14 decode = 15 total). Used in `_forward_decode()` and `_forward_talker_only()`. Only talker prefill uses per-request `generate()` due to variable lengths.

2. **Unroll the 15 decode steps** - Replace Python loops with explicit sequential code so CUDA can capture the entire flow.

3. **Pre-allocate all resources** - KV cache slots and request indices must be allocated once during graph capture and reused during replay:
   ```python
   code_predictor_kv_locs: torch.Tensor      # [max_bs, 17] - 2 prefill + 15 decode
   code_predictor_req_indices: torch.Tensor  # [max_bs] - request slot per batch item
   ```

4. **Vectorize req_to_token updates** - Replace Python indexing with GPU scatter operations that can be captured.

See the plan file at `/root/.claude/plans/cosmic-knitting-fairy.md` for detailed implementation design.

#### Overlap Scheduling (PARTIAL - HAS ISSUES)

Overlap scheduling is **partially supported** but introduces audio artifacts at the beginning and end of generated audio. Use `--disable-overlap-schedule` when starting the server for correct audio output.

**The Problem:**

With overlap scheduling, batch N+1's preparation runs before batch N's results processing completes:
```
Timeline with overlap:
  forward_stream: [--- Batch N forward ---][--- Batch N+1 forward ---]
  default_stream: [-- Batch N-1 results --][-- Batch N results --]
                  └─ overlaps with batch N ─┘
```

This causes `req.prev_sampled_thinker_token` (the look-ahead token for talker) to be stale when read during batch preparation, resulting in wrong audio tokens.

**What's Been Fixed (works with overlap disabled):**

1. **Pool Mutation**: Talker writes to shared `req_to_token_pool.req_to_token` during forward
   - Fix: Use `forward_batch.override_req_to_token` instead of mutating pool

2. **Stale KV Cache Locations**: `req.talker_kv_cache_locs` updated asynchronously
   - Fix: Capture to `pending_talker_state` immediately after forward

3. **Double Talker Prefill**: `req.talker_needs_prefill` also stale → two prefills → memory leak
   - Fix: If pending has KV locs, set `talker_needs_prefill=False`

4. **Wrong Decode Input**: `req.talker_codec_ids` and `req.prev_residual_codes` stale
   - Fix: Capture `codec_frame` to pending, use as `prev_codec_id`/`prev_residual_codes`

5. **Wrong Position Calculation**: `req.talker_prefill_len` and `req.talker_step` stale
   - Fix: Capture `prefill_len` and `talker_step` to pending

6. **Missing Last Frame**: Request finishes before last `codec_frame` stored
   - Fix: Store pending `codec_frame` to `req.talker_output_codes` when request finishes

**What's NOT Fixed (requires overlap disabled):**

7. **Stale prev_sampled_thinker_token**: The look-ahead token for talker is read from `req.prev_sampled_thinker_token` which is written by results processing. With overlap, this value is stale causing wrong audio at beginning and end of generation.
   - Attempted fix: Capture sampled token to pending state after sampling
   - Issue: `batch.reqs` order doesn't match `batch_result.next_token_ids` order, causing token mixing between requests
   - Future work needed to properly capture and align sampled tokens

### Testing

Prompt the user to start the server to test any changes. Do not attempt to start the server yourself.

```sh
# Start server (MUST disable CUDA graphs and overlap scheduling)
# Add --grpc-mode for streaming audio input support
python -m sglang.launch_server --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --disable-cuda-graph --disable-overlap-schedule --grpc-mode

# Test with curl (text-only, HTTP API)
curl -X POST http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
         "messages": [{"role": "user", "content": "Say hello"}}'

# Test gRPC streaming audio (requires grpcurl or custom client)
# See srt/grpc/sglang_scheduler.proto for StreamingAudio message definitions
```

Audio output:
- HTTP API: Saved to `/tmp/{rid}.wav`
- gRPC streaming: Sent as PCM16 chunks via `StreamingAudioResponse.audio_chunk`

## Key files

Reference model implementation from `transformers`: /workspace/transformers/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py
Working model implementation: srt/models/qwen3_omni_moe.py
Model config definition: srt/configs/qwen3_omni.py
Shared modules:
* srt/models/qwen3_moe.py
* srt/models/qwen3_vl.py
* srt/models/qwen3_vl_moe.py
* srt/models/qwen2_moe.py

gRPC streaming audio:
* srt/entrypoints/grpc_server.py - `StreamingAudio` RPC handler
* srt/grpc/grpc_request_manager.py - Request state management, token decoding
* srt/grpc/sglang_scheduler.proto - Protocol definitions
* srt/audio/streaming_buffer.py - Audio input buffering
* srt/managers/io_struct.py - `StreamingAudio*ReqInput` structs
* srt/managers/scheduler_output_processor_mixin.py - `maybe_decode_code2wav_chunk()`, `flush_code2wav_chunk()`

The model BF16 safetensor has HF repo id `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
The weights directory is at `/root/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695/`

## Development

Use jujutsu `jj` for version control instead of Git.

Common commands:
```sh
# Read commit logs
jj log

# Read current changes (diff)
jj show --git

# Read commit diff
jj show <change_id> --git

# Commit
jj commit -m "<message>"
```

Run python commands with `python`. You are already in the correct virtualenv.
Ask the user to install packages, don't install them yourself.