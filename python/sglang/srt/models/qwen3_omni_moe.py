# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Qwen3-Omni model compatible with HuggingFace weights."""

import copy
import functools
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerResizeMLP,
)

from sglang.srt.configs.qwen3_omni import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.qwen3_omni_code2wav_kernels import (
    causal_transconv1d_phase_inner,
    code2wav_rmsnorm,
    dense_causal_conv1d_kernel7,
    snakebeta,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.managers.mm_utils import embed_mm_inputs
from sglang.srt.managers.utils import (
    Qwen3OmniAudioPhasePayload,
    Qwen3OmniCodePredictorPayload,
    Qwen3OmniCodecFrame,
    Qwen3OmniTalkerDecodePayload,
    Qwen3OmniTalkerPrefillPayload,
)
from sglang.srt.managers.schedule_batch import (
    AUDIO_CODE2WAV_CHUNK_FRAMES,
    MM_PAD_SHIFT_VALUE,
    MultimodalDataItem,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.models.qwen3_vl_moe import (
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
    load_fused_expert_weights,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from sglang.srt.utils import add_prefix, is_hip, is_npu, logger


TALKER_TEMPERATURE_DEFAULT = 0.9
TALKER_TOP_K_DEFAULT = 50
TALKER_TOP_P_DEFAULT = 1.0
TALKER_REPETITION_PENALTY_DEFAULT = 1.05
_CODE_PREDICTOR_TOP_K = 50
_CODE2WAV_LEFT_CONTEXT_FRAMES = 25
_CODE2WAV_CAPTURE_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_CODE2WAV_SHORT_WINDOW_CAPTURE_BATCH_BUCKETS = (1, 2, 4)
_CODE2WAV_CAPTURE_MAX_FRAMES = 64
_TALKER_DECODE_CAPTURE_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)
_TALKER_DECODE_CAPTURE_SEQ_LEN_BUCKETS = (8192,)
_TALKER_PREFILL_CAPTURE_BATCH_SIZES = tuple(range(1, 9))
_TALKER_PREFILL_CAPTURE_MAX_TOKENS = 512
_TALKER_PREFILL_CAPTURE_MAX_Q_LEN = 128
_THINKER_PREFILL_CAPTURE_BATCH_SIZES = tuple(range(1, 33))
_THINKER_PREFILL_CAPTURE_MAX_TOKENS = 32
_THINKER_PREFILL_CAPTURE_KV_BUCKETS = (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
_TALKER_PREFILL_CAPTURE_KV_BUCKET = _THINKER_PREFILL_CAPTURE_KV_BUCKETS[-1]
_QWEN3_OMNI_TALKER_PREFILL_PREWARM_MS = (
    # Observed cached-prefix Talker-prefill continuation totals in bs=50
    # soaks. Missing AITER configs for these exact Ms create ~100ms first-hit
    # stalls that show up as first-audio and fixed-overhead tails.
    36,
    42,
    49,
    59,
    65,
    72,
    78,
    84,
    85,
    91,
    95,
    98,
    101,
    107,
    108,
    114,
    118,
    124,
    154,
    172,
    179,
    334,
    1093,
)
_QWEN3_OMNI_TALKER_PREFILL_PREWARM_GEMMS = tuple(
    (m, n, k)
    for m in _QWEN3_OMNI_TALKER_PREFILL_PREWARM_MS
    for n, k in (
        # Talker text/hidden projection MLP before the captured Talker model.
        # Missing these leaves EOI first-codec prep vulnerable to first-hit
        # AITER config stalls even when the Talker model body is warm.
        (1024, 2048),
        (1536, 1024),
        # Talker model/codec hot GEMMs observed in cached-prefix prefills.
        (1024, 768),
        (128, 1024),
    )
)
_QWEN3_OMNI_AITER_BF16_PREWARM_GEMMS = (
    # These late-session output shapes were still first-hitting AITER during
    # bs=50 soaks after the regular first-turn prewarm.
    (50, 128, 2048),
    (71, 128, 2048),
    (5075, 128, 2048),
) + _QWEN3_OMNI_TALKER_PREFILL_PREWARM_GEMMS
_VOICE_SUPPRESSED_THINKER_TOKEN_IDS = (
    151644,  # <|im_start|>
    151667,  # <think>
    151668,  # </think>
    151669,  # <|audio_bos|>
    151670,  # <|audio_eos|>
    151675,  # <|AUDIO|>
)


def _qwen3_omni_max_concurrent_audio_turns() -> int:
    max_concurrent = get_global_server_args().qwen3_omni_max_concurrent_audio_turns
    if max_concurrent is None:
        raise RuntimeError(
            "Qwen3-Omni CodePredictor scratch sizing requires "
            "--qwen3-omni-max-concurrent-audio-turns."
        )
    return max_concurrent


def _code_predictor_capture_batch_sizes() -> Tuple[int, ...]:
    return tuple(range(1, _qwen3_omni_max_concurrent_audio_turns() + 1))


def _code2wav_capture_batch_bucket(batch_size: int) -> Optional[int]:
    if batch_size <= 0:
        return None
    for bucket in _CODE2WAV_CAPTURE_BATCH_BUCKETS:
        if batch_size <= bucket:
            return bucket
    return None


def _talker_decode_capture_batch_bucket(
    batch_size: int, max_concurrent_audio_turns: int
) -> Optional[int]:
    max_bucket = min(
        max_concurrent_audio_turns, _TALKER_DECODE_CAPTURE_BATCH_BUCKETS[-1]
    )
    if batch_size <= 0 or batch_size > max_bucket:
        return None
    for bucket in _TALKER_DECODE_CAPTURE_BATCH_BUCKETS:
        if batch_size <= bucket:
            return min(bucket, max_bucket)
    return max_bucket


def _talker_decode_capture_batch_size(batch_size: int) -> Optional[int]:
    return _talker_decode_capture_batch_bucket(
        batch_size, _qwen3_omni_max_concurrent_audio_turns()
    )


def _code2wav_steady_chunk_frames() -> int:
    return AUDIO_CODE2WAV_CHUNK_FRAMES


def _attention_backend_uses_aiter(attn_backend) -> bool:
    backend_names = [
        attn_backend.__class__.__module__,
        attn_backend.__class__.__name__,
    ]
    for attr in ("prefill_backend", "decode_backend"):
        child = getattr(attn_backend, attr, None)
        if child is not None:
            backend_names.extend(
                [child.__class__.__module__, child.__class__.__name__]
            )
    return any("aiter" in name.lower() for name in backend_names)


def _server_uses_aiter_attention() -> bool:
    server_args = get_global_server_args()
    return (
        server_args.attention_backend == "aiter"
        or server_args.prefill_attention_backend == "aiter"
        or server_args.decode_attention_backend == "aiter"
    )


def _talker_decode_seq_len_bucket(seq_len: int) -> Optional[int]:
    if seq_len <= 0 or seq_len > _TALKER_DECODE_CAPTURE_SEQ_LEN_BUCKETS[-1]:
        return None
    for bucket in _TALKER_DECODE_CAPTURE_SEQ_LEN_BUCKETS:
        if seq_len <= bucket:
            return bucket
    return None


def _thinker_prefill_kv_bucket(seq_len: int) -> Optional[int]:
    if seq_len <= 0:
        return None
    for bucket in _THINKER_PREFILL_CAPTURE_KV_BUCKETS:
        if seq_len <= bucket:
            return bucket
    return None


@dataclass(frozen=True)
class Qwen3OmniTalkerPrefillUserChunk:
    source_start: int
    source_end: int
    talker_start: int
    talker_end: int


@dataclass(frozen=True)
class Qwen3OmniTalkerPrefillAssistantChunk:
    source_start: int
    source_end: int
    talker_start: int
    talker_end: int


Qwen3OmniTalkerPrefillChunk = Union[
    Qwen3OmniTalkerPrefillUserChunk,
    Qwen3OmniTalkerPrefillAssistantChunk,
]


@dataclass(frozen=True)
class Qwen3OmniTalkerPrefillChunkContract:
    chunks: Tuple[Qwen3OmniTalkerPrefillChunk, ...]
    talker_input_ids: torch.Tensor


def build_qwen3_omni_talker_prefill_chunk_contract(
    input_ids: torch.Tensor,
    position_input_ids: torch.Tensor,
    input_ids_cpu: Sequence[int],
    position_input_ids_cpu: Optional[Sequence[int]] = None,
    *,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    tts_pad_token_id: int,
    include_assistant_response: bool,
) -> Qwen3OmniTalkerPrefillChunkContract:
    if input_ids.shape != position_input_ids.shape:
        raise RuntimeError(
            "Qwen3-Omni Talker prefill chunk contract requires input ids and "
            "position ids with identical shapes"
        )
    if len(input_ids_cpu) != input_ids.shape[0]:
        raise RuntimeError(
            "Qwen3-Omni Talker prefill chunk contract requires CPU and GPU "
            f"input ids with identical length: cpu={len(input_ids_cpu)}, "
            f"gpu={input_ids.shape[0]}"
        )

    im_start_positions = [
        i for i, token_id in enumerate(input_ids_cpu) if token_id == im_start_token_id
    ]
    if not im_start_positions:
        raise RuntimeError(
            "Qwen3-Omni Talker prefill requires ChatML input with <|im_start|>"
        )

    im_start_indexes = im_start_positions + [len(input_ids_cpu)]

    chunks: List[Qwen3OmniTalkerPrefillChunk] = []
    talker_input_id_parts: List[torch.Tensor] = []
    talker_input_id_cpu_parts: List[int] = []
    talker_cursor = 0
    num_segments = len(im_start_indexes) - 1
    use_cpu_contract_ids = position_input_ids_cpu is not None
    if use_cpu_contract_ids and len(position_input_ids_cpu) != len(input_ids_cpu):
        raise RuntimeError(
            "Qwen3-Omni Talker prefill chunk contract requires CPU input ids "
            "and CPU position ids with identical length"
        )
    for i in range(num_segments):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        if im_start_index + 1 >= segment_end_index:
            raise RuntimeError(
                "Qwen3-Omni Talker prefill saw a ChatML segment without a role"
            )
        role_token = input_ids_cpu[im_start_index + 1]

        if role_token == system_token_id:
            continue
        if role_token == user_token_id:
            talker_ids = position_input_ids[im_start_index:segment_end_index]
            talker_end = talker_cursor + talker_ids.shape[0]
            chunks.append(
                Qwen3OmniTalkerPrefillUserChunk(
                    source_start=im_start_index,
                    source_end=segment_end_index,
                    talker_start=talker_cursor,
                    talker_end=talker_end,
                )
            )
            if use_cpu_contract_ids:
                talker_input_id_cpu_parts.extend(
                    position_input_ids_cpu[im_start_index:segment_end_index]
                )
            else:
                talker_input_id_parts.append(talker_ids)
            talker_cursor = talker_end
            continue
        if (
            include_assistant_response
            and role_token == assistant_token_id
            and i == num_segments - 1
        ):
            if use_cpu_contract_ids:
                talker_ids = None
                talker_len = 9
            else:
                talker_ids = torch.full(
                    (9,),
                    fill_value=tts_pad_token_id,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_len = talker_ids.shape[0]
            talker_end = talker_cursor + talker_len
            chunks.append(
                Qwen3OmniTalkerPrefillAssistantChunk(
                    source_start=im_start_index,
                    source_end=segment_end_index,
                    talker_start=talker_cursor,
                    talker_end=talker_end,
                )
            )
            if use_cpu_contract_ids:
                talker_input_id_cpu_parts.extend([tts_pad_token_id] * talker_len)
            else:
                talker_input_id_parts.append(talker_ids)
            talker_cursor = talker_end
            continue
        if role_token == assistant_token_id:
            continue
        raise RuntimeError(
            f"Qwen3-Omni Talker prefill saw unknown ChatML role token {role_token}"
        )

    if not talker_input_id_parts and not talker_input_id_cpu_parts:
        raise RuntimeError(
            "Qwen3-Omni Talker prefill chunk contract has no Talker chunks"
        )
    if use_cpu_contract_ids:
        talker_input_ids = torch.tensor(talker_input_id_cpu_parts, dtype=torch.long)
    else:
        talker_input_ids = torch.cat(talker_input_id_parts, dim=0)

    return Qwen3OmniTalkerPrefillChunkContract(
        chunks=tuple(chunks),
        talker_input_ids=talker_input_ids,
    )


def _qwen3_omni_speaker_id(talker_config, speaker: str) -> int:
    speaker_id = talker_config.speaker_id.get(speaker.lower())
    if speaker_id is None:
        raise NotImplementedError(f"Qwen3-Omni speaker {speaker} not implemented")
    return speaker_id


def _component_kv_allocator(forward_batch: ForwardBatch, name: str):
    return forward_batch.qwen3_omni_token_to_kv_pool_allocators[name]


def _qwen3_omni_audio_phase_batch_size(
    forward_batch: ForwardBatch, request_indices: List[int]
) -> int:
    if not request_indices:
        return forward_batch.batch_size
    return max(forward_batch.batch_size, max(request_indices) + 1)


def _mask_codec_eos(
    logits: torch.Tensor,
    codec_eos_token_id: int,
    allow_codec_eos: torch.Tensor,
) -> torch.Tensor:
    allow = allow_codec_eos.to(device=logits.device, dtype=torch.bool).view(-1)
    masked = logits.clone()
    masked[:, codec_eos_token_id] = torch.where(
        allow,
        logits[:, codec_eos_token_id],
        torch.full_like(logits[:, codec_eos_token_id], float("-inf")),
    )
    return masked


def _counter_uniform_noise(
    sampler_seed: torch.Tensor,
    sampler_step: torch.Tensor,
    sampler_offsets: torch.Tensor,
) -> torch.Tensor:
    seed = sampler_seed.to(dtype=torch.int64).view(-1, 1)
    step = sampler_step.to(dtype=torch.int64).view(-1, 1)
    offsets = sampler_offsets.to(device=seed.device, dtype=torch.int64).view(1, -1)
    # Capture-safe stochastic sampling still needs high-quality per-candidate
    # noise. A simple LCG made some Talker continuations fall into repeated
    # codec-token loops under multi-user load, so mix seed/step/candidate
    # with a fixed-shape integer hash before converting to uniforms.
    x = torch.bitwise_xor(seed, step * 0x1F123BB5)
    x = torch.bitwise_xor(x, offsets * 0x05491333)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = torch.bitwise_and(x * 0x045D9F3B, 0x7FFFFFFF)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = torch.bitwise_and(x * 0x045D9F3B, 0x7FFFFFFF)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = torch.bitwise_and(x, 0x00FFFFFF)
    return (x.to(torch.float32) + 1.0) * (1.0 / 16777217.0)


def _codec_repetition_mask(
    prev_tokens: Optional[Union[List[int], List[List[int]]]],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    if prev_tokens is None:
        return mask

    if batch_size == 1:
        prev_tokens_list = (
            prev_tokens
            if len(prev_tokens) > 0 and isinstance(prev_tokens[0], list)
            else [prev_tokens]
        )
    else:
        prev_tokens_list = prev_tokens
    if len(prev_tokens_list) != batch_size:
        raise RuntimeError("Qwen3-Omni codec sampler lost previous token rows")

    for row, token_ids in enumerate(prev_tokens_list):
        if not token_ids:
            continue
        unique_tokens = [
            int(token_id)
            for token_id in set(token_ids)
            if 0 <= int(token_id) < vocab_size
        ]
        if unique_tokens:
            mask[
                row,
                torch.tensor(unique_tokens, device=device, dtype=torch.long),
            ] = True
    return mask


def _stack_codec_repetition_masks(
    repetition_masks: Optional[List[Optional[torch.Tensor]]],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    if repetition_masks is None:
        return torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    if len(repetition_masks) != batch_size:
        raise RuntimeError("Qwen3-Omni codec sampler lost repetition-mask rows")
    rows = [
        mask.to(device=device, dtype=torch.bool)
        if mask is not None
        else torch.zeros((vocab_size,), dtype=torch.bool, device=device)
        for mask in repetition_masks
    ]
    return torch.stack(rows, dim=0)


def _update_codec_repetition_mask(
    repetition_mask: torch.Tensor,
    codec_tokens: torch.Tensor,
) -> torch.Tensor:
    return repetition_mask.scatter(
        1,
        codec_tokens.to(dtype=torch.long).view(-1, 1),
        True,
    )


def _sample_codec_tokens(
    logits: torch.Tensor,
    codec_eos_token_id: int,
    temperature: Union[float, torch.Tensor] = TALKER_TEMPERATURE_DEFAULT,
    top_k: Union[int, torch.Tensor] = TALKER_TOP_K_DEFAULT,
    top_p: Union[float, torch.Tensor] = TALKER_TOP_P_DEFAULT,
    repetition_penalty: Union[float, torch.Tensor] = TALKER_REPETITION_PENALTY_DEFAULT,
    prev_tokens: Optional[Union[List[int], List[List[int]]]] = None,
    repetition_mask: Optional[torch.Tensor] = None,
    sampler_seed: Optional[torch.Tensor] = None,
    sampler_step: Optional[torch.Tensor] = None,
    sampler_offsets: Optional[torch.Tensor] = None,
    uniform_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample codec tokens from logits. Works for both single and batched inputs.

    Args:
        logits: [vocab_size] or [batch_size, vocab_size]
        codec_eos_token_id: EOS token to preserve during suppression
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Repetition penalty for previously generated tokens
        prev_tokens: Previous codec tokens for repetition penalty
        repetition_mask: Capture-safe previous-token mask for repetition penalty
        sampler_seed: Per-request deterministic seed for capture-safe sampling
        sampler_step: Per-request codec step counter for capture-safe sampling
        sampler_offsets: Static candidate offsets used by the counter sampler
        uniform_noise: Optional pre-generated uniform noise for deterministic tests

    Returns:
        tokens: [] scalar or [batch_size] tensor of sampled token IDs
    """
    single_input = logits.dim() == 1
    if single_input:
        logits = logits.unsqueeze(0)

    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]

    def _batch_float(value, field: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            out = value.to(device=logits.device, dtype=logits.dtype).view(-1)
            if out.shape[0] != batch_size:
                raise RuntimeError(f"Qwen3-Omni codec sampler lost {field}")
            return out
        return torch.full(
            (batch_size,), float(value), device=logits.device, dtype=logits.dtype
        )

    def _uniform_value(value, field: str, cast):
        if isinstance(value, torch.Tensor):
            flat = value.view(-1)
            if flat.shape[0] != batch_size:
                raise RuntimeError(f"Qwen3-Omni codec sampler lost {field}")
            first_tensor = flat[0]
            if (flat != first_tensor).any().item():
                raise RuntimeError(f"Qwen3-Omni codec sampler cannot mix {field}")
            return cast(first_tensor.item())
        return cast(value)

    temperatures = _batch_float(temperature, "temperature")
    repetition_penalties = _batch_float(repetition_penalty, "repetition_penalty")
    k = _uniform_value(top_k, "top_k", int)
    p = _uniform_value(top_p, "top_p", float)

    work_logits = logits.clone()
    if repetition_mask is not None:
        repetition_mask = repetition_mask.to(device=logits.device, dtype=torch.bool)
        if tuple(repetition_mask.shape) != (batch_size, vocab_size):
            raise RuntimeError("Qwen3-Omni codec sampler repetition mask mismatch")
        penalized = torch.where(
            work_logits < 0,
            work_logits * repetition_penalties.unsqueeze(1),
            work_logits / repetition_penalties.unsqueeze(1),
        )
        work_logits = torch.where(repetition_mask, penalized, work_logits)
    else:
        if prev_tokens is None:
            prev_tokens_list = [None] * batch_size
        elif single_input:
            prev_tokens_list = [prev_tokens]
        else:
            prev_tokens_list = prev_tokens

        if len(prev_tokens_list) != batch_size:
            raise RuntimeError("Qwen3-Omni codec sampler lost previous token rows")

        penalty_rows = []
        penalty_tokens = []
        for row, token_ids in enumerate(prev_tokens_list):
            if not token_ids:
                continue
            unique_tokens = set(token_ids)
            penalty_rows.extend([row] * len(unique_tokens))
            penalty_tokens.extend(int(token_id) for token_id in unique_tokens)
        if penalty_rows:
            row_idx = torch.tensor(penalty_rows, device=logits.device, dtype=torch.long)
            token_idx = torch.tensor(penalty_tokens, device=logits.device, dtype=torch.long)
            penalized = work_logits[row_idx, token_idx]
            row_penalty = repetition_penalties[row_idx]
            work_logits[row_idx, token_idx] = torch.where(
                penalized < 0, penalized * row_penalty, penalized / row_penalty
            )

    suppress_start = vocab_size - 1024
    eos_values = work_logits[:, codec_eos_token_id].clone()
    work_logits[:, suppress_start:vocab_size] = float("-inf")
    work_logits[:, codec_eos_token_id] = eos_values

    work_logits = work_logits / temperatures.unsqueeze(1)
    if k <= 0 or k >= vocab_size:
        candidate_logits = work_logits.float()
        sample_indices = torch.arange(vocab_size, device=logits.device).expand(
            batch_size, vocab_size
        )
    else:
        candidate_logits, sample_indices = torch.topk(work_logits, k, dim=-1)
        candidate_logits = candidate_logits.float()

    if p < 1.0:
        sorted_logits, sorted_idx = torch.sort(candidate_logits, descending=True, dim=-1)
        sorted_p = F.softmax(sorted_logits, dim=-1)
        cumsum_p = torch.cumsum(sorted_p, dim=-1)
        mask = cumsum_p - sorted_p > p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        candidate_logits = torch.full_like(candidate_logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    # Gumbel-max preserves categorical sampling semantics while removing
    # torch.multinomial, which blocks the replay-safe captured Talker path.
    if uniform_noise is None and sampler_seed is not None:
        if sampler_step is None:
            raise RuntimeError("Qwen3-Omni codec sampler lost sampler step")
        if sampler_offsets is None:
            sampler_offsets = torch.arange(
                candidate_logits.shape[-1], device=logits.device, dtype=torch.int64
            )
        uniform_noise = _counter_uniform_noise(
            sampler_seed=sampler_seed,
            sampler_step=sampler_step,
            sampler_offsets=sampler_offsets,
        )

    if uniform_noise is None:
        uniform_noise = torch.rand(
            candidate_logits.shape, device=logits.device, dtype=torch.float32
        )
    else:
        uniform_noise = uniform_noise.to(device=logits.device, dtype=torch.float32)
        if tuple(uniform_noise.shape) != tuple(candidate_logits.shape):
            raise RuntimeError("Qwen3-Omni codec sampler noise shape mismatch")
    finfo = torch.finfo(torch.float32)
    gumbel = -torch.log(
        -torch.log(uniform_noise.clamp(min=finfo.tiny, max=1.0 - finfo.eps))
    )
    sampled_idx = torch.argmax(candidate_logits + gumbel, dim=-1, keepdim=True)

    tokens = sample_indices.gather(-1, sampled_idx).squeeze(-1)

    if single_input:
        return tokens.squeeze(0)
    return tokens


@dataclass
class Qwen3OmniCodePredictorInputs:
    request_indices: List[int]
    input_embeds: torch.Tensor
    codec_tokens: torch.Tensor


@dataclass
class ThinkerTalkerOutput:
    """Output from Qwen3-Omni forward pass with both thinker and talker results."""

    thinker_logits: LogitsProcessorOutput
    codec_frames: Optional[List[List[int]]] = None
    tts_pad_embed: Optional[torch.Tensor] = None
    talker_out_cache_loc_list: Optional[List[torch.Tensor]] = None
    updated_talker_kv_locs_list: Optional[List[torch.Tensor]] = None
    updated_talker_repetition_mask_list: Optional[List[torch.Tensor]] = None
    talker_mrope_position_delta_list: Optional[List[torch.Tensor]] = None
    talker_prefill_input_ids_list: Optional[List[torch.Tensor]] = None
    talker_user_prefill_len_list: Optional[List[int]] = None
    prev_codec_input_embeds_list: Optional[List[Optional[torch.Tensor]]] = None
    talker_needs_prefill: bool = False
    code_predictor_inputs: Optional[Qwen3OmniCodePredictorInputs] = None
    prompt_accept_hidden_states: Optional[List[torch.Tensor]] = None
    audio_phase_metrics: Optional[Dict[str, float]] = None

    # Delegate LogitsProcessorOutput attributes to thinker_logits for compatibility
    # with BatchResult.copy_to_cpu() which expects these attributes
    @property
    def hidden_states(self):
        return self.thinker_logits.hidden_states if self.thinker_logits else None

    @property
    def next_token_logprobs(self):
        return self.thinker_logits.next_token_logprobs if self.thinker_logits else None

    @property
    def input_token_logprobs(self):
        return self.thinker_logits.input_token_logprobs if self.thinker_logits else None

    @property
    def customized_info(self):
        return self.thinker_logits.customized_info if self.thinker_logits else None


@dataclass
class CodePredictorPhaseOutput:
    """Typed output of the code_predictor phase.

    Applied onto the umbrella `ThinkerTalkerOutput` via
    `_apply_code_predictor_phase_output`. `code_predictor_inputs` is cleared
    by the apply (the signal has been consumed).
    """

    codec_frames: List[Optional[Union[List[int], Qwen3OmniCodecFrame]]]
    prev_codec_input_embeds_list: List[Optional[torch.Tensor]]


@dataclass
class TalkerPrefillPhaseOutput:
    """Typed output of the talker_prefill phase.

    Applied onto the umbrella `ThinkerTalkerOutput` via
    `_apply_talker_prefill_phase_output`.
    """

    codec_frames: List[Optional[Union[List[int], Qwen3OmniCodecFrame]]]
    tts_pad_embed: Optional[torch.Tensor]
    talker_out_cache_loc_list: List[Optional[torch.Tensor]]
    talker_mrope_position_delta_list: List[Optional[torch.Tensor]]
    talker_prefill_input_ids_list: List[Optional[torch.Tensor]]
    talker_user_prefill_len_list: List[Optional[int]]
    prev_codec_input_embeds_list: List[Optional[torch.Tensor]]
    code_predictor_inputs: Optional[Qwen3OmniCodePredictorInputs]
    metrics: Optional[Dict[str, float]] = None


@dataclass
class TalkerDecodePhaseOutput:
    """Typed output of the talker_decode phase.

    `updated_talker_repetition_mask_list` is the canonical post-decode
    representation of the talker's codec-token history (M invariant);
    `talker_prev_codec_tokens` only appears in the prefill→first-decode
    boundary payload (`Qwen3OmniTalkerDecodePayload`/`...PrefillPayload`)
    and the batching-compatibility None-parity check, never in the typed
    output.
    """

    codec_frames: List[Optional[Union[List[int], Qwen3OmniCodecFrame]]]
    prev_codec_input_embeds_list: List[Optional[torch.Tensor]]
    updated_talker_kv_locs_list: List[Optional[torch.Tensor]]
    updated_talker_repetition_mask_list: List[Optional[torch.Tensor]]
    code_predictor_inputs: Optional[Qwen3OmniCodePredictorInputs]
    metrics: Optional[Dict[str, float]] = None


def _merge_phase_list(
    base_list: Optional[List[Optional[Any]]],
    phase_list: Optional[List[Optional[Any]]],
) -> Optional[List[Optional[Any]]]:
    if phase_list is None:
        return base_list
    merged = list(base_list) if base_list is not None else [None] * len(phase_list)
    for i, value in enumerate(phase_list):
        if value is not None:
            merged[i] = value
    return merged if any(value is not None for value in merged) else None


def _non_eos_code_predictor_inputs(
    request_indices: List[int],
    input_embeds: torch.Tensor,
    codec_tokens: torch.Tensor,
    codec_eos_token_id: int,
) -> Optional[Qwen3OmniCodePredictorInputs]:
    non_eos = codec_tokens != int(codec_eos_token_id)
    if non_eos.all():
        return Qwen3OmniCodePredictorInputs(
            request_indices=list(request_indices),
            input_embeds=input_embeds,
            codec_tokens=codec_tokens,
        )
    if not non_eos.any():
        return None

    keep = torch.nonzero(non_eos, as_tuple=True)[0]
    return Qwen3OmniCodePredictorInputs(
        request_indices=[request_indices[i] for i in keep.cpu().tolist()],
        input_embeds=input_embeds.index_select(0, keep),
        codec_tokens=codec_tokens.index_select(0, keep),
    )


def _stage_qwen3_omni_codec_ids(
    codec_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.cuda.Event]:
    codec_ids_cpu = torch.empty(
        codec_tokens.shape,
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )
    codec_ids_cpu.copy_(codec_tokens, non_blocking=True)
    codec_ids_event = torch.cuda.Event(enable_timing=False)
    codec_ids_event.record(torch.cuda.current_stream(codec_tokens.device))
    return codec_ids_cpu, codec_ids_event


def _stage_qwen3_omni_codec_frames(
    codec_tokens: torch.Tensor,
    residual_codes: torch.Tensor,
    codec_ids_cpu: torch.Tensor,
    codec_ids_event: torch.cuda.Event,
) -> List[Qwen3OmniCodecFrame]:
    codec_frames_tensor = torch.cat(
        [codec_tokens.unsqueeze(1), residual_codes], dim=1
    )
    codec_frames_cpu = torch.empty(
        codec_frames_tensor.shape,
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )
    codec_frames_cpu.copy_(codec_frames_tensor, non_blocking=True)
    codec_frames_event = torch.cuda.Event(enable_timing=False)
    codec_frames_event.record(torch.cuda.current_stream(codec_frames_tensor.device))
    return [
        Qwen3OmniCodecFrame(
            codec_id=codec_ids_cpu[local_i],
            codes_cpu=codec_frames_cpu[local_i],
            ready_event=codec_frames_event,
            codec_id_ready_event=codec_ids_event,
        )
        for local_i in range(codec_frames_tensor.shape[0])
    ]


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.d_model
        self.embed_dim = config.d_model
        self.self_attn = VisionAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            projection_size=embed_dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class Qwen3OmniMoeAudioEncoder(PreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeAudioEncoderLayer(config)
                for _ in range(config.encoder_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        conv_out_dim = config.downsample_hidden_size * (
            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        )
        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        # Introduce vectorized mask to avoid many small tensors
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        max_len_after_cnn = (
            int(feature_lens_after_cnn.max().item())
            if feature_lens_after_cnn.numel()
            else 0
        )

        idx = torch.arange(max_len_after_cnn, device=padded_feature.device)
        padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

        padded_feature = padded_feature.unsqueeze(1)
        # Add fast path + chunk normal path
        if padded_feature.size(0) <= self.conv_chunksize:
            padded_embed = F.gelu(self.conv2d1(padded_feature))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
        else:
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                x = F.gelu(self.conv2d1(chunk))
                x = F.gelu(self.conv2d2(x))
                x = F.gelu(self.conv2d3(x))
                padded_embeds.append(x)
            padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        # Use tolist() for efficient batch conversion from tensor to Python
        for cnn_len in aftercnn_lens.tolist():
            num_full_chunks = cnn_len // window_aftercnn
            remainder = cnn_len % window_aftercnn
            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
            if remainder:
                cu_chunk_lens.append(remainder)
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
            -1, dtype=torch.int32
        )
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen3OmniMoeVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_postshuffle_norm=False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else context_dim, eps=1e-6
        )
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x.view(-1, self.hidden_size)
            if self.use_postshuffle_norm
            else x.view(-1, x.shape[-1])
        )
        hidden = self.ln_q(x).view(-1, self.hidden_size)
        for layer in self.mlp:
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = layer(hidden)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        return hidden


class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionModel):
    config: Qwen3OmniMoeVisionEncoderConfig

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = None,
        **kwargs,
    ):
        super().__init__(
            vision_config=config,
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        self.merger = Qwen3OmniMoeVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            quant_config=quant_config,
            use_postshuffle_norm=False,
            prefix=add_prefix("merger", prefix),
        )
        self.merger_list = nn.ModuleList(
            [
                Qwen3OmniMoeVisionPatchMerger(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                    use_postshuffle_norm=True,
                    quant_config=quant_config,
                    prefix=add_prefix("merger_list", prefix),
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )
        del self.deepstack_merger_list

    @property
    def deepstack_merger_list(self):
        return self.merger_list

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device


class Qwen3OmniMoeThinkerSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Qwen3-Omni thinker MoE block with upstream-style native-precision router gate."""

    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )


class Qwen3OmniMoeThinkerDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_layer_sparse: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_layer_sparse=is_layer_sparse,
        )
        if self.is_layer_sparse:
            self.mlp = Qwen3OmniMoeThinkerSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )


class Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    config: Qwen3OmniMoeThinkerConfig

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        accept_hidden_layer: int = 24,
    ):
        super().__init__(
            config,
            quant_config,
            prefix,
            language_model_cls=functools.partial(
                Qwen3MoeLLMModel,
                decoder_layer_type=Qwen3OmniMoeThinkerDecoderLayer,
            ),
        )
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config.audio_config)
        self.visual = Qwen3OmniMoeVisionEncoder(
            config.vision_config,
            quant_config=None,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        # Enable layer capture for audio output (talker needs accept_hidden_layer states)
        self.model.layers_to_capture = [accept_hidden_layer]

    def get_audio_feature(self, items: List[MultimodalDataItem]):
        device = next(self.audio_tower.parameters()).device
        feature_attention_mask = (
            torch.cat([item.feature_attention_mask for item in items], dim=0)
            .type(torch.long)
            .to(device)
        )
        input_features = (
            torch.cat([item.feature for item in items])
            .type(self.audio_tower.dtype)
            .to(next(self.audio_tower.parameters()).device)
        )
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        # The audio encoder uses windowed attention via cu_seqlens.
        # Each chunk is processed independently with its own attention window.
        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )

        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features


# Per-index keys at these prefixes load into row i of the corresponding
# stacked Parameter on the code predictor (see load_weights below).
_CODE_PREDICTOR_STACK_PARAMS = (
    ("talker.code_predictor.lm_head.", "talker.code_predictor.lm_head_stack"),
    (
        "talker.code_predictor.model.embed_tokens.",
        "talker.code_predictor.model.embed_tokens_stack",
    ),
)


class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3MoeModel):
    """Code predictor model with stacked embeddings for each code group."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id_offset: int = 0,
    ):
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=functools.partial(
                Qwen3MoeDecoderLayer,
                is_layer_sparse=False,
            ),
            layer_id_offset=layer_id_offset,
        )
        # Stacked embedding for residual code groups: [num_code_groups - 1, V, H].
        # Replaces a ModuleList of VocabParallelEmbedding so the per-step weight
        # is a constant Parameter slice (graph-capture-safe). Assumes TP=1 for
        # the code predictor; vocab parallel sharding would need a per-rank
        # variant of this stack.
        if self.pp_group.is_first_rank:
            num_groups = config.num_code_groups - 1
            self.embed_tokens_stack = nn.Parameter(
                torch.empty(num_groups, config.vocab_size, config.hidden_size)
            )
        else:
            self.embed_tokens_stack = PPMissingLayer()


class Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(nn.Module):
    """Code predictor for conditional generation with ModuleList of lm_heads."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id_offset: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_id_offset = layer_id_offset

        self.model = Qwen3OmniMoeTalkerCodePredictorModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            layer_id_offset=layer_id_offset,
        )

        # Stacked lm_head for each code group: [num_code_groups - 1, V, H].
        # Replaces a ModuleList of nn.Linear so the per-step weight is a
        # constant Parameter slice (graph-capture-safe).
        num_heads = config.num_code_groups - 1
        self.lm_head_stack = nn.Parameter(
            torch.empty(num_heads, config.vocab_size, config.hidden_size)
        )
        self._override_req_to_token_buffer = None
        self._override_req_to_token_max_bs = None
        # Cached loop-invariant tensors for generate_batched.
        # Keyed by (max_bs, num_tokens, prefill_len, device).
        self._step_state_cache_key = None
        self._cached_seq_lens_per_step = None
        self._cached_seq_lens_per_step_cpu = None
        self._cached_positions_per_step = None
        # Persistent KV scratch slots reserved on first generate_batched call
        # and held forever. This keeps the hot code predictor path from doing
        # allocator.alloc/free per audio chunk. Larger batches still go through
        # allocator.alloc.
        self._scratch_kv_locs = None
        self._scratch_kv_capacity = 0
        self._captured_graphs = {}
        self._captured_graph_family_keys = set()

    @property
    def persistent_kv_reservation_size(self) -> int:
        return int(self._scratch_kv_capacity)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.model.embed_tokens_stack

    def _get_persistent_scratch_kv_locs(
        self, forward_batch: ForwardBatch, batch_size: int, kv_len: int
    ) -> Optional[torch.Tensor]:
        """Return a slice of the persistent KV scratch, or None if bs too large.

        Lazily reserves one CodePredictor scratch row per configured active
        audio turn. Returning None signals the caller to use the eager
        per-call alloc/free path; the captured-graph path requires this
        persistent scratch and fails loudly if it is unavailable.
        """
        scratch_max_bs = _qwen3_omni_max_concurrent_audio_turns()
        if batch_size > scratch_max_bs:
            return None
        allocator = _component_kv_allocator(forward_batch, "code_predictor")
        needed = scratch_max_bs * kv_len
        if self._scratch_kv_locs is None:
            new_locs = allocator.alloc(needed)
            if new_locs is None:
                raise RuntimeError(
                    "Qwen3-Omni CodePredictor persistent KV scratch allocation "
                    f"failed: needed={needed}, scratch_max_bs={scratch_max_bs}, "
                    f"kv_len={kv_len}"
                )
            self._scratch_kv_locs = new_locs
            self._scratch_kv_capacity = needed
        return self._scratch_kv_locs[: batch_size * kv_len]

    def _get_override_req_to_token_buffer(
        self, forward_batch: ForwardBatch, kv_len: int
    ) -> torch.Tensor:
        max_bs = forward_batch.req_to_token_pool.size
        if (
            self._override_req_to_token_buffer is None
            or self._override_req_to_token_max_bs != max_bs
            or self._override_req_to_token_buffer.shape[1] != kv_len
        ):
            self._override_req_to_token_buffer = torch.zeros(
                (max_bs, kv_len), dtype=torch.int32, device=forward_batch.seq_lens.device
            )
            self._override_req_to_token_max_bs = max_bs
        return self._override_req_to_token_buffer

    def _get_step_state_buffers(
        self,
        forward_batch: ForwardBatch,
        num_tokens: int,
        prefill_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (positions_per_step, seq_lens_per_step, seq_lens_per_step_cpu).

        Each is shape ``[num_tokens, max_bs]``; row ``k`` (k>=1) holds the
        per-decode-step values used inside the generate_batched decode loop:
        ``seq_lens = prefill_len + k`` and ``positions = prefill_len + k - 1``.
        Row 0 is unused (prefill path sets its own state).
        """
        max_bs = forward_batch.req_to_token_pool.size
        device = forward_batch.seq_lens.device
        cache_key = (max_bs, num_tokens, prefill_len, device)
        if self._step_state_cache_key != cache_key:
            seq_lens_col = torch.arange(
                prefill_len,
                prefill_len + num_tokens,
                dtype=torch.int32,
                device=device,
            )
            self._cached_seq_lens_per_step = (
                seq_lens_col.unsqueeze(1).expand(num_tokens, max_bs).contiguous()
            )
            self._cached_seq_lens_per_step_cpu = self._cached_seq_lens_per_step.cpu()
            self._cached_positions_per_step = (
                (seq_lens_col - 1).to(torch.long).unsqueeze(1).expand(num_tokens, max_bs).contiguous()
            )
            self._step_state_cache_key = cache_key
        return (
            self._cached_positions_per_step,
            self._cached_seq_lens_per_step,
            self._cached_seq_lens_per_step_cpu,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        generation_steps: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for code predictor.

        Args:
            input_ids: Input token ids
            positions: Position indices
            forward_batch: Forward batch info
            input_embeds: Optional pre-computed embeddings (for prefill stage)
            generation_steps: Current generation step (0 to num_code_groups-2),
                              selects which embedding and lm_head to use

        Returns:
            logits: Output logits from the selected lm_head
        """

        if input_embeds is not None and input_embeds.shape[0] > 1:
            # Prefill stage
            generation_steps = input_embeds.shape[0] - 2
        else:
            # Generation stage
            input_embeds = F.embedding(
                input_ids, self.model.embed_tokens_stack[generation_steps - 1]
            )

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        logits = F.linear(hidden_states, self.lm_head_stack[generation_steps])

        return logits

    def _sample(
        self,
        logits: torch.Tensor,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.8,
    ) -> torch.Tensor:
        # Launch uses CodePredictor argmax after constrained human A/B approval.
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _run_prefill_and_decode_body(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        kv_locs: torch.Tensor,
        num_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        require_capturable_metadata: bool = False,
    ) -> torch.Tensor:
        """The capture-eligible core of generate_batched: prefill + 14 decode iters.

        Caller has already:
          - allocated/persistent-scratched ``kv_locs`` of shape [batch_size, kv_len];
          - written kv_locs into both ``override_req_to_token`` and the
            ``req_to_token_pool`` row for the current ``req_pool_indices``;
          - set ``forward_batch.req_pool_indices`` and ``override_req_to_token``.

        Caller is responsible for save/restore of forward_batch attrs and for
        cleanup of the temporary writes after this returns.

        Returns:
            - [batch_size, num_tokens] sampled residual codec tokens.
            - [batch_size, hidden] left-context representation for the next Talker step.
        """
        batch_size = input_embeds.shape[0]
        device = input_embeds.device
        generated_tokens = []

        # === Prefill (batched) ===
        prefill_len = 2  # Each request has 2 tokens: hidden_state embed + codec embed
        total_prefill_tokens = batch_size * prefill_len
        prefill_locs = kv_locs[:, :prefill_len].reshape(-1).contiguous()

        forward_batch.seq_lens = torch.full((batch_size,), prefill_len, dtype=torch.int32, device=device)
        forward_batch.seq_lens_cpu = torch.full((batch_size,), prefill_len, dtype=torch.int32)
        forward_batch.seq_lens_sum = total_prefill_tokens
        forward_batch.out_cache_loc = prefill_locs
        forward_batch.forward_mode = ForwardMode.EXTEND
        forward_batch.extend_prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        forward_batch.extend_prefix_lens_cpu = [0] * batch_size
        forward_batch.extend_seq_lens = torch.full((batch_size,), prefill_len, dtype=torch.int32, device=device)
        forward_batch.extend_seq_lens_cpu = [prefill_len] * batch_size
        forward_batch.extend_num_tokens = total_prefill_tokens
        forward_batch.extend_start_loc = torch.arange(0, total_prefill_tokens, prefill_len, dtype=torch.int32, device=device)

        predictor_positions = torch.arange(0, prefill_len, dtype=torch.long, device=device).repeat(batch_size)
        prefill_embeds = input_embeds.view(total_prefill_tokens, -1)

        # AITER's regular extend-prefill init uses HIP-graph-incompatible ops
        # (.item(), tensor-indexed slice assigns). When the backend exposes a
        # capturable shim (`init_forward_metadata_audio_prefill`), use it; the
        # captured graph then never sees those ops. Backends without the shim
        # (e.g. Triton, whose regular init is already capturable) fall through.
        attn_backend = forward_batch.attn_backend
        capturable_prefill = getattr(
            attn_backend, "init_forward_metadata_audio_prefill", None
        )
        if capturable_prefill is None:
            attn_backend.init_forward_metadata(forward_batch)
        elif not capturable_prefill(forward_batch, prefill_len):
            if require_capturable_metadata:
                raise RuntimeError(
                    "CodePredictor capture requires audio-prefill attention "
                    "metadata, but init_forward_metadata_audio_prefill refused "
                    "this forward batch."
                )
            attn_backend.init_forward_metadata(forward_batch)

        hidden_states = self.model(
            input_ids=None,
            positions=predictor_positions,
            forward_batch=forward_batch,
            input_embeds=prefill_embeds.clone(),
        )  # [batch_size * 2, hidden]

        hidden_states = hidden_states.view(batch_size, prefill_len, -1)[:, -1, :]
        logits = F.linear(hidden_states, self.lm_head_stack[0])

        next_tokens = self._sample(logits, do_sample, top_k, top_p).squeeze(-1)
        generated_tokens.append(next_tokens)
        next_input_embeds = input_embeds[:, 1, :].clone()

        # === Decode loop (batched) ===
        positions_per_step, seq_lens_per_step, seq_lens_per_step_cpu = (
            self._get_step_state_buffers(forward_batch, num_tokens, prefill_len)
        )
        decode_locs_per_step = kv_locs[:, prefill_len:].t().contiguous()

        # override_req_to_token holds all kv_len slots already, so kv_indices
        # for step k is just override_req_to_token[:, :prefill_len+k]. Precompute
        # all (num_tokens-1) decode metadatas once instead of per-iter when the
        # selected attention backend supports it.
        use_multistep_attn = False
        if num_tokens > 1:
            forward_batch.forward_mode = ForwardMode.DECODE
            init_multistep = getattr(
                forward_batch.attn_backend,
                "init_forward_metadata_multistep",
                None,
            )
            if init_multistep is not None:
                use_multistep_attn = init_multistep(
                    forward_batch,
                    num_steps=num_tokens - 1,
                    base_seq_len=prefill_len + 1,
                )
            if (
                require_capturable_metadata
                and _attention_backend_uses_aiter(forward_batch.attn_backend)
                and not use_multistep_attn
            ):
                raise RuntimeError(
                    "CodePredictor capture on AITER requires precomputed "
                    "multistep decode attention metadata, but "
                    "init_forward_metadata_multistep refused this forward batch."
                )

        for step in range(1, num_tokens):
            step_embeds = F.embedding(
                next_tokens, self.model.embed_tokens_stack[step - 1]
            )

            current_seq_len = prefill_len + step
            forward_batch.seq_lens = seq_lens_per_step[step, :batch_size]
            forward_batch.seq_lens_cpu = seq_lens_per_step_cpu[step, :batch_size]
            forward_batch.seq_lens_sum = batch_size * current_seq_len
            forward_batch.out_cache_loc = decode_locs_per_step[step - 1]
            forward_batch.forward_mode = ForwardMode.DECODE

            predictor_positions = positions_per_step[step, :batch_size]

            if use_multistep_attn:
                forward_batch.attn_backend.select_step(step - 1)
            else:
                forward_batch.attn_backend.init_forward_metadata(forward_batch)

            # Save the unmutated embedding before the body forward in case the
            # model mutates input_embeds in-place. Then the carry sums the
            # ORIGINAL step embeddings, matching HF's `mid_residual_hiddens[k][0]`.
            step_embeds_for_carry = step_embeds.detach().clone()

            hidden_states = self.model(
                input_ids=None,
                positions=predictor_positions,
                forward_batch=forward_batch,
                input_embeds=step_embeds,
            )
            next_input_embeds = next_input_embeds + step_embeds_for_carry

            logits = F.linear(hidden_states, self.lm_head_stack[step])
            next_tokens = self._sample(logits, do_sample, top_k, top_p).squeeze(-1)
            generated_tokens.append(next_tokens)

        final_residual_embed = F.embedding(
            next_tokens, self.model.embed_tokens_stack[num_tokens - 1]
        )
        next_input_embeds = next_input_embeds + final_residual_embed

        return torch.stack(generated_tokens, dim=1), next_input_embeds

    def _should_use_captured_code_predictor(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        batch_size: int,
    ) -> bool:
        return (
            input_embeds.is_cuda
            and batch_size in _code_predictor_capture_batch_sizes()
            and _attention_backend_uses_aiter(forward_batch.attn_backend)
        )

    def _get_or_capture_code_predictor_graph(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        kv_locs: torch.Tensor,
        num_tokens: int,
        top_k: int,
        top_p: float,
    ):
        batch_size = input_embeds.shape[0]
        key = (batch_size, num_tokens, input_embeds.dtype, input_embeds.device)
        cached = self._captured_graphs.get(key)
        if cached is not None:
            return cached

        input_buffer = torch.empty_like(input_embeds)
        input_buffer.zero_()
        req_indices_buffer = torch.zeros(
            (batch_size,),
            dtype=forward_batch.req_pool_indices.dtype,
            device=input_embeds.device,
        )

        graph_forward_batch = forward_batch
        graph_forward_batch.batch_size = batch_size
        graph_forward_batch.req_pool_indices = req_indices_buffer
        graph_forward_batch.override_req_to_token = forward_batch.override_req_to_token

        def run_once():
            return self._run_prefill_and_decode_body(
                input_buffer,
                graph_forward_batch,
                kv_locs,
                num_tokens,
                False,
                top_k,
                top_p,
                require_capturable_metadata=True,
            )

        stream = torch.cuda.Stream(device=input_embeds.device)
        stream.wait_stream(torch.cuda.current_stream(device=input_embeds.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                run_once()
        torch.cuda.current_stream(device=input_embeds.device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sequences, next_input_embeds = run_once()

        captured = {
            "graph": graph,
            "input_buffer": input_buffer,
            "req_indices_buffer": req_indices_buffer,
            "sequences": sequences,
            "next_input_embeds": next_input_embeds,
        }
        self._captured_graphs[key] = captured
        return captured

    def _run_captured_prefill_and_decode_body(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        kv_locs: torch.Tensor,
        num_tokens: int,
        top_k: int,
        top_p: float,
        req_indices: torch.Tensor,
    ) -> torch.Tensor:
        graph_state = self._get_or_capture_code_predictor_graph(
            input_embeds,
            forward_batch,
            kv_locs,
            num_tokens,
            top_k,
            top_p,
        )
        graph_state["input_buffer"].copy_(input_embeds)
        graph_state["req_indices_buffer"].copy_(req_indices)
        graph_state["graph"].replay()

        # The graph output buffers are reused on the next replay. Clone before
        # returning because next_input_embeds is stored as the next Talker
        # dependency and must not be overwritten by later CodePredictor calls.
        return (
            graph_state["sequences"].clone(),
            graph_state["next_input_embeds"].clone(),
        )

    def _capture_code_predictor_graph_family(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        override_req_to_token: torch.Tensor,
        num_tokens: int,
        top_k: int,
        top_p: float,
    ) -> None:
        family_key = (num_tokens, input_embeds.dtype, input_embeds.device)
        if family_key in self._captured_graph_family_keys:
            return

        kv_len = num_tokens + 1
        original_batch_size = forward_batch.batch_size
        original_req_pool_indices = forward_batch.req_pool_indices
        original_override_req_to_token = forward_batch.override_req_to_token
        forward_batch.override_req_to_token = override_req_to_token
        for capture_bs in _code_predictor_capture_batch_sizes():
            kv_locs = self._get_persistent_scratch_kv_locs(
                forward_batch, capture_bs, kv_len
            )
            if kv_locs is None:
                raise RuntimeError(
                    "CodePredictor graph-family capture requires persistent "
                    f"scratch KV for batch size {capture_bs}"
                )
            fake_input_embeds = input_embeds.new_zeros(
                (capture_bs, input_embeds.shape[1], input_embeds.shape[2])
            )
            self._get_or_capture_code_predictor_graph(
                fake_input_embeds,
                forward_batch,
                kv_locs.view(capture_bs, kv_len),
                num_tokens,
                top_k,
                top_p,
            )
        forward_batch.batch_size = original_batch_size
        forward_batch.req_pool_indices = original_req_pool_indices
        forward_batch.override_req_to_token = original_override_req_to_token
        self._captured_graph_family_keys.add(family_key)

    @torch.no_grad()
    def generate_batched(
        self,
        input_embeds: torch.Tensor,  # [batch_size, 2, hidden]
        forward_batch: ForwardBatch,
        num_tokens: int,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.8,
        req_pool_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate num_tokens codes for a batch of requests.

        Runs 1 batched prefill + (num_tokens-1) batched decode steps,
        instead of batch_size × (1 prefill + num_tokens-1 decode).
        """
        batch_size = input_embeds.shape[0]
        kv_len = num_tokens + 1
        # === Save original forward_batch state ===
        original_seq_lens = forward_batch.seq_lens
        original_seq_lens_cpu = forward_batch.seq_lens_cpu
        original_seq_lens_sum = forward_batch.seq_lens_sum
        original_req_pool_indices = forward_batch.req_pool_indices
        original_out_cache_loc = forward_batch.out_cache_loc
        original_forward_mode = forward_batch.forward_mode
        original_extend_prefix_lens = forward_batch.extend_prefix_lens
        original_extend_prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
        original_extend_seq_lens = forward_batch.extend_seq_lens
        original_extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        original_extend_num_tokens = forward_batch.extend_num_tokens
        original_extend_start_loc = forward_batch.extend_start_loc
        original_override_req_to_token = forward_batch.override_req_to_token
        original_batch_size = forward_batch.batch_size

        allocator = _component_kv_allocator(forward_batch, "code_predictor")
        req_to_token_pool = forward_batch.req_to_token_pool
        if batch_size > forward_batch.req_to_token_pool.size:
            raise RuntimeError(
                f"Code predictor batch size {batch_size} exceeds max {forward_batch.req_to_token_pool.size}"
            )
        # Save only the [0:kv_len] window we'll overwrite below (one gather + clone).
        if req_pool_indices is None:
            req_indices = original_req_pool_indices[:batch_size]
        else:
            req_indices = req_pool_indices
        saved_req_to_tokens = req_to_token_pool.req_to_token[req_indices, :kv_len].clone()
        override_req_to_token = self._get_override_req_to_token_buffer(
            forward_batch, kv_len
        )
        kv_locs = self._get_persistent_scratch_kv_locs(
            forward_batch, batch_size, kv_len
        )
        used_persistent_scratch = kv_locs is not None
        use_captured_graph = self._should_use_captured_code_predictor(
            input_embeds, forward_batch, batch_size
        )
        if use_captured_graph and not used_persistent_scratch:
            raise RuntimeError(
                "CodePredictor capture requires persistent scratch KV, but "
                f"batch_size={batch_size} kv_len={kv_len} was not allocated."
            )
        if use_captured_graph:
            self._capture_code_predictor_graph_family(
                input_embeds,
                forward_batch,
                override_req_to_token,
                num_tokens,
                top_k,
                top_p,
            )
        if not used_persistent_scratch:
            kv_locs = allocator.alloc(batch_size * kv_len)
            if kv_locs is None:
                raise RuntimeError(
                    f"Code predictor KV pool exhausted: requested {batch_size * kv_len} "
                    f"slots ({batch_size} req(s) × {kv_len} per-req), "
                    f"available {allocator.available_size()}. "
                    f"Lower concurrency or raise --mem-fraction-static."
                )
        kv_locs = kv_locs.view(batch_size, kv_len)
        kv_locs_i32 = kv_locs.to(torch.int32)
        override_req_to_token[:batch_size, :kv_len] = kv_locs_i32
        req_to_token_pool.req_to_token[req_indices, :kv_len] = kv_locs_i32
        forward_batch.batch_size = batch_size
        forward_batch.req_pool_indices = req_indices
        forward_batch.override_req_to_token = override_req_to_token[:batch_size, :kv_len]

        if use_captured_graph:
            sequences, next_input_embeds = self._run_captured_prefill_and_decode_body(
                input_embeds,
                forward_batch,
                kv_locs,
                num_tokens,
                top_k,
                top_p,
                req_indices,
            )
        else:
            sequences, next_input_embeds = self._run_prefill_and_decode_body(
                input_embeds,
                forward_batch,
                kv_locs,
                num_tokens,
                do_sample,
                top_k,
                top_p,
            )

        # === Cleanup ===
        if not used_persistent_scratch:
            allocator.free(kv_locs.reshape(-1))
        req_to_token_pool.req_to_token[req_indices, :kv_len] = saved_req_to_tokens

        # Restore original forward_batch state
        forward_batch.seq_lens = original_seq_lens
        forward_batch.seq_lens_cpu = original_seq_lens_cpu
        forward_batch.seq_lens_sum = original_seq_lens_sum
        forward_batch.batch_size = original_batch_size
        forward_batch.req_pool_indices = original_req_pool_indices
        forward_batch.out_cache_loc = original_out_cache_loc
        forward_batch.forward_mode = original_forward_mode
        forward_batch.extend_prefix_lens = original_extend_prefix_lens
        forward_batch.extend_prefix_lens_cpu = original_extend_prefix_lens_cpu
        forward_batch.extend_seq_lens = original_extend_seq_lens
        forward_batch.extend_seq_lens_cpu = original_extend_seq_lens_cpu
        forward_batch.extend_num_tokens = original_extend_num_tokens
        forward_batch.extend_start_loc = original_extend_start_loc
        forward_batch.override_req_to_token = original_override_req_to_token
        forward_batch.attn_backend.init_forward_metadata(forward_batch)

        return sequences, next_input_embeds


class Qwen3OmniMoeTalkerSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Qwen3 Talker MoE with the checkpoint's shared expert branch."""

    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        shared_size = getattr(config, "shared_expert_intermediate_size", 0)
        if shared_size <= 0:
            raise RuntimeError(
                "Qwen3-Omni Talker sparse MoE requires shared_expert_intermediate_size"
            )
        self.shared_expert = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_expert", prefix),
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def _forward_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(self.shared_expert_gate(hidden_states)) * self.shared_expert(
            hidden_states
        )

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        output = super().forward_normal(
            hidden_states,
            should_allreduce_fusion=should_allreduce_fusion,
            use_reduce_scatter=use_reduce_scatter,
        )
        return output + self._forward_shared_expert(hidden_states)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        output = super().forward_deepep(hidden_states, forward_batch)
        return output + self._forward_shared_expert(hidden_states)


class Qwen3OmniMoeTalkerDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_layer_sparse: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_layer_sparse=is_layer_sparse,
        )
        if self.is_layer_sparse:
            # Older fork used Qwen2MoeSparseMoeBlock (with built-in shared
            # expert handling and softmax→topk→renorm routing). HF talker
            # MoE matches that semantics. Use it directly instead of the
            # Qwen3 sparse block + custom shared add — that combination
            # produced ~half-magnitude MoE output (Run 13 finding).
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )


class Qwen3OmniMoeTalkerForConditionalGeneration(nn.Module):
    """Top-level Talker module with projections, codec_head, model, and code_predictor."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id_offset: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_id_offset = layer_id_offset
        text_config = config.text_config

        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(config)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(config)

        self.codec_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.model = Qwen3MoeModel(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=Qwen3OmniMoeTalkerDecoderLayer,
            layer_id_offset=layer_id_offset,
        )

        # Code predictor - offset by talker's num_layers
        code_predictor_offset = layer_id_offset + text_config.num_hidden_layers
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
            config=config.code_predictor_config,
            quant_config=quant_config,
            prefix=add_prefix("code_predictor", prefix),
            layer_id_offset=code_predictor_offset,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens


class Qwen3OmniMoeCausalConvNet(nn.Module):
    """Causal 1D convolution with left-only padding for streaming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def _use_triton_code2wav_conv1d(self, hidden_state: torch.Tensor) -> bool:
        if hidden_state.device.type != "cuda" or hidden_state.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return False
        if (
            self.stride != 1
            or self.conv.groups != 1
            or self.conv.in_channels != self.conv.out_channels
            or self.conv.kernel_size != (7,)
            or self.conv.bias is None
        ):
            return False

        batch_size, channels, seq_len = hidden_state.shape
        if channels == 96 and seq_len == 55125:
            return batch_size in (1, 2, 4, 5, 6, 7, 8)
        if channels == 192 and seq_len == 18376:
            return batch_size in (4, 5, 8)
        return False

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self._use_triton_code2wav_conv1d(hidden_state):
            return dense_causal_conv1d_kernel7(
                hidden_state,
                self.conv.weight,
                self.conv.bias,
                self.dilation,
            )
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    """Causal transposed 1D convolution for upsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = self.left_pad

    # Bench-tuned shapes for the phase-inner Triton transconv kernel.
    _PHASE_INNER_TRANSCONV_SHAPES = {
        (3, 192, 96, 18376),   # decoder.4.block.1
        (4, 384, 192, 4595),   # decoder.3.block.1
    }

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if (
            hidden_state.device.type == "cuda"
            and hidden_state.dtype == torch.bfloat16
            and hidden_state.is_contiguous()
            and self.conv.groups == 1
            and self.conv.bias is not None
            and self.conv.kernel_size == (2 * self.conv.stride[0],)
        ):
            stride_val = self.conv.stride[0]
            shape_key = (stride_val, self.conv.in_channels, self.conv.out_channels, hidden_state.shape[2])
            if shape_key in self._PHASE_INNER_TRANSCONV_SHAPES:
                return causal_transconv1d_phase_inner(
                    hidden_state,
                    self.conv.weight,
                    self.conv.bias,
                    stride_val,
                )
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with causal depthwise conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=7, groups=dim, dilation=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.permute(0, 2, 1)
        return residual + hidden_states


class Qwen3OmniMoeCode2WavRMSNorm(nn.Module):
    """RMSNorm for Code2Wav transformer layers."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.bfloat16
        ):
            return code2wav_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3OmniMoeCode2WavLayerScale(nn.Module):
    """Layer scale for residual connections."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class Qwen3OmniMoeCode2WavMlp(nn.Module):
    """MLP for Code2Wav transformer layers."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        # Lazy gate+up weight cat on first forward — keeps checkpoint loading
        # unchanged while the forward path does one Linear+silu_and_mul split.
        # AMD bench `/tmp/bench_swiglu_amd.py`: 1.1-1.4x.
        self._gate_up_weight_buf: Optional[torch.Tensor] = None

    def _merged_gate_up(self) -> torch.Tensor:
        if self._gate_up_weight_buf is None:
            self._gate_up_weight_buf = torch.cat(
                [self.gate_proj.weight, self.up_proj.weight], dim=0
            )
        return self._gate_up_weight_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = F.linear(x, self._merged_gate_up())  # [..., 2*intermediate]
        gate, up = gu.split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3OmniMoeCode2WavRotaryEmbedding(nn.Module):
    """Rotary positional embedding for Code2Wav transformer."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.rope_theta, _ = get_rope_config(config)
        self.head_dim = config.hidden_size // config.num_attention_heads
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[1]:
            raise ValueError(
                f"Code2Wav rotary seq_len={seq_len} exceeds "
                f"max_position_embeddings={self.cos_cached.shape[1]}"
            )
        return (
            self.cos_cached[:, :seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :seq_len].to(dtype=x.dtype, device=x.device),
        )


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to q and k."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3OmniMoeCode2WavAttention(nn.Module):
    """Sliding window causal attention for Code2Wav transformer."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self._mask_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        # Lazy Q/K/V weight cat on first forward — keeps checkpoint loading
        # unchanged while the forward path does one merged Linear. AMD bench
        # `/tmp/bench_qkv_merge_amd.py`: 1.9x. Code2Wav has num_kv_heads ==
        # num_heads (no GQA), so output dims are identical and split is even.
        self._qkv_weight_buf: Optional[torch.Tensor] = None
        self._qkv_bias_buf: Optional[torch.Tensor] = None

    def _merged_qkv(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._qkv_weight_buf is None:
            self._qkv_weight_buf = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            )
            if self.q_proj.bias is not None:
                self._qkv_bias_buf = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                )
        return self._qkv_weight_buf, self._qkv_bias_buf

    def _attend_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return an SDPA attn_mask (True = ATTEND)."""
        key = (device, seq_len)
        mask = self._mask_cache.get(key)
        if mask is None:
            blocked = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            if self.sliding_window is not None:
                window_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                    diagonal=-self.sliding_window,
                )
                blocked = blocked | ~window_mask
            mask = ~blocked
            self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv_w, qkv_b = self._merged_qkv()
        qkv = F.linear(hidden_states, qkv_w, qkv_b)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query_states = qkv[:, :, 0].transpose(1, 2)
        key_states = qkv[:, :, 1].transpose(1, 2)
        value_states = qkv[:, :, 2].transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # SDPA: 1.7-1.8x vs hand-rolled at Code2Wav shapes on AMD MI300X
        # (`/tmp/bench_sdpa_amd.py`).
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=self._attend_mask(seq_len, hidden_states.device),
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3OmniMoeCode2WavTransformerLayer(nn.Module):
    """Single transformer layer for Code2Wav."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3OmniMoeCode2WavAttention(config, layer_idx)
        self.mlp = Qwen3OmniMoeCode2WavMlp(config)
        self.input_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.mlp_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Qwen3OmniMoeCode2WavTransformerModel(nn.Module):
    """Transformer model for Code2Wav (pre_transformer)."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Qwen3OmniMoeCode2WavTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeCode2WavRotaryEmbedding(config)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = inputs_embeds.shape
        position_embeddings = self.rotary_emb(inputs_embeds, seq_len)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        return self.norm(hidden_states)


class SnakeBeta(nn.Module):
    """Snake activation with learnable alpha and beta parameters."""

    def __init__(self, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.bfloat16
        ):
            return snakebeta(
                hidden_states,
                self.alpha,
                self.beta,
                self.no_div_by_zero,
            )
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(hidden_states * alpha), 2)


class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    """Residual unit for Code2Wav decoder."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Qwen3OmniMoeCode2WavDecoderBlock(nn.Module):
    """Upsampling decoder block for Code2Wav."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        in_dim = config.decoder_dim // (2 ** layer_idx)
        out_dim = config.decoder_dim // (2 ** (layer_idx + 1))
        upsample_rate = config.upsample_rates[layer_idx]

        self.block = nn.ModuleList([
            SnakeBeta(in_dim),
            Qwen3OmniMoeCausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ])
        for dilation in (1, 3, 9):
            self.block.append(Qwen3OmniMoeCode2WavDecoderResidualUnit(out_dim, dilation))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden = block(hidden)
        return hidden


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Code2Wav: Converts codec codes to waveform.

    Input: codes tensor of shape [batch, num_quantizers, num_frames]
    Output: waveform tensor of shape [batch, 1, num_samples] in range [-1, 1]
    """

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.config = config
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))

        # Code embedding: maps each quantizer's codes to hidden_size
        self.code_embedding = nn.Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        # Offset buffer: shift codes for each quantizer into its own embedding range
        self.register_buffer(
            "code_offset",
            torch.arange(config.num_quantizers).view(1, -1, 1) * config.codebook_size,
            persistent=False,
        )

        # Pre-transformer: sliding window causal attention
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel(config)

        # Upsample blocks (after transformer, before decoder)
        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(nn.ModuleList([
                Qwen3OmniMoeCausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                Qwen3OmniMoeConvNeXtBlock(config.hidden_size),
            ]))
        self.upsample = nn.ModuleList(upsample)

        # Decoder: progressive upsampling to waveform
        decoder = [Qwen3OmniMoeCausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(config, i))
        output_dim = config.decoder_dim // (2 ** len(config.upsample_rates))
        decoder.extend([
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, 7),
        ])
        self.decoder = nn.ModuleList(decoder)
        self._captured_forward_graphs: dict[tuple[int, int, torch.device], dict] = {}

    def output_num_samples(self, num_frames: int) -> int:
        if num_frames < 0:
            raise ValueError(f"num_frames must be non-negative, got {num_frames}")
        if num_frames == 0:
            return 0
        num_samples = num_frames
        for factor in self.config.upsampling_ratios:
            num_samples *= factor
        for rate in self.config.upsample_rates:
            num_samples = num_samples * rate - rate
        return num_samples

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: codes -> waveform.

        Args:
            codes: [batch, num_quantizers, num_frames] integer codes

        Returns:
            wav: [batch, 1, num_samples] float waveform in [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} quantizers, got {codes.shape[1]}"
            )

        wav = self._decode_hidden(self._encode_codes(codes))

        return wav.clamp(min=-1, max=1)

    def forward_split_decoder(
        self,
        codes: torch.Tensor,
        max_decoder_batch_size: int,
    ) -> torch.Tensor:
        if max_decoder_batch_size <= 0:
            raise ValueError(
                f"max_decoder_batch_size must be positive, got {max_decoder_batch_size}"
            )
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} quantizers, got {codes.shape[1]}"
            )

        hidden = self._encode_codes(codes)
        wavs = [
            self._decode_hidden(hidden[start : start + max_decoder_batch_size])
            for start in range(0, hidden.shape[0], max_decoder_batch_size)
        ]
        return torch.cat(wavs, dim=0).clamp(min=-1, max=1)

    def _get_or_capture_forward_graph(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        max_decoder_batch_size: int,
    ) -> dict:
        if num_frames <= 0 or num_frames > _CODE2WAV_CAPTURE_MAX_FRAMES:
            raise RuntimeError(
                f"Qwen3-Omni Code2Wav capture does not support {num_frames} frames"
            )
        batch_bucket = _code2wav_capture_batch_bucket(batch_size)
        if batch_bucket is None:
            raise RuntimeError(
                f"Qwen3-Omni Code2Wav capture has no batch bucket for {batch_size}"
            )
        key = (batch_bucket, num_frames, device)
        cached = self._captured_forward_graphs.get(key)
        if cached is not None:
            return cached

        codes = torch.zeros(
            (batch_bucket, self.config.num_quantizers, num_frames),
            dtype=torch.long,
            device=device,
        )

        def run_once() -> torch.Tensor:
            return self.forward_split_decoder(
                codes,
                max_decoder_batch_size=max_decoder_batch_size,
            )

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                run_once()
        torch.cuda.current_stream(device=device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            wav = run_once()

        graph_state = {"codes": codes, "graph": graph, "wav": wav}
        self._captured_forward_graphs[key] = graph_state
        return graph_state

    def forward_split_decoder_captured(
        self,
        codes: torch.Tensor,
        max_decoder_batch_size: int,
    ) -> torch.Tensor:
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} quantizers, got {codes.shape[1]}"
            )
        graph_state = self._get_or_capture_forward_graph(
            codes.shape[0],
            codes.shape[2],
            codes.device,
            max_decoder_batch_size,
        )
        graph_codes = graph_state["codes"]
        with torch.inference_mode():
            graph_codes[: codes.shape[0]].copy_(codes)
            if graph_codes.shape[0] > codes.shape[0]:
                graph_codes[codes.shape[0] :].zero_()
            graph_state["graph"].replay()
            return graph_state["wav"][: codes.shape[0]].clone()

    def should_capture_split_decoder(
        self,
        batch_size: int,
        num_frames: int,
        steady_chunk_frames: int,
    ) -> bool:
        max_window_frames = _CODE2WAV_LEFT_CONTEXT_FRAMES + steady_chunk_frames
        return (
            num_frames == max_window_frames
            or batch_size <= _CODE2WAV_SHORT_WINDOW_CAPTURE_BATCH_BUCKETS[-1]
        )

    def prewarm_runtime_shapes(
        self,
        batch_sizes: Iterable[int],
        steady_chunk_frames: int,
        max_decoder_batch_size: int,
        stream: Optional[torch.cuda.Stream] = None,
        capture_graphs: bool = True,
    ) -> int:
        batch_sizes = tuple(batch_sizes)
        max_window_frames = _CODE2WAV_LEFT_CONTEXT_FRAMES + steady_chunk_frames
        if capture_graphs:
            max_batch = max(batch_sizes)
            capture_batch_sizes = [
                bucket
                for bucket in _CODE2WAV_CAPTURE_BATCH_BUCKETS
                if bucket < max_batch
            ]
            capture_bucket = _code2wav_capture_batch_bucket(max_batch)
            if capture_bucket is None:
                raise RuntimeError(
                    f"Qwen3-Omni Code2Wav capture has no bucket for {max_batch}"
                )
            capture_batch_sizes.append(capture_bucket)
            # The steady 29-frame replay is bucket-captured for throughput.
            # First-audio and flush windows are usually small request groups;
            # keep those graph-warmed too so live turns do not pay eager cost.
            prewarm_specs = sorted(
                {(bs, max_window_frames) for bs in capture_batch_sizes}
                | {
                    (bs, frames)
                    for bs in _CODE2WAV_SHORT_WINDOW_CAPTURE_BATCH_BUCKETS
                    for frames in range(1, max_window_frames)
                }
            )
        else:
            rolling_window_frames = {4, 8, 12, 16, 20, 24, 28, 29}
            if steady_chunk_frames > AUDIO_CODE2WAV_CHUNK_FRAMES:
                window_frames = AUDIO_CODE2WAV_CHUNK_FRAMES
                while window_frames < max_window_frames:
                    rolling_window_frames.add(window_frames)
                    window_frames += steady_chunk_frames
                rolling_window_frames.add(max_window_frames)
            flush_window_frames = range(1, max_window_frames)
            prewarm_specs = sorted(
                {(bs, frames) for bs in batch_sizes for frames in rolling_window_frames}
                | {(1, frames) for frames in flush_window_frames}
            )

        codebook = self.config.codebook_size
        num_q = self.config.num_quantizers
        device = next(self.parameters()).device

        def run_specs() -> None:
            with torch.inference_mode():
                for bs, dframes in prewarm_specs:
                    codes = torch.randint(
                        0,
                        codebook,
                        (bs, num_q, dframes),
                        dtype=torch.long,
                        device=device,
                    )
                    if capture_graphs:
                        self.forward_split_decoder_captured(
                            codes, max_decoder_batch_size=max_decoder_batch_size
                        )
                    else:
                        self.forward_split_decoder(
                            codes, max_decoder_batch_size=max_decoder_batch_size
                        )

        if stream is None:
            run_specs()
            torch.cuda.synchronize()
        else:
            with torch.cuda.stream(stream):
                run_specs()
            stream.synchronize()
        return len(prewarm_specs)

    def _encode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        hidden = self.pre_transformer(hidden)
        # Keep decoder strides independent of the outer Code2Wav batch size.
        return hidden.permute(0, 2, 1).contiguous()

    def _decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        # Keep MI300X Code2Wav convs on validated small batches while sharing
        # the embedding/pre-transformer work across the full ready audio group.
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """
        Decode codes in chunks with left context overlap for streaming.

        Args:
            codes: [batch, num_quantizers, num_frames]
            chunk_size: number of frames per chunk
            left_context_size: number of frames of left context for overlap

        Returns:
            wav: [batch, 1, num_samples] concatenated waveform
        """
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            # Crop away the left context samples
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    def decode_streaming_chunk(
        self,
        codes: torch.Tensor,
        last_decoded_frame: int,
        left_context_size: int = 25,
    ) -> tuple[torch.Tensor, int]:
        """
        Decode a single streaming chunk from accumulated codes.

        Args:
            codes: [batch, num_quantizers, total_frames_so_far]
            last_decoded_frame: index of last frame that was decoded
            left_context_size: number of frames of left context

        Returns:
            (pcm_chunk, new_last_decoded_frame)
        """
        current_frame = codes.shape[-1]
        if current_frame <= last_decoded_frame:
            return torch.tensor([], device=codes.device), last_decoded_frame

        context_start = max(0, last_decoded_frame - left_context_size)
        context_size = last_decoded_frame - context_start

        codes_chunk = codes[..., context_start:current_frame]
        wav_chunk = self(codes_chunk)

        # Crop away the context samples
        pcm_chunk = wav_chunk[..., context_size * self.total_upsample :]
        return pcm_chunk, current_frame


class Qwen3OmniMoeForConditionalGeneration(PreTrainedModel):
    def __init__(
        self,
        config: Qwen3OmniMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.config = config

        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(
            config.thinker_config,
            quant_config=quant_config,
            prefix=prefix,
            accept_hidden_layer=config.talker_config.accept_hidden_layer,
        )

        thinker_num_layers = config.thinker_config.text_config.num_hidden_layers
        talker_quant_config = quant_config
        if quant_config is not None and quant_config.__class__.__name__ in (
            "GPTQConfig",
            "GPTQMarlinConfig",
        ):
            talker_quant_config = None
        self._talker_quant_config = talker_quant_config
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration(
            config=config.talker_config,
            quant_config=talker_quant_config,
            prefix="talker",
            layer_id_offset=thinker_num_layers,
        )

        # Code2Wav: codec codes to waveform decoder
        self.code2wav = Qwen3OmniMoeCode2Wav(config.code2wav_config)

        self.pad_input_ids = self.thinker.pad_input_ids
        self.accept_hidden_layer = config.talker_config.accept_hidden_layer
        self._talker_decode_req_to_token_buffer = None
        self._talker_decode_graphs = {}
        self._talker_prefill_graphs = {}
        self._thinker_prefill_graphs = {}
        self._talker_prefill_special_embed_cache = {}
        self.register_buffer(
            "_voice_suppressed_thinker_token_ids",
            torch.tensor(_VOICE_SUPPRESSED_THINKER_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )

    @property
    def persistent_kv_reservation_size(self) -> int:
        """KV slots held permanently by submodels — read by sglang's scheduler
        leak checker (see scheduler_runtime_checker_mixin)."""
        return (
            self.talker.code_predictor.persistent_kv_reservation_size
        )

    @staticmethod
    def _stabilize_thinker_logits(
        thinker_logits: LogitsProcessorOutput,
    ) -> LogitsProcessorOutput:
        if thinker_logits.next_token_logits is not None:
            thinker_logits.next_token_logits = thinker_logits.next_token_logits.clone()
        return thinker_logits

    def _suppress_voice_control_logits(
        self,
        thinker_logits: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        logits = thinker_logits.next_token_logits
        model_states = forward_batch.model_specific_states or {}
        voice_rows = model_states.get("qwen3_omni_voice_batch")
        if logits is None or voice_rows is None:
            return thinker_logits
        if logits.dim() != 2:
            raise RuntimeError("Qwen3-Omni voice Thinker logits must be rank 2")
        if voice_rows.numel() != logits.shape[0]:
            raise RuntimeError("Qwen3-Omni voice mask batch size mismatch")
        row_ids = voice_rows.to(device=logits.device).nonzero(as_tuple=True)[0]
        if row_ids.numel() == 0:
            return thinker_logits
        # Voice turns append the assistant ChatML header at EOI. The Thinker
        # output is assistant content only; fresh ChatML, audio placeholders,
        # or thinking markers poison parent-rid history and can make the next
        # Talker prefill parse an impossible segment such as
        # <|im_start|><think>. Suppress those few columns directly instead of
        # using generic logit_bias, which allocates a dense [batch, vocab]
        # tensor for every voice request.
        token_ids = self._voice_suppressed_thinker_token_ids.to(device=logits.device)
        logits[row_ids[:, None], token_ids[None, :]] = torch.finfo(logits.dtype).min
        return thinker_logits

    def _talker_prefill_special_embeds(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        speaker: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        talker_config = self.config.talker_config
        speaker_id = _qwen3_omni_speaker_id(talker_config, speaker)
        cache_key = (device.type, device.index, dtype, speaker_id)
        cached = self._talker_prefill_special_embed_cache.get(cache_key)
        if cached is not None:
            return cached

        tts_special_tokens = torch.tensor(
            [self.config.tts_bos_token_id, self.config.tts_pad_token_id],
            device=device,
            dtype=torch.long,
        )
        tts_projected = self.talker.text_projection(
            self.thinker.model.embed_tokens(tts_special_tokens)
        )
        codec_special_tokens = torch.tensor(
            [
                talker_config.codec_nothink_id,
                talker_config.codec_think_bos_id,
                talker_config.codec_think_eos_id,
                speaker_id,
                talker_config.codec_pad_id,
                talker_config.codec_bos_id,
            ],
            device=device,
            dtype=torch.long,
        )
        codec_embeds = self.talker.model.embed_tokens(codec_special_tokens)
        assistant_codec_hidden = torch.cat(
            [
                torch.zeros(
                    3,
                    self.config.talker_config.text_config.hidden_size,
                    device=device,
                    dtype=dtype,
                ),
                codec_embeds,
            ],
            dim=0,
        )
        cached = (tts_projected[0:1], tts_projected[1:2], assistant_codec_hidden)
        self._talker_prefill_special_embed_cache[cache_key] = cached
        return cached

    def _thinker_prefill_graph_attn_backend(self, forward_batch: ForwardBatch):
        return getattr(
            forward_batch.attn_backend,
            "prefill_backend",
            forward_batch.attn_backend,
        )

    def _should_use_captured_thinker_prefill(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> bool:
        if (
            input_embeds is None
            or not input_embeds.is_cuda
            or forward_batch.batch_size not in _THINKER_PREFILL_CAPTURE_BATCH_SIZES
            or input_ids.numel() > _THINKER_PREFILL_CAPTURE_MAX_TOKENS
            or forward_batch.extend_prefix_lens is None
            or forward_batch.extend_seq_lens is None
            or forward_batch.extend_start_loc is None
            or forward_batch.mrope_positions is None
            or not _attention_backend_uses_aiter(forward_batch.attn_backend)
        ):
            return False
        if not any(forward_batch.extend_prefix_lens_cpu):
            return False
        max_seq_len = int(forward_batch.seq_lens.max().item())
        return _thinker_prefill_kv_bucket(max_seq_len) is not None

    def _should_embed_text_thinker_prefill_for_capture(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> bool:
        if (
            input_embeds is not None
            or input_ids is None
            or forward_batch.contains_mm_inputs()
            or forward_batch.batch_size not in _THINKER_PREFILL_CAPTURE_BATCH_SIZES
            or input_ids.numel() > _THINKER_PREFILL_CAPTURE_MAX_TOKENS
            or forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.seq_lens is None
            or not _attention_backend_uses_aiter(forward_batch.attn_backend)
        ):
            return False
        if not any(forward_batch.extend_prefix_lens_cpu):
            return False
        max_seq_len = int(forward_batch.seq_lens.max().item())
        return _thinker_prefill_kv_bucket(max_seq_len) is not None

    def _copy_thinker_prefill_graph_inputs(
        self,
        graph_state: dict,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> None:
        graph_state["input_ids"].copy_(input_ids)
        graph_state["positions"].copy_(positions)
        graph_state["input_embeds"].copy_(input_embeds)
        graph_state["mrope_positions"].copy_(forward_batch.mrope_positions)
        graph_state["req_pool_indices"].copy_(forward_batch.req_pool_indices)
        graph_state["seq_lens"].copy_(forward_batch.seq_lens)
        graph_state["out_cache_loc"].copy_(forward_batch.out_cache_loc)
        graph_state["extend_prefix_lens"].copy_(forward_batch.extend_prefix_lens)
        graph_state["extend_seq_lens"].copy_(forward_batch.extend_seq_lens)
        graph_state["extend_start_loc"].copy_(forward_batch.extend_start_loc)

    def _build_thinker_prefill_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        graph_state: dict,
    ) -> None:
        from sglang.srt.layers.attention.aiter_backend import (
            ForwardMetadata as AiterForwardMetadata,
        )

        batch_size = graph_state["batch_size"]
        graph_state["kv_indptr"][0].zero_()
        torch.cumsum(
            graph_state["seq_lens"],
            dim=0,
            out=graph_state["kv_indptr"][1 : batch_size + 1],
        )
        graph_state["kv_indices"][:] = graph_state["out_cache_loc"][0].to(torch.int32)
        create_flashinfer_kv_indices_triton[(batch_size,)](
            graph_state["req_to_token"],
            graph_state["req_pool_indices"],
            graph_state["seq_lens"],
            graph_state["kv_indptr"],
            None,
            graph_state["kv_indices"],
            graph_state["req_to_token"].stride(0),
        )

        graph_state["qo_indptr"][0].zero_()
        torch.cumsum(
            graph_state["extend_seq_lens"],
            dim=0,
            out=graph_state["qo_indptr"][1 : batch_size + 1],
        )

        aiter_backend = self._thinker_prefill_graph_attn_backend(forward_batch)
        aiter_backend.forward_metadata = AiterForwardMetadata(
            kv_indptr=graph_state["kv_indptr"],
            kv_indices=graph_state["kv_indices"],
            qo_indptr=graph_state["qo_indptr"],
            kv_last_page_len=None,
            max_q_len=graph_state["max_q_len"],
            max_kv_len=graph_state["kv_bucket"],
            run_graph=False,
        )

    def _get_or_capture_thinker_prefill_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> dict:
        batch_size = forward_batch.batch_size
        num_tokens = input_ids.numel()
        max_q_len = int(forward_batch.extend_seq_lens.max().item())
        kv_bucket = _thinker_prefill_kv_bucket(int(forward_batch.seq_lens.max().item()))
        if kv_bucket is None:
            raise RuntimeError("Qwen3-Omni Thinker prefill capture has no KV bucket")

        key = (
            batch_size,
            num_tokens,
            max_q_len,
            kv_bucket,
            input_embeds.dtype,
            input_embeds.device,
        )
        cached = self._thinker_prefill_graphs.get(key)
        if cached is not None:
            return cached

        graph_state = {
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "max_q_len": max_q_len,
            "kv_bucket": kv_bucket,
            "input_ids": torch.empty_like(input_ids),
            "positions": torch.empty_like(positions),
            "input_embeds": torch.empty_like(input_embeds),
            "mrope_positions": torch.empty_like(forward_batch.mrope_positions),
            "req_pool_indices": torch.empty_like(forward_batch.req_pool_indices),
            "seq_lens": torch.empty_like(forward_batch.seq_lens),
            "out_cache_loc": torch.empty_like(forward_batch.out_cache_loc),
            "extend_prefix_lens": torch.empty_like(forward_batch.extend_prefix_lens),
            "extend_seq_lens": torch.empty_like(forward_batch.extend_seq_lens),
            "extend_start_loc": torch.empty_like(forward_batch.extend_start_loc),
            "req_to_token": forward_batch.req_to_token_pool.req_to_token,
            "kv_indptr": torch.empty(
                (batch_size + 1,), dtype=torch.int32, device=input_embeds.device
            ),
            "qo_indptr": torch.empty(
                (batch_size + 1,), dtype=torch.int32, device=input_embeds.device
            ),
            "kv_indices": torch.empty(
                (batch_size * kv_bucket + 256,),
                dtype=torch.int32,
                device=input_embeds.device,
            ),
        }
        self._copy_thinker_prefill_graph_inputs(
            graph_state, input_ids, positions, forward_batch, input_embeds
        )

        graph_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=graph_state["input_ids"],
            req_pool_indices=graph_state["req_pool_indices"],
            seq_lens=graph_state["seq_lens"],
            out_cache_loc=graph_state["out_cache_loc"],
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            req_to_token_pool=forward_batch.req_to_token_pool,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            attn_backend=forward_batch.attn_backend,
            extend_num_tokens=num_tokens,
            extend_seq_lens=graph_state["extend_seq_lens"],
            extend_prefix_lens=graph_state["extend_prefix_lens"],
            extend_start_loc=graph_state["extend_start_loc"],
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            positions=graph_state["positions"],
            mrope_positions=graph_state["mrope_positions"],
            qwen3_omni_token_to_kv_pool_allocators=(
                forward_batch.qwen3_omni_token_to_kv_pool_allocators
            ),
            global_forward_mode=ForwardMode.EXTEND,
        )

        def run_once():
            # Keep Qwen3 Omni semantics outside the graph. This replay only
            # covers the Thinker transformer body; logits/sampling and Talker
            # prefill payload construction stay on the existing eager path.
            self._build_thinker_prefill_graph_metadata(
                graph_forward_batch, graph_state
            )
            return self.thinker.model(
                input_ids=graph_state["input_ids"],
                positions=graph_state["positions"],
                forward_batch=graph_forward_batch,
                input_embeds=graph_state["input_embeds"],
            )

        stream = torch.cuda.Stream(device=input_embeds.device)
        stream.wait_stream(torch.cuda.current_stream(device=input_embeds.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                run_once()
        torch.cuda.current_stream(device=input_embeds.device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run_once()

        graph_state["graph"] = graph
        if not isinstance(outputs, tuple):
            raise RuntimeError("Qwen3-Omni Thinker prefill capture lost aux hidden states")
        graph_state["hidden"] = outputs[0]
        graph_state["aux_hidden"] = outputs[1]
        self._thinker_prefill_graphs[key] = graph_state
        return graph_state

    def _run_captured_thinker_prefill_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ):
        graph_state = self._get_or_capture_thinker_prefill_graph(
            input_ids, positions, forward_batch, input_embeds
        )
        self._copy_thinker_prefill_graph_inputs(
            graph_state, input_ids, positions, forward_batch, input_embeds
        )
        graph_state["graph"].replay()
        return (
            graph_state["hidden"][: input_ids.numel()].clone(),
            [
                hidden[: input_ids.numel()].clone()
                for hidden in graph_state["aux_hidden"]
            ],
        )

    def _should_use_captured_talker_prefill(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> bool:
        if (
            input_embeds is None
            or not input_embeds.is_cuda
            or not positions.is_cuda
            or forward_batch.batch_size not in _TALKER_PREFILL_CAPTURE_BATCH_SIZES
            or input_embeds.shape[0] > _TALKER_PREFILL_CAPTURE_MAX_TOKENS
            or forward_batch.extend_prefix_lens is None
            or forward_batch.extend_seq_lens is None
            or forward_batch.extend_start_loc is None
            or not _attention_backend_uses_aiter(forward_batch.attn_backend)
        ):
            return False
        if not any(forward_batch.extend_prefix_lens_cpu):
            return False
        max_q_len = int(forward_batch.extend_seq_lens.max().item())
        if max_q_len > _TALKER_PREFILL_CAPTURE_MAX_Q_LEN:
            return False
        return (
            _thinker_prefill_kv_bucket(int(forward_batch.seq_lens.max().item()))
            is not None
        )

    def _copy_talker_prefill_graph_inputs(
        self,
        graph_state: dict,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> None:
        graph_state["positions"].copy_(positions)
        graph_state["input_embeds"].copy_(input_embeds)
        graph_state["req_pool_indices"].copy_(forward_batch.req_pool_indices)
        graph_state["seq_lens"].copy_(forward_batch.seq_lens)
        graph_state["out_cache_loc"].copy_(forward_batch.out_cache_loc)
        graph_state["extend_prefix_lens"].copy_(forward_batch.extend_prefix_lens)
        graph_state["extend_seq_lens"].copy_(forward_batch.extend_seq_lens)
        graph_state["extend_start_loc"].copy_(forward_batch.extend_start_loc)

    def _build_talker_prefill_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        graph_state: dict,
    ) -> None:
        from sglang.srt.layers.attention.aiter_backend import (
            ForwardMetadata as AiterForwardMetadata,
        )

        batch_size = graph_state["batch_size"]
        graph_state["kv_indptr"][0].zero_()
        torch.cumsum(
            graph_state["seq_lens"],
            dim=0,
            out=graph_state["kv_indptr"][1 : batch_size + 1],
        )
        graph_state["kv_indices"][:] = graph_state["out_cache_loc"][0].to(torch.int32)
        create_flashinfer_kv_indices_triton[(batch_size,)](
            graph_state["req_to_token"],
            graph_state["req_pool_indices"],
            graph_state["seq_lens"],
            graph_state["kv_indptr"],
            None,
            graph_state["kv_indices"],
            graph_state["req_to_token"].stride(0),
        )

        graph_state["qo_indptr"][0].zero_()
        torch.cumsum(
            graph_state["extend_seq_lens"],
            dim=0,
            out=graph_state["qo_indptr"][1 : batch_size + 1],
        )

        aiter_backend = self._thinker_prefill_graph_attn_backend(forward_batch)
        aiter_backend.forward_metadata = AiterForwardMetadata(
            kv_indptr=graph_state["kv_indptr"],
            kv_indices=graph_state["kv_indices"],
            qo_indptr=graph_state["qo_indptr"],
            kv_last_page_len=None,
            max_q_len=graph_state["max_q_len"],
            max_kv_len=graph_state["kv_bucket"],
            run_graph=False,
        )

    def _get_or_capture_talker_prefill_graph(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> dict:
        batch_size = forward_batch.batch_size
        num_tokens = input_embeds.shape[0]
        max_q_len = int(forward_batch.extend_seq_lens.max().item())
        if _thinker_prefill_kv_bucket(int(forward_batch.seq_lens.max().item())) is None:
            raise RuntimeError("Qwen3-Omni Talker prefill capture has no KV bucket")
        # Use one large KV metadata bucket so a graph captured for a short
        # prefix can replay as the streaming session grows into later turns.
        kv_bucket = _TALKER_PREFILL_CAPTURE_KV_BUCKET

        key = (
            batch_size,
            num_tokens,
            max_q_len,
            kv_bucket,
            input_embeds.dtype,
            input_embeds.device,
        )
        cached = self._talker_prefill_graphs.get(key)
        if cached is not None:
            return cached

        graph_state = {
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "max_q_len": max_q_len,
            "kv_bucket": kv_bucket,
            "input_ids": torch.zeros(
                (num_tokens,), dtype=torch.long, device=input_embeds.device
            ),
            "positions": torch.empty_like(positions),
            "input_embeds": torch.empty_like(input_embeds),
            "req_pool_indices": torch.empty_like(forward_batch.req_pool_indices),
            "seq_lens": torch.empty_like(forward_batch.seq_lens),
            "out_cache_loc": torch.empty_like(forward_batch.out_cache_loc),
            "extend_prefix_lens": torch.empty_like(forward_batch.extend_prefix_lens),
            "extend_seq_lens": torch.empty_like(forward_batch.extend_seq_lens),
            "extend_start_loc": torch.empty_like(forward_batch.extend_start_loc),
            "req_to_token": forward_batch.req_to_token_pool.req_to_token,
            "kv_indptr": torch.empty(
                (batch_size + 1,), dtype=torch.int32, device=input_embeds.device
            ),
            "qo_indptr": torch.empty(
                (batch_size + 1,), dtype=torch.int32, device=input_embeds.device
            ),
            "kv_indices": torch.empty(
                (batch_size * kv_bucket + 256,),
                dtype=torch.int32,
                device=input_embeds.device,
            ),
        }
        self._copy_talker_prefill_graph_inputs(
            graph_state, positions, forward_batch, input_embeds
        )

        graph_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=graph_state["input_ids"],
            req_pool_indices=graph_state["req_pool_indices"],
            seq_lens=graph_state["seq_lens"],
            out_cache_loc=graph_state["out_cache_loc"],
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            req_to_token_pool=forward_batch.req_to_token_pool,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            attn_backend=forward_batch.attn_backend,
            extend_num_tokens=num_tokens,
            extend_seq_lens=graph_state["extend_seq_lens"],
            extend_prefix_lens=graph_state["extend_prefix_lens"],
            extend_start_loc=graph_state["extend_start_loc"],
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            positions=graph_state["positions"],
            qwen3_omni_token_to_kv_pool_allocators=(
                forward_batch.qwen3_omni_token_to_kv_pool_allocators
            ),
            global_forward_mode=ForwardMode.EXTEND,
        )

        def run_once():
            self._build_talker_prefill_graph_metadata(
                graph_forward_batch, graph_state
            )
            return self.talker.model(
                input_ids=None,
                positions=graph_state["positions"],
                forward_batch=graph_forward_batch,
                input_embeds=graph_state["input_embeds"],
            )

        stream = torch.cuda.Stream(device=input_embeds.device)
        stream.wait_stream(torch.cuda.current_stream(device=input_embeds.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                run_once()
        torch.cuda.current_stream(device=input_embeds.device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden = run_once()

        graph_state["graph"] = graph
        graph_state["hidden"] = hidden
        self._talker_prefill_graphs[key] = graph_state
        return graph_state

    def _run_captured_talker_prefill_model(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        graph_state = self._get_or_capture_talker_prefill_graph(
            positions, forward_batch, input_embeds
        )
        self._copy_talker_prefill_graph_inputs(
            graph_state, positions, forward_batch, input_embeds
        )
        graph_state["graph"].replay()
        return graph_state["hidden"][: input_embeds.shape[0]].clone()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Union[torch.Tensor, ThinkerTalkerOutput]:
        """
        Thinker forward pass for Qwen3-Omni.

        Talker/MTP audio work runs from run_qwen3_omni_audio_phase_after_sample()
        on a separate phase queue dispatched by the scheduler after sampling.
        """
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        is_prefill = forward_batch.forward_mode.is_extend()

        if is_prefill:
            return self._forward_prefill(input_ids, positions, forward_batch, input_embeds)
        if get_is_capture_mode() and forward_batch.forward_mode.is_decode():
            # CUDA graph capture only records the thinker path; talker runs after replay.
            thinker_result = self.thinker.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
            )
            if isinstance(thinker_result, tuple):
                thinker_hidden, _ = thinker_result
            else:
                thinker_hidden = thinker_result
            return self._stabilize_thinker_logits(
                self.thinker.logits_processor(
                    input_ids, thinker_hidden, self.thinker.lm_head, forward_batch
                )
            )

        if forward_batch.forward_mode.is_decode():
            model_states = forward_batch.model_specific_states or {}
            thinker_done = model_states.get("thinker_done")
            if thinker_done is not None and thinker_done.all():
                return ThinkerTalkerOutput(thinker_logits=None)

        return self._forward_thinker_decode(input_ids, positions, forward_batch)

    def _forward_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> ThinkerTalkerOutput:
        """Prefill both thinker and talker in parallel."""
        # 0. Process multimodal inputs (audio/image) if present
        # This computes embeddings and clamps input_ids to valid vocab range
        if input_embeds is None and forward_batch.contains_mm_inputs():
            mm_inputs_list = [
                mm_input
                for mm_input in forward_batch.mm_inputs
                if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            # Debug: log multimodal processing
            input_embeds, _ = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                multimodal_model=self.thinker,
                input_embedding=self.thinker.model.embed_tokens,
            )
            forward_batch.mm_inputs = None
        elif self._should_embed_text_thinker_prefill_for_capture(
            input_ids, forward_batch, input_embeds
        ):
            # The captured Thinker prefill graph starts after token embedding.
            # Text-only cached-prefix extends normally arrive with
            # input_embeds=None, so embedding the tiny suffix here lets the hot
            # late-session path reuse the same graph as audio-input prefills
            # without changing model semantics.
            input_embeds = self.thinker.model.embed_tokens(input_ids)

        # 1. Run thinker forward - model returns (hidden_states, aux_hidden_states)
        #    because layers_to_capture is set
        if self._should_use_captured_thinker_prefill(
            input_ids, forward_batch, input_embeds
        ):
            thinker_result = self._run_captured_thinker_prefill_model(
                input_ids, positions, forward_batch, input_embeds
            )
        else:
            thinker_result = self.thinker.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        # Unpack result - with layers_to_capture, returns tuple
        if isinstance(thinker_result, tuple):
            thinker_hidden, aux_hidden_states = thinker_result
            accept_hidden = aux_hidden_states[0]  # accept_hidden_layer (e.g., layer 24) hidden states
        else:
            thinker_hidden = thinker_result
            accept_hidden = None

        prompt_accept_hidden_states = None
        if accept_hidden is not None:
            prompt_accept_hidden_states = []
            model_states = forward_batch.model_specific_states or {}
            previous_prompt_accept_hidden_states = (
                model_states.get("prompt_accept_hidden_states")
            )
            origin_input_ids = model_states.get("origin_input_ids")
            starts = forward_batch.extend_start_loc.tolist()
            lens = forward_batch.extend_seq_lens_cpu
            prefix_lens = forward_batch.extend_prefix_lens_cpu
            for req_idx, (start, seq_len) in enumerate(zip(starts, lens, strict=True)):
                current = accept_hidden[start : start + seq_len].detach()
                previous = (
                    previous_prompt_accept_hidden_states[req_idx]
                    if previous_prompt_accept_hidden_states is not None
                    else None
                )
                if previous is not None:
                    current = torch.cat([previous, current], dim=0)
                elif prefix_lens[req_idx] > 0:
                    if origin_input_ids is None:
                        raise RuntimeError(
                            "Qwen3-Omni prefill with prefix cache requires "
                            "origin_input_ids to align prompt accept-hidden states"
                        )
                    cached_input_ids = origin_input_ids[req_idx][: prefix_lens[req_idx]]
                    if self._qwen3_omni_multimodal_mask(cached_input_ids).any():
                        raise RuntimeError(
                            "Qwen3-Omni prompt accept-hidden states are missing "
                            "for cached multimodal prompt tokens"
                        )
                    prefix_pad = current.new_zeros(
                        prefix_lens[req_idx], current.shape[-1]
                    )
                    current = torch.cat([prefix_pad, current], dim=0)
                prompt_accept_hidden_states.append(current)

        # 2. Get thinker logits
        thinker_logits = self._stabilize_thinker_logits(
            self.thinker.logits_processor(
                input_ids, thinker_hidden, self.thinker.lm_head, forward_batch
            )
        )
        thinker_logits = self._suppress_voice_control_logits(
            thinker_logits, forward_batch
        )

        # 3. Compute tts_pad_embed for later use (when thinker finishes)
        # tts_pad is a TEXT token - embed with thinker, then project through talker
        tts_pad_token_id = self.config.tts_pad_token_id
        tts_pad_text_embed = self.thinker.model.embed_tokens(
            torch.tensor([tts_pad_token_id], device=input_ids.device)
        )
        tts_pad_embed = self.talker.text_projection(tts_pad_text_embed).squeeze(0)  # [hidden_dim]

        # 4. With 1-step delay: DON'T run talker prefill here
        # Talker prefill will run at first decode step when we have actual sampled token
        # Set talker_needs_prefill=True to signal scheduler
        return ThinkerTalkerOutput(
            thinker_logits=thinker_logits,
            tts_pad_embed=tts_pad_embed,
            talker_needs_prefill=True,  # Signal to run talker prefill at first decode
            prompt_accept_hidden_states=prompt_accept_hidden_states,
        )

    def _forward_thinker_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> ThinkerTalkerOutput:
        thinker_result = self.thinker.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )
        if isinstance(thinker_result, tuple):
            thinker_hidden, _ = thinker_result
        else:
            thinker_hidden = thinker_result
        thinker_logits = self._stabilize_thinker_logits(
            self.thinker.logits_processor(
                input_ids, thinker_hidden, self.thinker.lm_head, forward_batch
            )
        )
        thinker_logits = self._suppress_voice_control_logits(
            thinker_logits, forward_batch
        )
        return ThinkerTalkerOutput(thinker_logits=thinker_logits)

    def _save_forward_batch_state(
        self,
        forward_batch: ForwardBatch,
        save_extend: bool = False,
        save_req_to_tokens: bool = True,
    ) -> dict:
        """Save forward_batch state for later restoration after talker forward."""
        saved_req_to_tokens = []
        if save_req_to_tokens:
            saved_req_to_tokens = [
                forward_batch.req_to_token_pool.req_to_token[
                    req_idx.item(), : forward_batch.seq_lens[i].item()
                ].clone()
                for i, req_idx in enumerate(forward_batch.req_pool_indices)
            ]
        state = {
            "batch_size": forward_batch.batch_size,
            "req_pool_indices": forward_batch.req_pool_indices.clone(),
            "saved_req_to_tokens": saved_req_to_tokens,
            "seq_lens": forward_batch.seq_lens,
            "seq_lens_cpu": forward_batch.seq_lens_cpu,
            "seq_lens_sum": forward_batch.seq_lens_sum,
            "out_cache_loc": forward_batch.out_cache_loc,
            "override_req_to_token": forward_batch.override_req_to_token,
        }

        if save_extend:
            state.update({
                "forward_mode": forward_batch.forward_mode,
                "extend_prefix_lens_cpu": forward_batch.extend_prefix_lens_cpu,
                "extend_seq_lens_cpu": forward_batch.extend_seq_lens_cpu,
                "extend_prefix_lens": forward_batch.extend_prefix_lens,
                "extend_seq_lens": forward_batch.extend_seq_lens,
                "extend_num_tokens": forward_batch.extend_num_tokens,
                "extend_start_loc": forward_batch.extend_start_loc,
            })

        return state

    @contextmanager
    def _temporary_audio_req_pool_indices(
        self, forward_batch: ForwardBatch, batch_size: int
    ) -> torch.Tensor:
        pool = forward_batch.req_to_token_pool
        if batch_size > len(pool.free_slots):
            raise RuntimeError(
                "Qwen3-Omni audio phase needs temporary request-token rows: "
                f"requested {batch_size}, available {len(pool.free_slots)}"
            )
        slots = pool.free_slots[:batch_size]
        del pool.free_slots[:batch_size]
        indices = torch.tensor(
            slots,
            dtype=forward_batch.req_pool_indices.dtype,
            device=forward_batch.req_pool_indices.device,
        )
        try:
            yield indices
        finally:
            pool.free_slots[:0] = slots

    def _get_talker_decode_req_to_token_buffer(
        self, forward_batch: ForwardBatch, batch_size: int, max_len: int
    ) -> torch.Tensor:
        buffer = self._talker_decode_req_to_token_buffer
        needs_new_buffer = (
            buffer is None
            or buffer.device != forward_batch.seq_lens.device
            or buffer.shape[0] < batch_size
            or buffer.shape[1] < max_len
        )
        if needs_new_buffer:
            buffer = torch.empty(
                batch_size,
                max_len,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            self._talker_decode_req_to_token_buffer = buffer
        return buffer[:batch_size, :max_len]

    def _restore_forward_batch_state(
        self,
        forward_batch: ForwardBatch,
        saved_state: dict,
        restore_attention_metadata: bool = True,
    ) -> None:
        """Restore forward_batch state after talker forward."""
        for req_idx, saved_tokens in zip(
            saved_state["req_pool_indices"], saved_state["saved_req_to_tokens"]
        ):
            forward_batch.req_to_token_pool.req_to_token[
                req_idx.item(), : len(saved_tokens)
            ] = saved_tokens

        forward_batch.override_req_to_token = saved_state["override_req_to_token"]
        forward_batch.batch_size = saved_state["batch_size"]
        forward_batch.req_pool_indices = saved_state["req_pool_indices"]
        forward_batch.seq_lens = saved_state["seq_lens"]
        forward_batch.seq_lens_cpu = saved_state["seq_lens_cpu"]
        forward_batch.seq_lens_sum = saved_state["seq_lens_sum"]
        forward_batch.out_cache_loc = saved_state["out_cache_loc"]

        if "forward_mode" in saved_state:
            forward_batch.forward_mode = saved_state["forward_mode"]
            forward_batch.extend_prefix_lens_cpu = saved_state["extend_prefix_lens_cpu"]
            forward_batch.extend_seq_lens_cpu = saved_state["extend_seq_lens_cpu"]
            forward_batch.extend_prefix_lens = saved_state["extend_prefix_lens"]
            forward_batch.extend_seq_lens = saved_state["extend_seq_lens"]
            forward_batch.extend_num_tokens = saved_state["extend_num_tokens"]
            forward_batch.extend_start_loc = saved_state["extend_start_loc"]

        if restore_attention_metadata:
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

    def _run_talker_model_body(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.talker.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

    def _run_talker_model_and_head_body(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self._run_talker_model_body(
            positions,
            forward_batch,
            input_embeds,
        )
        logits = self.talker.codec_head(hidden_states)
        return hidden_states, logits

    def _run_talker_model_head_and_sample_body(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        codec_eos_token_id: int,
        allow_codec_eos: torch.Tensor,
        temperature: torch.Tensor,
        top_k: int,
        top_p: float,
        repetition_penalty: torch.Tensor,
        repetition_mask: torch.Tensor,
        sampler_seed: torch.Tensor,
        sampler_step: torch.Tensor,
        sampler_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, logits = self._run_talker_model_and_head_body(
            positions,
            forward_batch,
            input_embeds,
        )
        logits = _mask_codec_eos(
            logits,
            codec_eos_token_id,
            allow_codec_eos,
        )
        codec_tokens = _sample_codec_tokens(
            logits,
            codec_eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_mask=repetition_mask,
            sampler_seed=sampler_seed,
            sampler_step=sampler_step,
            sampler_offsets=sampler_offsets,
        )
        updated_repetition_mask = _update_codec_repetition_mask(
            repetition_mask,
            codec_tokens,
        )
        return hidden_states, codec_tokens, updated_repetition_mask

    def _talker_decode_bucket_seq_len(
        self, forward_batch: ForwardBatch
    ) -> Optional[int]:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
            return None
        return _talker_decode_seq_len_bucket(int(seq_lens_cpu.max().item()))

    def _talker_decode_uniform_scalar(self, value, field: str, batch_size: int, cast):
        if isinstance(value, torch.Tensor):
            flat = value.view(-1)
            if flat.shape[0] != batch_size:
                raise RuntimeError(f"Qwen3-Omni Talker decode lost {field}")
            first_tensor = flat[0]
            if (flat != first_tensor).any().item():
                return None
            return cast(first_tensor.item())
        return cast(value)

    def _should_use_captured_talker_decode(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        top_k,
        top_p,
    ) -> bool:
        if (
            not input_embeds.is_cuda
            or _talker_decode_capture_batch_size(forward_batch.batch_size) is None
            or not _attention_backend_uses_aiter(forward_batch.attn_backend)
        ):
            return False

        if self._talker_decode_bucket_seq_len(forward_batch) is None:
            return False

        k = self._talker_decode_uniform_scalar(
            top_k, "top_k", forward_batch.batch_size, int
        )
        p = self._talker_decode_uniform_scalar(
            top_p, "top_p", forward_batch.batch_size, float
        )
        return (
            k is not None
            and 0 < k < self.talker.codec_head.out_features
            and p == 1.0
        )

    def _talker_decode_graph_attn_backend(self, forward_batch: ForwardBatch):
        return getattr(
            forward_batch.attn_backend,
            "decode_backend",
            forward_batch.attn_backend,
        )

    def _set_talker_decode_graph_batch(
        self,
        forward_batch: ForwardBatch,
        graph_state: dict,
    ) -> None:
        batch_size = graph_state["batch_size"]
        forward_batch.batch_size = batch_size
        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.seq_lens = graph_state["seq_lens"]
        bucket_seq_len = graph_state["bucket_seq_len"]
        forward_batch.seq_lens_sum = graph_state["seq_lens_sum"]
        forward_batch.out_cache_loc = graph_state["out_cache_loc"]
        forward_batch.req_pool_indices = graph_state["req_pool_indices"]
        forward_batch.override_req_to_token = graph_state["req_to_token"]

        from sglang.srt.layers.attention.aiter_backend import (
            ForwardMetadata as AiterForwardMetadata,
        )

        aiter_backend = self._talker_decode_graph_attn_backend(forward_batch)
        aiter_backend.forward_metadata = AiterForwardMetadata(
            kv_indptr=graph_state["kv_indptr"],
            kv_indices=graph_state["kv_indices"],
            qo_indptr=None,
            kv_last_page_len=None,
            max_q_len=1,
            max_kv_len=bucket_seq_len,
            run_graph=False,
        )

    def _build_talker_decode_graph_metadata(self, graph_state: dict) -> None:
        batch_size = graph_state["batch_size"]
        graph_state["kv_indptr"][0].zero_()
        torch.cumsum(
            graph_state["seq_lens"],
            dim=0,
            out=graph_state["kv_indptr"][1 : batch_size + 1],
        )
        # The graph buffer is bucket-sized; initialize padded slots so AITER
        # partition padding never reads stale KV indices.
        graph_state["kv_indices"][:] = graph_state["out_cache_loc"][0].to(torch.int32)
        # Match AITER eager decode metadata construction exactly, but make it
        # part of the graph replay so Talker decode does not pay this launch
        # on every audio frame.
        create_flashinfer_kv_indices_triton[(batch_size,)](
            graph_state["req_to_token"],
            graph_state["req_pool_indices"],
            graph_state["seq_lens"],
            graph_state["kv_indptr"],
            None,
            graph_state["kv_indices"],
            graph_state["req_to_token"].stride(0),
        )

    @staticmethod
    def _copy_talker_decode_graph_tensor(
        dst: torch.Tensor,
        src: torch.Tensor,
        actual_batch_size: int,
        graph_batch_size: int,
        batch_dim: int = 0,
    ) -> None:
        actual_slice = [slice(None)] * dst.ndim
        actual_slice[batch_dim] = slice(0, actual_batch_size)
        dst[tuple(actual_slice)].copy_(src)
        if actual_batch_size == graph_batch_size:
            return

        pad_slice = [slice(None)] * dst.ndim
        pad_slice[batch_dim] = slice(actual_batch_size, graph_batch_size)
        first_slice = [slice(None)] * src.ndim
        first_slice[batch_dim] = slice(0, 1)
        expand_shape = list(src.shape)
        expand_shape[batch_dim] = graph_batch_size - actual_batch_size
        dst[tuple(pad_slice)].copy_(src[tuple(first_slice)].expand(*expand_shape))

    def _copy_talker_decode_graph_inputs(
        self,
        graph_state: dict,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        allow_codec_eos: torch.Tensor,
        temperature: torch.Tensor,
        repetition_penalty: torch.Tensor,
        repetition_mask: torch.Tensor,
        sampler_seed: torch.Tensor,
        sampler_step: torch.Tensor,
    ) -> None:
        actual_batch_size = input_embeds.shape[0]
        graph_batch_size = graph_state["batch_size"]
        if actual_batch_size > graph_batch_size:
            raise RuntimeError(
                f"Talker decode graph batch {graph_batch_size} cannot replay "
                f"actual batch {actual_batch_size}"
            )
        graph_state["actual_batch_size"] = actual_batch_size

        copy_batch = self._copy_talker_decode_graph_tensor
        copy_batch(
            graph_state["input_embeds"],
            input_embeds,
            actual_batch_size,
            graph_batch_size,
        )
        if positions.ndim == 2:
            copy_batch(
                graph_state["positions"],
                positions,
                actual_batch_size,
                graph_batch_size,
                batch_dim=1,
            )
        else:
            copy_batch(
                graph_state["positions"],
                positions,
                actual_batch_size,
                graph_batch_size,
            )

        copy_batch(
            graph_state["out_cache_loc"],
            forward_batch.out_cache_loc,
            actual_batch_size,
            graph_batch_size,
        )
        copy_batch(
            graph_state["seq_lens"],
            forward_batch.seq_lens,
            actual_batch_size,
            graph_batch_size,
        )
        if forward_batch.seq_lens_cpu is None:
            raise RuntimeError("Talker decode graph replay requires CPU seq lengths")
        actual_seq_lens = forward_batch.seq_lens_cpu[:actual_batch_size].tolist()
        graph_state["seq_lens_sum"] = sum(actual_seq_lens) + actual_seq_lens[0] * (
            graph_batch_size - actual_batch_size
        )

        source_len = forward_batch.override_req_to_token.shape[1]
        graph_state["req_to_token"][:] = graph_state["out_cache_loc"][0].to(
            torch.int32
        )
        graph_state["req_to_token"][:actual_batch_size, :source_len].copy_(
            forward_batch.override_req_to_token[:, :source_len]
        )
        if actual_batch_size < graph_batch_size:
            graph_state["req_to_token"][actual_batch_size:, :source_len].copy_(
                forward_batch.override_req_to_token[:1, :source_len].expand(
                    graph_batch_size - actual_batch_size, -1
                )
            )

        for name, src in (
            ("allow_codec_eos", allow_codec_eos),
            ("temperature", temperature),
            ("repetition_penalty", repetition_penalty),
            ("repetition_mask", repetition_mask),
            ("sampler_seed", sampler_seed),
            ("sampler_step", sampler_step),
        ):
            copy_batch(
                graph_state[name],
                src,
                actual_batch_size,
                graph_batch_size,
            )

    def _get_or_capture_talker_decode_graph(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        bucket_seq_len: int,
        codec_eos_token_id: int,
        allow_codec_eos: torch.Tensor,
        temperature: torch.Tensor,
        top_k: int,
        top_p: float,
        repetition_penalty: torch.Tensor,
        repetition_mask: torch.Tensor,
        sampler_seed: torch.Tensor,
        sampler_step: torch.Tensor,
    ) -> dict:
        actual_batch_size = input_embeds.shape[0]
        batch_size = _talker_decode_capture_batch_size(actual_batch_size)
        if batch_size is None:
            raise RuntimeError(
                f"Talker decode capture cannot bucket batch {actual_batch_size}"
            )
        if positions.ndim == 2:
            graph_positions_shape = (positions.shape[0], batch_size)
        else:
            graph_positions_shape = (batch_size, *positions.shape[1:])
        key = (
            batch_size,
            bucket_seq_len,
            input_embeds.dtype,
            input_embeds.device,
            graph_positions_shape,
            positions.dtype,
            codec_eos_token_id,
            top_k,
            top_p,
        )
        cached = self._talker_decode_graphs.get(key)
        if cached is not None:
            return cached

        capture_start = time.perf_counter()
        vocab_size = self.talker.codec_head.out_features
        graph_state = {
            "batch_size": batch_size,
            "bucket_seq_len": bucket_seq_len,
            "codec_eos_token_id": codec_eos_token_id,
            "top_k": top_k,
            "top_p": top_p,
            "actual_batch_size": actual_batch_size,
            "seq_lens_sum": None,
            "input_embeds": torch.empty(
                (batch_size, *input_embeds.shape[1:]),
                dtype=input_embeds.dtype,
                device=input_embeds.device,
            ),
            "positions": torch.empty(
                graph_positions_shape,
                dtype=positions.dtype,
                device=positions.device,
            ),
            "allow_codec_eos": torch.empty(
                (batch_size,), dtype=torch.bool, device=input_embeds.device
            ),
            "temperature": torch.empty(
                (batch_size,), dtype=torch.float32, device=input_embeds.device
            ),
            "repetition_penalty": torch.empty(
                (batch_size,), dtype=torch.float32, device=input_embeds.device
            ),
            "repetition_mask": torch.empty(
                (batch_size, vocab_size), dtype=torch.bool, device=input_embeds.device
            ),
            "sampler_seed": torch.empty(
                (batch_size,), dtype=torch.int64, device=input_embeds.device
            ),
            "sampler_step": torch.empty(
                (batch_size,), dtype=torch.int64, device=input_embeds.device
            ),
            "sampler_offsets": torch.arange(
                top_k, dtype=torch.int64, device=input_embeds.device
            ),
            "seq_lens": torch.empty(
                (batch_size,),
                dtype=torch.int32,
                device=input_embeds.device,
            ),
            "out_cache_loc": torch.empty(
                (batch_size,),
                dtype=forward_batch.out_cache_loc.dtype,
                device=forward_batch.out_cache_loc.device,
            ),
            "req_pool_indices": torch.arange(
                batch_size,
                dtype=forward_batch.req_pool_indices.dtype,
                device=input_embeds.device,
            ),
            "req_to_token": torch.empty(
                (batch_size, bucket_seq_len),
                dtype=torch.int32,
                device=input_embeds.device,
            ),
            "kv_indptr": torch.empty(
                (batch_size + 1,), dtype=torch.int32, device=input_embeds.device
            ),
            "kv_indices": torch.empty(
                (batch_size * bucket_seq_len,),
                dtype=torch.int32,
                device=input_embeds.device,
            ),
        }

        self._copy_talker_decode_graph_inputs(
            graph_state,
            positions,
            forward_batch,
            input_embeds,
            allow_codec_eos,
            temperature,
            repetition_penalty,
            repetition_mask,
            sampler_seed,
            sampler_step,
        )
        saved_forward_batch_state = (
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.seq_lens,
            forward_batch.seq_lens_sum,
            forward_batch.out_cache_loc,
            forward_batch.req_pool_indices,
            forward_batch.override_req_to_token,
        )
        self._set_talker_decode_graph_batch(forward_batch, graph_state)

        def run_once():
            self._build_talker_decode_graph_metadata(graph_state)
            return self._run_talker_model_head_and_sample_body(
                graph_state["positions"],
                forward_batch,
                graph_state["input_embeds"],
                graph_state["codec_eos_token_id"],
                graph_state["allow_codec_eos"],
                graph_state["temperature"],
                graph_state["top_k"],
                graph_state["top_p"],
                graph_state["repetition_penalty"],
                graph_state["repetition_mask"],
                graph_state["sampler_seed"],
                graph_state["sampler_step"],
                graph_state["sampler_offsets"],
            )

        stream = torch.cuda.Stream(device=input_embeds.device)
        try:
            stream.wait_stream(torch.cuda.current_stream(device=input_embeds.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run_once()
            torch.cuda.current_stream(device=input_embeds.device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hidden_states, codec_tokens, updated_repetition_mask = run_once()
        finally:
            (
                forward_batch.batch_size,
                forward_batch.forward_mode,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                forward_batch.out_cache_loc,
                forward_batch.req_pool_indices,
                forward_batch.override_req_to_token,
            ) = saved_forward_batch_state

        graph_state["graph"] = graph
        graph_state["hidden_states"] = hidden_states
        graph_state["codec_tokens"] = codec_tokens
        graph_state["updated_repetition_mask"] = updated_repetition_mask
        self._talker_decode_graphs[key] = graph_state
        logger.info(
            "Qwen3-Omni Talker decode graph captured: batch_size=%s, "
            "bucket_seq_len=%s, dtype=%s, top_k=%s, graph_count=%s, "
            "capture_wall_ms=%.2f",
            batch_size,
            bucket_seq_len,
            input_embeds.dtype,
            top_k,
            len(self._talker_decode_graphs),
            (time.perf_counter() - capture_start) * 1000,
        )
        return graph_state

    def _run_captured_talker_decode_model_and_sample(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        codec_eos_token_id: int,
        allow_codec_eos: torch.Tensor,
        temperature: torch.Tensor,
        top_k: int,
        top_p: float,
        repetition_penalty: torch.Tensor,
        repetition_mask: torch.Tensor,
        sampler_seed: torch.Tensor,
        sampler_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bucket_seq_len = self._talker_decode_bucket_seq_len(forward_batch)
        if bucket_seq_len is None:
            raise RuntimeError("Talker decode capture has no seq-len bucket")

        graph_state = self._get_or_capture_talker_decode_graph(
            positions,
            forward_batch,
            input_embeds,
            bucket_seq_len,
            codec_eos_token_id,
            allow_codec_eos,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_mask,
            sampler_seed,
            sampler_step,
        )
        self._copy_talker_decode_graph_inputs(
            graph_state,
            positions,
            forward_batch,
            input_embeds,
            allow_codec_eos,
            temperature,
            repetition_penalty,
            repetition_mask,
            sampler_seed,
            sampler_step,
        )
        self._set_talker_decode_graph_batch(forward_batch, graph_state)
        graph_state["graph"].replay()
        actual_batch_size = graph_state["actual_batch_size"]
        return (
            graph_state["hidden_states"][:actual_batch_size].clone(),
            graph_state["codec_tokens"][:actual_batch_size].clone(),
            graph_state["updated_repetition_mask"][:actual_batch_size].clone(),
        )

    def _qwen3_omni_mrope_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_input,
    ) -> torch.Tensor:
        thinker_config = self.config.thinker_config
        if self._qwen3_omni_has_only_audio_multimodal_input(multimodal_input):
            # Voice-only Talker prefill only needs multimodal placeholders to
            # collapse to the audio token before mRoPE/chunk-contract parsing.
            # Image/video prompts stay on the generic exact mapper below.
            return torch.where(
                input_ids >= MM_PAD_SHIFT_VALUE,
                torch.full_like(input_ids, thinker_config.audio_token_id),
                input_ids,
            )
        mrope_input_ids = input_ids.clone()
        if multimodal_input is not None:
            for item in multimodal_input.mm_items:
                if item.is_audio():
                    token_id = thinker_config.audio_token_id
                elif item.is_image():
                    token_id = thinker_config.image_token_id
                elif item.is_video():
                    token_id = thinker_config.video_token_id
                else:
                    raise RuntimeError("Qwen3-Omni mRoPE received unknown modality item")
                for start, end in item.offsets:
                    mrope_input_ids[start : end + 1] = token_id

        audio_starts = torch.nonzero(
            input_ids == thinker_config.audio_start_token_id, as_tuple=True
        )[0]
        num_tokens = input_ids.numel()
        for audio_start_tensor in audio_starts:
            start = int(audio_start_tensor.item()) + 1
            if start >= num_tokens:
                continue
            audio_tail = (input_ids[start:] >= MM_PAD_SHIFT_VALUE) | (
                mrope_input_ids[start:] == thinker_config.audio_token_id
            )
            tail_end = torch.nonzero(~audio_tail, as_tuple=True)[0]
            end = (
                num_tokens
                if tail_end.numel() == 0
                else start + int(tail_end[0].item())
            )
            if end > start:
                mrope_input_ids[start:end] = thinker_config.audio_token_id

        unmapped_mask = (input_ids >= MM_PAD_SHIFT_VALUE) & (
            mrope_input_ids >= MM_PAD_SHIFT_VALUE
        )
        if unmapped_mask.any():
            first_unmapped = int(torch.nonzero(unmapped_mask, as_tuple=True)[0][0].item())
            raise RuntimeError(
                "Qwen3-Omni mRoPE could not map multimodal placeholder "
                f"at token index {first_unmapped}"
            )
        return mrope_input_ids

    def _qwen3_omni_audio_feature_lens(
        self,
        multimodal_input,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if multimodal_input is None:
            return None

        audio_feature_lens = []
        for item in multimodal_input.mm_items:
            if not item.is_audio():
                continue
            audio_feature_len = getattr(item, "audio_feature_lens", None)
            if audio_feature_len is None:
                audio_attention_mask = getattr(item, "feature_attention_mask", None)
                if audio_attention_mask is None:
                    raise RuntimeError(
                        "Qwen3-Omni talker mRoPE requires audio feature lengths "
                        "or feature_attention_mask"
                    )
                audio_feature_len = audio_attention_mask.sum(dim=-1)
            audio_feature_lens.append(
                torch.as_tensor(audio_feature_len, device=device).flatten().sum()
            )

        if not audio_feature_lens:
            return None
        return torch.stack(audio_feature_lens).sum().view(1)

    def _qwen3_omni_audio_token_lens(
        self,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        thinker_config = self.config.thinker_config
        audio_starts = torch.nonzero(
            input_ids == thinker_config.audio_start_token_id, as_tuple=True
        )[0]
        if audio_starts.numel() == 0:
            return None

        audio_lens = []
        for audio_start in audio_starts.tolist():
            if (
                audio_start + 1 >= input_ids.numel()
                or input_ids[audio_start + 1].item() != thinker_config.audio_token_id
            ):
                continue
            pos = audio_start + 1
            while (
                pos < input_ids.numel()
                and input_ids[pos].item() == thinker_config.audio_token_id
            ):
                pos += 1
            audio_lens.append(pos - audio_start - 1)
        if not audio_lens:
            return None
        return torch.tensor(audio_lens, dtype=torch.long, device=input_ids.device)

    def _qwen3_omni_has_only_audio_multimodal_input(self, multimodal_input) -> bool:
        if multimodal_input is None:
            return True
        for item in multimodal_input.mm_items:
            if not item.is_audio():
                return False
        return True

    def _qwen3_omni_mrope_positions(
        self,
        input_ids: torch.Tensor,
        multimodal_input,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        thinker_config = self.config.thinker_config
        if self._qwen3_omni_has_only_audio_multimodal_input(multimodal_input):
            # For voice-only Talker inputs the upstream Qwen3-Omni mRoPE path
            # degenerates to contiguous positions with zero delta, but reaches
            # that answer through CPU token lists and per-segment tensor cats.
            positions = torch.arange(
                input_ids.numel(), device=device, dtype=torch.long
            ).view(1, -1).expand(3, -1)
            return positions, torch.zeros((1, 1), device=device, dtype=torch.long)

        image_grid_thw = None
        video_grid_thw = None
        if multimodal_input is not None:
            for item in multimodal_input.mm_items:
                if "image_grid_thw" in item.model_specific_data:
                    image_grid_thw = item.model_specific_data["image_grid_thw"]
                if "video_grid_thw" in item.model_specific_data:
                    video_grid_thw = item.model_specific_data["video_grid_thw"]

        positions, position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=thinker_config.vision_config.spatial_merge_size,
            image_token_id=thinker_config.image_token_id,
            video_token_id=thinker_config.video_token_id,
            vision_start_token_id=thinker_config.vision_start_token_id,
            model_type="qwen3_omni_moe",
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=None,
            use_audio_in_video=False,
            audio_seqlens=self._qwen3_omni_audio_feature_lens(
                multimodal_input, device
            ),
            audio_token_lens=self._qwen3_omni_audio_token_lens(input_ids),
            audio_token_id=thinker_config.audio_token_id,
            audio_start_token_id=thinker_config.audio_start_token_id,
            position_id_per_seconds=thinker_config.position_id_per_seconds,
            tokens_per_second=thinker_config.vision_config.tokens_per_second,
        )
        return positions.squeeze(1).to(device=device, dtype=torch.long), position_delta

    def _switch_to_talker_kv_cache_batched(
        self,
        forward_batch: ForwardBatch,
        talker_kv_locs_list: List[Optional[torch.Tensor]],
        init_metadata: bool = True,
    ) -> List[torch.Tensor]:
        """Switch forward_batch to talker KV caches for all requests."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        allocator = _component_kv_allocator(forward_batch, "talker")
        new_seq_lens = []
        updated_kv_locs_list = []

        # Bulk alloc one KV slot per request so OOM raises clearly instead
        # of letting per-iter `None` returns crash with NoneType downstream.
        new_locs_bulk = allocator.alloc(batch_size)
        if new_locs_bulk is None:
            raise RuntimeError(
                f"Talker KV pool exhausted in decode step: requested {batch_size} "
                f"slots, available {allocator.available_size()}"
            )

        for i in range(batch_size):
            talker_kv_locs = talker_kv_locs_list[i] if i < len(talker_kv_locs_list) else None
            new_loc = new_locs_bulk[i : i + 1]

            if talker_kv_locs is not None:
                # Append the new KV slot to this request's growing talker KV chain.
                updated_locs = torch.cat([talker_kv_locs, new_loc])
                new_seq_lens.append(len(updated_locs))
                updated_kv_locs_list.append(updated_locs)
            else:
                new_seq_lens.append(1)
                updated_kv_locs_list.append(new_loc)

        max_len = max(new_seq_lens)
        talker_req_to_token = self._get_talker_decode_req_to_token_buffer(
            forward_batch, batch_size, max_len
        )
        talker_req_to_token.zero_()
        req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()
        for i, locs in enumerate(updated_kv_locs_list):
            req_idx = req_pool_indices_cpu[i]
            forward_batch.req_to_token_pool.req_to_token[
                req_idx, : len(locs)
            ] = locs.to(torch.int32)
            talker_req_to_token[i, : len(locs)] = locs

        forward_batch.override_req_to_token = talker_req_to_token
        uniform_seq_len = new_seq_lens[0]
        if all(seq_len == uniform_seq_len for seq_len in new_seq_lens):
            forward_batch.seq_lens = torch.full(
                (batch_size,), uniform_seq_len, dtype=torch.int32, device=device
            )
            forward_batch.seq_lens_cpu = torch.full(
                (batch_size,), uniform_seq_len, dtype=torch.int32
            )
        else:
            forward_batch.seq_lens = torch.tensor(
                new_seq_lens, dtype=torch.int32, device=device
            )
            forward_batch.seq_lens_cpu = torch.tensor(new_seq_lens, dtype=torch.int32)
        forward_batch.seq_lens_sum = sum(new_seq_lens)
        forward_batch.out_cache_loc = new_locs_bulk

        if init_metadata:
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

        return updated_kv_locs_list

    def _run_talker_prefill_batched_subset(
        self,
        forward_batch: ForwardBatch,
        payload: Qwen3OmniTalkerPrefillPayload,
        request_indices: List[int],
        run_code_predictor: bool = True,
        sample_codec: bool = True,
        restore_forward_metadata: bool = True,
    ) -> Tuple[
        List[Optional[List[int]]],
        Optional[torch.Tensor],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        List[Optional[int]],
        List[Optional[torch.Tensor]],
        Optional[Qwen3OmniCodePredictorInputs],
    ]:
        """Run batched talker prefill for a subset of requests."""
        batch_size = forward_batch.batch_size
        if not request_indices:
            return (
                [None] * batch_size,
                None,
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                None,
            )

        device = forward_batch.seq_lens.device
        prev_sampled_token = payload.prev_sampled_thinker_token
        prompt_accept_hidden_states = payload.prompt_accept_hidden_states
        tts_pad_embed = payload.tts_pad_embed
        talker_kv_locs_list = payload.talker_kv_cache_locs_list
        talker_prefill_input_ids_list = payload.talker_prefill_input_ids_list
        codec_eos_token_id = payload.codec_eos_token_id
        talker_temperature = payload.talker_temperature
        talker_top_k = payload.talker_top_k
        talker_top_p = payload.talker_top_p
        talker_repetition_penalty = payload.talker_repetition_penalty
        talker_prev_codec_tokens = payload.talker_prev_codec_tokens
        origin_input_ids = payload.origin_input_ids
        origin_input_ids_cpu = payload.origin_input_ids_cpu
        multimodal_inputs = payload.multimodal_inputs
        speakers = payload.speakers
        if speakers is None:
            raise RuntimeError("Qwen3-Omni Talker prefill requires speaker state")
        allocator = _component_kv_allocator(forward_batch, "talker")
        vocab_size = self.thinker.model.embed_tokens.num_embeddings
        output_batch_size = max(request_indices) + 1
        timing_start = time.perf_counter()
        prep_done = alloc_done = metadata_done = model_done = sample_done = None
        mrope_input_ms = prepare_inputs_ms = mrope_positions_ms = prefix_check_ms = 0.0

        def log_talker_prefill_timing(
            sample_done: float, code_predictor_done: float
        ) -> None:
            logger.info(
                "Qwen3-Omni talker prefill substage: batch_size=%s, "
                "total_tokens=%s, min_m=%s, max_m=%s, has_prefix=%s, "
                "sample_codec=%s, prep_ms=%.2f, alloc_ms=%.2f, "
                "metadata_ms=%.2f, model_ms=%.2f, sample_ms=%.2f, "
                "code_predictor_ms=%.2f, total_ms=%.2f, "
                "mrope_input_ms=%.2f, prepare_inputs_ms=%.2f, "
                "mrope_positions_ms=%.2f, prefix_check_ms=%.2f",
                len(request_indices),
                total_tokens,
                min(seq_lens),
                max(seq_lens),
                any(prefix_len > 0 for prefix_len in prefix_lens),
                sample_codec,
                (prep_done - timing_start) * 1000,
                (alloc_done - prep_done) * 1000,
                (metadata_done - alloc_done) * 1000,
                (model_done - metadata_done) * 1000,
                (sample_done - model_done) * 1000,
                (code_predictor_done - sample_done) * 1000,
                (code_predictor_done - timing_start) * 1000,
                mrope_input_ms,
                prepare_inputs_ms,
                mrope_positions_ms,
                prefix_check_ms,
            )

        talker_input_embeds_list = []
        talker_positions_list = []
        seq_lens = []
        full_seq_lens = []
        prefix_lens = []
        start_locs = []
        last_indices = []
        req_prev_tokens = []
        previous_kv_locs = []
        talker_prefill_input_ids_out: List[Optional[torch.Tensor]] = (
            [None] * output_batch_size
        )
        tts_pad_embeds_out: List[Optional[torch.Tensor]] = (
            [None] * output_batch_size
        )
        codec_frames: List[Optional[Union[List[int], Qwen3OmniCodecFrame]]] = (
            [None] * output_batch_size
        )
        talker_out_cache_locs: List[Optional[torch.Tensor]] = (
            [None] * output_batch_size
        )
        talker_mrope_position_deltas: List[Optional[torch.Tensor]] = (
            [None] * output_batch_size
        )
        talker_user_prefill_len_out: List[Optional[int]] = (
            [None] * output_batch_size
        )
        prev_codec_input_embeds: List[Optional[torch.Tensor]] = (
            [None] * output_batch_size
        )

        # Pass 1: prepare embeds + record seq_lens (no KV alloc yet).
        total_tokens = 0
        for idx in request_indices:
            raw_input_ids = origin_input_ids[idx]
            raw_input_ids_cpu = origin_input_ids_cpu[idx]
            full_input_ids = raw_input_ids.clamp(min=0, max=vocab_size - 1)
            multimodal_input = (
                multimodal_inputs[idx] if multimodal_inputs is not None else None
            )
            step_start = time.perf_counter()
            talker_position_input_ids = self._qwen3_omni_mrope_input_ids(
                raw_input_ids, multimodal_input
            )
            voice_only_input = (
                multimodal_input is not None
                and self._qwen3_omni_has_only_audio_multimodal_input(multimodal_input)
            )
            talker_position_input_ids_cpu = None
            if voice_only_input:
                audio_token_id = self.config.thinker_config.audio_token_id
                talker_position_input_ids_cpu = [
                    audio_token_id if token_id >= MM_PAD_SHIFT_VALUE else token_id
                    for token_id in raw_input_ids_cpu
                ]
            step_done = time.perf_counter()
            mrope_input_ms += (step_done - step_start) * 1000
            step_start = step_done
            first_token = prev_sampled_token[idx] if prev_sampled_token is not None else None
            accept_hidden = (
                prompt_accept_hidden_states[idx]
                if prompt_accept_hidden_states is not None
                else None
            )
            if idx >= len(speakers) or not speakers[idx]:
                raise RuntimeError(
                    f"Qwen3-Omni Talker prefill missing speaker for batch index {idx}"
                )
            speaker = speakers[idx]
            previous_locs = (
                talker_kv_locs_list[idx]
                if idx < len(talker_kv_locs_list)
                else None
            )
            previous_input_ids = (
                talker_prefill_input_ids_list[idx]
                if idx < len(talker_prefill_input_ids_list)
                else None
            )
            prefix_len = len(previous_locs) if previous_locs is not None else 0
            (
                talker_input_embeds,
                talker_input_ids,
                tts_pad_embed_new,
                user_prefill_len,
            ) = self._prepare_talker_prefill_1step(
                accept_hidden=accept_hidden,
                input_ids=raw_input_ids,
                clamped_input_ids=full_input_ids,
                position_input_ids=talker_position_input_ids,
                input_ids_cpu=raw_input_ids_cpu,
                position_input_ids_cpu=talker_position_input_ids_cpu,
                first_response_token=first_token,
                speaker=speaker,
                include_assistant_response=sample_codec,
                skip_talker_prefix_len=prefix_len,
            )
            step_done = time.perf_counter()
            prepare_inputs_ms += (step_done - step_start) * 1000
            step_start = step_done
            if voice_only_input:
                talker_positions = (
                    torch.arange(
                        talker_input_ids.shape[0],
                        device=device,
                        dtype=torch.long,
                    )
                    .view(1, -1)
                    .expand(3, -1)
                )
                talker_position_delta = torch.zeros(
                    (1, 1), device=device, dtype=torch.long
                )
            else:
                talker_positions, talker_position_delta = (
                    self._qwen3_omni_mrope_positions(talker_input_ids, multimodal_input)
                )
            if prefix_len > 0:
                talker_positions = talker_positions[:, prefix_len:]
            step_done = time.perf_counter()
            mrope_positions_ms += (step_done - step_start) * 1000
            step_start = step_done

            if previous_locs is not None:
                if previous_input_ids is None:
                    raise RuntimeError(
                        "Qwen3-Omni Talker prefill has cached KV without "
                        f"input ids for batch index {idx}"
                    )
                if len(previous_locs) != len(previous_input_ids):
                    raise RuntimeError(
                        "Qwen3-Omni Talker prefill cached KV/input-id length "
                        f"mismatch for batch index {idx}: "
                        f"kv_len={len(previous_locs)}, ids_len={len(previous_input_ids)}"
                    )
                matches_final_prefix = (
                    len(previous_input_ids) <= talker_input_ids.shape[0]
                    and torch.equal(
                        previous_input_ids.to(device=talker_input_ids.device),
                        talker_input_ids[: len(previous_input_ids)],
                    )
                )
                if not matches_final_prefix:
                    previous_ids = previous_input_ids.to(device=talker_input_ids.device)
                    compare_len = min(len(previous_ids), talker_input_ids.shape[0])
                    first_diff = None
                    if compare_len > 0:
                        diff = torch.nonzero(
                            previous_ids[:compare_len] != talker_input_ids[:compare_len],
                            as_tuple=True,
                        )[0]
                        if diff.numel() > 0:
                            diff_idx = int(diff[0].item())
                            first_diff = (
                                diff_idx,
                                int(previous_ids[diff_idx].item()),
                                int(talker_input_ids[diff_idx].item()),
                            )
                    raise RuntimeError(
                        "Qwen3-Omni Talker cached prefill input ids are not "
                        f"a prefix of the current prefill for batch index {idx}: "
                        f"cached_len={len(previous_input_ids)}, "
                        f"current_len={talker_input_ids.shape[0]}, "
                        f"first_diff={first_diff}"
                )
            if prefix_len >= talker_input_ids.shape[0]:
                raise RuntimeError(
                    "Qwen3-Omni Talker prefill has no new tokens to extend "
                    f"for batch index {idx}: prefix_len={prefix_len}, "
                    f"prepared_len={talker_input_ids.shape[0]}"
                )
            prefix_check_ms += (time.perf_counter() - step_start) * 1000

            seq_len = talker_input_embeds.shape[0]
            start_locs.append(total_tokens)
            last_indices.append(total_tokens + seq_len - 1)
            total_tokens += seq_len

            talker_input_embeds_list.append(talker_input_embeds)
            talker_positions_list.append(talker_positions)
            seq_lens.append(seq_len)
            full_seq_lens.append(prefix_len + seq_len)
            prefix_lens.append(prefix_len)
            previous_kv_locs.append(previous_locs)
            talker_prefill_input_ids_out[idx] = talker_input_ids.detach()
            talker_user_prefill_len_out[idx] = user_prefill_len
            talker_mrope_position_deltas[idx] = talker_position_delta.view(-1)[0].detach()

            if tts_pad_embed_new is not None:
                tts_pad_embeds_out[idx] = tts_pad_embed_new

            if talker_prev_codec_tokens is not None:
                req_prev_tokens.append(talker_prev_codec_tokens[idx])
            else:
                req_prev_tokens.append(None)
        prep_done = time.perf_counter()

        # Bulk alloc all talker KV slots in one call. On OOM raise rather
        # than return-empty: returning empty leaves talker_needs_prefill=True,
        # KV stays full, scheduler retries forever with no progress.
        kv_locs_bulk = allocator.alloc(total_tokens)
        if kv_locs_bulk is None:
            raise RuntimeError(
                f"Talker KV pool exhausted in prefill: requested {total_tokens} "
                f"slots for {len(request_indices)} request(s), "
                f"available {allocator.available_size()}. "
                f"Lower concurrency or raise --mem-fraction-static."
            )

        # Pass 2: split bulk alloc into per-request chunks.
        out_cache_loc_chunks = []
        offset = 0
        for i, idx in enumerate(request_indices):
            sl = seq_lens[i]
            chunk = kv_locs_bulk[offset : offset + sl]
            offset += sl
            if previous_kv_locs[i] is not None:
                talker_out_cache_locs[idx] = torch.cat([previous_kv_locs[i], chunk])
            else:
                talker_out_cache_locs[idx] = chunk
            out_cache_loc_chunks.append(chunk)
        alloc_done = time.perf_counter()

        saved_state = self._save_forward_batch_state(
            forward_batch,
            save_extend=True,
            save_req_to_tokens=False,
        )

        input_embeds = torch.cat(talker_input_embeds_list, dim=0)
        positions = torch.cat(talker_positions_list, dim=1)

        extend_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        extend_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        extend_start_loc = torch.tensor(start_locs, dtype=torch.int32, device=device)
        max_len = max(full_seq_lens)
        talker_req_to_token = torch.zeros(
            len(request_indices), max_len, dtype=torch.int32, device=device
        )
        for i, kv_locs in enumerate(out_cache_loc_chunks):
            previous_locs = previous_kv_locs[i]
            if previous_locs is not None:
                talker_req_to_token[i, : len(previous_locs)] = previous_locs
                talker_req_to_token[
                    i, len(previous_locs) : len(previous_locs) + len(kv_locs)
                ] = kv_locs
            else:
                talker_req_to_token[i, : len(kv_locs)] = kv_locs

        forward_batch.batch_size = len(request_indices)
        req_pool_context = self._temporary_audio_req_pool_indices(
            forward_batch, len(request_indices)
        )
        req_pool_indices = req_pool_context.__enter__()
        req_pool_entered = True
        forward_batch.req_pool_indices = req_pool_indices
        forward_batch.req_to_token_pool.req_to_token[
            req_pool_indices, :max_len
        ] = talker_req_to_token
        forward_batch.override_req_to_token = talker_req_to_token
        forward_batch.seq_lens = torch.tensor(
            full_seq_lens, dtype=torch.int32, device=device
        )
        forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()
        forward_batch.seq_lens_sum = sum(full_seq_lens)
        forward_batch.out_cache_loc = torch.cat(out_cache_loc_chunks)
        forward_batch.forward_mode = ForwardMode.EXTEND
        forward_batch.extend_prefix_lens = extend_prefix_lens
        forward_batch.extend_prefix_lens_cpu = extend_prefix_lens.cpu().tolist()
        forward_batch.extend_seq_lens = extend_seq_lens
        forward_batch.extend_seq_lens_cpu = extend_seq_lens.cpu().tolist()
        forward_batch.extend_start_loc = extend_start_loc
        forward_batch.extend_num_tokens = total_tokens

        logger.info(
            "Qwen3-Omni talker prefill shape: batch_size=%s, total_tokens=%s, "
            "min_m=%s, max_m=%s, uniform=%s, has_prefix=%s, sample_codec=%s, "
            "seq_lens=%s, prefix_lens=%s",
            len(request_indices),
            total_tokens,
            min(seq_lens),
            max(seq_lens),
            all(seq_len == seq_lens[0] for seq_len in seq_lens),
            any(prefix_len > 0 for prefix_len in prefix_lens),
            sample_codec,
            seq_lens,
            prefix_lens,
        )
        forward_batch.attn_backend.init_forward_metadata(forward_batch)
        metadata_done = time.perf_counter()

        try:
            use_captured_talker_prefill = self._should_use_captured_talker_prefill(
                positions,
                forward_batch,
                input_embeds,
            )
            if use_captured_talker_prefill:
                hidden_states = self._run_captured_talker_prefill_model(
                    positions,
                    forward_batch,
                    input_embeds,
                )
            else:
                hidden_states = self._run_talker_model_body(
                    positions,
                    forward_batch,
                    input_embeds,
                )
            model_done = time.perf_counter()
            if not sample_codec:
                tts_pad_embed_out = None
                if any(e is not None for e in tts_pad_embeds_out):
                    if all(e is not None for e in tts_pad_embeds_out):
                        tts_pad_embed_out = torch.stack(tts_pad_embeds_out, dim=0)
                end_done = time.perf_counter()
                log_talker_prefill_timing(model_done, end_done)
                return (
                    codec_frames,
                    tts_pad_embed_out,
                    talker_out_cache_locs,
                    talker_mrope_position_deltas,
                    talker_prefill_input_ids_out,
                    talker_user_prefill_len_out,
                    prev_codec_input_embeds,
                    None,
                )

            last_hidden = hidden_states[torch.tensor(last_indices, device=device)]

            if isinstance(talker_temperature, torch.Tensor):
                temperature = talker_temperature[request_indices]
            else:
                temperature = talker_temperature
            if isinstance(talker_top_k, torch.Tensor):
                top_k = talker_top_k[request_indices]
            else:
                top_k = talker_top_k
            if isinstance(talker_top_p, torch.Tensor):
                top_p = talker_top_p[request_indices]
            else:
                top_p = talker_top_p
            if isinstance(talker_repetition_penalty, torch.Tensor):
                repetition_penalty = talker_repetition_penalty[request_indices]
            else:
                repetition_penalty = talker_repetition_penalty

            logits = self.talker.codec_head(last_hidden)
            logits = logits.clone()
            logits[:, int(codec_eos_token_id)] = float("-inf")
            codec_tokens = _sample_codec_tokens(
                logits,
                codec_eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=req_prev_tokens,
            )
            sample_done = time.perf_counter()
            if run_code_predictor:
                codec_ids_cpu, codec_ids_event = _stage_qwen3_omni_codec_ids(
                    codec_tokens
                )
            else:
                codec_ids = codec_tokens.tolist()

            codec_embeds = self.talker.model.embed_tokens(codec_tokens)
            predictor_inputs = torch.stack([last_hidden, codec_embeds], dim=1)
            code_predictor_inputs = None
            if run_code_predictor:
                residual_codes, next_codec_input_embeds = self.talker.code_predictor.generate_batched(
                    input_embeds=predictor_inputs,
                    forward_batch=forward_batch,
                    num_tokens=self.talker.config.code_predictor_config.num_code_groups - 1,
                    do_sample=True,
                    top_k=_CODE_PREDICTOR_TOP_K,
                    top_p=0.8,
                )
                staged_codec_frames = _stage_qwen3_omni_codec_frames(
                    codec_tokens,
                    residual_codes,
                    codec_ids_cpu,
                    codec_ids_event,
                )
                for local_i, req_idx in enumerate(request_indices):
                    codec_frames[req_idx] = staged_codec_frames[local_i]
                    prev_codec_input_embeds[req_idx] = next_codec_input_embeds[
                        local_i
                    ].detach()
            else:
                code_predictor_inputs = _non_eos_code_predictor_inputs(
                    request_indices=list(request_indices),
                    input_embeds=predictor_inputs,
                    codec_tokens=codec_tokens,
                    codec_eos_token_id=int(codec_eos_token_id),
                )
                for local_i, req_idx in enumerate(request_indices):
                    if int(codec_ids[local_i]) == int(codec_eos_token_id):
                        codec_frames[req_idx] = [int(codec_ids[local_i])]
            code_predictor_done = time.perf_counter()
            log_talker_prefill_timing(sample_done, code_predictor_done)
        finally:
            self._restore_forward_batch_state(
                forward_batch,
                saved_state,
                restore_attention_metadata=restore_forward_metadata,
            )
            if req_pool_entered:
                req_pool_context.__exit__(None, None, None)

        tts_pad_embed_out = None
        if any(e is not None for e in tts_pad_embeds_out):
            if tts_pad_embed is not None:
                for i, embed in enumerate(tts_pad_embeds_out):
                    if embed is None:
                        if tts_pad_embed.dim() == 2:
                            tts_pad_embeds_out[i] = tts_pad_embed[i]
                        else:
                            tts_pad_embeds_out[i] = tts_pad_embed
            if all(e is not None for e in tts_pad_embeds_out):
                tts_pad_embed_out = torch.stack(tts_pad_embeds_out, dim=0)

        return (
            codec_frames,
            tts_pad_embed_out,
            talker_out_cache_locs,
            talker_mrope_position_deltas,
            talker_prefill_input_ids_out,
            talker_user_prefill_len_out,
            prev_codec_input_embeds,
            code_predictor_inputs,
        )

    def _run_talker_decode_batched_subset(
        self,
        forward_batch: ForwardBatch,
        payload: Qwen3OmniTalkerDecodePayload,
        request_indices: List[int],
        run_code_predictor: bool = True,
        restore_forward_metadata: bool = True,
    ) -> Tuple[
        List[Optional[List[int]]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        Optional[Qwen3OmniCodePredictorInputs],
        Dict[str, float],
    ]:
        """Run batched talker decode for a subset of requests."""
        batch_size = forward_batch.batch_size
        if not request_indices:
            return (
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                [None] * batch_size,
                None,
                {},
            )

        device = forward_batch.seq_lens.device
        index_tensor = torch.tensor(request_indices, device=device, dtype=torch.long)

        prev_codec_input_embeds = payload.prev_codec_input_embeds[index_tensor]

        talker_positions = payload.talker_positions
        if talker_positions.ndim == 2:
            talker_positions = talker_positions[:, index_tensor]
        else:
            talker_positions = talker_positions[index_tensor]

        codec_eos_token_id = payload.codec_eos_token_id
        temperature = payload.talker_temperature
        top_k = payload.talker_top_k
        top_p = payload.talker_top_p
        talker_repetition_penalty = payload.talker_repetition_penalty

        thinker_done_tensor = (
            payload.thinker_done[index_tensor]
            if payload.thinker_done is not None
            else None
        )
        talker_post_thinker_started = (
            payload.talker_post_thinker_started[index_tensor]
            if payload.talker_post_thinker_started is not None
            else None
        )
        talker_has_text_lag = (
            payload.talker_has_text_lag[index_tensor]
            if payload.talker_has_text_lag is not None
            else None
        )
        talker_needs_tts_eos = (
            payload.talker_needs_tts_eos[index_tensor]
            if payload.talker_needs_tts_eos is not None
            else None
        )

        if isinstance(talker_repetition_penalty, torch.Tensor):
            repetition_penalty = talker_repetition_penalty[index_tensor]
        else:
            repetition_penalty = talker_repetition_penalty
        if isinstance(temperature, torch.Tensor):
            temperature = temperature[index_tensor]
        if isinstance(top_k, torch.Tensor):
            top_k = top_k[index_tensor]
        if isinstance(top_p, torch.Tensor):
            top_p = top_p[index_tensor]

        talker_sampler_seed = payload.talker_sampler_seed[index_tensor]
        talker_sampler_step = payload.talker_sampler_step[index_tensor]

        req_prev_tokens = None
        if payload.talker_prev_codec_tokens is not None:
            req_prev_tokens = [
                payload.talker_prev_codec_tokens[i]
                if i < len(payload.talker_prev_codec_tokens)
                else []
                for i in request_indices
            ]
        req_repetition_masks = None
        if payload.talker_repetition_mask is not None:
            req_repetition_masks = [
                payload.talker_repetition_mask[i]
                if i < len(payload.talker_repetition_mask)
                else None
                for i in request_indices
            ]

        tts_pad_embed = payload.tts_pad_embed
        if tts_pad_embed is not None and tts_pad_embed.dim() == 2:
            tts_pad_embed = tts_pad_embed[index_tensor]
        tts_eos_embed = None
        if talker_needs_tts_eos is not None and talker_needs_tts_eos.any():
            tts_eos_token = torch.tensor(
                [self.config.tts_eos_token_id],
                dtype=torch.long,
                device=device,
            )
            tts_eos_embed = self.talker.text_projection(
                self.thinker.model.embed_tokens(tts_eos_token)
            )[0]

        prev_sampled_tokens = (
            payload.prev_sampled_thinker_token[index_tensor]
            if payload.prev_sampled_thinker_token is not None
            else None
        )

        input_embeds = prev_codec_input_embeds

        trailing_text_hidden = None
        if prev_sampled_tokens is not None:
            prev_embeds = self.thinker.model.embed_tokens(prev_sampled_tokens)
            trailing_text_hidden = self.talker.text_projection(prev_embeds)

        if (
            thinker_done_tensor is not None
            and thinker_done_tensor.all()
            and (talker_has_text_lag is None or not talker_has_text_lag.any())
        ):
            if talker_needs_tts_eos is not None and talker_needs_tts_eos.any():
                input_embeds = input_embeds + torch.where(
                    talker_needs_tts_eos.view(-1, 1),
                    tts_eos_embed.view(1, -1),
                    tts_pad_embed,
                )
            elif tts_pad_embed is not None:
                input_embeds = input_embeds + tts_pad_embed
        elif thinker_done_tensor is not None and thinker_done_tensor.any():
            for i in range(len(request_indices)):
                has_text_lag = (
                    talker_has_text_lag is not None and talker_has_text_lag[i]
                )
                if thinker_done_tensor[i] and not has_text_lag:
                    needs_tts_eos = (
                        talker_needs_tts_eos is not None and talker_needs_tts_eos[i]
                    )
                    if needs_tts_eos:
                        input_embeds[i] = input_embeds[i] + tts_eos_embed
                    elif tts_pad_embed is not None and tts_pad_embed.dim() == 2:
                        input_embeds[i] = input_embeds[i] + tts_pad_embed[i]
                else:
                    if trailing_text_hidden is not None:
                        input_embeds[i] = input_embeds[i] + trailing_text_hidden[i]
        elif trailing_text_hidden is not None:
            input_embeds = input_embeds + trailing_text_hidden

        talker_kv_locs_subset = [
            payload.talker_kv_cache_locs_list[i]
            if i < len(payload.talker_kv_cache_locs_list)
            else None
            for i in request_indices
        ]
        saved_state = self._save_forward_batch_state(
            forward_batch,
            save_req_to_tokens=False,
        )
        req_pool_context = self._temporary_audio_req_pool_indices(
            forward_batch, len(request_indices)
        )
        req_pool_entered = False
        try:
            forward_batch.forward_mode = ForwardMode.DECODE
            forward_batch.batch_size = len(request_indices)
            forward_batch.req_pool_indices = req_pool_context.__enter__()
            req_pool_entered = True
            updated_kv_locs_list = self._switch_to_talker_kv_cache_batched(
                forward_batch,
                talker_kv_locs_subset,
                init_metadata=False,
            )

            local_batch_size = len(request_indices)
            vocab_size = self.talker.codec_head.out_features
            temperature_tensor = (
                temperature.to(device=device, dtype=torch.float32).view(-1)
                if isinstance(temperature, torch.Tensor)
                else torch.full(
                    (local_batch_size,),
                    float(temperature),
                    device=device,
                    dtype=torch.float32,
                )
            )
            repetition_penalty_tensor = (
                repetition_penalty.to(device=device, dtype=torch.float32).view(-1)
                if isinstance(repetition_penalty, torch.Tensor)
                else torch.full(
                    (local_batch_size,),
                    float(repetition_penalty),
                    device=device,
                    dtype=torch.float32,
                )
            )
            if temperature_tensor.shape[0] != local_batch_size:
                raise RuntimeError("Qwen3-Omni Talker decode lost temperature rows")
            if repetition_penalty_tensor.shape[0] != local_batch_size:
                raise RuntimeError(
                    "Qwen3-Omni Talker decode lost repetition-penalty rows"
                )

            allow_codec_eos = (
                talker_post_thinker_started
                if talker_post_thinker_started is not None
                else False
            )
            if talker_has_text_lag is not None and not isinstance(allow_codec_eos, bool):
                allow_codec_eos = allow_codec_eos & ~talker_has_text_lag
            if talker_needs_tts_eos is not None and not isinstance(allow_codec_eos, bool):
                allow_codec_eos = allow_codec_eos & ~talker_needs_tts_eos
            if isinstance(allow_codec_eos, bool):
                allow_codec_eos_tensor = torch.full(
                    (local_batch_size,),
                    allow_codec_eos,
                    device=device,
                    dtype=torch.bool,
                )
            else:
                allow_codec_eos_tensor = allow_codec_eos.to(
                    device=device, dtype=torch.bool
                ).view(-1)
            if allow_codec_eos_tensor.shape[0] != local_batch_size:
                raise RuntimeError("Qwen3-Omni Talker decode lost EOS mask rows")

            if req_repetition_masks is not None and all(
                mask is not None for mask in req_repetition_masks
            ):
                repetition_mask = _stack_codec_repetition_masks(
                    req_repetition_masks,
                    local_batch_size,
                    vocab_size,
                    device,
                )
            else:
                repetition_mask = _codec_repetition_mask(
                    req_prev_tokens,
                    local_batch_size,
                    vocab_size,
                    device,
                )
            use_captured_talker_decode = self._should_use_captured_talker_decode(
                input_embeds, forward_batch, top_k, top_p
            )
            talker_model_start = time.perf_counter()
            if use_captured_talker_decode:
                captured_top_k = self._talker_decode_uniform_scalar(
                    top_k, "top_k", local_batch_size, int
                )
                captured_top_p = self._talker_decode_uniform_scalar(
                    top_p, "top_p", local_batch_size, float
                )
                if captured_top_k is None or captured_top_p is None:
                    raise RuntimeError("Qwen3-Omni Talker decode capture lost sampler")
                hidden_states, codec_tokens, updated_repetition_mask = (
                    self._run_captured_talker_decode_model_and_sample(
                        talker_positions,
                        forward_batch,
                        input_embeds,
                        int(codec_eos_token_id),
                        allow_codec_eos_tensor,
                        temperature_tensor,
                        captured_top_k,
                        captured_top_p,
                        repetition_penalty_tensor,
                        repetition_mask,
                        talker_sampler_seed,
                        talker_sampler_step,
                    )
                )
            else:
                forward_batch.attn_backend.init_forward_metadata(forward_batch)
                hidden_states, logits = self._run_talker_model_and_head_body(
                    talker_positions,
                    forward_batch,
                    input_embeds,
                )
                logits = _mask_codec_eos(
                    logits, int(codec_eos_token_id), allow_codec_eos_tensor
                )
                codec_tokens = _sample_codec_tokens(
                    logits,
                    codec_eos_token_id,
                    temperature=temperature_tensor,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty_tensor,
                    repetition_mask=repetition_mask,
                    sampler_seed=talker_sampler_seed,
                    sampler_step=talker_sampler_step,
                )
                updated_repetition_mask = _update_codec_repetition_mask(
                    repetition_mask,
                    codec_tokens,
                )
            talker_model_done = time.perf_counter()

            code_predictor_prep_start = time.perf_counter()
            codec_embeds = self.talker.model.embed_tokens(codec_tokens)
            predictor_inputs = torch.stack([hidden_states, codec_embeds], dim=1)
            code_predictor_prep_done = time.perf_counter()
            code_predictor_inputs = None
            code_predictor_start = code_predictor_prep_done
            code_predictor_done = code_predictor_prep_done
            codec_id_materialize_start = code_predictor_prep_done
            codec_id_materialize_done = code_predictor_prep_done
            frame_materialize_start = code_predictor_prep_done
            frame_materialize_done = code_predictor_prep_done
            if run_code_predictor:
                codec_id_materialize_start = time.perf_counter()
                codec_ids_cpu, codec_ids_event = _stage_qwen3_omni_codec_ids(
                    codec_tokens
                )
                codec_id_materialize_done = time.perf_counter()
                code_predictor_start = codec_id_materialize_done
                residual_codes, next_codec_input_embeds = self.talker.code_predictor.generate_batched(
                    input_embeds=predictor_inputs,
                    forward_batch=forward_batch,
                    num_tokens=self.talker.config.code_predictor_config.num_code_groups - 1,
                    do_sample=True,
                    top_k=_CODE_PREDICTOR_TOP_K,
                    top_p=0.8,
                )
                code_predictor_done = time.perf_counter()

                frame_materialize_start = code_predictor_done
                codec_frames_list = _stage_qwen3_omni_codec_frames(
                    codec_tokens,
                    residual_codes,
                    codec_ids_cpu,
                    codec_ids_event,
                )
                frame_materialize_done = time.perf_counter()
            else:
                codec_frames_list = []
                code_predictor_inputs = _non_eos_code_predictor_inputs(
                    request_indices=list(request_indices),
                    input_embeds=predictor_inputs,
                    codec_tokens=codec_tokens,
                    codec_eos_token_id=int(codec_eos_token_id),
                )
                frame_materialize_done = time.perf_counter()
            metrics = {
                "talker_model_sample_ms": (talker_model_done - talker_model_start) * 1000,
                "code_predictor_prep_ms": (code_predictor_prep_done - code_predictor_prep_start) * 1000,
                "codec_id_materialize_ms": (codec_id_materialize_done - codec_id_materialize_start) * 1000,
                "code_predictor_ms": (code_predictor_done - code_predictor_start) * 1000,
                "codec_frame_materialize_ms": (frame_materialize_done - frame_materialize_start) * 1000,
            }
        finally:
            self._restore_forward_batch_state(
                forward_batch,
                saved_state,
                restore_attention_metadata=restore_forward_metadata,
            )
            if req_pool_entered:
                req_pool_context.__exit__(None, None, None)

        codec_frames: List[Optional[Union[List[int], Qwen3OmniCodecFrame]]] = [None] * batch_size
        updated_kv_locs: List[Optional[torch.Tensor]] = [None] * batch_size
        updated_repetition_masks: List[Optional[torch.Tensor]] = [None] * batch_size
        prev_codec_input_embeds_out: List[Optional[torch.Tensor]] = [None] * batch_size
        for local_i, req_idx in enumerate(request_indices):
            if run_code_predictor:
                codec_frames[req_idx] = codec_frames_list[local_i]
                prev_codec_input_embeds_out[req_idx] = next_codec_input_embeds[
                    local_i
                ].detach()
            elif int(codec_tokens[local_i].item()) == int(codec_eos_token_id):
                codec_frames[req_idx] = [int(codec_tokens[local_i].item())]
            updated_kv_locs[req_idx] = updated_kv_locs_list[local_i]
            updated_repetition_masks[req_idx] = updated_repetition_mask[
                local_i
            ].detach()

        return (
            codec_frames,
            updated_kv_locs,
            updated_repetition_masks,
            prev_codec_input_embeds_out,
            code_predictor_inputs,
            metrics,
        )

    def post_cuda_graph_forward(
        self,
        forward_batch: ForwardBatch,
        thinker_logits: LogitsProcessorOutput,
    ) -> ThinkerTalkerOutput:
        thinker_logits = self._suppress_voice_control_logits(
            thinker_logits, forward_batch
        )
        return ThinkerTalkerOutput(thinker_logits=thinker_logits)

    def _run_code_predictor_phase(
        self,
        forward_batch: ForwardBatch,
        request_indices: List[int],
        payload: Qwen3OmniCodePredictorPayload,
    ) -> CodePredictorPhaseOutput:
        if payload.request_indices != request_indices:
            raise RuntimeError(
                "Qwen3-Omni code_predictor phase request indices do not match "
                "payload"
            )
        input_embeds = payload.input_embeds
        codec_tokens = payload.codec_tokens
        if input_embeds.shape[0] != len(request_indices):
            raise RuntimeError(
                "Qwen3-Omni code_predictor payload batch size does not match "
                "request indices"
        )

        codec_frames_list, next_codec_input_embeds = (
            self.run_qwen3_omni_code_predictor_payload_batch(
                forward_batch,
                Qwen3OmniCodePredictorPayload(
                    request_indices=list(range(len(request_indices))),
                    input_embeds=input_embeds,
                    codec_tokens=codec_tokens,
                ),
            )
        )

        batch_size = _qwen3_omni_audio_phase_batch_size(
            forward_batch, request_indices
        )
        codec_frames: List[Optional[Union[List[int], Qwen3OmniCodecFrame]]] = [None] * batch_size
        prev_codec_input_embeds: List[Optional[torch.Tensor]] = [None] * batch_size
        for local_i, req_idx in enumerate(request_indices):
            codec_frames[req_idx] = codec_frames_list[local_i]
            prev_codec_input_embeds[req_idx] = next_codec_input_embeds[
                local_i
            ].detach()

        return CodePredictorPhaseOutput(
            codec_frames=codec_frames,
            prev_codec_input_embeds_list=prev_codec_input_embeds,
        )

    def _apply_code_predictor_phase_output(
        self,
        base: ThinkerTalkerOutput,
        phase_output: CodePredictorPhaseOutput,
    ) -> ThinkerTalkerOutput:
        return replace(
            base,
            codec_frames=_merge_phase_list(base.codec_frames, phase_output.codec_frames),
            prev_codec_input_embeds_list=_merge_phase_list(
                base.prev_codec_input_embeds_list,
                phase_output.prev_codec_input_embeds_list,
            ),
            code_predictor_inputs=None,
            prompt_accept_hidden_states=None,
            audio_phase_metrics=None,
        )

    def _apply_talker_prefill_phase_output(
        self,
        base: ThinkerTalkerOutput,
        phase_output: TalkerPrefillPhaseOutput,
    ) -> ThinkerTalkerOutput:
        return replace(
            base,
            codec_frames=_merge_phase_list(base.codec_frames, phase_output.codec_frames),
            tts_pad_embed=(
                phase_output.tts_pad_embed
                if phase_output.tts_pad_embed is not None
                else base.tts_pad_embed
            ),
            talker_out_cache_loc_list=_merge_phase_list(
                base.talker_out_cache_loc_list,
                phase_output.talker_out_cache_loc_list,
            ),
            talker_mrope_position_delta_list=_merge_phase_list(
                base.talker_mrope_position_delta_list,
                phase_output.talker_mrope_position_delta_list,
            ),
            talker_prefill_input_ids_list=_merge_phase_list(
                base.talker_prefill_input_ids_list,
                phase_output.talker_prefill_input_ids_list,
            ),
            talker_user_prefill_len_list=_merge_phase_list(
                base.talker_user_prefill_len_list,
                phase_output.talker_user_prefill_len_list,
            ),
            prev_codec_input_embeds_list=_merge_phase_list(
                base.prev_codec_input_embeds_list,
                phase_output.prev_codec_input_embeds_list,
            ),
            code_predictor_inputs=(
                phase_output.code_predictor_inputs
                if phase_output.code_predictor_inputs is not None
                else base.code_predictor_inputs
            ),
            prompt_accept_hidden_states=None,
            audio_phase_metrics=phase_output.metrics,
        )

    def _apply_talker_decode_phase_output(
        self,
        base: ThinkerTalkerOutput,
        phase_output: TalkerDecodePhaseOutput,
    ) -> ThinkerTalkerOutput:
        return replace(
            base,
            codec_frames=_merge_phase_list(base.codec_frames, phase_output.codec_frames),
            prev_codec_input_embeds_list=_merge_phase_list(
                base.prev_codec_input_embeds_list,
                phase_output.prev_codec_input_embeds_list,
            ),
            updated_talker_kv_locs_list=_merge_phase_list(
                base.updated_talker_kv_locs_list,
                phase_output.updated_talker_kv_locs_list,
            ),
            updated_talker_repetition_mask_list=_merge_phase_list(
                base.updated_talker_repetition_mask_list,
                phase_output.updated_talker_repetition_mask_list,
            ),
            code_predictor_inputs=(
                phase_output.code_predictor_inputs
                if phase_output.code_predictor_inputs is not None
                else base.code_predictor_inputs
            ),
            prompt_accept_hidden_states=None,
            audio_phase_metrics=phase_output.metrics,
        )

    def run_qwen3_omni_code_predictor_payload_batch(
        self,
        forward_batch: ForwardBatch,
        payload: Qwen3OmniCodePredictorPayload,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        batch_size = payload.input_embeds.shape[0]
        if payload.request_indices != list(range(batch_size)):
            raise RuntimeError(
                "Qwen3-Omni batched code_predictor payload requires dense "
                "request indices"
            )
        if payload.codec_tokens.shape[0] != batch_size:
            raise RuntimeError(
                "Qwen3-Omni batched code_predictor payload has mismatched "
                "codec token count"
            )

        audio_forward_batch = copy.copy(forward_batch)
        with self._temporary_audio_req_pool_indices(
            forward_batch, batch_size
        ) as req_pool_indices:
            residual_codes, next_codec_input_embeds = self.talker.code_predictor.generate_batched(
                input_embeds=payload.input_embeds,
                forward_batch=audio_forward_batch,
                num_tokens=self.talker.config.code_predictor_config.num_code_groups - 1,
                do_sample=True,
                top_k=_CODE_PREDICTOR_TOP_K,
                top_p=0.8,
                req_pool_indices=req_pool_indices,
            )

        codec_frames_tensor = torch.cat(
            [payload.codec_tokens.unsqueeze(1), residual_codes], dim=1
        )
        return codec_frames_tensor.tolist(), next_codec_input_embeds.detach()

    def run_qwen3_omni_audio_phase_after_sample(
        self,
        forward_batch: ForwardBatch,
        logits_output: ThinkerTalkerOutput,
        next_token_ids: torch.Tensor,
        phase: str,
        request_indices: List[int],
        payload: Optional[Qwen3OmniAudioPhasePayload] = None,
    ) -> ThinkerTalkerOutput:
        if not isinstance(logits_output, ThinkerTalkerOutput):
            return logits_output

        if phase == "talker_prefill":
            if not isinstance(payload, Qwen3OmniTalkerPrefillPayload):
                raise RuntimeError(
                    "Qwen3-Omni talker_prefill phase requires a "
                    "Qwen3OmniTalkerPrefillPayload"
                )
            audio_forward_batch = copy.copy(forward_batch)
            audio_forward_batch.batch_size = _qwen3_omni_audio_phase_batch_size(
                forward_batch, request_indices
            )
            phase_output = self._run_talker_prefill_at_decode(
                thinker_logits=logits_output.thinker_logits,
                forward_batch=audio_forward_batch,
                payload=payload,
                request_indices=request_indices,
                run_code_predictor=True,
                sample_codec=payload.sample_codec,
                restore_forward_metadata=False,
            )
            return self._apply_talker_prefill_phase_output(logits_output, phase_output)

        if phase == "talker_decode":
            if not isinstance(payload, Qwen3OmniTalkerDecodePayload):
                raise RuntimeError(
                    "Qwen3-Omni talker_decode phase requires a "
                    "Qwen3OmniTalkerDecodePayload"
                )
            audio_forward_batch = copy.copy(forward_batch)
            audio_forward_batch.batch_size = _qwen3_omni_audio_phase_batch_size(
                forward_batch, request_indices
            )
            (
                codec_frames,
                updated_kv_locs_list,
                updated_repetition_masks,
                prev_codec_input_embeds,
                code_predictor_inputs,
                metrics,
            ) = self._run_talker_decode_batched_subset(
                forward_batch=audio_forward_batch,
                payload=payload,
                request_indices=request_indices,
                run_code_predictor=True,
                restore_forward_metadata=False,
            )
            phase_output = TalkerDecodePhaseOutput(
                codec_frames=codec_frames,
                prev_codec_input_embeds_list=prev_codec_input_embeds,
                updated_talker_kv_locs_list=updated_kv_locs_list,
                updated_talker_repetition_mask_list=updated_repetition_masks,
                code_predictor_inputs=code_predictor_inputs,
                metrics=metrics,
            )
            return self._apply_talker_decode_phase_output(logits_output, phase_output)

        if phase == "code_predictor":
            if not isinstance(payload, Qwen3OmniCodePredictorPayload):
                raise RuntimeError(
                    "Qwen3-Omni code_predictor phase requires a "
                    "Qwen3OmniCodePredictorPayload"
                )
            phase_output = self._run_code_predictor_phase(
                forward_batch, request_indices, payload
            )
            return self._apply_code_predictor_phase_output(logits_output, phase_output)

        raise RuntimeError(f"Unknown Qwen3-Omni audio phase: {phase}")

    def _run_talker_prefill_at_decode(
        self,
        thinker_logits: Optional[LogitsProcessorOutput],
        forward_batch: ForwardBatch,
        payload: Qwen3OmniTalkerPrefillPayload,
        request_indices: Optional[List[int]] = None,
        run_code_predictor: bool = True,
        sample_codec: bool = True,
        restore_forward_metadata: bool = True,
    ) -> TalkerPrefillPhaseOutput:
        """Run talker prefill at first decode step (1-step delay). Supports batching.

        Uses prev_sampled_thinker_token (token 0) as first response token,
        and current thinker input_ids (token 1) as look-ahead. Talker prefill
        samples codec and runs code predictor, returning complete first frame.
        When `thinker_logits is None`, runs the thinker forward as a side-effect
        to populate captured aux hidden states needed by downstream prefill —
        this is load-bearing for the second talker_prefill in a streaming-input
        request, not defensive programming.
        """
        batch_size = forward_batch.batch_size
        if thinker_logits is None:
            thinker_result = self.thinker.model(
                input_ids=forward_batch.input_ids,
                positions=forward_batch.positions,
                forward_batch=forward_batch,
            )
            thinker_hidden = (
                thinker_result[0]
                if isinstance(thinker_result, tuple)
                else thinker_result
            )
            self._stabilize_thinker_logits(
                self.thinker.logits_processor(
                    forward_batch.input_ids,
                    thinker_hidden,
                    self.thinker.lm_head,
                    forward_batch,
                )
            )
        indices = request_indices if request_indices is not None else list(range(batch_size))
        (
            codec_frames,
            tts_pad_embed_out,
            talker_out_cache_locs,
            talker_mrope_position_deltas,
            talker_prefill_input_ids_out,
            talker_user_prefill_len_out,
            prev_codec_input_embeds,
            code_predictor_inputs,
        ) = self._run_talker_prefill_batched_subset(
            forward_batch=forward_batch,
            payload=payload,
            request_indices=indices,
            run_code_predictor=run_code_predictor,
            sample_codec=sample_codec,
            restore_forward_metadata=restore_forward_metadata,
        )
        return TalkerPrefillPhaseOutput(
            codec_frames=codec_frames,
            tts_pad_embed=tts_pad_embed_out,
            talker_out_cache_loc_list=talker_out_cache_locs,
            talker_mrope_position_delta_list=talker_mrope_position_deltas,
            talker_prefill_input_ids_list=talker_prefill_input_ids_out,
            talker_user_prefill_len_list=talker_user_prefill_len_out,
            prev_codec_input_embeds_list=prev_codec_input_embeds,
            code_predictor_inputs=code_predictor_inputs,
        )

    def _prepare_talker_prefill_1step(
        self,
        accept_hidden: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        clamped_input_ids: torch.Tensor,
        position_input_ids: torch.Tensor,
        input_ids_cpu: Sequence[int],
        first_response_token: torch.Tensor,
        speaker: str,
        include_assistant_response: bool = True,
        skip_talker_prefix_len: int = 0,
        position_input_ids_cpu: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Build talker prefill input using the actual sampled first response token."""
        device = input_ids.device
        dtype = self.thinker.model.embed_tokens.weight.dtype
        talker_hidden_size = self.config.talker_config.text_config.hidden_size

        thinker_config = self.config.thinker_config
        im_start_token_id = self.config.im_start_token_id
        system_token_id = self.config.system_token_id
        user_token_id = self.config.user_token_id
        assistant_token_id = self.config.assistant_token_id
        tts_pad_token_id = self.config.tts_pad_token_id

        talker_config = self.config.talker_config
        speaker_id = _qwen3_omni_speaker_id(talker_config, speaker)
        logger.info(
            "Qwen3-Omni Talker prefill speaker conditioning: "
            "speaker=%s, speaker_id=%s",
            speaker,
            speaker_id,
        )

        tts_bos_embed, tts_pad_embed, assistant_codec_hidden_base = (
            self._talker_prefill_special_embeds(
                device=device,
                dtype=dtype,
                speaker=speaker,
            )
        )

        chunk_contract = build_qwen3_omni_talker_prefill_chunk_contract(
            input_ids=input_ids,
            position_input_ids=position_input_ids,
            input_ids_cpu=input_ids_cpu,
            position_input_ids_cpu=position_input_ids_cpu,
            im_start_token_id=im_start_token_id,
            system_token_id=system_token_id,
            user_token_id=user_token_id,
            assistant_token_id=assistant_token_id,
            tts_pad_token_id=tts_pad_token_id,
            include_assistant_response=include_assistant_response,
        )

        talker_input_parts = []

        for chunk in chunk_contract.chunks:
            if chunk.talker_end <= skip_talker_prefix_len:
                continue
            chunk_skip = max(0, skip_talker_prefix_len - chunk.talker_start)
            if isinstance(chunk, Qwen3OmniTalkerPrefillUserChunk):
                im_start_index = chunk.source_start + chunk_skip
                segment_end_index = chunk.source_end
                segment_ids = input_ids[im_start_index:segment_end_index]
                segment_embeds = self.thinker.model.embed_tokens(
                    clamped_input_ids[im_start_index:segment_end_index]
                )
                segment_ids_cpu = input_ids_cpu[im_start_index:segment_end_index]
                segment_has_mm = False
                segment_has_text = False
                for token_id in segment_ids_cpu:
                    is_mm_token = (
                        token_id >= MM_PAD_SHIFT_VALUE
                        or token_id == thinker_config.audio_token_id
                        or token_id == thinker_config.image_token_id
                        or token_id == thinker_config.video_token_id
                    )
                    segment_has_mm |= is_mm_token
                    segment_has_text |= not is_mm_token
                if segment_has_mm:
                    if (
                        accept_hidden is None
                        or accept_hidden.shape[0] < segment_end_index
                    ):
                        raise RuntimeError(
                            "Qwen3-Omni talker prefill requires prompt accept-hidden "
                            "states covering origin_input_ids when the user segment "
                            "contains multimodal placeholders"
                        )
                    mm_mask = self._qwen3_omni_multimodal_mask(segment_ids)
                    user_projected = torch.empty(
                        segment_embeds.shape[0],
                        talker_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    user_projected[mm_mask] = self.talker.hidden_projection(
                        accept_hidden[im_start_index:segment_end_index][mm_mask]
                    )
                    if segment_has_text:
                        user_projected[~mm_mask] = self.talker.text_projection(
                            segment_embeds[~mm_mask]
                        )
                else:
                    user_projected = self.talker.text_projection(segment_embeds)
                talker_input_parts.append(user_projected)
                continue

            if isinstance(chunk, Qwen3OmniTalkerPrefillAssistantChunk):
                im_start_index = chunk.source_start
                segment_end_index = chunk.source_end
                assistant_embeds = self.thinker.model.embed_tokens(
                    clamped_input_ids[im_start_index:segment_end_index]
                )
                assistant_hidden = self.talker.text_projection(assistant_embeds)

                # Use actual sampled token for first_text_embed (key difference from original)
                if first_response_token is not None:
                    first_token_embed = self.thinker.model.embed_tokens(first_response_token.unsqueeze(0))
                    first_text_embed = self.talker.text_projection(first_token_embed)
                else:
                    first_text_embed = tts_pad_embed

                # Build text side: [header, 4x pad, bos, first_text]
                assistant_text_hidden = torch.cat([
                    assistant_hidden[:3],
                    tts_pad_embed.expand(4, -1),
                    tts_bos_embed,
                    first_text_embed,
                ], dim=0)

                assistant_input = assistant_text_hidden + assistant_codec_hidden_base
                if chunk_skip > 0:
                    assistant_input = assistant_input[chunk_skip:]
                talker_input_parts.append(assistant_input)
                continue

            raise RuntimeError(
                f"Unknown Qwen3-Omni Talker prefill chunk {type(chunk).__name__}"
            )

        if not talker_input_parts:
            raise RuntimeError(
                "Qwen3-Omni Talker prefill has no suffix after cached prefix"
            )
        talker_input_embeds = torch.cat(talker_input_parts, dim=0)
        user_prefill_len = chunk_contract.talker_input_ids.shape[0]
        if isinstance(chunk_contract.chunks[-1], Qwen3OmniTalkerPrefillAssistantChunk):
            user_prefill_len = chunk_contract.chunks[-1].talker_start
        return (
            talker_input_embeds,
            chunk_contract.talker_input_ids,
            tts_pad_embed.squeeze(0),
            user_prefill_len,
        )

    def _qwen3_omni_multimodal_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        thinker_config = self.config.thinker_config
        return (
            (input_ids >= MM_PAD_SHIFT_VALUE)
            | (input_ids == thinker_config.audio_token_id)
            | (input_ids == thinker_config.image_token_id)
            | (input_ids == thinker_config.video_token_id)
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        # Get num_experts for thinker and talker (they may differ)
        thinker_num_experts = self.config.thinker_config.text_config.num_experts
        talker_num_experts = self.config.talker_config.text_config.num_experts

        thinker_expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=thinker_num_experts,
        )

        talker_expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=talker_num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
            # GPTQ-specific suffixes for non-quantized layers (e.g., mlp.gate)
            ".g_idx",
            ".qweight",
            ".qzeros",
            ".scales",
        )

        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict
        thinker_gate_quant_tensor_count = 0
        thinker_gate_weight_tensor_count = 0

        for name, loaded_weight in weights:
            if name.startswith("talker.") and loaded_weight.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ) and self._talker_quant_config is None:
                raise ValueError(
                    "Qwen3-Omni Talker is currently instantiated without a quant_config, "
                    f"but the checkpoint contains FP8 Talker tensor {name}. "
                    "Use a BF16 Talker checkpoint, or implement Talker quantized loading "
                    "before serving this checkpoint."
                )
            is_fused_expert = False
            if name.startswith("thinker.model.layers.") and ".mlp.gate." in name:
                if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
                    thinker_gate_quant_tensor_count += 1
                elif name.endswith(".weight"):
                    thinker_gate_weight_tensor_count += 1
            name = name.replace(r"model.language_model.", r"model.")

            # Code2Wav weights: load directly without sharding
            if "code2wav" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            # Rename codec_embedding to embed_tokens for talker model and code predictor
            if "talker.model.codec_embedding" in name:
                name = name.replace("talker.model.codec_embedding", "talker.model.embed_tokens")
            if "talker.code_predictor.model.codec_embedding" in name:
                name = name.replace(
                    "talker.code_predictor.model.codec_embedding",
                    "talker.code_predictor.model.embed_tokens",
                )

            # Code predictor stacked params: per-index keys at
            # *.lm_head.{i}.weight and *.model.embed_tokens.{i}.weight load
            # into row i of a single Parameter named *_stack.
            handled_stack = False
            for prefix, stack_name in _CODE_PREDICTOR_STACK_PARAMS:
                if not (name.startswith(prefix) and name.endswith(".weight")):
                    continue
                idx_str = name[len(prefix) : -len(".weight")]
                if idx_str.isdigit() and stack_name in params_dict:
                    with torch.no_grad():
                        params_dict[stack_name][int(idx_str)].copy_(loaded_weight)
                    handled_stack = True
                    break
            if handled_stack:
                continue

            # Determine if this is a talker weight and select appropriate expert mapping
            is_talker_weight = "talker" in name
            expert_params_mapping = (
                talker_expert_params_mapping if is_talker_weight else thinker_expert_params_mapping
            )
            num_experts = talker_num_experts if is_talker_weight else thinker_num_experts

            name = name.replace(".self_attn.out_proj", ".self_attn.proj")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Check for fused experts: ".experts.gate_up_proj" or ".experts.down_proj"
                # Use ".experts." prefix to avoid matching "shared_expert.down_proj"
                if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # [TODO] Skip layers that are on other devices (check if sglang has a similar function)
                # if is_pp_missing_parameter(name, self):
                #     continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or "audio_tower" in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            if (
                                name_mapped not in params_dict
                                or loaded_weight.shape
                                != params_dict[name_mapped].shape
                            ):
                                loaded_weight = loaded_weight.transpose(-1, -2)
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            if (
                                name_mapped not in params_dict
                                or loaded_weight.shape
                                != params_dict[name_mapped].shape
                            ):
                                loaded_weight = loaded_weight.transpose(-1, -2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        if name_mapped in params_dict.keys():
                            param = params_dict[name_mapped]
                        else:
                            continue
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue
                    if "visual" in name or "audio_tower" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")
                        name = name.replace(r"attn.out_proj.", r"attn.proj.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            f"Loaded weight with {name=} not found in params_dict"
                        )

        if thinker_gate_quant_tensor_count > 0 and thinker_gate_weight_tensor_count == 0:
            raise ValueError(
                "Incompatible GPTQ checkpoint: thinker MoE router gate appears quantized "
                f"({thinker_gate_quant_tensor_count} tensors like *.mlp.gate.qweight), "
                "but Qwen3 Omni serving keeps router gates in native precision to match "
                "upstream SGLang. Re-quantize with GPTQModel dynamic exclusion for router "
                "gate modules, e.g. QuantizeConfig(dynamic={r'-:.*\\.mlp\\.gate$': {}})."
            )

        self._prewarm_code2wav()
        if _server_uses_aiter_attention():
            if not is_hip():
                raise RuntimeError(
                    "Qwen3-Omni AITER prewarm requires ROCm. Use a CUDA "
                    "attention backend such as flashinfer or triton on Nvidia."
                )
            self._prewarm_aiter_bf16_gemms()

    def _prewarm_code2wav(self) -> None:
        """Run Code2Wav at production shapes before any client traffic."""
        code2wav = self.code2wav
        # Cover every production outer batch size. Even with split decoder
        # batches, the pre-transformer and hidden layout used to create new
        # first-hit variants when traffic reached a previously unseen outer bs.
        warm_batch_sizes = range(1, _qwen3_omni_max_concurrent_audio_turns() + 1)
        torch.cuda.synchronize()
        start = time.time()
        logger.info("[qwen3-omni-prewarm] Code2Wav full-path pre-warm start")
        prewarm_count = code2wav.prewarm_runtime_shapes(
            warm_batch_sizes,
            steady_chunk_frames=_code2wav_steady_chunk_frames(),
            max_decoder_batch_size=8,
            capture_graphs=False,
        )
        logger.info(
            f"[qwen3-omni-prewarm] Code2Wav pre-warm completed in "
            f"{time.time() - start:.2f}s over {prewarm_count} (bs, dframes) combos"
        )

    def _prewarm_aiter_bf16_gemms(self) -> None:
        """Load AITER kernels for observed Qwen3-Omni late-session GEMM shapes."""
        from aiter.tuned_gemm import tgemm

        torch.cuda.synchronize()
        start = time.time()
        logger.info("[qwen3-omni-prewarm] AITER BF16 GEMM pre-warm start")
        for m, n, k in _QWEN3_OMNI_AITER_BF16_PREWARM_GEMMS:
            hidden_states = torch.empty(
                (m, k), device="cuda", dtype=torch.bfloat16
            )
            weight = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
            tgemm.mm(hidden_states, weight, otype=torch.bfloat16)
        torch.cuda.synchronize()
        logger.info(
            f"[qwen3-omni-prewarm] AITER BF16 GEMM pre-warm completed in "
            f"{time.time() - start:.2f}s over "
            f"{len(_QWEN3_OMNI_AITER_BF16_PREWARM_GEMMS)} shapes"
        )


EntryClass = Qwen3OmniMoeForConditionalGeneration
