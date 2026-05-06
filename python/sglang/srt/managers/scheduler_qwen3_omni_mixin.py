from __future__ import annotations

import copy
import dataclasses
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Deque, List, Optional, Tuple, Union

import torch
from torch.cuda import Stream as CudaStream

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    AUDIO_BATCH_DELAY_MS,
    AUDIO_CODE2WAV_CHUNK_FRAMES,
    AUDIO_DEFAULT_TALKER_MAX_NEW_TOKENS,
    AUDIO_MAX_AUDIO_FRAME_LAG,
    AUDIO_MAX_TEXT_LAG,
    FINISH_LENGTH,
    FINISH_ABORT,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.managers.utils import (
    Qwen3OmniAudioPhaseItem,
    Qwen3OmniAudioPhasePayload,
    Qwen3OmniCodePredictorPayload,
    Qwen3OmniCodecFrame,
    Qwen3OmniTalkerDecodePayload,
    Qwen3OmniTalkerPrefillPayload,
)
if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        GenerationBatchResult,
        Scheduler,
    )

logger = logging.getLogger(__name__)


CODE2WAV_FLUSH_MAX_BATCH = 4


CODE2WAV_MAX_BATCH = 8


CODE2WAV_DECODER_MAX_BATCH = 8
CODE2WAV_STEADY_CHUNK_FRAMES = AUDIO_CODE2WAV_CHUNK_FRAMES
# Talker decode uses exact CUDA/HIP graph batch shapes. Keep the steady cap
# bounded so bursts do not create oversized audio waves that make downstream
# Code2Wav latency dominate the 30ms audio deadline.
TALKER_DECODE_STEADY_BATCH_CAP = 32
# Sampled Talker-prefill emits the first codec frame. Keep it inside the
# captured prefill batch family; larger live batches create first-audio risk.
TALKER_PREFILL_SAMPLED_BATCH_CAP = 8
# CodePredictor is graph-captured and remains cheap at launch-shape batches.
# The bs=50 tail misses show ~20ms CodePredictor queue waits after first PCM,
# while lowering the global audio deadline regressed Talker-prefill tails.
AUDIO_CODE_PREDICTOR_BATCH_DELAY_MS = 10
# Talker prefills that do not sample codec audio are background prefix work:
# incremental user-audio prefill or safety rollback rebuild. They may take
# longer, but must not steal the realtime deadline from live assistant audio.
AUDIO_BACKGROUND_TALKER_PREFILL_BATCH_DELAY_MS = 5000
QWEN3_OMNI_RUNTIME_SNAPSHOT_INTERVAL_S = 5.0


def _qwen3_omni_code2wav_chunk_size(req: Req, flush: bool) -> int:
    if flush or req.talker_last_decoded_frame == 0:
        return AUDIO_CODE2WAV_CHUNK_FRAMES
    return CODE2WAV_STEADY_CHUNK_FRAMES


class Qwen3OmniAudioPhase(Enum):
    THINKER_RUNNING = auto()
    TALKER_PREFILL_QUEUED = auto()
    TALKER_DECODE_READY = auto()
    MTP_QUEUED = auto()
    CODE2WAV_QUEUED = auto()
    BACKPRESSURED = auto()
    DONE = auto()


@dataclasses.dataclass(frozen=True)
class Qwen3OmniAudioDeadline:
    request_rid: str
    batch_index: int
    ready_at: float

    def queue_wait_ms(self, now: float) -> float:
        return (now - self.ready_at) * 1000


@dataclasses.dataclass(frozen=True)
class Qwen3OmniTalkerPrefillReady(Qwen3OmniAudioDeadline):
    committed_audio_input: bool
    audio_input_chunk_idx: int


@dataclasses.dataclass(frozen=True)
class Qwen3OmniTalkerDecodeReady(Qwen3OmniAudioDeadline):
    pass


@dataclasses.dataclass(frozen=True)
class Qwen3OmniCodePredictorReady(Qwen3OmniAudioDeadline):
    pass


@dataclasses.dataclass(frozen=True)
class Qwen3OmniCode2WavReady(Qwen3OmniAudioDeadline):
    pending_frames: int
    flush: bool
    current_frame: int
    new_frames: int
    codes_window_cpu: torch.Tensor


Qwen3OmniAudioReadyState = Union[
    Qwen3OmniTalkerPrefillReady,
    Qwen3OmniTalkerDecodeReady,
    Qwen3OmniCodePredictorReady,
    Qwen3OmniCode2WavReady,
]


def qwen3_omni_codec_frame_id(frame) -> int:
    if isinstance(frame, Qwen3OmniCodecFrame):
        return frame.codec_id_value()
    return int(frame[0])


def qwen3_omni_codec_frame_codes_ready(frame) -> bool:
    return not isinstance(frame, Qwen3OmniCodecFrame) or frame.is_ready()


def qwen3_omni_codec_frame_codes_cpu(frame) -> torch.Tensor:
    if isinstance(frame, Qwen3OmniCodecFrame):
        if not frame.is_ready():
            raise RuntimeError("Qwen3-Omni codec frame CPU staging is not ready")
        return frame.codes_cpu
    return torch.tensor(
        tuple(int(code) for code in frame),
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )


def qwen3_omni_code2wav_window_cpu(frames) -> torch.Tensor:
    if not frames:
        raise RuntimeError("Qwen3-Omni Code2Wav CPU window has no frames")
    first = qwen3_omni_codec_frame_codes_cpu(frames[0])
    window = torch.empty(
        (len(frames), int(first.numel())),
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )
    window[0].copy_(first)
    for i, frame in enumerate(frames[1:], start=1):
        row = qwen3_omni_codec_frame_codes_cpu(frame)
        if row.numel() != first.numel():
            raise RuntimeError("Qwen3-Omni Code2Wav frame quantizer mismatch")
        window[i].copy_(row)
    return window


def qwen3_omni_code2wav_frames_ready(frames) -> bool:
    return all(qwen3_omni_codec_frame_codes_ready(frame) for frame in frames)


def qwen3_omni_audio_deadline_ms(
    ready_state: Qwen3OmniAudioReadyState,
) -> int:
    if isinstance(ready_state, Qwen3OmniCodePredictorReady):
        return AUDIO_CODE_PREDICTOR_BATCH_DELAY_MS
    return AUDIO_BATCH_DELAY_MS


def qwen3_omni_audio_deadline_miss_ms(
    ready_state: Qwen3OmniAudioReadyState,
    now: float,
) -> float:
    return max(
        0.0,
        ready_state.queue_wait_ms(now) - qwen3_omni_audio_deadline_ms(ready_state),
    )


def qwen3_omni_audio_item_deadline_ms(item: Qwen3OmniAudioPhaseItem) -> int:
    if _qwen3_omni_background_talker_prefill_item(item):
        return AUDIO_BACKGROUND_TALKER_PREFILL_BATCH_DELAY_MS
    return min(qwen3_omni_audio_deadline_ms(state) for state in item.ready_states)


@dataclass(frozen=True)
class Qwen3OmniLaneDecision:
    action: str
    reason: str
    phase: str
    batch_size: int
    queue_wait_ms: float
    max_text_lag: int
    max_audio_frame_lag: int
    has_thinker_work: bool
    depends_on_thinker_lane: bool
    at_lag_limit: bool
    pending_audio_items: int
    deadline_ms: int = AUDIO_BATCH_DELAY_MS


QWEN3_OMNI_AUDIO_READY_BY_PHASE = {
    "talker_prefill": Qwen3OmniTalkerPrefillReady,
    "talker_decode": Qwen3OmniTalkerDecodeReady,
    "code_predictor": Qwen3OmniCodePredictorReady,
}


QWEN3_OMNI_AUDIO_PAYLOAD_BY_PHASE = {
    "talker_prefill": Qwen3OmniTalkerPrefillPayload,
    "talker_decode": Qwen3OmniTalkerDecodePayload,
    "code_predictor": Qwen3OmniCodePredictorPayload,
}


def wav_to_pcm16(wav: torch.Tensor) -> bytes:
    wav = (wav.squeeze().clamp(-1, 1) * 32767).to(torch.int16)
    return wav.cpu().numpy().tobytes()


def qwen3_omni_prefill_audio_done(req: Req) -> bool:
    if hasattr(req, "_prefill_audio_done_snapshot"):
        return req._prefill_audio_done_snapshot
    return req.audio_input_done


def build_qwen3_omni_audio_ready_state(
    req: Req,
    phase: str,
    batch_index: int,
    ready_at: float,
) -> Qwen3OmniAudioReadyState:
    if phase == "talker_prefill":
        return Qwen3OmniTalkerPrefillReady(
            request_rid=req.rid,
            batch_index=batch_index,
            ready_at=ready_at,
            committed_audio_input=qwen3_omni_prefill_audio_done(req),
            audio_input_chunk_idx=req.audio_input_chunk_idx,
        )
    if phase == "talker_decode":
        return Qwen3OmniTalkerDecodeReady(
            request_rid=req.rid,
            batch_index=batch_index,
            ready_at=ready_at,
        )
    if phase == "code_predictor":
        return Qwen3OmniCodePredictorReady(
            request_rid=req.rid,
            batch_index=batch_index,
            ready_at=ready_at,
        )
    raise RuntimeError(f"Unknown Qwen3-Omni audio phase {phase}")


def mark_qwen3_omni_code2wav_ready_state(
    req: Req,
    ready_at: float,
    flush: bool,
    chunk_size: int | None = None,
    left_context_size: int = 25,
) -> Qwen3OmniCode2WavReady | None:
    existing = next(
        (
            state
            for state in req.qwen3_omni_audio_ready_states
            if isinstance(state, Qwen3OmniCode2WavReady)
            and state.flush == flush
        ),
        None,
    )
    if existing is not None:
        return existing

    current_frame = len(req.talker_output_codes)
    new_frames = current_frame - req.talker_last_decoded_frame
    chunk_size = chunk_size or _qwen3_omni_code2wav_chunk_size(req, flush)
    if new_frames <= 0:
        raise RuntimeError(
            f"Qwen3-Omni Code2Wav ready state for {req.rid} has no new frames"
        )
    if not flush and new_frames > chunk_size:
        new_frames = chunk_size
        current_frame = req.talker_last_decoded_frame + chunk_size
    context_start = max(0, req.talker_last_decoded_frame - left_context_size)
    frames = req.talker_output_codes[context_start:current_frame]
    if not qwen3_omni_code2wav_frames_ready(frames):
        return None
    codes_window_cpu = qwen3_omni_code2wav_window_cpu(frames)

    ready_state = Qwen3OmniCode2WavReady(
        request_rid=req.rid,
        batch_index=0,
        ready_at=ready_at,
        pending_frames=len(req.talker_output_codes) - req.talker_last_decoded_frame,
        flush=flush,
        current_frame=current_frame,
        new_frames=new_frames,
        codes_window_cpu=codes_window_cpu,
    )
    req.qwen3_omni_audio_ready_state = ready_state
    req.qwen3_omni_audio_ready_states.append(ready_state)
    return ready_state


def mark_qwen3_omni_audio_ready_state(
    req: Req,
    phase: str,
    batch_index: int,
    ready_at: float,
) -> Qwen3OmniAudioReadyState:
    ready_state = build_qwen3_omni_audio_ready_state(
        req=req,
        phase=phase,
        batch_index=batch_index,
        ready_at=ready_at,
    )
    req.qwen3_omni_audio_ready_state = ready_state
    req.qwen3_omni_audio_ready_states.append(ready_state)
    return ready_state


def consume_qwen3_omni_audio_ready_state(
    req: Req,
    ready_state: Qwen3OmniAudioReadyState,
) -> None:
    if ready_state not in req.qwen3_omni_audio_ready_states:
        raise RuntimeError(
            "Qwen3-Omni audio ready-state cleanup lost ownership for "
            f"request {req.rid}"
        )
    req.qwen3_omni_audio_ready_states.remove(ready_state)
    if req.qwen3_omni_audio_ready_state == ready_state:
        req.qwen3_omni_audio_ready_state = (
            req.qwen3_omni_audio_ready_states[-1]
            if req.qwen3_omni_audio_ready_states
            else None
        )


def validate_qwen3_omni_audio_phase_item_ready_states(
    item: Qwen3OmniAudioPhaseItem,
) -> None:
    expected_payload_type = QWEN3_OMNI_AUDIO_PAYLOAD_BY_PHASE[item.phase]
    if not isinstance(item.payload, expected_payload_type):
        raise RuntimeError(
            "Qwen3-Omni audio phase item has incompatible payload "
            f"{type(item.payload).__name__} for phase {item.phase}"
        )
    if len(item.ready_states) != len(item.request_indices):
        raise RuntimeError(
            "Qwen3-Omni audio phase item has mismatched ready states and "
            "request indices"
        )
    expected_type = QWEN3_OMNI_AUDIO_READY_BY_PHASE[item.phase]
    for ready_state, request_index, request_rid in zip(
        item.ready_states, item.request_indices, item.request_rids
    ):
        req = item.reqs[request_index]
        if not isinstance(ready_state, expected_type):
            raise RuntimeError(
                "Qwen3-Omni audio phase item has incompatible ready state "
                f"{type(ready_state).__name__} for phase {item.phase}"
            )
        if (
            ready_state.batch_index != request_index
            or ready_state.request_rid != request_rid
        ):
            raise RuntimeError(
                "Qwen3-Omni audio phase item ready state does not match "
                "request index/rid"
            )
        if req.rid != request_rid:
            raise RuntimeError(
                "Qwen3-Omni audio phase item request list no longer matches "
                "ready-state ownership"
            )
        if ready_state not in req.qwen3_omni_audio_ready_states:
            raise RuntimeError(
                "Qwen3-Omni audio phase item is stale; request no longer owns "
                "the queued ready state"
            )


def _require_qwen3_omni_model_state(
    model_states: dict,
    key: str,
    phase: str,
):
    if key not in model_states:
        raise RuntimeError(
            f"Qwen3-Omni {phase} phase payload requires {key} in "
            "ForwardBatch.model_specific_states"
        )
    return model_states[key]


def _clone_qwen3_omni_value(value):
    # Deep-clone tensors in audio phase payloads. Pass through scalars (sampling
    # args may be Python ints/floats) and None; recurse into list payloads.
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_qwen3_omni_value(v) for v in value]
    return value


def _clone_qwen3_omni_codec_history(
    values: Optional[List[Optional[List[int]]]],
) -> Optional[List[Optional[List[int]]]]:
    if values is None:
        return None
    return [list(value) if value is not None else None for value in values]


def snapshot_qwen3_omni_audio_forward_batch(forward_batch):
    snapshot = copy.copy(forward_batch)
    for attr in (
        "input_ids",
        "positions",
        "seq_lens",
        "req_pool_indices",
        "out_cache_loc",
        "extend_prefix_lens",
        "extend_seq_lens",
        "extend_start_loc",
    ):
        value = getattr(forward_batch, attr, None)
        if torch.is_tensor(value):
            setattr(snapshot, attr, _clone_qwen3_omni_value(value))

    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if torch.is_tensor(seq_lens_cpu):
        snapshot.seq_lens_cpu = _clone_qwen3_omni_value(seq_lens_cpu)
    elif seq_lens_cpu is not None:
        snapshot.seq_lens_cpu = list(seq_lens_cpu)

    extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if extend_prefix_lens_cpu is not None:
        snapshot.extend_prefix_lens_cpu = list(extend_prefix_lens_cpu)

    extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_seq_lens_cpu is not None:
        snapshot.extend_seq_lens_cpu = list(extend_seq_lens_cpu)

    return snapshot


def is_qwen3_omni_thinker_stop_token(req: Req, token_id: int) -> bool:
    if req.eos_token_ids and token_id in req.eos_token_ids:
        return True
    if req.sampling_params.stop_token_ids and token_id in req.sampling_params.stop_token_ids:
        return True
    if req.tokenizer is None:
        return False
    additional_stop_token_ids = req.tokenizer.additional_stop_token_ids
    return token_id == req.tokenizer.eos_token_id or (
        additional_stop_token_ids is not None and token_id in additional_stop_token_ids
    )


def build_qwen3_omni_audio_phase_payload(
    phase: str,
    forward_batch,
    logits_output,
    next_token_ids,
    request_indices: List[int],
    sample_codec: bool = True,
) -> Qwen3OmniAudioPhasePayload:
    model_states = dict(forward_batch.model_specific_states or {})
    device = forward_batch.seq_lens.device

    prompt_accept_hidden_states = getattr(
        logits_output, "prompt_accept_hidden_states", None
    )
    if prompt_accept_hidden_states is not None:
        model_states["prompt_accept_hidden_states"] = prompt_accept_hidden_states

    if getattr(logits_output, "talker_needs_prefill", False):
        model_states["prev_sampled_thinker_token"] = next_token_ids.to(
            device=device, dtype=torch.long
        )

    if phase == "talker_prefill":
        return Qwen3OmniTalkerPrefillPayload(
            origin_input_ids=_clone_qwen3_omni_value(
                _require_qwen3_omni_model_state(
                    model_states, "origin_input_ids", phase
                )
            ),
            origin_input_ids_cpu=_clone_qwen3_omni_value(
                _require_qwen3_omni_model_state(
                    model_states, "origin_input_ids_cpu", phase
                )
            ),
            multimodal_inputs=(
                list(model_states["multimodal_inputs"])
                if model_states.get("multimodal_inputs") is not None
                else None
            ),
            prev_sampled_thinker_token=_clone_qwen3_omni_value(
                _require_qwen3_omni_model_state(
                    model_states, "prev_sampled_thinker_token", phase
                )
            ),
            prompt_accept_hidden_states=_clone_qwen3_omni_value(
                model_states.get("prompt_accept_hidden_states")
            ),
            tts_pad_embed=_clone_qwen3_omni_value(
                model_states.get("tts_pad_embed")
            ),
            codec_eos_token_id=model_states.get("codec_eos_token_id", 2150),
            talker_temperature=_clone_qwen3_omni_value(
                model_states.get("talker_temperature", 0.9)
            ),
            talker_top_k=_clone_qwen3_omni_value(
                model_states.get("talker_top_k", 50)
            ),
            talker_top_p=_clone_qwen3_omni_value(
                model_states.get("talker_top_p", 1.0)
            ),
            talker_repetition_penalty=_clone_qwen3_omni_value(
                model_states.get("talker_repetition_penalty", 1.05)
            ),
            talker_prev_codec_tokens=_clone_qwen3_omni_codec_history(
                model_states.get("talker_prev_codec_tokens")
            ),
            speakers=(
                list(model_states["speakers"])
                if model_states.get("speakers") is not None
                else None
            ),
            talker_kv_cache_locs_list=_clone_qwen3_omni_value(
                model_states.get("talker_kv_cache_locs_list", [])
            ),
            talker_prefill_input_ids_list=_clone_qwen3_omni_value(
                model_states.get("talker_prefill_input_ids_list", [])
            ),
            sample_codec=sample_codec,
        )

    if phase == "talker_decode":
        positions = model_states.get(
            "talker_positions", getattr(forward_batch, "positions", None)
        )
        if positions is None:
            raise RuntimeError(
                "Qwen3-Omni talker_decode phase payload requires talker_positions "
                "or ForwardBatch.positions"
            )
        current_thinker_token = next_token_ids.to(device=device, dtype=torch.long)
        prev_sampled_thinker_token = model_states.get(
            "prev_sampled_thinker_token",
            current_thinker_token,
        )
        thinker_done = model_states.get("thinker_done")
        talker_has_text_lag = model_states.get("talker_has_text_lag")
        if thinker_done is not None:
            use_current_thinker_token = ~thinker_done
            if talker_has_text_lag is not None:
                use_current_thinker_token = (
                    use_current_thinker_token & ~talker_has_text_lag
                )
            prev_sampled_thinker_token = torch.where(
                use_current_thinker_token,
                current_thinker_token,
                prev_sampled_thinker_token,
            )
        return Qwen3OmniTalkerDecodePayload(
            prev_codec_input_embeds=_clone_qwen3_omni_value(
                _require_qwen3_omni_model_state(
                    model_states, "prev_codec_input_embeds", phase
                )
            ),
            talker_positions=_clone_qwen3_omni_value(positions),
            talker_kv_cache_locs_list=_clone_qwen3_omni_value(
                model_states.get("talker_kv_cache_locs_list", [])
            ),
            thinker_done=_clone_qwen3_omni_value(
                model_states.get("thinker_done")
            ),
            talker_post_thinker_started=_clone_qwen3_omni_value(
                model_states.get("talker_post_thinker_started")
            ),
            talker_has_text_lag=_clone_qwen3_omni_value(
                model_states.get("talker_has_text_lag")
            ),
            talker_needs_tts_eos=_clone_qwen3_omni_value(
                model_states.get("talker_needs_tts_eos")
            ),
            tts_pad_embed=_clone_qwen3_omni_value(
                model_states.get("tts_pad_embed")
            ),
            prev_sampled_thinker_token=_clone_qwen3_omni_value(
                prev_sampled_thinker_token
            ),
            codec_eos_token_id=model_states.get("codec_eos_token_id", 2150),
            talker_temperature=_clone_qwen3_omni_value(
                model_states.get("talker_temperature", 0.9)
            ),
            talker_top_k=_clone_qwen3_omni_value(
                model_states.get("talker_top_k", 50)
            ),
            talker_top_p=_clone_qwen3_omni_value(
                model_states.get("talker_top_p", 1.0)
            ),
            talker_repetition_penalty=_clone_qwen3_omni_value(
                model_states.get("talker_repetition_penalty", 1.05)
            ),
            talker_prev_codec_tokens=_clone_qwen3_omni_codec_history(
                model_states.get("talker_prev_codec_tokens")
            ),
            talker_repetition_mask=_clone_qwen3_omni_value(
                model_states.get("talker_repetition_mask")
            ),
            talker_sampler_seed=_clone_qwen3_omni_value(
                _require_qwen3_omni_model_state(
                    model_states, "talker_sampler_seed", phase
                )
            ),
            talker_sampler_step=_clone_qwen3_omni_value(
                _require_qwen3_omni_model_state(
                    model_states, "talker_sampler_step", phase
                )
            ),
        )

    if phase == "code_predictor":
        pending = getattr(logits_output, "code_predictor_inputs", None)
        if pending is None:
            raise RuntimeError("Qwen3-Omni code_predictor phase has no pending inputs")
        if pending.request_indices != request_indices:
            raise RuntimeError(
                "Qwen3-Omni code_predictor payload request indices do not match "
                "pending Talker output"
            )
        return Qwen3OmniCodePredictorPayload(
            request_indices=list(request_indices),
            input_embeds=_clone_qwen3_omni_value(pending.input_embeds),
            codec_tokens=_clone_qwen3_omni_value(pending.codec_tokens),
        )

    raise RuntimeError(f"Unknown Qwen3-Omni audio phase {phase}")


def filter_qwen3_omni_code_predictor_payload_by_request_indices(
    payload: Qwen3OmniCodePredictorPayload,
    request_indices: List[int],
) -> Qwen3OmniCodePredictorPayload:
    if payload.request_indices == request_indices:
        return payload
    row_by_request_index = {
        request_index: row
        for row, request_index in enumerate(payload.request_indices)
    }
    missing = [
        request_index
        for request_index in request_indices
        if request_index not in row_by_request_index
    ]
    if missing:
        raise RuntimeError(
            "Qwen3-Omni code_predictor payload cleanup request indices do not "
            f"match pending inputs: missing={missing}"
        )
    input_rows = torch.tensor(
        [row_by_request_index[request_index] for request_index in request_indices],
        device=payload.input_embeds.device,
        dtype=torch.long,
    )
    return Qwen3OmniCodePredictorPayload(
        request_indices=list(request_indices),
        input_embeds=payload.input_embeds[input_rows],
        codec_tokens=payload.codec_tokens[input_rows],
    )


def filter_qwen3_omni_audio_phase_item_by_request_indices(
    item: Qwen3OmniAudioPhaseItem,
    request_indices: List[int],
) -> bool:
    if not request_indices:
        for req_idx, ready_state in zip(item.request_indices, item.ready_states):
            consume_qwen3_omni_audio_ready_state(item.reqs[req_idx], ready_state)
        return False
    keep_set = set(request_indices)
    for req_idx, ready_state in zip(item.request_indices, item.ready_states):
        if req_idx not in keep_set:
            consume_qwen3_omni_audio_ready_state(item.reqs[req_idx], ready_state)
    kept_pairs = [
        (req_idx, rid)
        for req_idx, rid in zip(item.request_indices, item.request_rids)
        if req_idx in keep_set
    ]
    if [req_idx for req_idx, _ in kept_pairs] != request_indices:
        raise RuntimeError(
            "Qwen3-Omni audio phase item cleanup request indices do not "
            "match queued item order"
        )

    if len(kept_pairs) == len(item.request_indices):
        return True

    item.request_indices = [req_idx for req_idx, _ in kept_pairs]
    item.request_rids = [rid for _, rid in kept_pairs]
    item.ready_states = [
        ready_state
        for ready_state in item.ready_states
        if ready_state.batch_index in keep_set
    ]
    if len(item.ready_states) != len(item.request_indices):
        raise RuntimeError(
            "Qwen3-Omni audio phase cleanup lost ready-state ownership "
            "for a retained request"
        )
    item.queued_at = min(ready_state.ready_at for ready_state in item.ready_states)

    if item.phase == "code_predictor":
        filter_qwen3_omni_code_predictor_inputs_by_request_indices(
            item.logits_output,
            item.request_indices,
        )
        item.payload = filter_qwen3_omni_code_predictor_payload_by_request_indices(
            item.payload,
            item.request_indices,
        )
    return True


def _replace_qwen3_omni_audio_ready_state(
    req: Req,
    old_state: Qwen3OmniAudioReadyState,
    new_state: Qwen3OmniAudioReadyState,
) -> None:
    if old_state not in req.qwen3_omni_audio_ready_states:
        raise RuntimeError(
            "Qwen3-Omni audio ready-state reindex lost ownership for "
            f"request {req.rid}"
        )
    state_index = req.qwen3_omni_audio_ready_states.index(old_state)
    req.qwen3_omni_audio_ready_states[state_index] = new_state
    if req.qwen3_omni_audio_ready_state == old_state:
        req.qwen3_omni_audio_ready_state = new_state


def _qwen3_omni_sampling_values_compatible(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right)
    return left == right


def _qwen3_omni_talker_decode_items_compatible(
    left: Qwen3OmniAudioPhaseItem,
    right: Qwen3OmniAudioPhaseItem,
) -> bool:
    if right.phase != "talker_decode" or right.return_logprob != left.return_logprob:
        return False
    left_payload = left.payload
    right_payload = right.payload
    if not isinstance(left_payload, Qwen3OmniTalkerDecodePayload) or not isinstance(
        right_payload, Qwen3OmniTalkerDecodePayload
    ):
        return False
    if left_payload.codec_eos_token_id != right_payload.codec_eos_token_id:
        return False
    for field in (
        "thinker_done",
        "talker_post_thinker_started",
        "talker_has_text_lag",
        "talker_needs_tts_eos",
        "tts_pad_embed",
        "prev_sampled_thinker_token",
        "talker_prev_codec_tokens",
    ):
        if (getattr(left_payload, field) is None) != (
            getattr(right_payload, field) is None
        ):
            return False
    if left_payload.talker_positions.dim() != right_payload.talker_positions.dim():
        return False
    for field in (
        "talker_temperature",
        "talker_top_k",
        "talker_top_p",
        "talker_repetition_penalty",
    ):
        if not _qwen3_omni_sampling_values_compatible(
            getattr(left_payload, field), getattr(right_payload, field)
        ):
            return False
    return True


def _qwen3_omni_talker_prefill_items_compatible(
    left: Qwen3OmniAudioPhaseItem,
    right: Qwen3OmniAudioPhaseItem,
) -> bool:
    if right.phase != "talker_prefill" or right.return_logprob != left.return_logprob:
        return False
    left_payload = left.payload
    right_payload = right.payload
    if not isinstance(left_payload, Qwen3OmniTalkerPrefillPayload) or not isinstance(
        right_payload, Qwen3OmniTalkerPrefillPayload
    ):
        return False
    if (
        left_payload.sample_codec != right_payload.sample_codec
        or left_payload.codec_eos_token_id != right_payload.codec_eos_token_id
    ):
        return False
    for field in (
        "multimodal_inputs",
        "prompt_accept_hidden_states",
        "tts_pad_embed",
        "talker_prev_codec_tokens",
    ):
        if (getattr(left_payload, field) is None) != (
            getattr(right_payload, field) is None
        ):
            return False
    for field in (
        "talker_temperature",
        "talker_top_k",
        "talker_top_p",
        "talker_repetition_penalty",
    ):
        if not _qwen3_omni_sampling_values_compatible(
            getattr(left_payload, field), getattr(right_payload, field)
        ):
            return False
    return True


def _qwen3_omni_frame_producing_phase(phase: str) -> bool:
    return phase in ("talker_decode", "code_predictor")


def _qwen3_omni_frame_budget(req: Req) -> int:
    return min(
        AUDIO_CODE2WAV_CHUNK_FRAMES,
        AUDIO_MAX_AUDIO_FRAME_LAG - req.qwen3_omni_audio_frame_lag,
    )


def _qwen3_omni_audio_items_fit_frame_budget(
    items: List[Qwen3OmniAudioPhaseItem],
) -> bool:
    for item in items:
        if not _qwen3_omni_frame_producing_phase(item.phase):
            raise RuntimeError(
                "Qwen3-Omni frame budget only applies to Talker decode and "
                f"CodePredictor, got {item.phase}"
            )
    return _qwen3_omni_request_rows_fit_frame_budget(items)


def _qwen3_omni_request_rows_fit_frame_budget(
    items: List[Qwen3OmniAudioPhaseItem],
) -> bool:
    produced_by_rid: dict[str, int] = {}
    budget_by_rid: dict[str, int] = {}
    for item in items:
        for req_idx in item.request_indices:
            req = item.reqs[req_idx]
            budget_by_rid.setdefault(req.rid, _qwen3_omni_frame_budget(req))
            produced_by_rid[req.rid] = produced_by_rid.get(req.rid, 0) + 1
            if produced_by_rid[req.rid] > budget_by_rid[req.rid]:
                return False
    return True


def _qwen3_omni_can_extend_frame_producing_batch(
    items: List[Qwen3OmniAudioPhaseItem],
    candidate: Qwen3OmniAudioPhaseItem,
) -> bool:
    return _qwen3_omni_audio_items_fit_frame_budget([*items, candidate])


def _qwen3_omni_can_extend_talker_decode_batch(
    items: List[Qwen3OmniAudioPhaseItem],
    candidate: Qwen3OmniAudioPhaseItem,
) -> bool:
    rows = sum(len(item.request_indices) for item in items)
    if rows + len(candidate.request_indices) > TALKER_DECODE_STEADY_BATCH_CAP:
        return False
    return _qwen3_omni_can_extend_frame_producing_batch(items, candidate)


def _qwen3_omni_talker_decode_batch_chunks(
    request_indices: List[int],
) -> List[List[int]]:
    return [
        request_indices[i : i + TALKER_DECODE_STEADY_BATCH_CAP]
        for i in range(0, len(request_indices), TALKER_DECODE_STEADY_BATCH_CAP)
    ]


def _qwen3_omni_talker_prefill_item_samples_codec(
    item: Qwen3OmniAudioPhaseItem,
) -> bool:
    if item.phase != "talker_prefill":
        raise RuntimeError(
            "Qwen3-Omni Talker prefill frame-budget check got "
            f"{item.phase}"
        )
    if not isinstance(item.payload, Qwen3OmniTalkerPrefillPayload):
        raise RuntimeError("Qwen3-Omni Talker prefill item lost its payload")
    return item.payload.sample_codec


def _qwen3_omni_background_talker_prefill_item(
    item: Qwen3OmniAudioPhaseItem,
) -> bool:
    return item.phase == "talker_prefill" and not (
        _qwen3_omni_talker_prefill_item_samples_codec(item)
    )


def _qwen3_omni_can_extend_talker_prefill_batch(
    items: List[Qwen3OmniAudioPhaseItem],
    candidate: Qwen3OmniAudioPhaseItem,
) -> bool:
    items_sample_codec = _qwen3_omni_talker_prefill_item_samples_codec(items[0])
    candidate_sample_codec = _qwen3_omni_talker_prefill_item_samples_codec(candidate)
    if items_sample_codec != candidate_sample_codec:
        raise RuntimeError(
            "Qwen3-Omni Talker prefill batching mixed codec-sampling states"
        )
    if not candidate_sample_codec:
        return True
    rows = sum(len(item.request_indices) for item in items)
    if rows + len(candidate.request_indices) > TALKER_PREFILL_SAMPLED_BATCH_CAP:
        return False
    return _qwen3_omni_request_rows_fit_frame_budget([*items, candidate])


def _qwen3_omni_row_tensor(
    tensor: torch.Tensor, row: int, field: str, phase: str
) -> torch.Tensor:
    if tensor.dim() == 0 or row >= tensor.shape[0]:
        raise RuntimeError(f"Qwen3-Omni {phase} payload lost {field}")
    return tensor[row : row + 1]


def _qwen3_omni_sampling_value(field: str, source_rows, phase: str):
    values = [(getattr(payload, field), row) for payload, row in source_rows]
    if all(torch.is_tensor(value) for value, _ in values):
        return torch.cat(
            [_qwen3_omni_row_tensor(value, row, field, phase) for value, row in values],
            dim=0,
        )
    if any(torch.is_tensor(value) for value, _ in values):
        raise RuntimeError(
            f"Qwen3-Omni {phase} batch cannot mix tensor/scalar {field}"
        )
    first = values[0][0]
    if any(value != first for value, _ in values):
        raise RuntimeError(
            f"Qwen3-Omni {phase} batch cannot mix scalar {field}"
        )
    return first


def _build_qwen3_omni_talker_prefill_batch_item(
    items: List[Qwen3OmniAudioPhaseItem],
) -> Qwen3OmniAudioPhaseItem:
    if not items:
        raise RuntimeError("Qwen3-Omni talker_prefill batch is empty")
    for item in items:
        if not _qwen3_omni_talker_prefill_items_compatible(items[0], item):
            raise RuntimeError("Qwen3-Omni talker_prefill batch is incompatible")

    rows = [
        (item, req_idx, ready_state)
        for item in items
        for req_idx, ready_state in zip(item.request_indices, item.ready_states)
    ]
    request_indices = list(range(len(rows)))
    reqs = [item.reqs[req_idx] for item, req_idx, _ in rows]
    request_rids = [req.rid for req in reqs]
    source_rows = [(item.payload, req_idx) for item, req_idx, _ in rows]

    def list_rows(field: str) -> Optional[List]:
        values = [getattr(payload, field) for payload, _ in source_rows]
        if all(value is None for value in values):
            return None
        rows_out = []
        for value, (_, row) in zip(values, source_rows):
            if value is None:
                raise RuntimeError(
                    f"Qwen3-Omni talker_prefill payload lost {field}"
                )
            if row >= len(value):
                if field in (
                    "talker_kv_cache_locs_list",
                    "talker_prefill_input_ids_list",
                ):
                    rows_out.append(None)
                    continue
                raise RuntimeError(
                    f"Qwen3-Omni talker_prefill payload lost {field}"
                )
            rows_out.append(value[row])
        return rows_out

    def optional_embed_rows(field: str) -> Optional[torch.Tensor]:
        values = [getattr(payload, field) for payload, _ in source_rows]
        if all(value is None for value in values):
            return None
        embeds = []
        for value, (_, row) in zip(values, source_rows):
            if value is None:
                raise RuntimeError(
                    f"Qwen3-Omni talker_prefill batch cannot mix missing {field}"
                )
            if value.dim() == 1:
                embeds.append(value)
            elif value.dim() == 2 and row < value.shape[0]:
                embeds.append(value[row])
            else:
                raise RuntimeError(
                    f"Qwen3-Omni talker_prefill payload lost {field}"
                )
        return torch.stack(embeds, dim=0)

    first_payload = source_rows[0][0]
    payload = Qwen3OmniTalkerPrefillPayload(
        origin_input_ids=list_rows("origin_input_ids"),
        origin_input_ids_cpu=list_rows("origin_input_ids_cpu"),
        multimodal_inputs=list_rows("multimodal_inputs"),
        prev_sampled_thinker_token=torch.cat(
            [
                _qwen3_omni_row_tensor(
                    payload.prev_sampled_thinker_token,
                    row,
                    "prev_sampled_thinker_token",
                    "talker_prefill",
                )
                for payload, row in source_rows
            ],
            dim=0,
        ),
        prompt_accept_hidden_states=list_rows("prompt_accept_hidden_states"),
        tts_pad_embed=optional_embed_rows("tts_pad_embed"),
        codec_eos_token_id=first_payload.codec_eos_token_id,
        talker_temperature=_qwen3_omni_sampling_value("talker_temperature", source_rows, "talker_prefill"),
        talker_top_k=_qwen3_omni_sampling_value("talker_top_k", source_rows, "talker_prefill"),
        talker_top_p=_qwen3_omni_sampling_value("talker_top_p", source_rows, "talker_prefill"),
        talker_repetition_penalty=_qwen3_omni_sampling_value("talker_repetition_penalty", source_rows, "talker_prefill"),
        talker_prev_codec_tokens=list_rows("talker_prev_codec_tokens"),
        speakers=list_rows("speakers"),
        talker_kv_cache_locs_list=list_rows("talker_kv_cache_locs_list"),
        talker_prefill_input_ids_list=list_rows("talker_prefill_input_ids_list"),
        sample_codec=first_payload.sample_codec,
    )
    next_token_ids = torch.cat(
        [_qwen3_omni_row_tensor(item.next_token_ids, req_idx, "next_token_ids", "talker_prefill") for item, req_idx, _ in rows],
        dim=0,
    )
    logits_output = copy.copy(items[0].logits_output)
    for attr in (
        "codec_frames",
        "tts_pad_embed",
        "talker_out_cache_loc_list",
        "talker_mrope_position_delta_list",
        "talker_prefill_input_ids_list",
        "talker_user_prefill_len_list",
        "prev_codec_input_embeds_list",
        "code_predictor_inputs",
    ):
        if hasattr(logits_output, attr):
            setattr(logits_output, attr, None)

    ready_states = []
    for dense_idx, (item, req_idx, old_ready_state) in enumerate(rows):
        req = item.reqs[req_idx]
        new_ready_state = replace(old_ready_state, batch_index=dense_idx)
        _replace_qwen3_omni_audio_ready_state(req, old_ready_state, new_ready_state)
        ready_states.append(new_ready_state)

    return Qwen3OmniAudioPhaseItem(
        logits_output=logits_output,
        forward_batch=items[0].forward_batch,
        next_token_ids=next_token_ids,
        reqs=reqs,
        return_logprob=items[0].return_logprob,
        phase="talker_prefill",
        request_indices=request_indices,
        request_rids=request_rids,
        payload=payload,
        ready_states=ready_states,
        queued_at=min(item.queued_at for item in items),
    )


def _build_qwen3_omni_talker_decode_batch_item(
    items: List[Qwen3OmniAudioPhaseItem],
) -> Qwen3OmniAudioPhaseItem:
    if not items:
        raise RuntimeError("Qwen3-Omni talker_decode batch is empty")
    for item in items:
        if item.phase != "talker_decode":
            raise RuntimeError(
                "Qwen3-Omni talker_decode batch contains a different phase"
            )
        if not isinstance(item.payload, Qwen3OmniTalkerDecodePayload):
            raise RuntimeError(
                "Qwen3-Omni talker_decode batch requires owned decode payloads"
            )
        if item.return_logprob != items[0].return_logprob:
            raise RuntimeError(
                "Qwen3-Omni talker_decode batch cannot mix return_logprob values"
            )

    rows = [
        (item, req_idx, ready_state)
        for item in items
        for req_idx, ready_state in zip(item.request_indices, item.ready_states)
    ]
    request_indices = list(range(len(rows)))
    reqs = [item.reqs[req_idx] for item, req_idx, _ in rows]
    request_rids = [req.rid for req in reqs]
    source_rows = [(item.payload, req_idx) for item, req_idx, _ in rows]

    def tensor_rows(field: str) -> torch.Tensor:
        return torch.cat(
            [
                _qwen3_omni_row_tensor(
                    getattr(payload, field), row, field, "talker_decode"
                )
                for payload, row in source_rows
            ],
            dim=0,
        )

    def optional_tensor_rows(field: str) -> Optional[torch.Tensor]:
        values = [getattr(payload, field) for payload, _ in source_rows]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError(
                f"Qwen3-Omni talker_decode batch cannot mix missing {field}"
            )
        return tensor_rows(field)

    def optional_embed_rows(field: str) -> Optional[torch.Tensor]:
        values = [getattr(payload, field) for payload, _ in source_rows]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError(
                f"Qwen3-Omni talker_decode batch cannot mix missing {field}"
            )
        embeds = []
        for value, (_, row) in zip(values, source_rows):
            if value.dim() == 1:
                embeds.append(value)
            elif value.dim() == 2 and row < value.shape[0]:
                embeds.append(value[row])
            else:
                raise RuntimeError(
                    f"Qwen3-Omni talker_decode payload lost {field}"
                )
        return torch.stack(embeds, dim=0)

    def position_rows() -> torch.Tensor:
        first = source_rows[0][0].talker_positions
        if first.dim() == 2:
            return torch.cat(
                [
                    payload.talker_positions[:, row : row + 1]
                    for payload, row in source_rows
                ],
                dim=1,
            )
        return tensor_rows("talker_positions")

    def list_rows(field: str) -> Optional[List]:
        values = [getattr(payload, field) for payload, _ in source_rows]
        if all(value is None for value in values):
            return None
        rows_out = []
        for value, (_, row) in zip(values, source_rows):
            if value is None or row >= len(value):
                raise RuntimeError(
                    f"Qwen3-Omni talker_decode payload lost {field}"
                )
            rows_out.append(value[row])
        return rows_out

    codec_eos_token_id = source_rows[0][0].codec_eos_token_id
    if any(payload.codec_eos_token_id != codec_eos_token_id for payload, _ in source_rows):
        raise RuntimeError(
            "Qwen3-Omni talker_decode batch cannot mix codec EOS token ids"
        )

    payload = Qwen3OmniTalkerDecodePayload(
        prev_codec_input_embeds=tensor_rows("prev_codec_input_embeds"),
        talker_positions=position_rows(),
        talker_kv_cache_locs_list=list_rows("talker_kv_cache_locs_list"),
        thinker_done=optional_tensor_rows("thinker_done"),
        talker_post_thinker_started=optional_tensor_rows("talker_post_thinker_started"),
        talker_has_text_lag=optional_tensor_rows("talker_has_text_lag"),
        talker_needs_tts_eos=optional_tensor_rows("talker_needs_tts_eos"),
        tts_pad_embed=optional_embed_rows("tts_pad_embed"),
        prev_sampled_thinker_token=optional_tensor_rows("prev_sampled_thinker_token"),
        codec_eos_token_id=codec_eos_token_id,
        talker_temperature=_qwen3_omni_sampling_value("talker_temperature", source_rows, "talker_decode"),
        talker_top_k=_qwen3_omni_sampling_value("talker_top_k", source_rows, "talker_decode"),
        talker_top_p=_qwen3_omni_sampling_value("talker_top_p", source_rows, "talker_decode"),
        talker_repetition_penalty=_qwen3_omni_sampling_value("talker_repetition_penalty", source_rows, "talker_decode"),
        talker_prev_codec_tokens=list_rows("talker_prev_codec_tokens"),
        talker_repetition_mask=list_rows("talker_repetition_mask"),
        talker_sampler_seed=tensor_rows("talker_sampler_seed"),
        talker_sampler_step=tensor_rows("talker_sampler_step"),
    )
    next_token_ids = torch.cat(
        [_qwen3_omni_row_tensor(item.next_token_ids, req_idx, "next_token_ids", "talker_decode") for item, req_idx, _ in rows],
        dim=0,
    )
    logits_output = copy.copy(items[0].logits_output)
    for attr in (
        "codec_frames",
        "prev_codec_input_embeds_list",
        "updated_talker_kv_locs_list",
        "code_predictor_inputs",
    ):
        if hasattr(logits_output, attr):
            setattr(logits_output, attr, None)

    ready_states = []
    for dense_idx, (item, req_idx, old_ready_state) in enumerate(rows):
        req = item.reqs[req_idx]
        new_ready_state = replace(old_ready_state, batch_index=dense_idx)
        _replace_qwen3_omni_audio_ready_state(req, old_ready_state, new_ready_state)
        ready_states.append(new_ready_state)

    return Qwen3OmniAudioPhaseItem(
        logits_output=logits_output,
        forward_batch=items[0].forward_batch,
        next_token_ids=next_token_ids,
        reqs=reqs,
        return_logprob=items[0].return_logprob,
        phase="talker_decode",
        request_indices=request_indices,
        request_rids=request_rids,
        payload=payload,
        ready_states=ready_states,
        queued_at=min(item.queued_at for item in items),
    )


@dataclass
class Code2WavWorkItem:
    req: Req
    codes_window_cpu: torch.Tensor
    current_frame: int
    new_frames: int
    ready_state: Qwen3OmniCode2WavReady


@dataclass
class Code2WavPendingWork:
    items: List[Code2WavWorkItem]
    wav_chunks: torch.Tensor
    launched_at: float
    event: torch.cuda.Event | None
    enqueue_wall_ms: float
    codes_tensor_ms: float
    forward_enqueue_ms: float
    event_record_ms: float


@dataclass(frozen=True)
class Code2WavCompletedWork:
    decoded_reqs: int
    decoded_frames: int
    emitted_chunks: int
    batch_groups: int
    event_wait_ms: float
    materialize_ms: float


def should_run_code2wav_phase(
    req: Req,
    flush: bool = False,
    chunk_size: int | None = None,
) -> bool:
    if req.qwen3_omni_code2wav_pending:
        return False
    if req.codec_eos_token_id is None or not req.talker_output_codes:
        return False
    new_frames = len(req.talker_output_codes) - req.talker_last_decoded_frame
    if flush:
        if new_frames <= 0:
            return False
        current_frame = len(req.talker_output_codes)
        context_start = max(0, req.talker_last_decoded_frame - 25)
        return qwen3_omni_code2wav_frames_ready(
            req.talker_output_codes[context_start:current_frame]
        )
    chunk_size = chunk_size or _qwen3_omni_code2wav_chunk_size(req, flush=False)
    if new_frames < chunk_size:
        return False
    current_frame = req.talker_last_decoded_frame + min(new_frames, chunk_size)
    context_start = max(0, req.talker_last_decoded_frame - 25)
    return qwen3_omni_code2wav_frames_ready(
        req.talker_output_codes[context_start:current_frame]
    )


def qwen3_omni_finished_waiting_for_code2wav(req: Req) -> bool:
    if not req.finished() or req.codec_eos_token_id is None:
        return False
    if req.qwen3_omni_code2wav_pending:
        return True
    return len(req.talker_output_codes) > req.talker_last_decoded_frame


def build_code2wav_work_item(
    req: Req,
    flush: bool = False,
    chunk_size: int | None = None,
    left_context_size: int = 25,
) -> Code2WavWorkItem | None:
    if len(req.talker_pcm16_chunks) >= req.talker_pcm16_chunk_cap:
        return None

    ready_state = next(
        (
            state
            for state in req.qwen3_omni_audio_ready_states
            if isinstance(state, Qwen3OmniCode2WavReady)
            and state.flush == flush
        ),
        None,
    )
    if ready_state is None:
        if not should_run_code2wav_phase(
            req,
            flush=flush,
            chunk_size=chunk_size,
        ):
            return None
        ready_state = mark_qwen3_omni_code2wav_ready_state(
            req,
            time.perf_counter(),
            flush=flush,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )
        if ready_state is None:
            return None

    if ready_state.codes_window_cpu.numel() == 0:
        raise RuntimeError(
            f"Qwen3-Omni Code2Wav ready state for {req.rid} has no owned codes"
        )
    if ready_state.current_frame <= req.talker_last_decoded_frame:
        raise RuntimeError(
            f"Qwen3-Omni Code2Wav ready state for {req.rid} is stale: "
            f"current_frame={ready_state.current_frame}, "
            f"last_decoded={req.talker_last_decoded_frame}"
        )
    if ready_state.new_frames <= 0:
        raise RuntimeError(
            f"Qwen3-Omni Code2Wav ready state for {req.rid} has no new frames"
        )

    return Code2WavWorkItem(
        req=req,
        codes_window_cpu=ready_state.codes_window_cpu,
        current_frame=ready_state.current_frame,
        new_frames=ready_state.new_frames,
        ready_state=ready_state,
    )


def _iter_code2wav_phase_batches(
    reqs: List[Req],
    flush: bool,
    chunk_size: int | None,
    left_context_size: int,
    max_batch_size: int | None,
):
    work_groups = build_code2wav_work_groups(
        reqs,
        flush=flush,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
    )
    for items in sorted(
        work_groups.values(),
        key=lambda group: min(item.ready_state.ready_at for item in group),
    ):
        items.sort(key=lambda item: item.ready_state.ready_at)
        if max_batch_size is None:
            yield items
        else:
            for start in range(0, len(items), max_batch_size):
                yield items[start : start + max_batch_size]


def run_code2wav_phase_batch(
    reqs: List[Req],
    code2wav,
    flush: bool = False,
    chunk_size: int | None = None,
    left_context_size: int = 25,
    max_batch_size: int | None = None,
) -> Tuple[int, int, int, int]:
    decoded_reqs = decoded_frames = emitted_chunks = batch_groups = 0
    for items_batch in _iter_code2wav_phase_batches(
        reqs, flush, chunk_size, left_context_size, max_batch_size
    ):
        sub_dr, sub_df, sub_ec, sub_bg = run_code2wav_work_group(
            code2wav, items_batch
        )
        decoded_reqs += sub_dr
        decoded_frames += sub_df
        emitted_chunks += sub_ec
        batch_groups += sub_bg
    return decoded_reqs, decoded_frames, emitted_chunks, batch_groups


def launch_code2wav_phase_batch(
    reqs: List[Req],
    code2wav,
    stream: torch.cuda.Stream,
    flush: bool = False,
    chunk_size: int | None = None,
    left_context_size: int = 25,
    max_batch_size: int | None = None,
    decoder_max_batch_size: int | None = None,
    event_pool: List[torch.cuda.Event] | None = None,
    max_pending_groups: int | None = None,
) -> Tuple[List[Code2WavPendingWork], int, int, int, int, float]:
    pending_work: List[Code2WavPendingWork] = []
    decoded_reqs = decoded_frames = emitted_chunks = batch_groups = 0
    build_started_at = time.perf_counter()
    work_groups = build_code2wav_work_groups(
        reqs,
        flush=flush,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
    )
    work_group_build_ms = (time.perf_counter() - build_started_at) * 1000
    for items in sorted(
        work_groups.values(),
        key=lambda group: min(item.ready_state.ready_at for item in group),
    ):
        items.sort(key=lambda item: item.ready_state.ready_at)
        if max_batch_size is None:
            items_batches = (items,)
        else:
            items_batches = (
                items[start : start + max_batch_size]
                for start in range(0, len(items), max_batch_size)
            )
        for items_batch in items_batches:
            pending_work.append(
                launch_code2wav_work_group(
                    code2wav,
                    items_batch,
                    stream,
                    decoder_max_batch_size=decoder_max_batch_size,
                    completion_event=(
                        event_pool.pop()
                        if event_pool is not None and event_pool
                        else None
                    ),
                )
            )
            decoded_reqs += len(items_batch)
            decoded_frames += sum(item.new_frames for item in items_batch)
            emitted_chunks += len(items_batch)
            batch_groups += 1
            if max_pending_groups is not None and batch_groups >= max_pending_groups:
                return (
                    pending_work,
                    decoded_reqs,
                    decoded_frames,
                    emitted_chunks,
                    batch_groups,
                    work_group_build_ms,
                )
    return (
        pending_work,
        decoded_reqs,
        decoded_frames,
        emitted_chunks,
        batch_groups,
        work_group_build_ms,
    )


def _code2wav_codes_tensor(
    codes_windows_cpu: List[torch.Tensor], device: torch.device
) -> torch.Tensor:
    max_window_len = max(int(window.shape[0]) for window in codes_windows_cpu)
    num_quantizers = int(codes_windows_cpu[0].shape[1])
    padded_codes_windows = torch.empty(
        (len(codes_windows_cpu), max_window_len, num_quantizers),
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )
    for i, window in enumerate(codes_windows_cpu):
        window_len = int(window.shape[0])
        padded_codes_windows[i, :window_len].copy_(window)
        if window_len < max_window_len:
            padded_codes_windows[i, window_len:].copy_(window[-1])
    return (
        padded_codes_windows.to(device=device, non_blocking=True)
        .transpose(1, 2)
        .contiguous()
    )


def _code2wav_forward(
    code2wav,
    codes_tensor: torch.Tensor,
    decoder_max_batch_size: int | None,
) -> torch.Tensor:
    batch_size = codes_tensor.shape[0]
    num_frames = codes_tensor.shape[2]
    if code2wav.should_capture_split_decoder(
        batch_size, num_frames, CODE2WAV_STEADY_CHUNK_FRAMES
    ):
        if decoder_max_batch_size is None:
            raise RuntimeError("Qwen3-Omni Code2Wav capture requires split decoder")
        return code2wav.forward_split_decoder_captured(
            codes_tensor,
            max_decoder_batch_size=decoder_max_batch_size,
        )
    if decoder_max_batch_size is None:
        return code2wav(codes_tensor)
    return code2wav.forward_split_decoder(
        codes_tensor,
        decoder_max_batch_size,
    )


def build_code2wav_work_groups(
    reqs: List[Req],
    flush: bool,
    chunk_size: int | None,
    left_context_size: int,
) -> dict[int, List[Code2WavWorkItem]]:
    work_items: List[Code2WavWorkItem] = []
    for req in reqs:
        item = build_code2wav_work_item(
            req,
            flush=flush,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )
        if item is None:
            continue
        work_items.append(item)

    work_groups: dict[int, List[Code2WavWorkItem]] = {}
    for item in work_items:
        key = 0 if flush else item.new_frames
        work_groups.setdefault(key, []).append(item)
    return work_groups


def run_code2wav_work_group(
    code2wav,
    items: List[Code2WavWorkItem],
) -> Tuple[int, int, int, int]:
    pending = launch_code2wav_work_group(code2wav, items, stream=None)
    result = complete_code2wav_pending_work(code2wav, pending, block=True)
    if result is None:
        raise RuntimeError("Synchronous Qwen3-Omni Code2Wav work did not complete")
    return (
        result.decoded_reqs,
        result.decoded_frames,
        result.emitted_chunks,
        result.batch_groups,
    )


def launch_code2wav_work_group(
    code2wav,
    items: List[Code2WavWorkItem],
    stream: torch.cuda.Stream | None,
    decoder_max_batch_size: int | None = None,
    completion_event: torch.cuda.Event | None = None,
) -> Code2WavPendingWork:
    device = next(code2wav.parameters()).device
    if items[0].codes_window_cpu.numel() == 0:
        raise ValueError(f"Code2Wav work item for {items[0].req.rid} has no codes")
    num_quantizers = int(items[0].codes_window_cpu.shape[1])
    codes_windows_cpu: List[torch.Tensor] = []
    for item in items:
        if item.codes_window_cpu.numel() == 0:
            raise ValueError(f"Code2Wav work item for {item.req.rid} has no codes")
        if int(item.codes_window_cpu.shape[1]) != num_quantizers:
            raise ValueError(
                f"Code2Wav work item for {item.req.rid} has "
                f"{int(item.codes_window_cpu.shape[1])} quantizers, expected {num_quantizers}"
            )
        codes_windows_cpu.append(item.codes_window_cpu)

    enqueue_started_at = time.perf_counter()
    codes_tensor_ms = 0.0
    forward_enqueue_ms = 0.0
    event_record_ms = 0.0
    if stream is None:
        codes_tensor_started_at = time.perf_counter()
        codes_tensor = _code2wav_codes_tensor(codes_windows_cpu, device)
        codes_tensor_ms = (time.perf_counter() - codes_tensor_started_at) * 1000
        with torch.no_grad():
            forward_started_at = time.perf_counter()
            wav_chunks = _code2wav_forward(
                code2wav, codes_tensor, decoder_max_batch_size
            )
            forward_enqueue_ms = (time.perf_counter() - forward_started_at) * 1000
        event = None
    else:
        with torch.cuda.stream(stream), torch.no_grad():
            codes_tensor_started_at = time.perf_counter()
            codes_tensor = _code2wav_codes_tensor(codes_windows_cpu, device)
            codes_tensor_ms = (time.perf_counter() - codes_tensor_started_at) * 1000
            forward_started_at = time.perf_counter()
            wav_chunks = _code2wav_forward(
                code2wav, codes_tensor, decoder_max_batch_size
            )
            forward_enqueue_ms = (time.perf_counter() - forward_started_at) * 1000
            event_started_at = time.perf_counter()
            event = completion_event or torch.cuda.Event(enable_timing=False)
            event.record(stream)
            event_record_ms = (time.perf_counter() - event_started_at) * 1000
    launched_at = time.perf_counter()
    for item in items:
        item.req.qwen3_omni_code2wav_pending = True
    return Code2WavPendingWork(
        items=items,
        wav_chunks=wav_chunks,
        launched_at=launched_at,
        event=event,
        enqueue_wall_ms=(launched_at - enqueue_started_at) * 1000,
        codes_tensor_ms=codes_tensor_ms,
        forward_enqueue_ms=forward_enqueue_ms,
        event_record_ms=event_record_ms,
    )


def complete_code2wav_pending_work(
    code2wav,
    pending: Code2WavPendingWork,
    block: bool,
) -> Code2WavCompletedWork | None:
    event_wait_started_at = time.perf_counter()
    if pending.event is not None:
        if not block and not pending.event.query():
            return None
        pending.event.synchronize()
    event_wait_ms = (time.perf_counter() - event_wait_started_at) * 1000

    materialize_started_at = time.perf_counter()
    decoded_reqs = 0
    decoded_frames = 0
    emitted_chunks = 0
    for i, item in enumerate(pending.items):
        crop_start_frame = int(item.codes_window_cpu.shape[0]) - item.new_frames
        crop_start = (
            code2wav.output_num_samples(crop_start_frame)
            if crop_start_frame > 0
            else 0
        )
        crop_end = code2wav.output_num_samples(int(item.codes_window_cpu.shape[0]))
        expected_new_samples = crop_end - crop_start
        wav_new = pending.wav_chunks[i : i + 1, :, crop_start:crop_end]
        if wav_new.shape[-1] != expected_new_samples:
            raise RuntimeError(
                f"Code2Wav crop for {item.req.rid} produced "
                f"{wav_new.shape[-1]} samples, expected {expected_new_samples}"
            )
        item.req.talker_pcm16_chunks.append(wav_to_pcm16(wav_new))
        item.req.talker_last_decoded_frame = item.current_frame
        item.req.qwen3_omni_code2wav_pending = False
        decoded_reqs += 1
        emitted_chunks += 1
        decoded_frames += item.new_frames
        item.req.qwen3_omni_audio_frame_lag = (
            len(item.req.talker_output_codes) - item.current_frame
        )
        item.req.qwen3_omni_audio_phase = (
            Qwen3OmniAudioPhase.DONE
            if item.ready_state.flush
            else Qwen3OmniAudioPhase.CODE2WAV_QUEUED
        )
        consume_qwen3_omni_audio_ready_state(item.req, item.ready_state)
        if should_run_code2wav_phase(item.req):
            mark_qwen3_omni_code2wav_ready_state(
                item.req,
                time.perf_counter(),
                flush=False,
            )

    materialize_ms = (time.perf_counter() - materialize_started_at) * 1000
    return Code2WavCompletedWork(
        decoded_reqs=decoded_reqs,
        decoded_frames=decoded_frames,
        emitted_chunks=emitted_chunks,
        batch_groups=1,
        event_wait_ms=event_wait_ms,
        materialize_ms=materialize_ms,
    )


def _update_qwen3_omni_state(
    req: Req,
    logits_output,
    req_idx: int,
    thinker_token_id: int | None = None,
) -> None:
    if not hasattr(logits_output, "talker_needs_prefill"):
        return

    _update_qwen3_omni_prompt_accept_hidden(req, logits_output, req_idx)
    if logits_output.talker_needs_prefill:
        req.talker_needs_prefill = True
        req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.TALKER_PREFILL_QUEUED

    if thinker_token_id is not None:
        req.prev_sampled_thinker_token = thinker_token_id
        if (
            req.codec_eos_token_id is not None
            and not req.thinker_done
            and not is_qwen3_omni_thinker_stop_token(req, thinker_token_id)
        ):
            req.qwen3_omni_text_lag_tokens.append(thinker_token_id)
            if len(req.qwen3_omni_text_lag_tokens) > AUDIO_MAX_TEXT_LAG:
                req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.BACKPRESSURED
                raise RuntimeError(
                    f"Qwen3-Omni audio text lag exceeded {AUDIO_MAX_TEXT_LAG} "
                    f"tokens for request {req.rid}"
                )

    frame = None
    if logits_output.codec_frames is not None and req_idx < len(logits_output.codec_frames):
        frame = logits_output.codec_frames[req_idx]
    if frame is not None:
        req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.MTP_QUEUED
        if req.thinker_done:
            req.qwen3_omni_talker_post_thinker_started = True
        codec_token = qwen3_omni_codec_frame_id(frame)
        req.talker_codec_ids.append(codec_token)
        updated_repetition_masks = getattr(
            logits_output,
            "updated_talker_repetition_mask_list",
            None,
        )
        if (
            updated_repetition_masks is not None
            and req_idx < len(updated_repetition_masks)
            and updated_repetition_masks[req_idx] is not None
        ):
            req.talker_repetition_mask = updated_repetition_masks[req_idx].detach()
        next_step = req.talker_step + (
            1 if logits_output.talker_out_cache_loc_list is None else 0
        )
        if (
            codec_token == req.codec_eos_token_id
            or next_step >= 220
            or next_step % 32 == 0
        ):
            logger.info(
                "Qwen3-Omni talker codec: rid=%s step=%s codec=%s "
                "codec_eos=%s thinker_done=%s post_thinker=%s text_lag=%s "
                "tts_eos_pending=%s output_frames=%s frame_lag=%s",
                req.rid,
                next_step,
                codec_token,
                req.codec_eos_token_id,
                req.thinker_done,
                req.qwen3_omni_talker_post_thinker_started,
                len(req.qwen3_omni_text_lag_tokens),
                req.qwen3_omni_tts_eos_pending,
                len(req.talker_output_codes),
                req.qwen3_omni_audio_frame_lag,
            )
        if codec_token != req.codec_eos_token_id:
            if logits_output.prev_codec_input_embeds_list is None:
                raise RuntimeError(
                    "Qwen3-Omni Talker output requires code-predictor hidden-state "
                    "carry for the next codec input"
                )
            prev_codec_input_embed = logits_output.prev_codec_input_embeds_list[req_idx]
            if prev_codec_input_embed is None:
                raise RuntimeError(
                    "Qwen3-Omni Talker output missing code-predictor hidden-state "
                    f"carry for request index {req_idx}"
                )
            req.prev_codec_input_embed = prev_codec_input_embed.detach()
            req.talker_output_codes.append(frame)
            if req.qwen3_omni_text_lag_tokens:
                req.qwen3_omni_text_lag_tokens.popleft()
            elif req.thinker_done and req.qwen3_omni_tts_eos_pending:
                req.qwen3_omni_tts_eos_pending = False
            req.qwen3_omni_audio_frame_lag += 1
            if req.qwen3_omni_audio_frame_lag > AUDIO_MAX_AUDIO_FRAME_LAG:
                req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.BACKPRESSURED
                raise RuntimeError(
                    f"Qwen3-Omni audio frame lag exceeded "
                    f"{AUDIO_MAX_AUDIO_FRAME_LAG} frames for request {req.rid}"
                )
            if should_run_code2wav_phase(req):
                req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.CODE2WAV_QUEUED
                mark_qwen3_omni_code2wav_ready_state(
                    req,
                    time.perf_counter(),
                    flush=False,
                )

    if logits_output.tts_pad_embed is not None:
        if logits_output.tts_pad_embed.dim() == 2:
            req.tts_pad_embed = logits_output.tts_pad_embed[req_idx].detach()
        else:
            req.tts_pad_embed = logits_output.tts_pad_embed.detach()

    if logits_output.talker_out_cache_loc_list is not None:
        if req_idx < len(logits_output.talker_out_cache_loc_list):
            kv_locs = logits_output.talker_out_cache_loc_list[req_idx]
            if kv_locs is not None:
                req.talker_kv_cache_locs = kv_locs
                req.talker_prefill_len = len(kv_locs)
                req.talker_needs_prefill = False
                req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.TALKER_DECODE_READY
    talker_prefill_input_ids = getattr(
        logits_output, "talker_prefill_input_ids_list", None
    )
    if (
        talker_prefill_input_ids is not None
        and req_idx < len(talker_prefill_input_ids)
        and talker_prefill_input_ids[req_idx] is not None
    ):
        req.talker_prefill_input_ids = talker_prefill_input_ids[req_idx].detach()

    talker_user_prefill_lens = getattr(
        logits_output, "talker_user_prefill_len_list", None
    )
    if (
        talker_user_prefill_lens is not None
        and req_idx < len(talker_user_prefill_lens)
        and talker_user_prefill_lens[req_idx] is not None
    ):
        req.talker_user_prefill_len = int(talker_user_prefill_lens[req_idx])

    talker_position_deltas = getattr(
        logits_output, "talker_mrope_position_delta_list", None
    )
    if talker_position_deltas is not None and req_idx < len(talker_position_deltas):
        delta = talker_position_deltas[req_idx]
        if delta is not None:
            req.talker_mrope_position_delta = int(delta.detach().cpu().item())

    if logits_output.updated_talker_kv_locs_list is not None:
        if req_idx < len(logits_output.updated_talker_kv_locs_list):
            kv_locs = logits_output.updated_talker_kv_locs_list[req_idx]
            if kv_locs is not None:
                req.talker_kv_cache_locs = kv_locs
                req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.TALKER_DECODE_READY

    if frame is not None and logits_output.talker_out_cache_loc_list is None:
        req.talker_step += 1


def _update_qwen3_omni_prompt_accept_hidden(
    req: Req,
    logits_output,
    req_idx: int,
) -> None:
    prompt_accept_hidden_states = getattr(
        logits_output, "prompt_accept_hidden_states", None
    )
    if (
        prompt_accept_hidden_states is not None
        and req_idx < len(prompt_accept_hidden_states)
        and prompt_accept_hidden_states[req_idx] is not None
    ):
        prompt_hidden = prompt_accept_hidden_states[req_idx].detach()
        req.qwen3_omni_prompt_accept_hidden = prompt_hidden


def release_skipped_qwen3_omni_audio_prefill_locs(
    logits_output,
    req_idx: int,
    talker_allocator,
) -> bool:
    talker_locs = getattr(logits_output, "talker_out_cache_loc_list", None)
    if talker_locs is None or talker_locs[req_idx] is None:
        return False

    talker_allocator.free(talker_locs[req_idx])
    talker_locs[req_idx] = None
    return True


def release_qwen3_omni_component_kv(req: Req, model_runner) -> bool:
    if req.codec_eos_token_id is None or req.talker_kv_cache_locs is None:
        return False

    talker_allocator = model_runner.qwen3_omni_token_to_kv_pool_allocators["talker"]
    keep_len = (
        0
        if isinstance(req.finished_reason, FINISH_ABORT)
        else req.talker_user_prefill_len
    )
    if keep_len > 0:
        if keep_len > len(req.talker_kv_cache_locs):
            raise RuntimeError(
                f"Qwen3-Omni Talker user-prefix cache length exceeds KV length "
                f"for request {req.rid}: keep_len={keep_len}, "
                f"kv_len={len(req.talker_kv_cache_locs)}"
            )
        if req.talker_prefill_input_ids is None:
            raise RuntimeError(
                "Qwen3-Omni Talker user-prefix cache requires prefill input ids "
                f"for request {req.rid}"
            )
        if keep_len > len(req.talker_prefill_input_ids):
            raise RuntimeError(
                f"Qwen3-Omni Talker user-prefix cache length exceeds input-id "
                f"length for request {req.rid}: keep_len={keep_len}, "
                f"ids_len={len(req.talker_prefill_input_ids)}"
            )

        # Preserve only the user-prefix part for the next turn. The final
        # assistant bootstrap and generated codec-token KV are response-local.
        if keep_len < len(req.talker_kv_cache_locs):
            talker_allocator.free(req.talker_kv_cache_locs[keep_len:])
        req.talker_kv_cache_locs = req.talker_kv_cache_locs[:keep_len].detach()
        req.talker_prefill_input_ids = req.talker_prefill_input_ids[:keep_len].detach()
        req.talker_prefill_len = keep_len
        return True

    talker_allocator.free(req.talker_kv_cache_locs)
    req.talker_kv_cache_locs = None
    req.talker_prefill_len = 0
    req.talker_user_prefill_len = 0
    req.talker_prefill_input_ids = None
    return True


def release_qwen3_omni_post_thinker_kv(req: Req, tree_cache) -> bool:
    if req.codec_eos_token_id is None or not req.thinker_done:
        return False

    thinker_kv_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
    if thinker_kv_len < req.cache_protected_len:
        raise RuntimeError(
            f"Qwen3-Omni request {req.rid} has protected Thinker KV past the "
            f"actual Thinker token space: protected={req.cache_protected_len}, "
            f"actual={thinker_kv_len}"
        )
    if thinker_kv_len > req.kv_committed_len:
        raise RuntimeError(
            f"Qwen3-Omni request {req.rid} committed fewer Thinker KV slots than "
            f"actual Thinker tokens: committed={req.kv_committed_len}, "
            f"actual={thinker_kv_len}"
        )
    if thinker_kv_len == req.kv_allocated_len:
        return False

    if req.req_pool_idx is None:
        raise RuntimeError(
            f"Qwen3-Omni request {req.rid} lost req_pool_idx before Thinker KV "
            "tail release"
        )

    page_size = getattr(tree_cache.token_to_kv_pool_allocator, "page_size", None)
    if page_size is None:
        page_size = tree_cache.page_size
    free_start = thinker_kv_len
    if page_size > 1:
        free_start = ((thinker_kv_len + page_size - 1) // page_size) * page_size
    if free_start < req.kv_allocated_len:
        tail = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            free_start : req.kv_allocated_len
        ]
        tree_cache.token_to_kv_pool_allocator.free(tail)
    req.kv_committed_len = thinker_kv_len
    req.kv_allocated_len = min(req.kv_allocated_len, max(thinker_kv_len, free_start))
    return True


def filter_uncommitted_qwen3_omni_code_predictor_inputs(
    logits_output,
    reqs: List[Req],
):
    """Drop MTP work for Talker prefill outputs that will not reach Req state."""
    pending = getattr(logits_output, "code_predictor_inputs", None)
    if pending is None:
        return None

    keep_local_indices = []
    keep_request_indices = []
    for local_i, req_idx in enumerate(pending.request_indices):
        req = reqs[req_idx]
        prefill_audio_done = qwen3_omni_prefill_audio_done(req)
        if prefill_audio_done:
            keep_local_indices.append(local_i)
            keep_request_indices.append(req_idx)

    if len(keep_local_indices) == len(pending.request_indices):
        return pending

    if not keep_local_indices:
        logits_output.code_predictor_inputs = None
        return None

    keep = torch.tensor(
        keep_local_indices,
        dtype=torch.long,
        device=pending.input_embeds.device,
    )
    logits_output.code_predictor_inputs = type(pending)(
        request_indices=keep_request_indices,
        input_embeds=pending.input_embeds.index_select(0, keep),
        codec_tokens=pending.codec_tokens.index_select(0, keep),
    )
    return logits_output.code_predictor_inputs


def filter_qwen3_omni_code_predictor_inputs_by_request_indices(
    logits_output,
    keep_request_indices: List[int],
):
    pending = getattr(logits_output, "code_predictor_inputs", None)
    if pending is None:
        raise RuntimeError("Qwen3-Omni code_predictor phase has no pending inputs")

    keep_set = set(keep_request_indices)
    keep_local_indices = []
    filtered_request_indices = []
    for local_i, req_idx in enumerate(pending.request_indices):
        if req_idx in keep_set:
            keep_local_indices.append(local_i)
            filtered_request_indices.append(req_idx)

    if filtered_request_indices != keep_request_indices:
        raise RuntimeError(
            "Qwen3-Omni code_predictor cleanup request indices do not match "
            "pending Talker output"
        )

    if len(keep_local_indices) == len(pending.request_indices):
        return pending

    keep = torch.tensor(
        keep_local_indices,
        dtype=torch.long,
        device=pending.input_embeds.device,
    )
    logits_output.code_predictor_inputs = type(pending)(
        request_indices=filtered_request_indices,
        input_embeds=pending.input_embeds.index_select(0, keep),
        codec_tokens=pending.codec_tokens.index_select(0, keep),
    )
    return logits_output.code_predictor_inputs


def filter_qwen3_omni_committed_talker_prefills(
    reqs: List[Req],
    request_indices: List[int],
) -> List[int]:
    committed_indices = []
    for req_idx in request_indices:
        req = reqs[req_idx]
        prefill_audio_done = qwen3_omni_prefill_audio_done(req)
        if prefill_audio_done:
            committed_indices.append(req_idx)
    return committed_indices


def check_qwen3_omni_finish(req: Req, new_accepted_tokens: List[int]) -> bool:
    if req.codec_eos_token_id is None:
        return False

    if req.thinker_done:
        custom = req.sampling_params.custom_params
        talker_limit = (
            custom.get(
                "talker_max_new_tokens", AUDIO_DEFAULT_TALKER_MAX_NEW_TOKENS
            )
            if custom
            else AUDIO_DEFAULT_TALKER_MAX_NEW_TOKENS
        )
        if req.talker_step >= talker_limit:
            req.finished_reason = FINISH_LENGTH(length=talker_limit)
            req.finished_len = len(req.output_ids)
            logger.info(
                "Qwen3-Omni audio finish length: rid=%s talker_step=%s "
                "talker_limit=%s last_codec=%s codec_eos=%s "
                "post_thinker=%s text_lag=%s tts_eos_pending=%s",
                req.rid,
                req.talker_step,
                talker_limit,
                req.talker_codec_ids[-1] if req.talker_codec_ids else None,
                req.codec_eos_token_id,
                req.qwen3_omni_talker_post_thinker_started,
                len(req.qwen3_omni_text_lag_tokens),
                req.qwen3_omni_tts_eos_pending,
            )
            return True

        if (
            req.talker_codec_ids
            and req.talker_codec_ids[-1] == req.codec_eos_token_id
            and req.qwen3_omni_talker_post_thinker_started
        ):
            req.finished_reason = FINISH_MATCHED_TOKEN(matched=req.codec_eos_token_id)
            req.finished_len = len(req.output_ids)
            logger.info(
                "Qwen3-Omni audio finish codec-eos: rid=%s talker_step=%s "
                "codec_eos=%s text_lag=%s tts_eos_pending=%s",
                req.rid,
                req.talker_step,
                req.codec_eos_token_id,
                len(req.qwen3_omni_text_lag_tokens),
                req.qwen3_omni_tts_eos_pending,
            )
            return True

        return True

    matched_eos = False
    if req.eos_token_ids:
        matched_eos = any(token_id in req.eos_token_ids for token_id in new_accepted_tokens)
    if (
        not matched_eos
        and req.sampling_params.stop_token_ids
        and any(token_id in req.sampling_params.stop_token_ids for token_id in new_accepted_tokens)
    ):
        matched_eos = True
    if not matched_eos and req.tokenizer is not None:
        additional_stop_token_ids = req.tokenizer.additional_stop_token_ids
        matched_eos = any(
            token_id == req.tokenizer.eos_token_id
            or (
                additional_stop_token_ids is not None
                and token_id in additional_stop_token_ids
            )
            for token_id in new_accepted_tokens
        )

    if matched_eos:
        req.thinker_done = True
        req.qwen3_omni_tts_eos_pending = True
        return True

    return False


class SchedulerQwen3OmniMixin:
    """Sibling mixin holding all Qwen3-Omni audio scheduler logic.

    Methods access scheduler state via ``self`` (Scheduler attributes are
    available because both mixins share the Scheduler instance).
    """

    def _init_qwen3_omni_state(self: Scheduler) -> None:
        self.qwen3_omni_audio_phase_queue: Deque[Qwen3OmniAudioPhaseItem] = deque()
        self.qwen3_omni_code2wav_pending_work: List[Any] = []
        self.qwen3_omni_code2wav_stream: Optional[CudaStream] = None
        self.qwen3_omni_code2wav_stream_prewarmed = False
        self.qwen3_omni_code2wav_event_pool: List[Any] = []
        self.qwen3_omni_last_runtime_snapshot_at = 0.0

    def _get_code2wav_model(self: Scheduler):
        if getattr(self.model_config.hf_config, "model_type", None) != "qwen3_omni_moe":
            return None
        return self.tp_worker.model_runner.model.code2wav

    def _qwen3_omni_component_kv_stats(self: Scheduler) -> str:
        allocators = (
            self.tp_worker.model_runner.qwen3_omni_token_to_kv_pool_allocators
        )
        stats = []
        for name, allocator in sorted(allocators.items()):
            free = allocator.available_size()
            stats.append(f"{name}:{allocator.size - free}/{free}")
        return ",".join(stats)

    def _qwen3_omni_runtime_snapshot(self: Scheduler) -> dict[str, Any]:
        active_audio_turns, queued_audio_turns = self._qwen3_omni_audio_turn_stats()
        component_allocators = (
            self.tp_worker.model_runner.qwen3_omni_token_to_kv_pool_allocators
        )
        component_kv = {}
        for name, allocator in sorted(component_allocators.items()):
            free = allocator.available_size()
            component_kv[name] = {
                "used": allocator.size - free,
                "free": free,
                "size": allocator.size,
            }

        stored_talker_user_prefix_turns = 0
        stored_talker_user_prefix_tokens = 0
        for session_id in self.audio_sessions:
            streaming_session = self.session_controller.get(session_id)
            if streaming_session is None:
                continue
            talker_prefix = streaming_session.get_component_kv(
                "qwen3_omni_talker_user_prefix"
            )
            if talker_prefix is not None:
                stored_talker_user_prefix_turns += 1
                stored_talker_user_prefix_tokens += len(talker_prefix.kv_cache_locs)

        active_talker_user_prefix_turns = 0
        active_talker_user_prefix_tokens = 0
        for req in self.audio_turn_reqs.values():
            if req.talker_kv_cache_locs is not None:
                active_talker_user_prefix_turns += 1
                active_talker_user_prefix_tokens += len(req.talker_kv_cache_locs)

        return {
            "audio_sessions": len(self.audio_sessions),
            "completed_audio_turns": sum(
                session.completed_turn_count
                for session in self.audio_sessions.values()
            ),
            "active_audio_turns": active_audio_turns,
            "queued_audio_turns": queued_audio_turns,
            "waiting_queue": len(self.waiting_queue),
            "running_batch": len(self.running_batch.reqs),
            "audio_phase_queue": len(self.qwen3_omni_audio_phase_queue),
            "code2wav_pending": len(self.qwen3_omni_code2wav_pending_work),
            "tree_cache": {
                "type": type(self.tree_cache).__name__,
                "disable": getattr(self.tree_cache, "disable", None),
                "available": self.token_to_kv_pool_allocator.available_size(),
                "used": (
                    self.token_to_kv_pool_allocator.size
                    - self.token_to_kv_pool_allocator.available_size()
                ),
                "size": self.token_to_kv_pool_allocator.size,
                "evictable": self.tree_cache.evictable_size(),
                "protected": self.tree_cache.protected_size(),
                "full_evictable": self.tree_cache.full_evictable_size(),
                "full_protected": self.tree_cache.full_protected_size(),
            },
            "streaming_session": {
                "enabled": self.server_args.enable_streaming_session,
                "sessions": self._streaming_session_count(),
                "held_tokens": self._session_held_tokens(),
                "held_req_slots": self._session_held_req_count(),
            },
            "talker_user_prefix_cache": {
                "stored_turns": stored_talker_user_prefix_turns,
                "stored_tokens": stored_talker_user_prefix_tokens,
                "active_turns": active_talker_user_prefix_turns,
                "active_tokens": active_talker_user_prefix_tokens,
            },
            "component_kv": component_kv,
        }

    def _log_qwen3_omni_runtime_snapshot(
        self: Scheduler,
        event: str,
    ) -> None:
        logger.info(
            "Qwen3-Omni runtime snapshot: event=%s stats=%s",
            event,
            json.dumps(self._qwen3_omni_runtime_snapshot(), sort_keys=True),
        )

    def _maybe_log_qwen3_omni_runtime_snapshot(self: Scheduler) -> None:
        now = time.perf_counter()
        if (
            now - self.qwen3_omni_last_runtime_snapshot_at
            < QWEN3_OMNI_RUNTIME_SNAPSHOT_INTERVAL_S
        ):
            return
        self.qwen3_omni_last_runtime_snapshot_at = now
        self._log_qwen3_omni_runtime_snapshot("periodic")

    def _qwen3_omni_audio_turn_stats(self: Scheduler) -> Tuple[int, int]:
        active_audio_turns = len(self.audio_turn_reqs)
        queued_audio_turns = sum(
            1 for req in self.waiting_queue if req.rid in self.audio_turn_reqs
        )
        return active_audio_turns, queued_audio_turns

    def _log_qwen3_omni_audio_phase_drained(
        self: Scheduler,
        scheduler_action: str,
        phase: str,
        fields: dict,
    ) -> None:
        active_audio_turns, queued_audio_turns = self._qwen3_omni_audio_turn_stats()
        field_str = ", ".join(f"{k}={v}" for k, v in fields.items())
        logger.info(
            "Qwen3-Omni audio phase drained: lane=audio, "
            "scheduler_action=%s, phase=%s, %s, "
            "component_kv_used_free=%s, active_audio_turns=%s, "
            "queued_audio_turns=%s",
            scheduler_action,
            phase,
            field_str,
            self._qwen3_omni_component_kv_stats(),
            active_audio_turns,
            queued_audio_turns,
        )

    def _build_qwen3_omni_audio_phase_item(
        self: Scheduler,
        result: GenerationBatchResult,
        forward_batch,
        next_token_ids,
        return_logprob: bool,
        phase: str,
        request_indices: List[int],
        reqs: Optional[List[Req]],
    ) -> Qwen3OmniAudioPhaseItem:
        if reqs is None:
            raise RuntimeError("Qwen3-Omni audio phase items require request rids")
        if phase not in QWEN3_OMNI_AUDIO_READY_BY_PHASE:
            raise RuntimeError(f"Unknown Qwen3-Omni audio phase {phase}")
        ready_at = time.perf_counter()
        ready_states = [
            mark_qwen3_omni_audio_ready_state(
                reqs[req_idx],
                phase,
                req_idx,
                ready_at,
            )
            for req_idx in request_indices
        ]
        sample_codec = True
        if phase == "talker_prefill":
            rebuild_rows = [
                getattr(reqs[req_idx], "qwen3_omni_rebuild_request", False)
                for req_idx in request_indices
            ]
            has_rebuild = any(rebuild_rows)
            if has_rebuild:
                if not all(rebuild_rows):
                    raise RuntimeError(
                        "Qwen3-Omni talker_prefill phase cannot mix rollback "
                        "rebuild rows with normal audio rows"
                    )
                sample_codec = False
            else:
                committed_audio = [
                    qwen3_omni_prefill_audio_done(reqs[req_idx])
                    for req_idx in request_indices
                ]
                sample_codec = all(committed_audio)
                if any(committed_audio) and not sample_codec:
                    raise RuntimeError(
                        "Qwen3-Omni talker_prefill phase cannot mix committed "
                        "and streaming audio input"
                    )
        payload = build_qwen3_omni_audio_phase_payload(
            phase=phase,
            forward_batch=forward_batch,
            logits_output=result.logits_output,
            next_token_ids=next_token_ids,
            request_indices=request_indices,
            sample_codec=sample_codec,
        )
        if isinstance(payload, Qwen3OmniTalkerPrefillPayload):
            speakers = []
            for req in reqs:
                if not req.speaker:
                    raise RuntimeError(
                        f"Qwen3-Omni talker_prefill requires speaker state for "
                        f"request {req.rid}"
                    )
                speakers.append(req.speaker)
            payload = replace(
                payload,
                speakers=speakers,
                sample_codec=sample_codec,
            )
        return Qwen3OmniAudioPhaseItem(
            logits_output=result.logits_output,
            forward_batch=snapshot_qwen3_omni_audio_forward_batch(forward_batch),
            next_token_ids=_clone_qwen3_omni_value(next_token_ids),
            reqs=reqs,
            return_logprob=return_logprob,
            phase=phase,
            request_indices=request_indices,
            request_rids=[reqs[req_idx].rid for req_idx in request_indices],
            payload=payload,
            ready_states=ready_states,
            queued_at=min(state.ready_at for state in ready_states),
        )

    def _purge_qwen3_omni_audio_phase_queue(self: Scheduler, abort_req: AbortReq):
        if not self.qwen3_omni_audio_phase_queue:
            return
        if abort_req.abort_all:
            for item in self.qwen3_omni_audio_phase_queue:
                filter_qwen3_omni_audio_phase_item_by_request_indices(item, [])
            self.qwen3_omni_audio_phase_queue.clear()
            return

        retained_items = []
        for item in self.qwen3_omni_audio_phase_queue:
            if len(item.request_indices) != len(item.request_rids):
                raise RuntimeError(
                    "Qwen3-Omni audio phase item has mismatched indices and rids"
                )

            kept_pairs = [
                (req_idx, rid)
                for req_idx, rid in zip(item.request_indices, item.request_rids)
                if not rid.startswith(abort_req.rid)
            ]
            keep_request_indices = [req_idx for req_idx, _ in kept_pairs]
            if filter_qwen3_omni_audio_phase_item_by_request_indices(
                item,
                keep_request_indices,
            ):
                retained_items.append(item)

        self.qwen3_omni_audio_phase_queue.clear()
        self.qwen3_omni_audio_phase_queue.extend(retained_items)

    def _build_qwen3_omni_audio_phase_items(
        self: Scheduler,
        result: GenerationBatchResult,
        return_logprob: bool,
        reqs: Optional[List[Req]] = None,
    ) -> List[Qwen3OmniAudioPhaseItem]:
        forward_batch = result.qwen3_omni_audio_forward_batch
        assert forward_batch is not None
        next_token_ids = result.next_token_ids
        assert next_token_ids is not None
        model_states = forward_batch.model_specific_states or {}
        batch_size = forward_batch.batch_size

        if getattr(result.logits_output, "talker_needs_prefill", False):
            request_indices = list(range(batch_size))
            if reqs is not None:
                request_indices = [
                    req_idx
                    for req_idx in request_indices
                    if reqs[req_idx].inflight_middle_chunks <= 0
                ]
            if reqs is not None:
                request_indices = filter_qwen3_omni_committed_talker_prefills(
                    reqs,
                    request_indices,
                )
            if not request_indices:
                return []
            if reqs is None:
                raise RuntimeError("Qwen3-Omni talker_prefill requires request state")
            rebuild_indices = [
                req_idx
                for req_idx in request_indices
                if getattr(reqs[req_idx], "qwen3_omni_rebuild_request", False)
            ]
            normal_indices = [
                req_idx
                for req_idx in request_indices
                if not getattr(reqs[req_idx], "qwen3_omni_rebuild_request", False)
            ]
            items = [
                self._build_qwen3_omni_audio_phase_item(
                    result=result,
                    forward_batch=forward_batch,
                    next_token_ids=next_token_ids,
                    return_logprob=return_logprob,
                    phase="talker_prefill",
                    request_indices=indices,
                    reqs=reqs,
                )
                for indices in (rebuild_indices,)
                if indices
            ]
            items.extend(
                self._build_qwen3_omni_audio_phase_item(
                    result=result,
                    forward_batch=forward_batch,
                    next_token_ids=next_token_ids,
                    return_logprob=return_logprob,
                    phase="talker_prefill",
                    request_indices=[req_idx],
                    reqs=reqs,
                )
                for req_idx in normal_indices
            )
            return items

        needs_prefill = model_states.get("talker_needs_prefill")
        can_decode = model_states.get("talker_can_decode")
        if can_decode is not None:
            can_decode_indices = [
                i for i, can_decode_i in enumerate(can_decode.tolist()) if can_decode_i
            ]
        else:
            can_decode_indices = []
        items: List[Qwen3OmniAudioPhaseItem] = []
        if needs_prefill is not None and needs_prefill.any():
            prefill_indices = [
                i for i, needs_prefill_i in enumerate(needs_prefill.tolist())
                if needs_prefill_i
            ]
            original_prefill_set = set(prefill_indices)
            if reqs is not None:
                prefill_indices = [
                    req_idx
                    for req_idx in prefill_indices
                    if reqs[req_idx].inflight_middle_chunks <= 0
                ]
            if reqs is not None:
                prefill_indices = filter_qwen3_omni_committed_talker_prefills(
                    reqs,
                    prefill_indices,
                )
            if prefill_indices:
                items.extend(
                    self._build_qwen3_omni_audio_phase_item(
                        result=result,
                        forward_batch=forward_batch,
                        next_token_ids=next_token_ids,
                        return_logprob=return_logprob,
                        phase="talker_prefill",
                        request_indices=[req_idx],
                        reqs=reqs,
                    )
                    for req_idx in prefill_indices
                )
            decode_indices = [
                i
                for i in can_decode_indices
                if i not in original_prefill_set
            ]
            if decode_indices:
                items.extend(
                    self._build_qwen3_omni_audio_phase_item(
                        result=result,
                        forward_batch=forward_batch,
                        next_token_ids=next_token_ids,
                        return_logprob=return_logprob,
                        phase="talker_decode",
                        request_indices=decode_chunk,
                        reqs=reqs,
                    )
                    for decode_chunk in _qwen3_omni_talker_decode_batch_chunks(
                        decode_indices
                    )
                )
            return items

        if can_decode_indices:
            return [
                self._build_qwen3_omni_audio_phase_item(
                    result=result,
                    forward_batch=forward_batch,
                    next_token_ids=next_token_ids,
                    return_logprob=return_logprob,
                    phase="talker_decode",
                    request_indices=decode_chunk,
                    reqs=reqs,
                )
                for decode_chunk in _qwen3_omni_talker_decode_batch_chunks(
                    can_decode_indices
                )
            ]
        return []

    def _enqueue_qwen3_omni_audio_phase_queue(
        self: Scheduler,
        result: GenerationBatchResult,
        return_logprob: bool,
        reqs: Optional[List[Req]] = None,
    ) -> List[Qwen3OmniAudioPhaseItem]:
        if getattr(result, "qwen3_omni_audio_forward_batch", None) is None:
            return []

        phase_items = self._build_qwen3_omni_audio_phase_items(
            result, return_logprob, reqs
        )
        if not phase_items:
            result.qwen3_omni_audio_forward_batch = None
            return []

        self.qwen3_omni_audio_phase_queue.extend(phase_items)
        result.qwen3_omni_audio_forward_batch = None
        return phase_items

    def _apply_qwen3_omni_audio_phase_output(
        self: Scheduler,
        item: Qwen3OmniAudioPhaseItem,
    ) -> List[Req]:
        logits_output = item.logits_output
        affected_reqs = []

        for req_idx in item.request_indices:
            req = item.reqs[req_idx]
            prefill_audio_done = qwen3_omni_prefill_audio_done(req)
            if item.phase == "talker_prefill" and not prefill_audio_done:
                _update_qwen3_omni_state(req, logits_output, req_idx)
                req.talker_needs_prefill = True
                req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.TALKER_PREFILL_QUEUED
                ready_state = item.ready_states[item.request_indices.index(req_idx)]
                req.qwen3_omni_partial_talker_prefill_chunk_idx = max(
                    req.qwen3_omni_partial_talker_prefill_chunk_idx,
                    ready_state.audio_input_chunk_idx,
                )
                continue

            _update_qwen3_omni_state(req, logits_output, req_idx)
            frame = None
            if (
                logits_output.codec_frames is not None
                and req_idx < len(logits_output.codec_frames)
            ):
                frame = logits_output.codec_frames[req_idx]
            if (
                item.phase == "talker_prefill"
                and getattr(req, "qwen3_omni_rebuild_request", False)
            ):
                if frame is not None:
                    raise RuntimeError(
                        "Qwen3-Omni rollback rebuild Talker prefill produced "
                        f"codec audio for request {req.rid}"
                    )
                req.finished_reason = FINISH_LENGTH(length=0)
                req.finished_len = len(req.output_ids)
                self._handle_finished_req(req, req_idx, logits_output)
                affected_reqs.append(req)
                continue
            if frame is not None:
                req.update_finish_state()
                if req.finished():
                    self._handle_finished_req(req, req_idx, logits_output)
            affected_reqs.append(req)

        if not affected_reqs:
            return []
        return affected_reqs

    def _set_qwen3_omni_code_predictor_rows(
        self: Scheduler,
        item: Qwen3OmniAudioPhaseItem,
        codec_frames: List[List[int]],
        prev_codec_input_embeds: torch.Tensor,
    ) -> None:
        batch_size = len(item.reqs)
        if len(codec_frames) != len(item.request_indices):
            raise RuntimeError(
                "Qwen3-Omni batched code_predictor returned the wrong "
                "number of codec frames"
            )
        if prev_codec_input_embeds.shape[0] != len(item.request_indices):
            raise RuntimeError(
                "Qwen3-Omni batched code_predictor returned the wrong "
                "number of carry embeddings"
            )

        logits_output = item.logits_output
        output_codec_frames = getattr(logits_output, "codec_frames", None)
        if output_codec_frames is None:
            output_codec_frames = [None] * batch_size
        else:
            output_codec_frames = list(output_codec_frames)
        output_prev_embeds = getattr(logits_output, "prev_codec_input_embeds_list", None)
        if output_prev_embeds is None:
            output_prev_embeds = [None] * batch_size
        else:
            output_prev_embeds = list(output_prev_embeds)
        if len(output_codec_frames) != batch_size or len(output_prev_embeds) != batch_size:
            raise RuntimeError(
                "Qwen3-Omni code_predictor output list length does not match "
                "the original request batch"
            )

        for local_i, req_idx in enumerate(item.request_indices):
            output_codec_frames[req_idx] = codec_frames[local_i]
            output_prev_embeds[req_idx] = prev_codec_input_embeds[local_i].detach()
        logits_output.codec_frames = output_codec_frames
        logits_output.prev_codec_input_embeds_list = output_prev_embeds
        logits_output.code_predictor_inputs = None

    def _drain_qwen3_omni_code_predictor_items_batched(
        self: Scheduler,
        items: List[Qwen3OmniAudioPhaseItem],
        apply_outputs: bool,
        applied_reqs_by_rid: dict,
        applied_return_logprob_by_rid: dict,
    ) -> None:
        payloads = []
        for item in items:
            if item.phase != "code_predictor":
                raise RuntimeError(
                    "Qwen3-Omni code_predictor batch contains a non-code_predictor item"
                )
            if not isinstance(item.payload, Qwen3OmniCodePredictorPayload):
                raise RuntimeError(
                    "Qwen3-Omni code_predictor phase requires owned payload"
                )
            payloads.append(item.payload)

        total_batch_size = sum(len(item.request_indices) for item in items)
        if total_batch_size == 0:
            raise RuntimeError("Qwen3-Omni code_predictor batch is empty")

        input_embeds = torch.cat([payload.input_embeds for payload in payloads], dim=0)
        codec_tokens = torch.cat([payload.codec_tokens for payload in payloads], dim=0)
        payload = Qwen3OmniCodePredictorPayload(
            request_indices=list(range(total_batch_size)),
            input_embeds=input_embeds,
            codec_tokens=codec_tokens,
        )

        start_time = time.perf_counter()
        codec_frames, prev_codec_input_embeds = (
            self.tp_worker.model_runner.model.run_qwen3_omni_code_predictor_payload_batch(
                items[0].forward_batch,
                payload,
            )
        )
        if len(codec_frames) != total_batch_size:
            raise RuntimeError(
                "Qwen3-Omni batched code_predictor returned a mismatched "
                "codec frame batch"
            )

        offset = 0
        for item in items:
            next_offset = offset + len(item.request_indices)
            self._set_qwen3_omni_code_predictor_rows(
                item,
                codec_frames[offset:next_offset],
                prev_codec_input_embeds[offset:next_offset],
            )
            for req_idx, ready_state in zip(item.request_indices, item.ready_states):
                consume_qwen3_omni_audio_ready_state(item.reqs[req_idx], ready_state)
            offset = next_offset

        now = time.perf_counter()
        queue_wait_ms = max(
            ready_state.queue_wait_ms(now)
            for item in items
            for ready_state in item.ready_states
        )
        deadline_miss_ms = max(
            qwen3_omni_audio_deadline_miss_ms(ready_state, now)
            for item in items
            for ready_state in item.ready_states
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        fields = {
            "source_forward_mode": ",".join(
                sorted({item.forward_batch.forward_mode.name for item in items})
            ),
            "batch_size": total_batch_size,
            "prefill_reqs": 0,
            "decode_reqs": 0,
            "code_predictor_reqs": total_batch_size,
            "queue_wait_ms": f"{queue_wait_ms:.2f}",
            "deadline_miss_ms": f"{deadline_miss_ms:.2f}",
            "wall_ms": f"{elapsed_ms:.2f}",
            "item_count": len(items),
        }
        apply_elapsed_ms = 0.0
        if apply_outputs:
            apply_start_time = time.perf_counter()
            for item in items:
                for req in self._apply_qwen3_omni_audio_phase_output(item):
                    if (
                        req.rid in applied_return_logprob_by_rid
                        and applied_return_logprob_by_rid[req.rid]
                        != item.return_logprob
                    ):
                        raise RuntimeError(
                            "Qwen3-Omni deferred audio output saw mixed "
                            f"return_logprob values for request {req.rid}"
                        )
                    applied_reqs_by_rid[req.rid] = req
                    applied_return_logprob_by_rid[req.rid] = item.return_logprob
            apply_elapsed_ms = (time.perf_counter() - apply_start_time) * 1000
        fields["apply_ms"] = f"{apply_elapsed_ms:.2f}"
        fields["wall_ms"] = f"{(time.perf_counter() - start_time) * 1000:.2f}"
        self._log_qwen3_omni_audio_phase_drained(
            scheduler_action="drain_pending_audio",
            phase="code_predictor",
            fields=fields,
        )

    def _drain_pending_qwen3_omni_audio_phase_queue(
        self: Scheduler,
        apply_outputs: bool = False,
        max_items: Optional[int] = None,
        drain_dependent_code_predictor: bool = False,
    ) -> int:
        applied_reqs_by_rid = {}
        applied_return_logprob_by_rid = {}
        drained_items = 0
        dependent_code_predictor_budget = 0
        while self.qwen3_omni_audio_phase_queue:
            if max_items is not None and drained_items >= max_items:
                if (
                    dependent_code_predictor_budget <= 0
                    or self.qwen3_omni_audio_phase_queue[0].phase != "code_predictor"
                ):
                    break
                dependent_code_predictor_budget -= 1
            item = self.qwen3_omni_audio_phase_queue.popleft()
            validate_qwen3_omni_audio_phase_item_ready_states(item)
            if item.phase == "code_predictor":
                if not _qwen3_omni_audio_items_fit_frame_budget([item]):
                    self.qwen3_omni_audio_phase_queue.appendleft(item)
                    break
                code_predictor_items = [item]
                while (
                    self.qwen3_omni_audio_phase_queue
                    and self.qwen3_omni_audio_phase_queue[0].phase == "code_predictor"
                ):
                    # CodePredictor emits codec frames, so a large same-request
                    # batch must yield to Code2Wav at chunk boundaries instead
                    # of outrunning the audio stream and tripping frame lag.
                    if not _qwen3_omni_can_extend_frame_producing_batch(
                        code_predictor_items,
                        self.qwen3_omni_audio_phase_queue[0],
                    ):
                        break
                    code_predictor_item = self.qwen3_omni_audio_phase_queue.popleft()
                    validate_qwen3_omni_audio_phase_item_ready_states(
                        code_predictor_item
                    )
                    code_predictor_items.append(code_predictor_item)
                self._drain_qwen3_omni_code_predictor_items_batched(
                    code_predictor_items,
                    apply_outputs,
                    applied_reqs_by_rid,
                    applied_return_logprob_by_rid,
                )
                drained_items += len(code_predictor_items)
                continue

            batched_items = [item]
            if item.phase == "talker_prefill":
                while (
                    self.qwen3_omni_audio_phase_queue
                    and _qwen3_omni_talker_prefill_items_compatible(
                        item, self.qwen3_omni_audio_phase_queue[0]
                    )
                    # A committed Talker prefill samples the first codec
                    # token through a dependent CodePredictor item. Treat
                    # those rows as frame-producing for batching purposes so
                    # we do not create an oversized CodePredictor item before
                    # Code2Wav has a chance to reclaim frame lag.
                    and _qwen3_omni_can_extend_talker_prefill_batch(
                        batched_items,
                        self.qwen3_omni_audio_phase_queue[0],
                    )
                ):
                    talker_prefill_item = self.qwen3_omni_audio_phase_queue.popleft()
                    validate_qwen3_omni_audio_phase_item_ready_states(
                        talker_prefill_item
                    )
                    batched_items.append(talker_prefill_item)
                if len(batched_items) > 1:
                    item = _build_qwen3_omni_talker_prefill_batch_item(batched_items)
                    validate_qwen3_omni_audio_phase_item_ready_states(item)
            elif item.phase == "talker_decode":
                while (
                    self.qwen3_omni_audio_phase_queue
                    and _qwen3_omni_talker_decode_items_compatible(
                        item, self.qwen3_omni_audio_phase_queue[0]
                    )
                    # Talker decode creates one dependent CodePredictor frame
                    # per row. Keep per-request bursts to a Code2Wav chunk so
                    # the semantic frame order stays intact while Code2Wav can
                    # reclaim lag between bursts.
                    and _qwen3_omni_can_extend_talker_decode_batch(
                        batched_items,
                        self.qwen3_omni_audio_phase_queue[0],
                    )
                ):
                    talker_decode_item = self.qwen3_omni_audio_phase_queue.popleft()
                    validate_qwen3_omni_audio_phase_item_ready_states(
                        talker_decode_item
                    )
                    batched_items.append(talker_decode_item)
                if len(batched_items) > 1:
                    item = _build_qwen3_omni_talker_decode_batch_item(batched_items)
                    validate_qwen3_omni_audio_phase_item_ready_states(item)

            drained_items += len(batched_items)
            forward_batch = item.forward_batch
            source_forward_mode = forward_batch.forward_mode.name
            phase_batch_size = len(item.request_indices)
            prefill_count = (
                phase_batch_size if item.phase == "talker_prefill" else 0
            )
            decode_count = (
                phase_batch_size if item.phase == "talker_decode" else 0
            )
            code_predictor_count = (
                phase_batch_size if item.phase == "code_predictor" else 0
            )
            now = time.perf_counter()
            queue_wait_ms = max(
                ready_state.queue_wait_ms(now) for ready_state in item.ready_states
            )
            deadline_miss_ms = max(
                qwen3_omni_audio_deadline_miss_ms(ready_state, now)
                for ready_state in item.ready_states
            )
            start_time = time.perf_counter()
            model_start_time = time.perf_counter()
            item.logits_output = (
                self.tp_worker.model_runner.model.run_qwen3_omni_audio_phase_after_sample(
                    forward_batch,
                    item.logits_output,
                    item.next_token_ids,
                    item.phase,
                    item.request_indices,
                    item.payload,
                )
            )
            model_elapsed_ms = (time.perf_counter() - model_start_time) * 1000
            post_start_time = time.perf_counter()
            for req_idx, ready_state in zip(item.request_indices, item.ready_states):
                consume_qwen3_omni_audio_ready_state(item.reqs[req_idx], ready_state)
            if item.phase == "talker_prefill":
                for req_idx, ready_state in zip(
                    item.request_indices, item.ready_states
                ):
                    req = item.reqs[req_idx]
                    prefill_audio_done = qwen3_omni_prefill_audio_done(req)
                    if not prefill_audio_done:
                        req.qwen3_omni_partial_talker_prefill_chunk_idx = max(
                            req.qwen3_omni_partial_talker_prefill_chunk_idx,
                            ready_state.audio_input_chunk_idx,
                        )
            pending_code_predictor = getattr(
                item.logits_output, "code_predictor_inputs", None
            )
            if (
                pending_code_predictor is not None
                and item.phase == "talker_prefill"
            ):
                pending_code_predictor = (
                    filter_uncommitted_qwen3_omni_code_predictor_inputs(
                        item.logits_output,
                        item.reqs,
                    )
                )
            if pending_code_predictor is not None:
                rid_by_index = dict(zip(item.request_indices, item.request_rids))
                code_predictor_ready_at = time.perf_counter()
                code_predictor_ready_states = [
                    mark_qwen3_omni_audio_ready_state(
                        item.reqs[req_idx],
                        "code_predictor",
                        req_idx,
                        code_predictor_ready_at,
                    )
                    for req_idx in pending_code_predictor.request_indices
                ]
                self.qwen3_omni_audio_phase_queue.appendleft(
                    Qwen3OmniAudioPhaseItem(
                        logits_output=item.logits_output,
                        forward_batch=forward_batch,
                        next_token_ids=item.next_token_ids,
                        reqs=item.reqs,
                        return_logprob=item.return_logprob,
                        phase="code_predictor",
                        request_indices=pending_code_predictor.request_indices,
                        request_rids=[
                            rid_by_index[req_idx]
                            for req_idx in pending_code_predictor.request_indices
                        ],
                        payload=build_qwen3_omni_audio_phase_payload(
                            phase="code_predictor",
                            forward_batch=forward_batch,
                            logits_output=item.logits_output,
                            next_token_ids=item.next_token_ids,
                            request_indices=pending_code_predictor.request_indices,
                        ),
                        ready_states=code_predictor_ready_states,
                        queued_at=min(
                            ready_state.ready_at
                            for ready_state in code_predictor_ready_states
                        ),
                    )
                )
                if drain_dependent_code_predictor:
                    dependent_code_predictor_budget += 1
            post_elapsed_ms = (time.perf_counter() - post_start_time) * 1000
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            fields = {
                "source_forward_mode": source_forward_mode,
                "batch_size": phase_batch_size,
                "prefill_reqs": prefill_count,
                "decode_reqs": decode_count,
                "code_predictor_reqs": code_predictor_count,
                "queue_wait_ms": f"{queue_wait_ms:.2f}",
                "deadline_miss_ms": f"{deadline_miss_ms:.2f}",
                "wall_ms": f"{elapsed_ms:.2f}",
                "model_ms": f"{model_elapsed_ms:.2f}",
                "post_ms": f"{post_elapsed_ms:.2f}",
            }
            audio_phase_metrics = getattr(
                item.logits_output, "audio_phase_metrics", None
            )
            if audio_phase_metrics is not None:
                fields.update(
                    {
                        name: f"{value:.2f}"
                        for name, value in audio_phase_metrics.items()
                    }
                )
            if item.phase == "talker_decode":
                phase_reqs = [item.reqs[req_idx] for req_idx in item.request_indices]
                thinker_done_reqs = sum(1 for req in phase_reqs if req.thinker_done)
                text_lag_reqs = sum(
                    1 for req in phase_reqs if req.qwen3_omni_text_lag_tokens
                )
                fields.update(
                    {
                        "thinker_done_reqs": thinker_done_reqs,
                        "text_lag_reqs": text_lag_reqs,
                        "post_thinker_pad_reqs": sum(
                            1
                            for req in phase_reqs
                            if req.thinker_done and not req.qwen3_omni_text_lag_tokens
                        ),
                        "tts_eos_reqs": sum(
                            1 for req in phase_reqs if req.qwen3_omni_tts_eos_pending
                        ),
                    }
                )
            apply_elapsed_ms = 0.0
            if apply_outputs:
                apply_start_time = time.perf_counter()
                for req in self._apply_qwen3_omni_audio_phase_output(item):
                    if (
                        req.rid in applied_return_logprob_by_rid
                        and applied_return_logprob_by_rid[req.rid]
                        != item.return_logprob
                    ):
                        raise RuntimeError(
                            "Qwen3-Omni deferred audio output saw mixed "
                            f"return_logprob values for request {req.rid}"
                        )
                    applied_reqs_by_rid[req.rid] = req
                    applied_return_logprob_by_rid[req.rid] = item.return_logprob
                apply_elapsed_ms = (time.perf_counter() - apply_start_time) * 1000
            fields["apply_ms"] = f"{apply_elapsed_ms:.2f}"
            fields["wall_ms"] = f"{(time.perf_counter() - start_time) * 1000:.2f}"
            self._log_qwen3_omni_audio_phase_drained(
                scheduler_action="drain_pending_audio",
                phase=item.phase,
                fields=fields,
            )

        if apply_outputs and applied_reqs_by_rid:
            affected_reqs = list(applied_reqs_by_rid.values())
            code2wav_reqs = self._qwen3_omni_code2wav_ready_reqs()
            self._drain_qwen3_omni_code2wav_phase(
                code2wav_reqs, self._get_code2wav_model()
            )
            output_reqs_by_rid = {req.rid: req for req in affected_reqs}
            output_reqs_by_rid.update({req.rid: req for req in code2wav_reqs})
            output_reqs = list(output_reqs_by_rid.values())
            # We bucket by req.return_logprob (set at turn-start, immutable
            # thereafter). For reqs that came through the apply path the
            # value must match the apply-time bucket; the assertion below
            # makes that contract explicit so a future refactor cannot
            # silently misroute streams.
            for req in affected_reqs:
                expected = applied_return_logprob_by_rid.get(req.rid)
                if expected is not None and req.return_logprob != expected:
                    raise RuntimeError(
                        f"Qwen3-Omni req {req.rid} return_logprob mutated "
                        f"between apply ({expected}) and stream "
                        f"({req.return_logprob})"
                    )
            for return_logprob in (False, True):
                streamed_reqs = [
                    req
                    for req in output_reqs
                    if (req.talker_pcm16_chunks or req.finished())
                    and req.return_logprob == return_logprob
                ]
                if streamed_reqs:
                    self.stream_output(streamed_reqs, return_logprob)
        return drained_items

    def _qwen3_omni_has_thinker_lane_work(self: Scheduler) -> bool:
        return (
            bool(self.waiting_queue)
            or self.chunked_req is not None
            or not self.running_batch.is_empty()
            or (
                self.last_batch is not None
                and not self.last_batch.is_empty()
            )
        )

    def _qwen3_omni_has_thinker_prefill_work(self: Scheduler) -> bool:
        return (
            bool(self.waiting_queue)
            or self.chunked_req is not None
            or (
                self.last_batch is not None
                and not self.last_batch.is_empty()
            )
        )

    def _qwen3_omni_audio_item_depends_on_thinker_lane(
        self: Scheduler,
        item: Qwen3OmniAudioPhaseItem,
    ) -> bool:
        if item.phase == "talker_prefill" and all(
            isinstance(state, Qwen3OmniTalkerPrefillReady)
            and not state.committed_audio_input
            for state in item.ready_states
        ):
            return False

        thinker_reqs = list(self.waiting_queue)
        if self.chunked_req is not None:
            thinker_reqs.append(self.chunked_req)
        thinker_reqs.extend(self.running_batch.reqs)
        if self.last_batch is not None:
            thinker_reqs.extend(self.last_batch.reqs)

        thinker_rids = {req.rid for req in thinker_reqs}
        overlapping_indices = [
            req_idx
            for req_idx, rid in zip(item.request_indices, item.request_rids)
            if rid in thinker_rids
        ]
        if item.phase == "talker_decode":
            return any(
                item.reqs[req_idx].qwen3_omni_text_lag_tokens
                or item.reqs[req_idx].thinker_done
                or item.reqs[req_idx].qwen3_omni_tts_eos_pending
                for req_idx in overlapping_indices
            )
        return bool(overlapping_indices)

    def _qwen3_omni_audio_item_at_lag_limit(
        self: Scheduler,
        item: Qwen3OmniAudioPhaseItem,
    ) -> bool:
        return any(
            len(item.reqs[req_idx].qwen3_omni_text_lag_tokens) >= AUDIO_MAX_TEXT_LAG
            or item.reqs[req_idx].qwen3_omni_audio_frame_lag
            >= AUDIO_MAX_AUDIO_FRAME_LAG
            for req_idx in item.request_indices
        )

    def _qwen3_omni_audio_item_lag_stats(
        self: Scheduler,
        item: Qwen3OmniAudioPhaseItem,
    ) -> Tuple[int, int]:
        max_text_lag = 0
        max_audio_frame_lag = 0
        for req_idx in item.request_indices:
            req = item.reqs[req_idx]
            max_text_lag = max(
                max_text_lag,
                len(req.qwen3_omni_text_lag_tokens),
            )
            max_audio_frame_lag = max(
                max_audio_frame_lag,
                req.qwen3_omni_audio_frame_lag,
            )
        return max_text_lag, max_audio_frame_lag

    def _qwen3_omni_audio_item_needs_first_pcm(
        self: Scheduler,
        item: Qwen3OmniAudioPhaseItem,
    ) -> bool:
        if item.phase == "talker_prefill" and not any(
            isinstance(state, Qwen3OmniTalkerPrefillReady)
            and state.committed_audio_input
            for state in item.ready_states
        ):
            return False
        if item.phase in ("talker_decode", "code_predictor") and not any(
            len(item.reqs[req_idx].qwen3_omni_text_lag_tokens) > 0
            or item.reqs[req_idx].thinker_done
            or item.reqs[req_idx].qwen3_omni_tts_eos_pending
            for req_idx in item.request_indices
        ):
            return False
        return any(
            not item.reqs[req_idx].talker_pcm16_chunks
            for req_idx in item.request_indices
        )

    def _qwen3_omni_code2wav_reqs_at_lag_limit(
        self: Scheduler,
        reqs: List[Req],
    ) -> bool:
        return any(
            req.qwen3_omni_audio_frame_lag >= AUDIO_MAX_AUDIO_FRAME_LAG
            for req in reqs
        )

    def _qwen3_omni_code2wav_lag_stats(
        self: Scheduler,
        reqs: List[Req],
    ) -> Tuple[int, int]:
        return (
            0,
            max((req.qwen3_omni_audio_frame_lag for req in reqs), default=0),
        )

    def _qwen3_omni_code2wav_reqs_need_first_pcm(
        self: Scheduler,
        reqs: List[Req],
    ) -> bool:
        return any(not req.talker_pcm16_chunks for req in reqs)

    def _qwen3_omni_lane_decision_reason(
        self: Scheduler,
        at_lag_limit: bool,
        depends_on_thinker_lane: bool,
        has_thinker_work: bool,
        queue_wait_ms: float,
        deadline_ms: int = AUDIO_BATCH_DELAY_MS,
        has_thinker_prefill_work: bool = False,
        needs_first_pcm: bool = False,
    ) -> str:
        if at_lag_limit:
            return "audio_lag_limit"
        if depends_on_thinker_lane:
            return "phase_depends_on_thinker_lane"
        if not has_thinker_work:
            return "no_thinker_work"
        if needs_first_pcm:
            return "first_pcm_before_thinker"
        if has_thinker_prefill_work:
            return "code2wav_before_thinker_prefill"
        if queue_wait_ms >= deadline_ms:
            return "audio_deadline_due"
        return "audio_has_slack"

    def _qwen3_omni_code2wav_queue_wait_ms(
        self: Scheduler,
        reqs: List[Req],
        now: float,
    ) -> float:
        return max(
            self._qwen3_omni_code2wav_ready_state(req).queue_wait_ms(now)
            for req in reqs
        )

    def _qwen3_omni_code2wav_should_preempt_phase_queue(
        self: Scheduler,
        reqs: List[Req],
        now: float,
    ) -> bool:
        if not reqs:
            return False
        return (
            self._qwen3_omni_code2wav_reqs_at_lag_limit(reqs)
            or self._qwen3_omni_has_thinker_prefill_work()
            or self._qwen3_omni_code2wav_reqs_need_first_pcm(reqs)
            or self._qwen3_omni_code2wav_queue_wait_ms(reqs, now)
            >= AUDIO_BATCH_DELAY_MS
        )

    def _promote_qwen3_omni_urgent_audio_phase_item(self: Scheduler) -> None:
        # If an item further back in the queue is at the audio deadline, at the
        # audio-frame-lag limit, or blocking the thinker lane, bubble it to the
        # front — but only if no earlier item shares a request with it, so
        # per-request frame order is preserved.
        # Queue depth is bounded by phase concurrency (~few × bs); the
        # O(n) list rebuild on the deque is cheap at those depths.
        if len(self.qwen3_omni_audio_phase_queue) <= 1:
            return

        now = time.perf_counter()
        first_is_background_prefill = _qwen3_omni_background_talker_prefill_item(
            self.qwen3_omni_audio_phase_queue[0]
        )
        earlier_rids: set[str] = set()
        for item_index, item in enumerate(self.qwen3_omni_audio_phase_queue):
            validate_qwen3_omni_audio_phase_item_ready_states(item)
            item_rids = set(item.request_rids)
            if item_index > 0 and earlier_rids.isdisjoint(item_rids):
                deadline_due = (
                    max(
                        ready_state.queue_wait_ms(now)
                        for ready_state in item.ready_states
                    )
                    >= qwen3_omni_audio_item_deadline_ms(item)
                )
                if (
                    (
                        first_is_background_prefill
                        and not _qwen3_omni_background_talker_prefill_item(item)
                    )
                    or deadline_due
                    or self._qwen3_omni_audio_item_at_lag_limit(item)
                    or self._qwen3_omni_audio_item_depends_on_thinker_lane(item)
                    or self._qwen3_omni_audio_item_needs_first_pcm(item)
                ):
                    items = list(self.qwen3_omni_audio_phase_queue)
                    urgent_item = items.pop(item_index)
                    items.insert(0, urgent_item)
                    self.qwen3_omni_audio_phase_queue.clear()
                    self.qwen3_omni_audio_phase_queue.extend(items)
                    return
            earlier_rids.update(item_rids)

    def _drop_finished_qwen3_omni_audio_phase_items(self: Scheduler) -> None:
        if not self.qwen3_omni_audio_phase_queue:
            return

        retained_items = []
        for item in self.qwen3_omni_audio_phase_queue:
            live_request_indices = [
                req_idx
                for req_idx in item.request_indices
                if not item.reqs[req_idx].finished()
            ]
            if filter_qwen3_omni_audio_phase_item_by_request_indices(
                item,
                live_request_indices,
            ):
                retained_items.append(item)
        self.qwen3_omni_audio_phase_queue.clear()
        self.qwen3_omni_audio_phase_queue.extend(retained_items)

    def _qwen3_omni_code2wav_ready_state(
        self: Scheduler,
        req: Req,
    ) -> Qwen3OmniCode2WavReady | None:
        if req.qwen3_omni_code2wav_pending:
            return None
        for ready_state in req.qwen3_omni_audio_ready_states:
            if isinstance(ready_state, Qwen3OmniCode2WavReady) and should_run_code2wav_phase(
                req,
                flush=ready_state.flush,
            ):
                return ready_state
        # Fallback: a req can newly satisfy `should_run_code2wav_phase` between
        # the upstream lane-decision (which adds explicit Code2WavReady ready
        # states) and the lane drain (which re-queries). Synthesize a ready
        # state on the spot so the Code2Wav lane drains it; otherwise we'd
        # spin waiting for an explicit mark that won't be added until the
        # next lane-decision tick.
        if qwen3_omni_finished_waiting_for_code2wav(req):
            return mark_qwen3_omni_code2wav_ready_state(
                req,
                time.perf_counter(),
                flush=True,
            )
        if should_run_code2wav_phase(req):
            return mark_qwen3_omni_code2wav_ready_state(
                req,
                time.perf_counter(),
                flush=False,
            )
        return None

    def _qwen3_omni_code2wav_ready_reqs(self: Scheduler) -> List[Req]:
        return [
            req
            for req in self.audio_turn_reqs.values()
            if self._qwen3_omni_code2wav_ready_state(req) is not None
        ]

    def _qwen3_omni_code2wav_stream(
        self: Scheduler,
        code2wav,
    ) -> torch.cuda.Stream | None:
        device = next(code2wav.parameters()).device
        if device.type != "cuda":
            return None
        if self.qwen3_omni_code2wav_stream is None:
            with torch.cuda.device(device):
                self.qwen3_omni_code2wav_stream = torch.cuda.Stream()
        if not self.qwen3_omni_code2wav_stream_prewarmed:
            max_concurrent = self.server_args.qwen3_omni_max_concurrent_audio_turns
            if max_concurrent is None:
                raise RuntimeError(
                    "Qwen3-Omni Code2Wav runtime stream prewarm requires "
                    "--qwen3-omni-max-concurrent-audio-turns."
                )
            start = time.perf_counter()
            prewarm_count = code2wav.prewarm_runtime_shapes(
                range(1, max_concurrent + 1),
                steady_chunk_frames=CODE2WAV_STEADY_CHUNK_FRAMES,
                max_decoder_batch_size=CODE2WAV_DECODER_MAX_BATCH,
                stream=self.qwen3_omni_code2wav_stream,
            )
            self.qwen3_omni_code2wav_stream_prewarmed = True
            logger.info(
                "Qwen3-Omni Code2Wav runtime stream prewarm completed: "
                "specs=%s, elapsed_s=%.2f",
                prewarm_count,
                time.perf_counter() - start,
            )
        return self.qwen3_omni_code2wav_stream

    def _qwen3_omni_has_completed_code2wav_work(self: Scheduler) -> bool:
        for pending in self.qwen3_omni_code2wav_pending_work:
            if pending.event is None or pending.event.query():
                return True
        return False

    def _qwen3_omni_pending_code2wav_blocks_thinker(
        self: Scheduler,
        now: float,
    ) -> bool:
        for pending in self.qwen3_omni_code2wav_pending_work:
            pending_wait_ms = (now - pending.launched_at) * 1000
            for item in pending.items:
                req = item.req
                if (
                    not req.talker_pcm16_chunks
                    or req.qwen3_omni_audio_frame_lag >= AUDIO_MAX_AUDIO_FRAME_LAG
                    or qwen3_omni_audio_deadline_miss_ms(item.ready_state, now) > 0
                    or pending_wait_ms >= AUDIO_BATCH_DELAY_MS
                ):
                    return True
        return False

    def _complete_qwen3_omni_code2wav_work(
        self: Scheduler,
        block: bool = False,
    ) -> List[Req]:
        code2wav = self._get_code2wav_model()
        completed_reqs: List[Req] = []
        remaining_work: List[Code2WavPendingWork] = []
        for pending in self.qwen3_omni_code2wav_pending_work:
            result = complete_code2wav_pending_work(code2wav, pending, block=block)
            if result is None:
                remaining_work.append(pending)
                continue
            completed_at = time.perf_counter()
            pending_elapsed_ms = (completed_at - pending.launched_at) * 1000
            completed_reqs.extend(item.req for item in pending.items)
            if pending.event is not None:
                self.qwen3_omni_code2wav_event_pool.append(pending.event)
            self._log_qwen3_omni_audio_phase_drained(
                scheduler_action="complete_async_code2wav",
                phase="code2wav",
                fields={
                    "batch_size": len(pending.items),
                    "decoded_reqs": result.decoded_reqs,
                    "decoded_frames": result.decoded_frames,
                    "emitted_chunks": result.emitted_chunks,
                    "flush": False,
                    "batch_groups": result.batch_groups,
                    "queue_wait_ms": "0.00",
                    "deadline_miss_ms": "0.00",
                    "wall_ms": f"{pending_elapsed_ms:.2f}",
                    "pending_elapsed_ms": f"{pending_elapsed_ms:.2f}",
                    "enqueue_wall_ms": f"{pending.enqueue_wall_ms:.2f}",
                    "event_wait_ms": f"{result.event_wait_ms:.2f}",
                    "materialize_ms": f"{result.materialize_ms:.2f}",
                    "event_pool_size": len(self.qwen3_omni_code2wav_event_pool),
                },
            )
        self.qwen3_omni_code2wav_pending_work = remaining_work
        return completed_reqs

    def _log_qwen3_omni_lane_decision(
        self: Scheduler,
        decision: Qwen3OmniLaneDecision,
    ) -> None:
        self._maybe_log_qwen3_omni_runtime_snapshot()
        active_audio_turns, queued_audio_turns = self._qwen3_omni_audio_turn_stats()
        logger.info(
            "Qwen3-Omni scheduler action selected: action=%s, "
            "reason=%s, "
            "oldest_audio_phase=%s, oldest_audio_batch_size=%s, "
            "oldest_audio_queue_wait_ms=%.2f, "
            "oldest_audio_deadline_slack_ms=%.2f, "
            "max_text_lag_tokens=%s, max_audio_frame_lag=%s, "
            "has_thinker_work=%s, "
            "depends_on_thinker_lane=%s, audio_at_lag_limit=%s, "
            "pending_audio_items=%s, "
            "component_kv_used_free=%s, active_audio_turns=%s, "
            "queued_audio_turns=%s",
            decision.action,
            decision.reason,
            decision.phase,
            decision.batch_size,
            decision.queue_wait_ms,
            decision.deadline_ms - decision.queue_wait_ms,
            decision.max_text_lag,
            decision.max_audio_frame_lag,
            decision.has_thinker_work,
            decision.depends_on_thinker_lane,
            decision.at_lag_limit,
            decision.pending_audio_items,
            self._qwen3_omni_component_kv_stats(),
            active_audio_turns,
            queued_audio_turns,
        )

    def _select_qwen3_omni_scheduler_action(self: Scheduler) -> str:
        self._drop_finished_qwen3_omni_audio_phase_items()
        if self._qwen3_omni_has_completed_code2wav_work():
            self._log_qwen3_omni_lane_decision(
                Qwen3OmniLaneDecision(
                    action="audio",
                    reason="code2wav_async_complete",
                    phase="code2wav",
                    batch_size=0,
                    queue_wait_ms=0.0,
                    max_text_lag=0,
                    max_audio_frame_lag=0,
                    has_thinker_work=self._qwen3_omni_has_thinker_lane_work(),
                    depends_on_thinker_lane=False,
                    at_lag_limit=False,
                    pending_audio_items=len(self.qwen3_omni_audio_phase_queue),
                )
            )
            return "audio"
        now = time.perf_counter()
        if (
            self.qwen3_omni_code2wav_pending_work
            and self._qwen3_omni_has_thinker_prefill_work()
            and self._qwen3_omni_pending_code2wav_blocks_thinker(now)
        ):
            self._log_qwen3_omni_lane_decision(
                Qwen3OmniLaneDecision(
                    action="audio",
                    reason="code2wav_pending_before_thinker_prefill",
                    phase="code2wav",
                    batch_size=0,
                    queue_wait_ms=0.0,
                    max_text_lag=0,
                    max_audio_frame_lag=0,
                    has_thinker_work=self._qwen3_omni_has_thinker_lane_work(),
                    depends_on_thinker_lane=False,
                    at_lag_limit=False,
                    pending_audio_items=len(self.qwen3_omni_audio_phase_queue),
                )
            )
            return "audio"
        code2wav_ready_reqs = self._qwen3_omni_code2wav_ready_reqs()
        if not self.qwen3_omni_audio_phase_queue:
            if not code2wav_ready_reqs:
                return "thinker"
            has_thinker_work = self._qwen3_omni_has_thinker_lane_work()
            now = time.perf_counter()
            queue_wait_ms = self._qwen3_omni_code2wav_queue_wait_ms(
                code2wav_ready_reqs, now
            )
            at_lag_limit = self._qwen3_omni_code2wav_reqs_at_lag_limit(
                code2wav_ready_reqs
            )
            needs_first_pcm = self._qwen3_omni_code2wav_reqs_need_first_pcm(
                code2wav_ready_reqs
            )
            has_thinker_prefill_work = self._qwen3_omni_has_thinker_prefill_work()
            max_text_lag, max_audio_frame_lag = self._qwen3_omni_code2wav_lag_stats(
                code2wav_ready_reqs
            )
            action = (
                "audio"
                if at_lag_limit
                or not has_thinker_work
                or has_thinker_prefill_work
                or needs_first_pcm
                or queue_wait_ms >= AUDIO_BATCH_DELAY_MS
                else "thinker"
            )
            reason = self._qwen3_omni_lane_decision_reason(
                at_lag_limit=at_lag_limit,
                depends_on_thinker_lane=False,
                has_thinker_work=has_thinker_work,
                queue_wait_ms=queue_wait_ms,
                has_thinker_prefill_work=has_thinker_prefill_work,
                needs_first_pcm=needs_first_pcm,
            )
            self._log_qwen3_omni_lane_decision(
                Qwen3OmniLaneDecision(
                    action=action,
                    reason=reason,
                    phase="code2wav",
                    batch_size=len(code2wav_ready_reqs),
                    queue_wait_ms=queue_wait_ms,
                    max_text_lag=max_text_lag,
                    max_audio_frame_lag=max_audio_frame_lag,
                    has_thinker_work=has_thinker_work,
                    depends_on_thinker_lane=False,
                    at_lag_limit=at_lag_limit,
                    pending_audio_items=0,
                )
            )
            return action

        item = self.qwen3_omni_audio_phase_queue[0]
        validate_qwen3_omni_audio_phase_item_ready_states(item)
        now = time.perf_counter()
        code2wav_preempts = self._qwen3_omni_code2wav_should_preempt_phase_queue(
            code2wav_ready_reqs, now
        )
        if code2wav_preempts:
            code2wav_queue_wait_ms = self._qwen3_omni_code2wav_queue_wait_ms(
                code2wav_ready_reqs, now
            )
            max_text_lag, max_audio_frame_lag = self._qwen3_omni_code2wav_lag_stats(
                code2wav_ready_reqs
            )
            at_lag_limit = self._qwen3_omni_code2wav_reqs_at_lag_limit(
                code2wav_ready_reqs
            )
            needs_first_pcm = self._qwen3_omni_code2wav_reqs_need_first_pcm(
                code2wav_ready_reqs
            )
            has_thinker_prefill_work = self._qwen3_omni_has_thinker_prefill_work()
            reason = self._qwen3_omni_lane_decision_reason(
                at_lag_limit=at_lag_limit,
                depends_on_thinker_lane=False,
                has_thinker_work=True,
                queue_wait_ms=code2wav_queue_wait_ms,
                has_thinker_prefill_work=has_thinker_prefill_work,
                needs_first_pcm=needs_first_pcm,
            )
            self._log_qwen3_omni_lane_decision(
                Qwen3OmniLaneDecision(
                    action="audio",
                    reason=reason,
                    phase="code2wav",
                    batch_size=len(code2wav_ready_reqs),
                    queue_wait_ms=code2wav_queue_wait_ms,
                    max_text_lag=max_text_lag,
                    max_audio_frame_lag=max_audio_frame_lag,
                    has_thinker_work=self._qwen3_omni_has_thinker_lane_work(),
                    depends_on_thinker_lane=False,
                    at_lag_limit=at_lag_limit,
                    pending_audio_items=len(self.qwen3_omni_audio_phase_queue),
                )
            )
            return "audio"

        self._promote_qwen3_omni_urgent_audio_phase_item()
        item = self.qwen3_omni_audio_phase_queue[0]
        validate_qwen3_omni_audio_phase_item_ready_states(item)
        now = time.perf_counter()
        queue_wait_ms = max(
            ready_state.queue_wait_ms(now) for ready_state in item.ready_states
        )
        has_thinker_work = self._qwen3_omni_has_thinker_lane_work()
        depends_on_thinker_lane = (
            self._qwen3_omni_audio_item_depends_on_thinker_lane(item)
        )
        at_lag_limit = self._qwen3_omni_audio_item_at_lag_limit(item)
        has_thinker_prefill_work = self._qwen3_omni_has_thinker_prefill_work()
        needs_first_pcm = self._qwen3_omni_audio_item_needs_first_pcm(item)
        max_text_lag, max_audio_frame_lag = self._qwen3_omni_audio_item_lag_stats(
            item
        )
        action = (
            "audio"
            if at_lag_limit
            or depends_on_thinker_lane
            or not has_thinker_work
            or needs_first_pcm
            or queue_wait_ms >= qwen3_omni_audio_item_deadline_ms(item)
            else "thinker"
        )
        reason = self._qwen3_omni_lane_decision_reason(
            at_lag_limit=at_lag_limit,
            depends_on_thinker_lane=depends_on_thinker_lane,
            has_thinker_work=has_thinker_work,
            queue_wait_ms=queue_wait_ms,
            deadline_ms=qwen3_omni_audio_item_deadline_ms(item),
            has_thinker_prefill_work=has_thinker_prefill_work,
            needs_first_pcm=needs_first_pcm,
        )
        self._log_qwen3_omni_lane_decision(
            Qwen3OmniLaneDecision(
                action=action,
                reason=reason,
                phase=item.phase,
                batch_size=len(item.request_indices),
                queue_wait_ms=queue_wait_ms,
                max_text_lag=max_text_lag,
                max_audio_frame_lag=max_audio_frame_lag,
                has_thinker_work=has_thinker_work,
                depends_on_thinker_lane=depends_on_thinker_lane,
                at_lag_limit=at_lag_limit,
                pending_audio_items=len(self.qwen3_omni_audio_phase_queue),
                deadline_ms=qwen3_omni_audio_item_deadline_ms(item),
            )
        )
        return action

    def _run_qwen3_omni_code2wav_lane_if_ready(self: Scheduler) -> bool:
        completed_reqs = self._complete_qwen3_omni_code2wav_work(block=False)
        if completed_reqs:
            for return_logprob in (False, True):
                streamed_reqs = [
                    req
                    for req in completed_reqs
                    if req.talker_pcm16_chunks
                    and req.return_logprob == return_logprob
                ]
                if streamed_reqs:
                    self.stream_output(streamed_reqs, return_logprob)
            return True

        if self.qwen3_omni_code2wav_pending_work:
            completed_reqs = self._complete_qwen3_omni_code2wav_work(block=True)
            if not completed_reqs:
                raise RuntimeError(
                    "Qwen3-Omni blocked on in-flight Code2Wav work but no "
                    "Code2Wav work completed"
                )
            for return_logprob in (False, True):
                streamed_reqs = [
                    req
                    for req in completed_reqs
                    if req.talker_pcm16_chunks and req.return_logprob == return_logprob
                ]
                if streamed_reqs:
                    self.stream_output(streamed_reqs, return_logprob)
            return True

        reqs = self._qwen3_omni_code2wav_ready_reqs()
        if not reqs:
            return False
        code2wav = self._get_code2wav_model()
        flush_reqs = [
            req for req in reqs if qwen3_omni_finished_waiting_for_code2wav(req)
        ]
        non_flush_reqs = [
            req for req in reqs if should_run_code2wav_phase(req, flush=False)
        ]
        now = time.perf_counter()
        non_flush_is_urgent = bool(non_flush_reqs) and (
            self._qwen3_omni_code2wav_reqs_need_first_pcm(non_flush_reqs)
            or self._qwen3_omni_code2wav_reqs_at_lag_limit(non_flush_reqs)
            or self._qwen3_omni_code2wav_queue_wait_ms(non_flush_reqs, now)
            >= AUDIO_BATCH_DELAY_MS
        )
        # Finished-turn flush audio is tail work. When live audio is waiting,
        # drain non-flush Code2Wav first so a rare flush launch stall cannot
        # block first PCM or dependent CodePredictor frame production.
        frame_producing_head_waits = (
            bool(self.qwen3_omni_audio_phase_queue)
            and _qwen3_omni_frame_producing_phase(
                self.qwen3_omni_audio_phase_queue[0].phase
            )
        )
        if non_flush_reqs and (non_flush_is_urgent or frame_producing_head_waits):
            self._drain_qwen3_omni_code2wav_phase(non_flush_reqs, code2wav)
            reqs = non_flush_reqs
        elif flush_reqs:
            self._drain_qwen3_omni_code2wav_phase(flush_reqs, code2wav, flush=True)
            reqs = flush_reqs
        else:
            self._drain_qwen3_omni_code2wav_phase(non_flush_reqs, code2wav)
            reqs = non_flush_reqs
        for return_logprob in (False, True):
            streamed_reqs = [
                req
                for req in reqs
                if req.talker_pcm16_chunks and req.return_logprob == return_logprob
            ]
            if streamed_reqs:
                self.stream_output(streamed_reqs, return_logprob)
        return True

    def _run_qwen3_omni_audio_lane_if_selected(self: Scheduler) -> bool:
        if self._select_qwen3_omni_scheduler_action() != "audio":
            return False
        if self._qwen3_omni_has_completed_code2wav_work():
            if not self._run_qwen3_omni_code2wav_lane_if_ready():
                raise RuntimeError(
                    "Qwen3-Omni audio scheduler selected completed Code2Wav "
                    "but no completed Code2Wav work ran"
                )
            return True
        if (
            self.qwen3_omni_code2wav_pending_work
            and self._qwen3_omni_has_thinker_prefill_work()
            and self._qwen3_omni_pending_code2wav_blocks_thinker(time.perf_counter())
        ):
            if not self._run_qwen3_omni_code2wav_lane_if_ready():
                raise RuntimeError(
                    "Qwen3-Omni audio scheduler selected pending Code2Wav "
                    "before thinker prefill but no Code2Wav work ran"
                )
            return True
        code2wav_ready_reqs = self._qwen3_omni_code2wav_ready_reqs()
        if self._qwen3_omni_code2wav_should_preempt_phase_queue(
            code2wav_ready_reqs, time.perf_counter()
        ):
            if not self._run_qwen3_omni_code2wav_lane_if_ready():
                raise RuntimeError(
                    "Qwen3-Omni audio scheduler selected preemptive Code2Wav "
                    "but no Code2Wav work ran"
                )
            return True
        if not self.qwen3_omni_audio_phase_queue:
            if not self._run_qwen3_omni_code2wav_lane_if_ready():
                raise RuntimeError(
                    "Qwen3-Omni audio scheduler selected code2wav but no "
                    "Code2Wav work ran"
                )
            return True
        head_item = self.qwen3_omni_audio_phase_queue[0]
        if (
            _qwen3_omni_frame_producing_phase(head_item.phase)
            and not _qwen3_omni_audio_items_fit_frame_budget([head_item])
        ):
            pending_reqs = [
                head_item.reqs[req_idx]
                for req_idx in head_item.request_indices
                if head_item.reqs[req_idx].qwen3_omni_code2wav_pending
            ]
            # Invariant: a frame-producing item only reaches the head with an
            # exceeded frame budget if at least one of its reqs already has a
            # Code2Wav launch in flight to block on. The batch-builder enforces
            # this via _qwen3_omni_can_extend_frame_producing_batch. If we ever
            # hit this branch with no pending Code2Wav work, the queueing
            # invariant has been violated upstream — load-bearing assertion.
            if not pending_reqs:
                raise RuntimeError(
                    "Qwen3-Omni frame-producing audio phase reached frame "
                    "lag budget without pending Code2Wav work"
                )
            completed_reqs = self._complete_qwen3_omni_code2wav_work(block=True)
            if not completed_reqs:
                raise RuntimeError(
                    "Qwen3-Omni blocked for Code2Wav frame-lag relief but no "
                    "Code2Wav work completed"
                )
            for return_logprob in (False, True):
                streamed_reqs = [
                    req
                    for req in completed_reqs
                    if req.talker_pcm16_chunks and req.return_logprob == return_logprob
                ]
                if streamed_reqs:
                    self.stream_output(streamed_reqs, return_logprob)
            return True
        drained_items = self._drain_pending_qwen3_omni_audio_phase_queue(
            apply_outputs=True,
            max_items=1,
            drain_dependent_code_predictor=True,
        )
        if drained_items < 1:
            raise RuntimeError(
                "Qwen3-Omni audio scheduler selected audio but did not run an item"
            )
        return True

    def _drain_qwen3_omni_code2wav_phase(
        self: Scheduler,
        reqs: List[Req],
        code2wav,
        flush: bool = False,
    ) -> None:
        if code2wav is None:
            return
        if self.qwen3_omni_code2wav_pending_work:
            return
        phase_reqs = [
            req for req in reqs if should_run_code2wav_phase(req, flush=flush)
        ]
        if not phase_reqs:
            return
        ready_at = time.perf_counter()
        ready_states = []
        for req in phase_reqs:
            ready_state = next(
                (
                    state
                    for state in req.qwen3_omni_audio_ready_states
                    if isinstance(state, Qwen3OmniCode2WavReady)
                    and state.flush == flush
                ),
                None,
            )
            if ready_state is None:
                ready_state = mark_qwen3_omni_code2wav_ready_state(
                    req,
                    ready_at,
                    flush=flush,
                )
            ready_states.append(ready_state)

        min_code2wav_window_frames = min(
            int(ready_state.codes_window_cpu.shape[0]) for ready_state in ready_states
        )
        max_code2wav_window_frames = max(
            int(ready_state.codes_window_cpu.shape[0]) for ready_state in ready_states
        )
        initial_phase_req_count = len(phase_reqs)
        total_decoded_reqs = 0
        total_decoded_frames = 0
        total_emitted_chunks = 0
        total_batch_groups = 0
        start_time = time.perf_counter()
        queue_wait_ms = max(
            ready_state.queue_wait_ms(start_time) for ready_state in ready_states
        )
        oldest_ready_state = min(ready_states, key=lambda state: state.ready_at)
        deadline_miss_ms = max(
            qwen3_omni_audio_deadline_miss_ms(ready_state, start_time)
            for ready_state in ready_states
        )
        code2wav_stream = self._qwen3_omni_code2wav_stream(code2wav)
        if code2wav_stream is not None:
            (
                pending_work,
                total_decoded_reqs,
                total_decoded_frames,
                total_emitted_chunks,
                total_batch_groups,
                work_group_build_ms,
            ) = launch_code2wav_phase_batch(
                phase_reqs,
                code2wav,
                code2wav_stream,
                flush=flush,
                max_batch_size=(
                    CODE2WAV_FLUSH_MAX_BATCH if flush else None
                ),
                decoder_max_batch_size=CODE2WAV_DECODER_MAX_BATCH,
                event_pool=self.qwen3_omni_code2wav_event_pool,
                max_pending_groups=1,
            )
            self.qwen3_omni_code2wav_pending_work.extend(pending_work)
            now = time.perf_counter()
            self._log_qwen3_omni_audio_phase_drained(
                scheduler_action="launch_async_code2wav",
                phase="code2wav",
                fields={
                    "batch_size": initial_phase_req_count,
                    "decoded_reqs": total_decoded_reqs,
                    "decoded_frames": total_decoded_frames,
                    "emitted_chunks": total_emitted_chunks,
                    "flush": flush,
                    "min_window_frames": min_code2wav_window_frames,
                    "max_window_frames": max_code2wav_window_frames,
                    "batch_groups": total_batch_groups,
                    "queue_wait_ms": f"{queue_wait_ms:.2f}",
                    "oldest_ready_rid": oldest_ready_state.request_rid,
                    "deadline_miss_ms": f"{deadline_miss_ms:.2f}",
                    "wall_ms": f"{(now - start_time) * 1000:.2f}",
                    "work_group_build_ms": f"{work_group_build_ms:.2f}",
                    "max_enqueue_wall_ms": (
                        f"{max(p.enqueue_wall_ms for p in pending_work):.2f}"
                    ),
                    "max_codes_tensor_ms": (
                        f"{max(p.codes_tensor_ms for p in pending_work):.2f}"
                    ),
                    "max_forward_enqueue_ms": (
                        f"{max(p.forward_enqueue_ms for p in pending_work):.2f}"
                    ),
                    "max_event_record_ms": (
                        f"{max(p.event_record_ms for p in pending_work):.2f}"
                    ),
                },
            )
            return
        while phase_reqs:
            decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
                run_code2wav_phase_batch(
                    phase_reqs,
                    code2wav,
                    flush=flush,
                    max_batch_size=(
                        CODE2WAV_FLUSH_MAX_BATCH if flush else CODE2WAV_MAX_BATCH
                    ),
                )
            )
            if decoded_reqs == 0:
                break
            total_decoded_reqs += decoded_reqs
            total_decoded_frames += decoded_frames
            total_emitted_chunks += emitted_chunks
            total_batch_groups += batch_groups
            if flush:
                break
            phase_reqs = [
                req for req in reqs if should_run_code2wav_phase(req, flush=flush)
            ]

        now = time.perf_counter()
        elapsed_ms = (now - start_time) * 1000
        self._log_qwen3_omni_audio_phase_drained(
            scheduler_action="drain_code2wav",
            phase="code2wav",
            fields={
                "batch_size": initial_phase_req_count,
                "decoded_reqs": total_decoded_reqs,
                "decoded_frames": total_decoded_frames,
                "emitted_chunks": total_emitted_chunks,
                "flush": flush,
                "min_window_frames": min_code2wav_window_frames,
                "max_window_frames": max_code2wav_window_frames,
                "batch_groups": total_batch_groups,
                "queue_wait_ms": f"{queue_wait_ms:.2f}",
                "oldest_ready_rid": oldest_ready_state.request_rid,
                "deadline_miss_ms": f"{deadline_miss_ms:.2f}",
                "wall_ms": f"{elapsed_ms:.2f}",
            },
        )
