import array
from collections import deque
from dataclasses import replace
import logging
import time
from types import MethodType, SimpleNamespace

import pytest
import sglang.srt.managers.scheduler_qwen3_omni_mixin as scheduler_qwen3_omni_mixin
import sglang.srt.managers.scheduler_output_processor_mixin as scheduler_output_processor_mixin
import torch

from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    get_rope_index_qwen3_omni,
)
from sglang.srt.managers.schedule_batch import (
    AUDIO_BATCH_DELAY_MS,
    AUDIO_MAX_AUDIO_FRAME_LAG,
    AUDIO_MAX_TEXT_LAG,
    MM_PAD_SHIFT_VALUE,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_qwen3_omni_mixin import (
    Qwen3OmniAudioPhase,
    Qwen3OmniCode2WavReady,
    Qwen3OmniCodePredictorReady,
    Qwen3OmniTalkerDecodeReady,
    Qwen3OmniTalkerPrefillReady,
    SchedulerQwen3OmniMixin,
    TALKER_DECODE_STEADY_BATCH_CAP,
    _qwen3_omni_can_extend_talker_decode_batch,
    _qwen3_omni_talker_decode_batch_chunks,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
    _update_qwen3_omni_state,
    build_code2wav_work_item,
    build_qwen3_omni_audio_phase_payload,
    filter_qwen3_omni_code_predictor_inputs_by_request_indices,
    filter_uncommitted_qwen3_omni_code_predictor_inputs,
    filter_qwen3_omni_committed_talker_prefills,
    launch_code2wav_phase_batch,
    launch_code2wav_work_group,
    qwen3_omni_finished_waiting_for_code2wav,
    release_qwen3_omni_component_kv,
    release_qwen3_omni_post_thinker_kv,
    release_skipped_qwen3_omni_audio_prefill_locs,
    run_code2wav_phase_batch,
)
from sglang.srt.managers.utils import (
    Qwen3OmniAudioPhaseItem,
    Qwen3OmniCodePredictorPayload,
    Qwen3OmniTalkerDecodePayload,
    Qwen3OmniTalkerPrefillPayload,
)
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeCode2WavConfig
from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniCodePredictorInputs,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniTalkerPrefillAssistantChunk,
    Qwen3OmniTalkerPrefillUserChunk,
    TalkerPrefillPhaseOutput,
    ThinkerTalkerOutput,
    _apply_rotary_pos_emb,
    _mask_codec_eos,
    _sample_codec_tokens,
    _talker_decode_capture_batch_bucket,
    build_qwen3_omni_talker_prefill_chunk_contract,
)


def _talker_sampler_state(batch_size: int):
    return {
        "talker_sampler_seed": torch.arange(
            100, 100 + batch_size, dtype=torch.int64
        ),
        "talker_sampler_step": torch.zeros(batch_size, dtype=torch.int64),
    }


def _fake_frame_item(phase: str, rid: str, *, rows: int = 1):
    req = SimpleNamespace(rid=rid, qwen3_omni_audio_frame_lag=0)
    return SimpleNamespace(
        phase=phase,
        request_indices=list(range(rows)),
        reqs=[req] * rows,
    )


class AddProjection:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, x):
        return x + self.offset


class FakeCode2Wav(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.total_upsample = 2
        self.calls = []

    def output_num_samples(self, num_frames: int) -> int:
        return num_frames * self.total_upsample

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.detach().cpu())
        batch_size = codes.shape[0]
        samples = codes.shape[2] * self.total_upsample
        return torch.arange(
            batch_size * samples, dtype=torch.float32, device=codes.device
        ).view(batch_size, 1, samples) / 100


class OffsetLengthFakeCode2Wav(FakeCode2Wav):
    def output_num_samples(self, num_frames: int) -> int:
        if num_frames == 0:
            return 0
        return num_frames * self.total_upsample + 5

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.detach().cpu())
        batch_size = codes.shape[0]
        samples = codes.shape[2] * self.total_upsample + 5
        return torch.arange(
            batch_size * samples, dtype=torch.float32, device=codes.device
        ).view(batch_size, 1, samples) / 1000


class SplitDecoderFakeCode2Wav(FakeCode2Wav):
    def __init__(self):
        super().__init__()
        self.decoder_max_batch_sizes = []

    def forward_split_decoder(
        self,
        codes: torch.Tensor,
        max_decoder_batch_size: int,
    ) -> torch.Tensor:
        self.decoder_max_batch_sizes.append(max_decoder_batch_size)
        return self(codes)


def test_code2wav_output_num_samples_matches_decoder_crop():
    code2wav = SimpleNamespace(
        config=SimpleNamespace(upsampling_ratios=(2, 2), upsample_rates=(8, 5, 4, 3))
    )

    assert Qwen3OmniMoeCode2Wav.output_num_samples(code2wav, 0) == 0
    assert Qwen3OmniMoeCode2Wav.output_num_samples(code2wav, 1) == 1365
    assert Qwen3OmniMoeCode2Wav.output_num_samples(code2wav, 4) == 7125
    with pytest.raises(ValueError, match="non-negative"):
        Qwen3OmniMoeCode2Wav.output_num_samples(code2wav, -1)


def test_code2wav_forward_caches_attention_mask_and_broadcasts_rope():
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        decoder_dim=16,
        upsample_rates=[2],
        upsampling_ratios=[2],
        codebook_size=32,
        num_quantizers=2,
        sliding_window=4,
    )
    code2wav = Qwen3OmniMoeCode2Wav(config).eval()
    codes = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(2, config.num_quantizers, 5),
    )
    codes[1] = codes[0]

    with torch.no_grad():
        wav = code2wav(codes)

    assert wav.shape[0] == 2
    torch.testing.assert_close(wav[0], wav[1])
    attn = code2wav.pre_transformer.layers[0].self_attn
    assert (codes.device, codes.shape[-1]) in attn._mask_cache


def test_code2wav_split_decoder_matches_forward():
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        decoder_dim=16,
        upsample_rates=[2],
        upsampling_ratios=[2],
        codebook_size=32,
        num_quantizers=2,
        sliding_window=4,
    )
    code2wav = Qwen3OmniMoeCode2Wav(config).eval()
    codes = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(5, config.num_quantizers, 5),
    )

    with torch.no_grad():
        full_wav = code2wav(codes)
        split_wav = code2wav.forward_split_decoder(codes, max_decoder_batch_size=2)

    torch.testing.assert_close(split_wav, full_wav)


def test_sample_codec_tokens_batches_suppression_and_eos_preservation():
    logits = torch.zeros((2, 2200), dtype=torch.float32)
    logits[:, 100] = 5
    logits[0, 2199] = 10
    logits[1, 2150] = 11

    tokens = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        prev_tokens=[[], []],
    )

    assert tokens.tolist() == [100, 2150]


def test_talker_decode_batching_caps_exact_graph_first_use_tail():
    items = [
        _fake_frame_item("talker_decode", f"r{i}")
        for i in range(TALKER_DECODE_STEADY_BATCH_CAP)
    ]

    assert not _qwen3_omni_can_extend_talker_decode_batch(
        items,
        _fake_frame_item("talker_decode", "overflow"),
    )
    assert _qwen3_omni_can_extend_talker_decode_batch(
        items[:-1],
        _fake_frame_item("talker_decode", "fits"),
    )
    assert not _qwen3_omni_can_extend_talker_decode_batch(
        [
            _fake_frame_item(
                "talker_decode",
                "batch",
                rows=TALKER_DECODE_STEADY_BATCH_CAP,
            )
        ],
        _fake_frame_item("talker_decode", "overflow"),
    )
    assert _qwen3_omni_talker_decode_batch_chunks(
        list(range(TALKER_DECODE_STEADY_BATCH_CAP + 3))
    ) == [
        list(range(TALKER_DECODE_STEADY_BATCH_CAP)),
        list(
            range(
                TALKER_DECODE_STEADY_BATCH_CAP,
                TALKER_DECODE_STEADY_BATCH_CAP + 3,
            )
        ),
    ]


def test_talker_decode_capture_uses_buckets_for_large_audio_groups():
    assert _talker_decode_capture_batch_bucket(1, 50) == 1
    assert _talker_decode_capture_batch_bucket(3, 50) == 4
    assert _talker_decode_capture_batch_bucket(17, 50) == 32
    assert _talker_decode_capture_batch_bucket(28, 50) == 32
    assert _talker_decode_capture_batch_bucket(33, 50) is None
    assert _talker_decode_capture_batch_bucket(17, 30) == 30
    assert _talker_decode_capture_batch_bucket(31, 30) is None


def test_sample_codec_tokens_batches_repetition_penalty():
    logits = torch.zeros((2, 2200), dtype=torch.float32)
    logits[0, 10] = 10
    logits[0, 20] = 9
    logits[1, 30] = 10
    logits[1, 40] = 9

    tokens = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=torch.tensor([2.0, 1.0]),
        prev_tokens=[[10, 10], []],
    )

    assert tokens.tolist() == [20, 30]


def test_sample_codec_tokens_uses_gumbel_draw_not_argmax():
    logits = torch.zeros((2, 2200), dtype=torch.float32)
    logits[:, 10] = 2
    logits[:, 20] = 1

    tokens = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        repetition_penalty=1.0,
        prev_tokens=[[], []],
        uniform_noise=torch.tensor([[0.50, 0.99], [0.50, 0.50]]),
    )

    assert tokens.tolist() == [20, 10]


def test_sample_codec_tokens_seeded_sampler_replays():
    logits = torch.zeros((2, 2200), dtype=torch.float32)
    logits[:, 10] = 2
    logits[:, 20] = 1
    sampler_seed = torch.tensor([123, 456], dtype=torch.int64)
    sampler_step = torch.tensor([7, 8], dtype=torch.int64)
    sampler_offsets = torch.arange(2, dtype=torch.int64)

    tokens_a = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        repetition_penalty=1.0,
        prev_tokens=[[], []],
        sampler_seed=sampler_seed,
        sampler_step=sampler_step,
        sampler_offsets=sampler_offsets,
    )
    tokens_b = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        repetition_penalty=1.0,
        prev_tokens=[[], []],
        sampler_seed=sampler_seed,
        sampler_step=sampler_step,
        sampler_offsets=sampler_offsets,
    )

    assert torch.equal(tokens_a, tokens_b)


def test_sample_codec_tokens_repetition_mask_matches_prev_tokens():
    logits = torch.zeros((2, 2200), dtype=torch.float32)
    logits[0, 10] = 10
    logits[0, 20] = 9
    logits[1, 30] = 10
    logits[1, 40] = 9
    repetition_mask = torch.zeros((2, 2200), dtype=torch.bool)
    repetition_mask[0, 10] = True

    prev_tokens_result = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=torch.tensor([2.0, 1.0]),
        prev_tokens=[[10, 10], []],
    )
    mask_result = _sample_codec_tokens(
        logits,
        codec_eos_token_id=2150,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=torch.tensor([2.0, 1.0]),
        repetition_mask=repetition_mask,
    )

    assert torch.equal(mask_result, prev_tokens_result)


class FakeAllocator:
    def __init__(self):
        self.freed = []
        self.page_size = 1

    def free(self, locs):
        self.freed.append(locs)


class FakeSequentialAllocator:
    def __init__(self, start=10):
        self.next_loc = start
        self.allocated = []

    def alloc(self, count):
        locs = torch.arange(self.next_loc, self.next_loc + count, dtype=torch.int64)
        self.next_loc += count
        self.allocated.append(locs)
        return locs

    def available_size(self):
        return 1000

    def free(self, locs):
        self.allocated.append(("free", locs))


class FakeKVAllocator:
    def __init__(self, size, free):
        self.size = size
        self.free = free

    def available_size(self):
        return self.free


class FakeAttnBackend:
    def __init__(self):
        self.metadata_batch_sizes = []
        self.metadata_req_pool_indices = []

    def init_forward_metadata(self, forward_batch):
        self.metadata_batch_sizes.append(forward_batch.batch_size)
        self.metadata_req_pool_indices.append(
            forward_batch.req_pool_indices.detach().cpu().tolist()
        )


class FakeCodePredictor:
    def __init__(self):
        self.seen_batch_size = None
        self.seen_req_pool_indices = None
        self.seen_explicit_req_pool_indices = None
        self.seen_forward_batch = None

    def generate_batched(
        self,
        input_embeds,
        forward_batch,
        num_tokens,
        do_sample,
        top_k,
        top_p,
        req_pool_indices=None,
    ):
        self.seen_forward_batch = forward_batch
        self.seen_batch_size = forward_batch.batch_size
        self.seen_req_pool_indices = forward_batch.req_pool_indices.cpu().tolist()
        self.seen_explicit_req_pool_indices = req_pool_indices.cpu().tolist()
        return (
            torch.tensor([[101, 201], [102, 202]]),
            torch.arange(input_embeds.shape[0] * input_embeds.shape[-1]).view(
                input_embeds.shape[0], input_embeds.shape[-1]
            ),
        )


class FakeAudioPhaseScheduler(SchedulerQwen3OmniMixin, SchedulerOutputProcessorMixin):
    def __init__(self):
        self.enable_overlap = False
        self.qwen3_omni_audio_phase_queue = deque()
        self.qwen3_omni_code2wav_pending_work = []
        self.qwen3_omni_code2wav_event_pool = []
        self.qwen3_omni_code2wav_stream = None
        self.audio_turn_reqs = {}
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
        self.last_batch = None
        self.code2wav = FakeCode2Wav()
        self.streams = []
        self.finished_audio_reqs = []
        self.code_predictor_batch_calls = []
        self.tp_worker = SimpleNamespace(
            calls=[],
            run_qwen3_omni_audio_phase_item=self.run_audio_phase_item,
            run_qwen3_omni_code_predictor_payload_batch=(
                self.run_code_predictor_payload_batch
            ),
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    run_qwen3_omni_audio_phase_after_sample=(
                        lambda *args: self.tp_worker.run_qwen3_omni_audio_phase_item(
                            *args
                        )
                    ),
                    run_qwen3_omni_code_predictor_payload_batch=(
                        lambda *args: self.tp_worker.run_qwen3_omni_code_predictor_payload_batch(
                            *args
                        )
                    ),
                ),
                qwen3_omni_token_to_kv_pool_allocators={
                    "talker": FakeAllocator(),
                }
            ),
        )

    def run_audio_phase_item(
        self,
        forward_batch,
        logits_output,
        next_token_ids,
        phase,
        request_indices,
        payload,
    ):
        self.tp_worker.calls.append(
            (forward_batch, logits_output, next_token_ids, phase, request_indices, payload)
        )
        return logits_output

    def run_code_predictor_payload_batch(self, forward_batch, payload):
        self.code_predictor_batch_calls.append((forward_batch, payload))
        codec_frames = [
            [int(token), int(token) + 10] for token in payload.codec_tokens.tolist()
        ]
        prev_embeds = torch.arange(
            payload.input_embeds.shape[0] * payload.input_embeds.shape[-1],
            dtype=payload.input_embeds.dtype,
        ).view(payload.input_embeds.shape[0], payload.input_embeds.shape[-1])
        return codec_frames, prev_embeds

    def _qwen3_omni_component_kv_stats(self):
        return "talker:0/10"

    def _qwen3_omni_audio_turn_stats(self):
        return (0, 0)

    def _get_code2wav_model(self):
        return self.code2wav

    def _drain_qwen3_omni_audio_phase_queue(self, result, return_logprob, reqs):
        for req in reqs:
            req.audio_input_chunk_idx = getattr(req, "audio_input_chunk_idx", 0)
        return self._enqueue_qwen3_omni_audio_phase_queue(
            result, return_logprob, reqs
        )

    def stream_output(self, reqs, return_logprob, skip_req=None):
        self.streams.append(
            (
                [req.rid for req in reqs],
                return_logprob,
                [req.talker_pcm16_chunks.copy() for req in reqs],
            )
        )
        for req in reqs:
            req.talker_pcm16_chunks.clear()

    def _handle_finished_req(self, req, i, logits_output, finished_code2wav_reqs=None):
        self.finished_audio_reqs.append(req.rid)
        self._drain_qwen3_omni_code2wav_phase([req], self.code2wav, flush=True)
        req.qwen3_omni_audio_phase = Qwen3OmniAudioPhase.DONE


def make_req(frames, last_decoded_frame=0, rid="turn-a"):
    return SimpleNamespace(
        rid=rid,
        codec_eos_token_id=2150,
        talker_output_codes=frames,
        talker_last_decoded_frame=last_decoded_frame,
        talker_pcm16_chunks=[],
        talker_pcm16_chunk_cap=1000,
        qwen3_omni_audio_frame_lag=4,
        qwen3_omni_audio_phase=None,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        qwen3_omni_code2wav_pending=False,
        finished=lambda: False,
    )


def set_audio_ready_state(req, ready_state):
    req.qwen3_omni_audio_ready_state = ready_state
    req.qwen3_omni_audio_ready_states = [ready_state]
    text_lag_tokens = getattr(req, "qwen3_omni_text_lag_tokens", deque())
    if isinstance(text_lag_tokens, list):
        text_lag_tokens = deque(text_lag_tokens)
    req.qwen3_omni_text_lag_tokens = text_lag_tokens
    if not hasattr(req, "audio_input_done"):
        req.audio_input_done = (
            ready_state.committed_audio_input
            if isinstance(ready_state, Qwen3OmniTalkerPrefillReady)
            else True
        )
    req.qwen3_omni_partial_talker_prefill_chunk_idx = getattr(
        req, "qwen3_omni_partial_talker_prefill_chunk_idx", 0
    )
    req.qwen3_omni_audio_frame_lag = getattr(req, "qwen3_omni_audio_frame_lag", 0)
    req.qwen3_omni_code2wav_pending = getattr(
        req, "qwen3_omni_code2wav_pending", False
    )
    req.thinker_done = getattr(req, "thinker_done", False)
    req.qwen3_omni_tts_eos_pending = getattr(
        req, "qwen3_omni_tts_eos_pending", False
    )
    req.talker_pcm16_chunks = getattr(req, "talker_pcm16_chunks", [])
    req.return_logprob = getattr(req, "return_logprob", False)
    return ready_state


def test_qwen3_omni_streamed_audio_mrope_uses_token_lens():
    audio_start_token_id = 151646
    audio_token_id = 151647
    input_ids = torch.tensor(
        [
            [
                10,
                audio_start_token_id,
                11,
                audio_start_token_id,
                *([audio_token_id] * 30),
                12,
            ]
        ],
        dtype=torch.long,
    )

    positions, delta = get_rope_index_qwen3_omni(
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        input_ids=input_ids,
        audio_seqlens=torch.tensor([438], dtype=torch.long),
        audio_token_lens=torch.tensor([30], dtype=torch.long),
        audio_token_id=audio_token_id,
        audio_start_token_id=audio_start_token_id,
        position_id_per_seconds=25,
        tokens_per_second=25,
    )

    assert positions.shape == (3, 1, input_ids.shape[1])
    assert delta.shape == (1, 1)


def test_qwen3_omni_streamed_audio_mrope_accepts_open_audio_span():
    audio_start_token_id = 151646
    audio_token_id = 151647
    input_ids = torch.tensor(
        [[10, audio_start_token_id, *([audio_token_id] * 30)]],
        dtype=torch.long,
    )

    positions, delta = get_rope_index_qwen3_omni(
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        input_ids=input_ids,
        audio_seqlens=torch.tensor([438], dtype=torch.long),
        audio_token_lens=torch.tensor([30], dtype=torch.long),
        audio_token_id=audio_token_id,
        audio_start_token_id=audio_start_token_id,
        position_id_per_seconds=25,
        tokens_per_second=25,
    )

    assert positions.shape == (3, 1, input_ids.shape[1])
    assert delta.shape == (1, 1)


def test_code2wav_phase_batches_matching_windows():
    code2wav = FakeCode2Wav()
    frames = [[frame, frame + 10] for frame in range(4)]
    reqs = [make_req(frames), make_req(frames)]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(reqs, code2wav)
    )

    assert decoded_reqs == 2
    assert decoded_frames == 8
    assert emitted_chunks == 2
    assert batch_groups == 1
    assert len(code2wav.calls) == 1
    assert code2wav.calls[0].shape == (2, 2, 4)
    assert [req.talker_last_decoded_frame for req in reqs] == [4, 4]
    assert [len(req.talker_pcm16_chunks) for req in reqs] == [1, 1]
    assert all(
        req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.CODE2WAV_QUEUED
        for req in reqs
    )


def test_code2wav_flush_batches_matching_tail_windows():
    code2wav = FakeCode2Wav()
    frames = [[frame, frame + 10] for frame in range(2)]
    reqs = [make_req(frames), make_req(frames)]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(reqs, code2wav, flush=True)
    )

    assert decoded_reqs == 2
    assert decoded_frames == 4
    assert emitted_chunks == 2
    assert batch_groups == 1
    assert len(code2wav.calls) == 1
    assert code2wav.calls[0].shape == (2, 2, 2)
    assert [req.talker_last_decoded_frame for req in reqs] == [2, 2]


def test_code2wav_flush_honors_max_batch_size():
    code2wav = FakeCode2Wav()
    frames = [[frame, frame + 10] for frame in range(2)]
    reqs = [make_req(frames) for _ in range(8)]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(
            reqs,
            code2wav,
            flush=True,
            max_batch_size=4,
        )
    )

    assert decoded_reqs == 8
    assert decoded_frames == 16
    assert emitted_chunks == 8
    assert batch_groups == 2
    assert [call.shape for call in code2wav.calls] == [(4, 2, 2), (4, 2, 2)]


def test_code2wav_flush_batches_ragged_tail_windows():
    code2wav = FakeCode2Wav()
    reqs = [
        make_req([[frame, frame + 10] for frame in range(1)], rid="turn-a"),
        make_req([[frame, frame + 10] for frame in range(3)], rid="turn-b"),
        make_req([[frame, frame + 10] for frame in range(4)], rid="turn-c"),
    ]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(reqs, code2wav, flush=True)
    )

    assert decoded_reqs == 3
    assert decoded_frames == 8
    assert emitted_chunks == 3
    assert batch_groups == 1
    assert len(code2wav.calls) == 1
    assert code2wav.calls[0].shape == (3, 2, 4)
    assert [req.talker_last_decoded_frame for req in reqs] == [1, 3, 4]
    assert [len(req.talker_pcm16_chunks[0]) for req in reqs] == [4, 12, 16]


def test_code2wav_ragged_crop_uses_output_length_map():
    code2wav = OffsetLengthFakeCode2Wav()
    reqs = [
        make_req(
            [[frame, frame + 10] for frame in range(3)],
            last_decoded_frame=1,
            rid="turn-a",
        ),
        make_req([[frame, frame + 10] for frame in range(4)], rid="turn-b"),
    ]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(reqs, code2wav, flush=True)
    )

    assert decoded_reqs == 2
    assert decoded_frames == 6
    assert emitted_chunks == 2
    assert batch_groups == 1
    assert code2wav.calls[0].shape == (2, 2, 4)
    pcm = array.array("h")
    pcm.frombytes(reqs[0].talker_pcm16_chunks[0])
    assert list(pcm) == [
        int(0.007 * 32767),
        int(0.008 * 32767),
        int(0.009 * 32767),
        int(0.010 * 32767),
    ]


def test_code2wav_phase_batches_different_windows_by_right_padding():
    code2wav = FakeCode2Wav()
    reqs = [
        make_req([[frame, frame + 10] for frame in range(4)]),
        make_req([[frame, frame + 10] for frame in range(5)], last_decoded_frame=1),
    ]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(reqs, code2wav, chunk_size=4)
    )

    assert decoded_reqs == 2
    assert decoded_frames == 8
    assert emitted_chunks == 2
    assert batch_groups == 1
    assert len(code2wav.calls) == 1
    assert code2wav.calls[0].shape == (2, 2, 5)
    assert [req.talker_last_decoded_frame for req in reqs] == [4, 5]


def test_code2wav_phase_decodes_one_streaming_chunk_at_a_time():
    code2wav = FakeCode2Wav()
    frames = [[frame, frame + 10] for frame in range(8)]
    req = make_req(frames)

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch([req], code2wav)
    )

    assert decoded_reqs == 1
    assert decoded_frames == 4
    assert emitted_chunks == 1
    assert batch_groups == 1
    assert code2wav.calls[0].shape == (1, 2, 4)
    assert req.talker_last_decoded_frame == 4
    assert req.qwen3_omni_audio_frame_lag == 4


def test_code2wav_steady_chunk_keeps_first_chunk_small(monkeypatch):
    monkeypatch.setattr(
        scheduler_qwen3_omni_mixin,
        "CODE2WAV_STEADY_CHUNK_FRAMES",
        8,
    )
    first_req = make_req([[frame, frame + 10] for frame in range(4)])
    steady_req = make_req(
        [[frame, frame + 10] for frame in range(11)],
        last_decoded_frame=4,
    )

    assert scheduler_output_processor_mixin.should_run_code2wav_phase(first_req)
    assert not scheduler_output_processor_mixin.should_run_code2wav_phase(steady_req)

    steady_req.talker_output_codes.append([11, 21])

    assert scheduler_output_processor_mixin.should_run_code2wav_phase(steady_req)
    first_item = build_code2wav_work_item(first_req)
    steady_item = build_code2wav_work_item(steady_req)
    assert first_item is not None
    assert steady_item is not None
    assert first_item.new_frames == 4
    assert first_item.current_frame == 4
    assert steady_item.new_frames == 8
    assert steady_item.current_frame == 12


def test_code2wav_scheduler_drain_catches_up_in_streaming_chunks():
    scheduler = FakeAudioPhaseScheduler()
    frames = [[frame, frame + 10] for frame in range(8)]
    req = make_req(frames)

    scheduler._drain_qwen3_omni_code2wav_phase([req], scheduler.code2wav)

    assert [call.shape for call in scheduler.code2wav.calls] == [
        (1, 2, 4),
        (1, 2, 8),
    ]
    assert req.talker_last_decoded_frame == 8
    assert req.qwen3_omni_audio_frame_lag == 0
    assert len(req.talker_pcm16_chunks) == 2
    assert req.qwen3_omni_audio_ready_state is None


def test_code2wav_scheduler_drain_caps_steady_batch_size():
    scheduler = FakeAudioPhaseScheduler()
    frames = [[frame, frame + 10] for frame in range(4)]
    reqs = [make_req(frames, rid=f"turn-{idx}") for idx in range(9)]

    scheduler._drain_qwen3_omni_code2wav_phase(reqs, scheduler.code2wav)

    assert [call.shape for call in scheduler.code2wav.calls] == [
        (8, 2, 4),
        (1, 2, 4),
    ]
    assert [req.talker_last_decoded_frame for req in reqs] == [4] * 9


def test_async_code2wav_launch_splits_decoder_only():
    code2wav = SplitDecoderFakeCode2Wav()
    frames = [[frame, frame + 10] for frame in range(4)]
    reqs = [make_req(frames, rid=f"turn-{idx}") for idx in range(9)]

    (
        pending_work,
        decoded_reqs,
        decoded_frames,
        emitted_chunks,
        batch_groups,
        work_group_build_ms,
    ) = (
        launch_code2wav_phase_batch(
            reqs,
            code2wav,
            stream=None,
            decoder_max_batch_size=8,
        )
    )

    assert decoded_reqs == 9
    assert decoded_frames == 36
    assert emitted_chunks == 9
    assert batch_groups == 1
    assert work_group_build_ms >= 0
    assert len(pending_work) == 1
    assert [call.shape for call in code2wav.calls] == [(9, 2, 4)]
    assert code2wav.decoder_max_batch_sizes == [8]
    assert all(req.qwen3_omni_code2wav_pending for req in reqs)


def test_code2wav_ready_reqs_excludes_pending_async_work():
    scheduler = FakeAudioPhaseScheduler()
    req = make_req([[frame, frame + 10] for frame in range(4)], rid="turn-pending")
    req.qwen3_omni_code2wav_pending = True
    scheduler.audio_turn_reqs = {req.rid: req}

    assert scheduler._qwen3_omni_code2wav_ready_reqs() == []


def test_finished_req_waits_for_residual_code2wav_flush():
    req = make_req(
        [[frame, frame + 10] for frame in range(35)],
        last_decoded_frame=32,
        rid="turn-flush",
    )
    req.finished = lambda: True

    assert qwen3_omni_finished_waiting_for_code2wav(req)

    req.talker_last_decoded_frame = len(req.talker_output_codes)
    assert not qwen3_omni_finished_waiting_for_code2wav(req)


def test_finished_req_waits_for_inflight_code2wav_flush():
    req = make_req([[frame, frame + 10] for frame in range(4)], rid="turn-inflight")
    req.finished = lambda: True
    req.talker_last_decoded_frame = len(req.talker_output_codes)
    req.qwen3_omni_code2wav_pending = True

    assert qwen3_omni_finished_waiting_for_code2wav(req)


def test_code2wav_ready_reqs_synthesizes_finished_flush_state():
    scheduler = FakeAudioPhaseScheduler()
    req = make_req(
        [[frame, frame + 10] for frame in range(35)],
        last_decoded_frame=32,
        rid="turn-finished-flush",
    )
    req.finished = lambda: True
    scheduler.audio_turn_reqs = {req.rid: req}

    ready_reqs = scheduler._qwen3_omni_code2wav_ready_reqs()

    assert ready_reqs == [req]
    ready_state = req.qwen3_omni_audio_ready_states[0]
    assert isinstance(ready_state, Qwen3OmniCode2WavReady)
    assert ready_state.flush


def test_code2wav_lane_drains_finished_flush_ready_req():
    scheduler = FakeAudioPhaseScheduler()
    req = make_req(
        [[frame, frame + 10] for frame in range(35)],
        last_decoded_frame=32,
        rid="turn-finished-flush",
    )
    req.finished = lambda: True
    req.return_logprob = False
    scheduler.audio_turn_reqs = {req.rid: req}

    assert scheduler._run_qwen3_omni_code2wav_lane_if_ready()

    assert req.talker_last_decoded_frame == 35
    assert req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.DONE
    assert len(scheduler.streams) == 1
    assert scheduler.streams[0][0] == ["turn-finished-flush"]


def test_code2wav_lane_streams_completed_async_work_once():
    scheduler = FakeAudioPhaseScheduler()
    req = make_req([[frame, frame + 10] for frame in range(4)], rid="turn-async")
    req.return_logprob = False
    item = build_code2wav_work_item(req)
    pending = launch_code2wav_work_group(scheduler.code2wav, [item], stream=None)
    scheduler.qwen3_omni_code2wav_pending_work = [pending]

    assert req.qwen3_omni_code2wav_pending
    assert scheduler._run_qwen3_omni_code2wav_lane_if_ready()

    assert not req.qwen3_omni_code2wav_pending
    assert scheduler.qwen3_omni_code2wav_pending_work == []
    assert len(scheduler.streams) == 1
    assert scheduler.streams[0][0] == ["turn-async"]
    assert req.talker_pcm16_chunks == []
    assert req.talker_last_decoded_frame == 4
    assert not scheduler._run_qwen3_omni_code2wav_lane_if_ready()


def test_scheduler_audio_lane_drains_code2wav_ready_turns():
    scheduler = FakeAudioPhaseScheduler()
    frames = [[frame, frame + 10] for frame in range(4)]
    reqs = [
        make_req(frames, rid="turn-a"),
        make_req(frames, rid="turn-b"),
    ]
    ready_at = time.perf_counter() - (AUDIO_BATCH_DELAY_MS / 1000) - 0.001
    for req in reqs:
        req.return_logprob = False
        ready_state = (
            scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
                req,
                ready_at,
                flush=False,
            )
        )
        assert ready_state.codes_window == tuple(tuple(frame) for frame in frames)
    scheduler.audio_turn_reqs = {req.rid: req for req in reqs}

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    assert scheduler._run_qwen3_omni_audio_lane_if_selected() is True
    assert len(scheduler.code2wav.calls) == 1
    assert scheduler.code2wav.calls[0].shape == (2, 2, 4)
    [(rids, return_logprob, audio_chunks)] = scheduler.streams
    assert rids == ["turn-a", "turn-b"]
    assert return_logprob is False
    assert [len(chunks) for chunks in audio_chunks] == [1, 1]
    assert [req.talker_last_decoded_frame for req in reqs] == [4, 4]


def test_code2wav_phase_groups_first_chunk_from_uneven_lag():
    code2wav = FakeCode2Wav()
    reqs = [
        make_req([[frame, frame + 10] for frame in range(8)]),
        make_req([[frame, frame + 10] for frame in range(4)]),
    ]

    decoded_reqs, decoded_frames, emitted_chunks, batch_groups = (
        run_code2wav_phase_batch(reqs, code2wav)
    )

    assert decoded_reqs == 2
    assert decoded_frames == 8
    assert emitted_chunks == 2
    assert batch_groups == 1
    assert code2wav.calls[0].shape == (2, 2, 4)
    assert [req.talker_last_decoded_frame for req in reqs] == [4, 4]
    assert [req.qwen3_omni_audio_frame_lag for req in reqs] == [4, 0]


def test_code2wav_rotary_broadcasts_batch_and_heads():
    q = torch.ones(2, 4, 3, 8)
    k = torch.ones(2, 4, 3, 8)
    cos = torch.ones(2, 3, 8)
    sin = torch.zeros(2, 3, 8)

    q_embed, k_embed = _apply_rotary_pos_emb(q, k, cos, sin)

    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape


def test_skipped_audio_prefill_releases_talker_locs():
    locs = torch.tensor([3, 4, 5])
    logits_output = SimpleNamespace(talker_out_cache_loc_list=[None, locs])
    allocator = FakeAllocator()

    released = release_skipped_qwen3_omni_audio_prefill_locs(
        logits_output,
        1,
        allocator,
    )

    assert released
    assert len(allocator.freed) == 1
    assert allocator.freed[0] is locs
    assert logits_output.talker_out_cache_loc_list == [None, None]


def test_skipped_audio_prefill_release_is_noop_without_locs():
    logits_output = SimpleNamespace(talker_out_cache_loc_list=[None])
    allocator = FakeAllocator()

    released = release_skipped_qwen3_omni_audio_prefill_locs(
        logits_output,
        0,
        allocator,
    )

    assert not released
    assert allocator.freed == []


def test_finished_audio_req_releases_talker_component_kv(monkeypatch):
    locs = torch.tensor([11, 12, 13])
    allocator = FakeAllocator()
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        ),
        enable_hisparse=False,
        tree_cache=SimpleNamespace(),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                qwen3_omni_token_to_kv_pool_allocators={"talker": allocator}
            )
        ),
        maybe_collect_routed_experts=lambda req: None,
        maybe_collect_indexer_topk=lambda req: None,
        maybe_collect_customized_info=lambda i, req, logits_output: None,
        _drain_qwen3_omni_code2wav_phase=lambda reqs, code2wav, flush=False: None,
        _get_code2wav_model=lambda: None,
    )
    released_normal_kv = []
    monkeypatch.setattr(
        scheduler_output_processor_mixin,
        "release_kv_cache",
        lambda req, tree_cache: released_normal_kv.append(req.rid),
    )
    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        talker_kv_cache_locs=locs,
        talker_prefill_len=3,
        talker_user_prefill_len=0,
        talker_prefill_input_ids=torch.tensor([1, 2, 3]),
        finished_reason=None,
        thinker_done=True,
        origin_input_ids=[1, 2, 3],
        output_ids=[4],
        cache_protected_len=0,
        kv_committed_len=3,
        kv_allocated_len=3,
        req_pool_idx=0,
        multimodal_inputs=None,
        session=None,
        qwen3_omni_code2wav_pending=False,
        talker_output_codes=[],
        talker_last_decoded_frame=0,
        finished=lambda: True,
        time_stats=SimpleNamespace(set_completion_time=lambda: None),
    )

    SchedulerOutputProcessorMixin._handle_finished_req(
        scheduler, req, 0, SimpleNamespace()
    )

    assert allocator.freed == [locs]
    assert req.talker_kv_cache_locs is None
    assert req.talker_prefill_len == 0
    assert req.talker_prefill_input_ids is None
    assert released_normal_kv == ["turn-a"]


def test_qwen3_omni_component_kv_release_is_audio_only():
    allocator = FakeAllocator()
    req = SimpleNamespace(
        codec_eos_token_id=None,
        talker_kv_cache_locs=torch.tensor([1]),
    )
    model_runner = SimpleNamespace(
        qwen3_omni_token_to_kv_pool_allocators={"talker": allocator}
    )

    assert not release_qwen3_omni_component_kv(req, model_runner)
    assert allocator.freed == []


def test_qwen3_omni_finished_req_releases_post_thinker_kv_tail():
    allocator = FakeAllocator()
    req_to_token = torch.arange(100, 120, dtype=torch.int64).unsqueeze(0)
    tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool_allocator=allocator,
    )
    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=True,
        origin_input_ids=[1, 2, 3, 4, 5],
        output_ids=[6, 7, 8],
        cache_protected_len=5,
        kv_committed_len=11,
        kv_allocated_len=11,
        req_pool_idx=0,
    )

    assert release_qwen3_omni_post_thinker_kv(req, tree_cache)

    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == [107, 108, 109, 110]
    assert req.kv_committed_len == 7
    assert req.kv_allocated_len == 7


def test_qwen3_omni_post_thinker_kv_release_requires_actual_token_prefix():
    allocator = FakeAllocator()
    tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(20, dtype=torch.int64).unsqueeze(0)
        ),
        token_to_kv_pool_allocator=allocator,
    )
    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=True,
        origin_input_ids=[1, 2, 3],
        output_ids=[4],
        cache_protected_len=5,
        kv_committed_len=7,
        kv_allocated_len=7,
        req_pool_idx=0,
    )

    with pytest.raises(RuntimeError, match="protected Thinker KV"):
        release_qwen3_omni_post_thinker_kv(req, tree_cache)


def test_partial_audio_prefill_preserves_thinker_accept_hidden(monkeypatch):
    cached_reqs = []

    def fake_cache(req, tree_cache, chunked=False):
        cached_reqs.append((req, chunked))

    monkeypatch.setattr(
        scheduler_output_processor_mixin,
        "maybe_cache_unfinished_req",
        fake_cache,
    )

    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    logits_output = SimpleNamespace(
        talker_needs_prefill=True,
        prompt_accept_hidden_states=[hidden],
        codec_frames=None,
        tts_pad_embed=None,
        talker_out_cache_loc_list=[None],
        talker_mrope_position_delta_list=None,
        updated_talker_kv_locs_list=None,
        code_predictor_inputs=None,
        next_token_logprobs=None,
        input_token_logprobs=None,
        next_token_top_logprobs_val=None,
        next_token_token_ids_logprobs_val=None,
        customized_info=None,
        hidden_states=None,
    )
    req = SimpleNamespace(
        rid="turn-a",
        is_retracted=False,
        is_chunked=0,
        _prefill_audio_done_snapshot=False,
        audio_input_done=False,
        audio_input_prefill_inflight=True,
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        fill_ids=[1],
        qwen3_omni_prompt_accept_hidden=None,
        talker_needs_prefill=False,
        qwen3_omni_audio_phase=None,
        session=SimpleNamespace(streaming=True),
        req_pool_idx=None,
        finished=lambda: False,
    )
    freed_reqs = []
    scheduler = SimpleNamespace(
        is_generation=True,
        enable_overlap=False,
        qwen3_omni_audio_phase_queue=deque(),
        waiting_queue=[],
        tree_cache=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(
            free=lambda freed_req: freed_reqs.append(freed_req)
        ),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                qwen3_omni_token_to_kv_pool_allocators={"talker": FakeAllocator()}
            )
        ),
        stream_output=lambda reqs, return_logprob, skip_req=None: None,
        report_prefill_stats=lambda **kwargs: None,
        _enqueue_qwen3_omni_audio_phase_queue=lambda *args, **kwargs: [],
        _drain_qwen3_omni_audio_phase_queue=lambda *args, **kwargs: None,
        _drain_qwen3_omni_code2wav_phase=lambda *args, **kwargs: None,
        _get_code2wav_model=lambda: None,
    )
    batch = SimpleNamespace(
        reqs=[req],
        return_logprob=False,
        decoding_reqs=[],
        prefill_stats=None,
        dp_cooperation_info=None,
    )
    result = SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=logits_output,
        next_token_ids=torch.tensor([17]),
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        can_run_cuda_graph=False,
    )

    SchedulerOutputProcessorMixin.process_batch_result_prefill(
        scheduler,
        batch,
        result,
    )

    assert torch.equal(req.qwen3_omni_prompt_accept_hidden, hidden)
    assert req.talker_needs_prefill is True
    assert req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.TALKER_PREFILL_QUEUED
    assert req.audio_input_prefill_inflight is False
    assert scheduler.waiting_queue == [req]
    assert cached_reqs == [(req, True)]
    assert freed_reqs == []


def test_uncommitted_audio_prefill_drops_code_predictor_inputs():
    logits_output = SimpleNamespace(
        code_predictor_inputs=SimpleNamespace(
            request_indices=[0, 1],
            input_embeds=torch.arange(8).view(2, 2, 2),
            codec_tokens=torch.tensor([11, 12]),
        )
    )
    reqs = [
        SimpleNamespace(_prefill_audio_done_snapshot=False),
        SimpleNamespace(_prefill_audio_done_snapshot=False),
    ]

    pending = filter_uncommitted_qwen3_omni_code_predictor_inputs(
        logits_output,
        reqs,
    )

    assert pending is None
    assert logits_output.code_predictor_inputs is None


def test_uncommitted_audio_prefill_filters_code_predictor_inputs():
    input_embeds = torch.arange(12).view(3, 2, 2)
    logits_output = SimpleNamespace(
        code_predictor_inputs=SimpleNamespace(
            request_indices=[0, 1, 2],
            input_embeds=input_embeds,
            codec_tokens=torch.tensor([11, 12, 13]),
        )
    )
    reqs = [
        SimpleNamespace(_prefill_audio_done_snapshot=True),
        SimpleNamespace(_prefill_audio_done_snapshot=False),
        SimpleNamespace(_prefill_audio_done_snapshot=True),
    ]

    pending = filter_uncommitted_qwen3_omni_code_predictor_inputs(
        logits_output,
        reqs,
    )

    assert pending is logits_output.code_predictor_inputs
    assert pending.request_indices == [0, 2]
    assert pending.input_embeds.tolist() == input_embeds[[0, 2]].tolist()
    assert pending.codec_tokens.tolist() == [11, 13]


def test_code_predictor_inputs_filter_by_request_indices():
    input_embeds = torch.arange(12).view(3, 2, 2)
    logits_output = SimpleNamespace(
        code_predictor_inputs=SimpleNamespace(
            request_indices=[0, 1, 2],
            input_embeds=input_embeds,
            codec_tokens=torch.tensor([11, 12, 13]),
        )
    )

    pending = filter_qwen3_omni_code_predictor_inputs_by_request_indices(
        logits_output,
        [0, 2],
    )

    assert pending is logits_output.code_predictor_inputs
    assert pending.request_indices == [0, 2]
    assert pending.input_embeds.tolist() == input_embeds[[0, 2]].tolist()
    assert pending.codec_tokens.tolist() == [11, 13]


def test_deferred_code_predictor_phase_uses_request_index_subset():
    code_predictor = FakeCodePredictor()
    model = SimpleNamespace(
        talker=SimpleNamespace(
            config=SimpleNamespace(
                code_predictor_config=SimpleNamespace(num_code_groups=3)
            ),
            code_predictor=code_predictor,
        ),
    )
    model._temporary_audio_req_pool_indices = MethodType(
        Qwen3OmniMoeForConditionalGeneration._temporary_audio_req_pool_indices,
        model,
    )
    model.run_qwen3_omni_code_predictor_payload_batch = MethodType(
        Qwen3OmniMoeForConditionalGeneration.run_qwen3_omni_code_predictor_payload_batch,
        model,
    )
    attn_backend = FakeAttnBackend()
    forward_batch = SimpleNamespace(
        batch_size=3,
        req_pool_indices=torch.tensor([10, 11, 12]),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((16, 8), dtype=torch.int32),
            free_slots=[1, 2, 3],
        ),
        attn_backend=attn_backend,
    )
    payload = Qwen3OmniCodePredictorPayload(
        request_indices=[0, 2],
        input_embeds=torch.zeros(2, 2, 4),
        codec_tokens=torch.tensor([31, 32]),
    )

    phase_output = Qwen3OmniMoeForConditionalGeneration._run_code_predictor_phase(
        model,
        forward_batch,
        [0, 2],
        payload,
    )

    assert code_predictor.seen_batch_size == 3
    assert code_predictor.seen_forward_batch is not forward_batch
    assert code_predictor.seen_req_pool_indices == [10, 11, 12]
    assert code_predictor.seen_explicit_req_pool_indices == [1, 2]
    assert forward_batch.req_to_token_pool.free_slots == [1, 2, 3]
    assert forward_batch.batch_size == 3
    assert forward_batch.req_pool_indices.tolist() == [10, 11, 12]
    assert attn_backend.metadata_batch_sizes == []
    assert attn_backend.metadata_req_pool_indices == []
    assert phase_output.codec_frames == [[31, 101, 201], None, [32, 102, 202]]


def test_deferred_talker_prefill_phase_uses_audio_forward_batch_view():
    seen = {}

    def run_prefill(
        self,
        thinker_logits,
        forward_batch,
        payload,
        request_indices,
        run_code_predictor,
        sample_codec,
        restore_forward_metadata,
    ):
        seen["forward_batch"] = forward_batch
        seen["thinker_logits"] = thinker_logits
        seen["payload"] = payload
        seen["request_indices"] = request_indices
        seen["run_code_predictor"] = run_code_predictor
        seen["sample_codec"] = sample_codec
        seen["restore_forward_metadata"] = restore_forward_metadata
        forward_batch.batch_size = 99
        return TalkerPrefillPhaseOutput(
            codec_frames=[[7]],
            tts_pad_embed=None,
            talker_out_cache_loc_list=[None],
            talker_mrope_position_delta_list=[None],
            talker_prefill_input_ids_list=[None],
            talker_user_prefill_len_list=[None],
            prev_codec_input_embeds_list=[None],
            code_predictor_inputs=None,
        )

    model = SimpleNamespace()
    model._run_talker_prefill_at_decode = MethodType(run_prefill, model)
    model._apply_talker_prefill_phase_output = MethodType(
        Qwen3OmniMoeForConditionalGeneration._apply_talker_prefill_phase_output,
        model,
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        model_specific_states={},
    )
    logits_output = ThinkerTalkerOutput(thinker_logits=None)
    payload = Qwen3OmniTalkerPrefillPayload(
        origin_input_ids=[torch.tensor([101, 102])],
        origin_input_ids_cpu=[[101, 102]],
        multimodal_inputs=None,
        prev_sampled_thinker_token=torch.tensor([17]),
        prompt_accept_hidden_states=None,
        tts_pad_embed=None,
        codec_eos_token_id=2150,
        talker_temperature=0.9,
        talker_top_k=50,
        talker_top_p=1.0,
        talker_repetition_penalty=1.05,
        talker_prev_codec_tokens=None,
        speakers=["chelsie"],
        talker_kv_cache_locs_list=[],
        talker_prefill_input_ids_list=[],
    )

    merged = Qwen3OmniMoeForConditionalGeneration.run_qwen3_omni_audio_phase_after_sample(
        model,
        forward_batch,
        logits_output,
        torch.tensor([17]),
        "talker_prefill",
        [0],
        payload,
    )

    assert seen["forward_batch"] is not forward_batch
    assert seen["thinker_logits"] is None
    assert seen["request_indices"] == [0]
    assert seen["run_code_predictor"] is False
    assert seen["sample_codec"] is True
    assert seen["restore_forward_metadata"] is False
    assert seen["payload"].origin_input_ids == payload.origin_input_ids
    assert forward_batch.batch_size == 1
    assert merged.codec_frames == [[7]]


def test_deferred_talker_prefill_phase_uses_owned_audio_batch_size():
    seen = {}

    def run_prefill(
        self,
        thinker_logits,
        forward_batch,
        payload,
        request_indices,
        run_code_predictor,
        sample_codec,
        restore_forward_metadata,
    ):
        seen["forward_batch_size"] = forward_batch.batch_size
        seen["payload_origin_len"] = len(payload.origin_input_ids)
        seen["request_indices"] = list(request_indices)
        seen["restore_forward_metadata"] = restore_forward_metadata
        return TalkerPrefillPhaseOutput(
            codec_frames=[[7], [8]],
            tts_pad_embed=None,
            talker_out_cache_loc_list=[None, None],
            talker_mrope_position_delta_list=[None, None],
            talker_prefill_input_ids_list=[None, None],
            talker_user_prefill_len_list=[None, None],
            prev_codec_input_embeds_list=[None, None],
            code_predictor_inputs=None,
        )

    model = SimpleNamespace()
    model._run_talker_prefill_at_decode = MethodType(run_prefill, model)
    model._apply_talker_prefill_phase_output = MethodType(
        Qwen3OmniMoeForConditionalGeneration._apply_talker_prefill_phase_output,
        model,
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        model_specific_states={},
    )
    payload = Qwen3OmniTalkerPrefillPayload(
        origin_input_ids=[torch.tensor([101]), torch.tensor([102])],
        origin_input_ids_cpu=[[101], [102]],
        multimodal_inputs=None,
        prev_sampled_thinker_token=torch.tensor([17, 18]),
        prompt_accept_hidden_states=None,
        tts_pad_embed=None,
        codec_eos_token_id=2150,
        talker_temperature=0.9,
        talker_top_k=50,
        talker_top_p=1.0,
        talker_repetition_penalty=1.05,
        talker_prev_codec_tokens=None,
        speakers=["chelsie", "chelsie"],
        talker_kv_cache_locs_list=[None, None],
        talker_prefill_input_ids_list=[None, None],
    )

    merged = Qwen3OmniMoeForConditionalGeneration.run_qwen3_omni_audio_phase_after_sample(
        model,
        forward_batch,
        ThinkerTalkerOutput(thinker_logits=None),
        torch.tensor([17, 18]),
        "talker_prefill",
        [0, 1],
        payload,
    )

    assert seen["forward_batch_size"] == 2
    assert seen["payload_origin_len"] == 2
    assert seen["request_indices"] == [0, 1]
    assert seen["restore_forward_metadata"] is False
    assert forward_batch.batch_size == 1
    assert merged.codec_frames == [[7], [8]]


def test_deferred_talker_decode_phase_uses_owned_audio_batch_size():
    seen = {}

    def run_decode(
        self,
        forward_batch,
        payload,
        request_indices,
        run_code_predictor,
        restore_forward_metadata,
    ):
        seen["forward_batch_size"] = forward_batch.batch_size
        seen["request_indices"] = list(request_indices)
        seen["restore_forward_metadata"] = restore_forward_metadata
        return (
            [[7], [8]],
            [torch.tensor([1]), torch.tensor([2])],
            [torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)],
            [torch.ones(2), torch.ones(2) * 2],
            None,
        )

    model = SimpleNamespace()
    model._run_talker_decode_batched_subset = MethodType(run_decode, model)
    model._apply_talker_decode_phase_output = MethodType(
        Qwen3OmniMoeForConditionalGeneration._apply_talker_decode_phase_output,
        model,
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    payload = Qwen3OmniTalkerDecodePayload(
        prev_codec_id=torch.tensor([1, 2]),
        prev_residual_codes=torch.tensor([[3, 4], [5, 6]]),
        prev_codec_input_embeds=torch.ones(2, 2),
        talker_positions=torch.tensor([[0, 1]]),
        talker_kv_cache_locs_list=[torch.tensor([1]), torch.tensor([2])],
        thinker_done=None,
        talker_post_thinker_started=None,
        talker_has_text_lag=None,
        talker_needs_tts_eos=None,
        tts_pad_embed=None,
        prev_sampled_thinker_token=None,
        codec_eos_token_id=2150,
        talker_temperature=0.9,
        talker_top_k=50,
        talker_top_p=1.0,
        talker_repetition_penalty=1.05,
        talker_prev_codec_tokens=[[], []],
        **_talker_sampler_state(2),
    )

    merged = Qwen3OmniMoeForConditionalGeneration.run_qwen3_omni_audio_phase_after_sample(
        model,
        forward_batch,
        ThinkerTalkerOutput(thinker_logits=None),
        torch.tensor([1, 2]),
        "talker_decode",
        [0, 1],
        payload,
    )

    assert seen["forward_batch_size"] == 2
    assert seen["request_indices"] == [0, 1]
    assert seen["restore_forward_metadata"] is False
    assert forward_batch.batch_size == 1
    assert merged.codec_frames == [[7], [8]]
    assert torch.equal(merged.updated_talker_kv_locs_list[1], torch.tensor([2]))


def test_deferred_code_predictor_phase_uses_owned_audio_batch_size():
    def run_code_predictor(self, forward_batch, payload):
        assert payload.request_indices == [0, 1]
        return [[7, 1], [8, 2]], torch.ones(2, 3)

    model = SimpleNamespace()
    model.run_qwen3_omni_code_predictor_payload_batch = MethodType(
        run_code_predictor, model
    )
    model._apply_code_predictor_phase_output = MethodType(
        Qwen3OmniMoeForConditionalGeneration._apply_code_predictor_phase_output,
        model,
    )
    forward_batch = SimpleNamespace(batch_size=1)
    payload = Qwen3OmniCodePredictorPayload(
        request_indices=[0, 1],
        input_embeds=torch.ones(2, 2, 3),
        codec_tokens=torch.tensor([7, 8]),
    )

    phase_output = Qwen3OmniMoeForConditionalGeneration._run_code_predictor_phase(
        model,
        forward_batch,
        [0, 1],
        payload,
    )
    merged = Qwen3OmniMoeForConditionalGeneration._apply_code_predictor_phase_output(
        model,
        ThinkerTalkerOutput(thinker_logits=None),
        phase_output,
    )

    assert merged.codec_frames == [[7, 1], [8, 2]]
    assert len(merged.prev_codec_input_embeds_list) == 2


def test_talker_prefill_extends_existing_partial_talker_kv_prefix():
    seen = {}

    def prepare_prefill(
        self,
        accept_hidden,
        input_ids,
        clamped_input_ids,
        position_input_ids,
        input_ids_cpu,
        position_input_ids_cpu,
        first_response_token,
        speaker,
        include_assistant_response,
        skip_talker_prefix_len=0,
    ):
        embeds = torch.arange(10, dtype=torch.float32).view(5, 2)
        return (
            embeds[skip_talker_prefix_len:],
            torch.arange(5, dtype=torch.long),
            torch.ones(2),
            3,
        )

    def mrope_positions(self, talker_input_ids, multimodal_input):
        return talker_input_ids.unsqueeze(0), torch.tensor([0])

    def mrope_input_ids(self, raw_input_ids, multimodal_input):
        return raw_input_ids

    def save_state(self, forward_batch, save_extend=False, save_req_to_tokens=True):
        return {
            "batch_size": forward_batch.batch_size,
            "req_pool_indices": forward_batch.req_pool_indices.clone(),
            "seq_lens": forward_batch.seq_lens.clone(),
            "seq_lens_cpu": forward_batch.seq_lens_cpu.clone(),
            "seq_lens_sum": forward_batch.seq_lens_sum,
            "out_cache_loc": forward_batch.out_cache_loc,
            "forward_mode": forward_batch.forward_mode,
        }

    def restore_state(
        self, forward_batch, saved_state, restore_attention_metadata=True
    ):
        seen["restore_attention_metadata"] = restore_attention_metadata
        for key, value in saved_state.items():
            setattr(forward_batch, key, value)

    class FakeTalkerBody:
        def __call__(self, input_ids, positions, forward_batch, input_embeds):
            seen["positions"] = positions.detach().clone()
            seen["input_embeds"] = input_embeds.detach().clone()
            seen["seq_lens"] = forward_batch.seq_lens.detach().clone()
            seen["req_pool_indices"] = forward_batch.req_pool_indices.cpu().tolist()
            seen["extend_seq_lens"] = forward_batch.extend_seq_lens.detach().clone()
            seen["extend_prefix_lens"] = (
                forward_batch.extend_prefix_lens.detach().clone()
            )
            seen["override_req_to_token"] = (
                forward_batch.override_req_to_token.detach().clone()
            )
            seen["out_cache_loc"] = forward_batch.out_cache_loc.detach().clone()
            return input_embeds

    model = SimpleNamespace()
    model.thinker = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(16, 2))
    )
    model.talker = SimpleNamespace(model=FakeTalkerBody())
    model._prepare_talker_prefill_1step = MethodType(prepare_prefill, model)
    model._qwen3_omni_mrope_input_ids = MethodType(mrope_input_ids, model)
    model._qwen3_omni_mrope_positions = MethodType(mrope_positions, model)
    model._save_forward_batch_state = MethodType(save_state, model)
    model._restore_forward_batch_state = MethodType(restore_state, model)
    model._temporary_audio_req_pool_indices = MethodType(
        Qwen3OmniMoeForConditionalGeneration._temporary_audio_req_pool_indices,
        model,
    )
    model._run_talker_model_body = MethodType(
        Qwen3OmniMoeForConditionalGeneration._run_talker_model_body,
        model,
    )

    allocator = FakeSequentialAllocator(start=10)
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        seq_lens_sum=1,
        req_pool_indices=torch.tensor([7]),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((8, 8), dtype=torch.int32),
            free_slots=[1, 2, 3],
        ),
        out_cache_loc=torch.tensor([99]),
        forward_mode=SimpleNamespace(name="DECODE"),
        qwen3_omni_token_to_kv_pool_allocators={"talker": allocator},
        attn_backend=FakeAttnBackend(),
    )
    previous_locs = torch.tensor([1, 2])
    payload = Qwen3OmniTalkerPrefillPayload(
        origin_input_ids=[torch.tensor([1, 2, 3, 4, 5])],
        origin_input_ids_cpu=[[1, 2, 3, 4, 5]],
        multimodal_inputs=None,
        prev_sampled_thinker_token=torch.tensor([17]),
        prompt_accept_hidden_states=None,
        tts_pad_embed=None,
        codec_eos_token_id=2150,
        talker_temperature=0.9,
        talker_top_k=50,
        talker_top_p=1.0,
        talker_repetition_penalty=1.05,
        talker_prev_codec_tokens=None,
        speakers=["chelsie"],
        talker_kv_cache_locs_list=[previous_locs],
        talker_prefill_input_ids_list=[torch.tensor([0, 1])],
    )

    (
        codec_frames,
        tts_pad_embed,
        talker_out_cache_locs,
        _talker_mrope_position_deltas,
        talker_prefill_input_ids,
        talker_user_prefill_lens,
        prev_codec_input_embeds,
        code_predictor_inputs,
    ) = Qwen3OmniMoeForConditionalGeneration._run_talker_prefill_batched_subset(
        model,
        forward_batch,
        payload,
        [0],
        run_code_predictor=False,
        sample_codec=False,
    )

    assert codec_frames == [None]
    assert torch.equal(tts_pad_embed, torch.ones(1, 2))
    assert torch.equal(talker_out_cache_locs[0], torch.tensor([1, 2, 10, 11, 12]))
    assert torch.equal(talker_prefill_input_ids[0], torch.arange(5))
    assert talker_user_prefill_lens == [3]
    assert prev_codec_input_embeds == [None]
    assert code_predictor_inputs is None
    assert torch.equal(seen["input_embeds"], torch.arange(4, 10).view(3, 2))
    assert torch.equal(seen["positions"], torch.tensor([[2, 3, 4]]))
    assert seen["req_pool_indices"] == [1]
    assert torch.equal(seen["seq_lens"], torch.tensor([5], dtype=torch.int32))
    assert torch.equal(seen["extend_seq_lens"], torch.tensor([3], dtype=torch.int32))
    assert torch.equal(seen["extend_prefix_lens"], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(seen["override_req_to_token"], torch.tensor([[1, 2, 10, 11, 12]]))
    assert torch.equal(seen["out_cache_loc"], torch.tensor([10, 11, 12]))
    assert torch.equal(
        forward_batch.req_to_token_pool.req_to_token[7, :5],
        torch.zeros(5, dtype=torch.int32),
    )
    assert seen["restore_attention_metadata"] is True
    assert forward_batch.req_to_token_pool.free_slots == [1, 2, 3]
    assert forward_batch.batch_size == 1
    assert torch.equal(forward_batch.req_pool_indices, torch.tensor([7]))


def test_talker_prefill_fails_fast_when_cached_input_ids_are_not_prefix():
    seen = {}

    def prepare_prefill(
        self,
        accept_hidden,
        input_ids,
        clamped_input_ids,
        position_input_ids,
        input_ids_cpu,
        position_input_ids_cpu,
        first_response_token,
        speaker,
        include_assistant_response,
        skip_talker_prefix_len=0,
    ):
        embeds = torch.arange(10, dtype=torch.float32).view(5, 2)
        return embeds, torch.arange(5, dtype=torch.long), torch.ones(2), 3

    def save_state(self, forward_batch, save_extend=False, save_req_to_tokens=True):
        return {
            "batch_size": forward_batch.batch_size,
            "req_pool_indices": forward_batch.req_pool_indices.clone(),
            "seq_lens": forward_batch.seq_lens.clone(),
            "seq_lens_cpu": forward_batch.seq_lens_cpu.clone(),
            "seq_lens_sum": forward_batch.seq_lens_sum,
            "out_cache_loc": forward_batch.out_cache_loc,
            "forward_mode": forward_batch.forward_mode,
        }

    def restore_state(
        self, forward_batch, saved_state, restore_attention_metadata=True
    ):
        for key, value in saved_state.items():
            setattr(forward_batch, key, value)

    class FakeTalkerBody:
        def __call__(self, input_ids, positions, forward_batch, input_embeds):
            seen["input_embeds"] = input_embeds.detach().clone()
            seen["extend_prefix_lens"] = (
                forward_batch.extend_prefix_lens.detach().clone()
            )
            seen["override_req_to_token"] = (
                forward_batch.override_req_to_token.detach().clone()
            )
            return input_embeds

    model = SimpleNamespace()
    model.thinker = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(16, 2))
    )
    model.talker = SimpleNamespace(model=FakeTalkerBody())
    model._prepare_talker_prefill_1step = MethodType(prepare_prefill, model)
    model._qwen3_omni_mrope_input_ids = MethodType(
        lambda self, raw_input_ids, multimodal_input: raw_input_ids,
        model,
    )
    model._qwen3_omni_mrope_positions = MethodType(
        lambda self, talker_input_ids, multimodal_input: (
            talker_input_ids.unsqueeze(0),
            torch.tensor([0]),
        ),
        model,
    )
    model._save_forward_batch_state = MethodType(save_state, model)
    model._restore_forward_batch_state = MethodType(restore_state, model)
    model._run_talker_model_body = MethodType(
        Qwen3OmniMoeForConditionalGeneration._run_talker_model_body,
        model,
    )

    allocator = FakeSequentialAllocator(start=10)
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        seq_lens_sum=1,
        req_pool_indices=torch.tensor([7]),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((8, 8), dtype=torch.int32),
            free_slots=[1, 2, 3],
        ),
        out_cache_loc=torch.tensor([99]),
        forward_mode=SimpleNamespace(name="DECODE"),
        qwen3_omni_token_to_kv_pool_allocators={"talker": allocator},
        attn_backend=FakeAttnBackend(),
    )
    previous_locs = torch.tensor([1, 2])
    payload = Qwen3OmniTalkerPrefillPayload(
        origin_input_ids=[torch.tensor([1, 2, 3, 4, 5])],
        origin_input_ids_cpu=[[1, 2, 3, 4, 5]],
        multimodal_inputs=None,
        prev_sampled_thinker_token=torch.tensor([17]),
        prompt_accept_hidden_states=None,
        tts_pad_embed=None,
        codec_eos_token_id=2150,
        talker_temperature=0.9,
        talker_top_k=50,
        talker_top_p=1.0,
        talker_repetition_penalty=1.05,
        talker_prev_codec_tokens=None,
        speakers=["chelsie"],
        talker_kv_cache_locs_list=[previous_locs],
        talker_prefill_input_ids_list=[torch.tensor([9, 9])],
    )

    with pytest.raises(RuntimeError, match="not a prefix"):
        Qwen3OmniMoeForConditionalGeneration._run_talker_prefill_batched_subset(
            model,
            forward_batch,
            payload,
            [0],
            run_code_predictor=False,
            sample_codec=False,
        )


def test_talker_prefill_chunk_contract_is_token_space():
    input_ids = torch.tensor(
        [10, 11, 91, 10, 12, 92, 93, 10, 13, 94],
        dtype=torch.long,
    )
    position_input_ids = torch.tensor(
        [10, 11, 91, 10, 12, 151675, 151675, 10, 13, 94],
        dtype=torch.long,
    )

    contract = build_qwen3_omni_talker_prefill_chunk_contract(
        input_ids=input_ids,
        position_input_ids=position_input_ids,
        input_ids_cpu=input_ids.tolist(),
        position_input_ids_cpu=position_input_ids.tolist(),
        im_start_token_id=10,
        system_token_id=11,
        user_token_id=12,
        assistant_token_id=13,
        tts_pad_token_id=21,
        include_assistant_response=True,
    )

    assert contract.chunks == (
        Qwen3OmniTalkerPrefillUserChunk(
            source_start=3,
            source_end=7,
            talker_start=0,
            talker_end=4,
        ),
        Qwen3OmniTalkerPrefillAssistantChunk(
            source_start=7,
            source_end=10,
            talker_start=4,
            talker_end=13,
        ),
    )
    assert contract.talker_input_ids.tolist() == [
        10,
        12,
        151675,
        151675,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
        21,
    ]
    assert contract.talker_input_ids.device.type == "cpu"


def test_talker_prefill_chunk_contract_requires_chatml():
    with pytest.raises(RuntimeError, match="requires ChatML"):
        build_qwen3_omni_talker_prefill_chunk_contract(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            position_input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            input_ids_cpu=[1, 2, 3],
            im_start_token_id=10,
            system_token_id=11,
            user_token_id=12,
            assistant_token_id=13,
            tts_pad_token_id=21,
            include_assistant_response=True,
        )


def test_mrope_input_ids_canonicalizes_streaming_audio_placeholders_without_offsets():
    model = SimpleNamespace(
        config=SimpleNamespace(
            thinker_config=SimpleNamespace(
                audio_start_token_id=151669,
                audio_token_id=151675,
                image_token_id=151655,
                video_token_id=151656,
            )
        )
    )
    model._qwen3_omni_has_only_audio_multimodal_input = MethodType(
        Qwen3OmniMoeForConditionalGeneration._qwen3_omni_has_only_audio_multimodal_input,
        model,
    )
    input_ids = torch.tensor(
        [10, 12, 198, 151669, MM_PAD_SHIFT_VALUE + 123, MM_PAD_SHIFT_VALUE + 124, 198],
        dtype=torch.long,
    )

    mrope_input_ids = Qwen3OmniMoeForConditionalGeneration._qwen3_omni_mrope_input_ids(
        model, input_ids, multimodal_input=None
    )

    assert mrope_input_ids.tolist() == [10, 12, 198, 151669, 151675, 151675, 198]


def test_mrope_input_ids_fails_on_unmapped_multimodal_placeholder():
    model = SimpleNamespace(
        config=SimpleNamespace(
            thinker_config=SimpleNamespace(
                audio_start_token_id=151669,
                audio_token_id=151675,
                image_token_id=151655,
                video_token_id=151656,
            )
        )
    )
    model._qwen3_omni_has_only_audio_multimodal_input = MethodType(
        Qwen3OmniMoeForConditionalGeneration._qwen3_omni_has_only_audio_multimodal_input,
        model,
    )

    with pytest.raises(RuntimeError, match="could not map multimodal placeholder"):
        Qwen3OmniMoeForConditionalGeneration._qwen3_omni_mrope_input_ids(
            model,
            torch.tensor([10, 12, MM_PAD_SHIFT_VALUE + 123], dtype=torch.long),
            multimodal_input=SimpleNamespace(
                mm_items=[
                    SimpleNamespace(
                        is_audio=lambda: False,
                        is_image=lambda: True,
                        is_video=lambda: False,
                        offsets=[],
                    )
                ]
            ),
        )


def test_talker_prefill_uses_accept_hidden_for_multimodal_user_tokens():
    hidden_size = 2
    model = SimpleNamespace()
    model.config = SimpleNamespace(
        im_start_token_id=10,
        system_token_id=11,
        user_token_id=12,
        assistant_token_id=13,
        tts_bos_token_id=20,
        tts_pad_token_id=21,
        thinker_config=SimpleNamespace(
            audio_token_id=151675,
            image_token_id=151655,
            video_token_id=151656,
        ),
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size),
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=5,
            codec_bos_id=6,
            speaker_id={"chelsie": 7},
        ),
    )
    model._qwen3_omni_multimodal_mask = MethodType(
        Qwen3OmniMoeForConditionalGeneration._qwen3_omni_multimodal_mask,
        model,
    )
    model.thinker = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(64, hidden_size))
    )
    model.talker = SimpleNamespace(
        text_projection=AddProjection(100),
        hidden_projection=AddProjection(200),
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(16, hidden_size)),
    )

    mm_pad_value = 1_000_123
    input_ids = torch.tensor(
        [10, 12, 198, mm_pad_value, 151670, 151645, 198, 10, 13, 198],
        dtype=torch.long,
    )
    thinker_embeds = torch.arange(
        input_ids.numel() * hidden_size, dtype=torch.float32
    ).view(input_ids.numel(), hidden_size)
    accept_hidden = thinker_embeds + 1000

    (
        talker_input,
        talker_input_ids,
        _tts_pad_embed,
        _user_prefill_len,
    ) = Qwen3OmniMoeForConditionalGeneration._prepare_talker_prefill_1step(
        model,
        accept_hidden=accept_hidden,
        input_ids=input_ids,
        clamped_input_ids=input_ids.clamp(
            min=0, max=model.thinker.model.embed_tokens.num_embeddings - 1
        ),
        position_input_ids=input_ids,
        input_ids_cpu=input_ids.tolist(),
        first_response_token=torch.tensor(30),
        speaker="chelsie",
    )

    expected_text = model.thinker.model.embed_tokens(input_ids[0:1])[0] + 100
    assert torch.equal(talker_input[0], expected_text)
    assert torch.equal(talker_input[3], accept_hidden[3] + 200)
    assert talker_input_ids.tolist()[-9:] == [21] * 9


def test_talker_prefill_requires_accept_hidden_for_multimodal_user_tokens():
    hidden_size = 2
    model = SimpleNamespace()
    model.config = SimpleNamespace(
        im_start_token_id=10,
        system_token_id=11,
        user_token_id=12,
        assistant_token_id=13,
        tts_bos_token_id=20,
        tts_pad_token_id=21,
        thinker_config=SimpleNamespace(
            audio_token_id=151675,
            image_token_id=151655,
            video_token_id=151656,
        ),
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size),
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=5,
            codec_bos_id=6,
            speaker_id={"chelsie": 7},
        ),
    )
    model._qwen3_omni_multimodal_mask = MethodType(
        Qwen3OmniMoeForConditionalGeneration._qwen3_omni_multimodal_mask,
        model,
    )
    model.thinker = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(64, hidden_size))
    )
    model.talker = SimpleNamespace(
        text_projection=AddProjection(100),
        hidden_projection=AddProjection(200),
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(16, hidden_size)),
    )
    input_ids = torch.tensor(
        [10, 12, 198, 1_000_123, 151670, 151645, 198, 10, 13, 198],
        dtype=torch.long,
    )
    thinker_embeds = torch.zeros(input_ids.numel(), hidden_size)

    with pytest.raises(RuntimeError, match="accept-hidden"):
        Qwen3OmniMoeForConditionalGeneration._prepare_talker_prefill_1step(
            model,
            accept_hidden=None,
            input_ids=input_ids,
            clamped_input_ids=input_ids.clamp(
                min=0, max=model.thinker.model.embed_tokens.num_embeddings - 1
            ),
            position_input_ids=input_ids,
            input_ids_cpu=input_ids.tolist(),
            first_response_token=torch.tensor(30),
            speaker="chelsie",
        )


def test_talker_prefill_selection_prefers_committed_prefill():
    reqs = [
        SimpleNamespace(
            _prefill_audio_done_snapshot=False,
            qwen3_omni_partial_talker_prefill_warmed=False,
        ),
        SimpleNamespace(
            _prefill_audio_done_snapshot=False,
            qwen3_omni_partial_talker_prefill_warmed=True,
        ),
        SimpleNamespace(
            _prefill_audio_done_snapshot=True,
            qwen3_omni_partial_talker_prefill_warmed=True,
        ),
    ]

    request_indices = filter_qwen3_omni_committed_talker_prefills(
        reqs,
        [0, 1, 2],
    )

    assert request_indices == [2]


def test_talker_prefill_selection_skips_uncommitted_prefill():
    reqs = [
        SimpleNamespace(
            _prefill_audio_done_snapshot=False,
            qwen3_omni_partial_talker_prefill_warmed=False,
        ),
        SimpleNamespace(
            _prefill_audio_done_snapshot=False,
            qwen3_omni_partial_talker_prefill_warmed=True,
        ),
        SimpleNamespace(
            _prefill_audio_done_snapshot=False,
            qwen3_omni_partial_talker_prefill_warmed=False,
            talker_kv_cache_locs=torch.tensor([3, 4, 5]),
        ),
    ]

    request_indices = filter_qwen3_omni_committed_talker_prefills(
        reqs,
        [0, 1, 2],
    )

    assert request_indices == []


def test_uncommitted_talker_prefill_is_not_enqueued():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(talker_needs_prefill=True)
    next_token_ids = torch.tensor([17])
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="EXTEND"),
        model_specific_states={
            "origin_input_ids": [torch.tensor([101, 102])],
            "origin_input_ids_cpu": [[101, 102]],
        },
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=next_token_ids,
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    reqs = [
        SimpleNamespace(
            rid="turn-a",
            audio_input_done=False,
            _prefill_audio_done_snapshot=False,
            qwen3_omni_partial_talker_prefill_warmed=False,
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
            speaker="chelsie",
            is_chunked=0,
        )
    ]

    scheduler._drain_qwen3_omni_audio_phase_queue(result, False, reqs)

    assert scheduler.qwen3_omni_audio_phase_queue == deque()
    assert reqs[0].qwen3_omni_audio_ready_state is None


def test_rebuild_talker_prefill_is_split_from_normal_prefill():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(
        talker_needs_prefill=True,
        prompt_accept_hidden_states=[torch.ones(2, 3), torch.ones(2, 3) * 2],
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="EXTEND"),
        model_specific_states={
            "origin_input_ids": [torch.tensor([101, 102]), torch.tensor([201, 202])],
            "origin_input_ids_cpu": [[101, 102], [201, 202]],
        },
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=torch.tensor([17, 18]),
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    reqs = [
        SimpleNamespace(
            rid="rollback-rebuild",
            qwen3_omni_rebuild_request=True,
            audio_input_done=True,
            _prefill_audio_done_snapshot=True,
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
            speaker="chelsie",
            is_chunked=0,
        ),
        SimpleNamespace(
            rid="normal-turn",
            audio_input_done=True,
            _prefill_audio_done_snapshot=True,
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
            speaker="chelsie",
            is_chunked=0,
        ),
    ]

    scheduler._drain_qwen3_omni_audio_phase_queue(result, False, reqs)

    queued = list(scheduler.qwen3_omni_audio_phase_queue)
    assert [item.request_indices for item in queued] == [[0], [1]]
    assert [item.request_rids for item in queued] == [
        ["rollback-rebuild"],
        ["normal-turn"],
    ]
    assert [item.payload.sample_codec for item in queued] == [False, True]


def test_streaming_talker_prefill_output_commits_kv_without_finishing_prefill():
    scheduler = FakeAudioPhaseScheduler()
    locs = torch.tensor([3, 4, 5])
    logits_output = SimpleNamespace(
        talker_needs_prefill=False,
        prompt_accept_hidden_states=None,
        codec_frames=None,
        prev_codec_input_embeds_list=None,
        tts_pad_embed=None,
        talker_out_cache_loc_list=[locs],
        talker_mrope_position_delta_list=None,
        updated_talker_kv_locs_list=None,
    )
    req = SimpleNamespace(
        rid="turn-a",
        audio_input_done=False,
        _prefill_audio_done_snapshot=False,
        qwen3_omni_partial_talker_prefill_warmed=False,
        qwen3_omni_partial_talker_prefill_chunk_idx=0,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        qwen3_omni_audio_phase=None,
        talker_kv_cache_locs=None,
        talker_prefill_len=0,
        talker_needs_prefill=True,
        audio_input_chunk_idx=0,
    )
    ready_state = Qwen3OmniTalkerPrefillReady("turn-a", 0, 1.0, False, 0)
    set_audio_ready_state(req, ready_state)
    item = Qwen3OmniAudioPhaseItem(
        logits_output=logits_output,
        forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="EXTEND")),
        next_token_ids=torch.tensor([17]),
        reqs=[req],
        return_logprob=False,
        phase="talker_prefill",
        request_indices=[0],
        request_rids=["turn-a"],
        payload=Qwen3OmniTalkerPrefillPayload(
            origin_input_ids=[torch.tensor([101, 102])],
            origin_input_ids_cpu=[[101, 102]],
            multimodal_inputs=None,
            prev_sampled_thinker_token=torch.tensor([17]),
            prompt_accept_hidden_states=None,
            tts_pad_embed=None,
            codec_eos_token_id=2150,
            talker_temperature=0.9,
            talker_top_k=50,
            talker_top_p=1.0,
            talker_repetition_penalty=1.05,
            talker_prev_codec_tokens=None,
            speakers=["chelsie"],
            talker_kv_cache_locs_list=[],
            talker_prefill_input_ids_list=[],
            sample_codec=False,
        ),
        ready_states=[ready_state],
    )

    affected_reqs = scheduler._apply_qwen3_omni_audio_phase_output(item)

    assert affected_reqs == []
    assert req.talker_kv_cache_locs is locs
    assert req.talker_prefill_len == 3
    assert req.talker_needs_prefill is True
    assert req.qwen3_omni_partial_talker_prefill_warmed is False
    assert req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.TALKER_PREFILL_QUEUED


def test_rebuild_talker_prefill_finishes_without_codec_sampling():
    scheduler = FakeAudioPhaseScheduler()
    locs = torch.tensor([3, 4, 5])
    input_ids = torch.tensor([13, 14, 15])
    logits_output = SimpleNamespace(
        talker_needs_prefill=False,
        prompt_accept_hidden_states=None,
        codec_frames=[None],
        prev_codec_input_embeds_list=None,
        tts_pad_embed=None,
        talker_out_cache_loc_list=[locs],
        talker_prefill_input_ids_list=[input_ids],
        talker_user_prefill_len_list=[3],
        talker_mrope_position_delta_list=None,
        updated_talker_kv_locs_list=None,
    )
    req = SimpleNamespace(
        rid="safe-rid",
        qwen3_omni_rebuild_request=True,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        qwen3_omni_audio_phase=None,
        talker_kv_cache_locs=None,
        talker_prefill_len=0,
        talker_user_prefill_len=0,
        talker_prefill_input_ids=None,
        talker_needs_prefill=True,
        output_ids=[],
        finished_reason=None,
        codec_eos_token_id=2150,
        talker_output_codes=[],
        talker_last_decoded_frame=0,
        talker_pcm16_chunks=[],
        talker_pcm16_chunk_cap=1000,
        qwen3_omni_audio_frame_lag=0,
        qwen3_omni_code2wav_pending=False,
    )
    req.finished = lambda: req.finished_reason is not None
    ready_state = Qwen3OmniTalkerPrefillReady("safe-rid", 0, 1.0, True, 0)
    set_audio_ready_state(req, ready_state)
    item = Qwen3OmniAudioPhaseItem(
        logits_output=logits_output,
        forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="EXTEND")),
        next_token_ids=torch.tensor([17]),
        reqs=[req],
        return_logprob=False,
        phase="talker_prefill",
        request_indices=[0],
        request_rids=["safe-rid"],
        payload=Qwen3OmniTalkerPrefillPayload(
            origin_input_ids=[torch.tensor([101, 102])],
            origin_input_ids_cpu=[[101, 102]],
            multimodal_inputs=None,
            prev_sampled_thinker_token=torch.tensor([17]),
            prompt_accept_hidden_states=None,
            tts_pad_embed=None,
            codec_eos_token_id=2150,
            talker_temperature=0.9,
            talker_top_k=50,
            talker_top_p=1.0,
            talker_repetition_penalty=1.05,
            talker_prev_codec_tokens=None,
            speakers=["chelsie"],
            talker_kv_cache_locs_list=[],
            talker_prefill_input_ids_list=[],
            sample_codec=False,
        ),
        ready_states=[ready_state],
    )

    affected_reqs = scheduler._apply_qwen3_omni_audio_phase_output(item)

    assert affected_reqs == [req]
    assert scheduler.finished_audio_reqs == ["safe-rid"]
    assert req.finished()
    assert req.finished_len == 0
    assert req.talker_kv_cache_locs is locs
    assert torch.equal(req.talker_prefill_input_ids, input_ids)
    assert req.talker_user_prefill_len == 3


def test_audio_phase_log_stats_report_component_kv_and_turn_queue():
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                qwen3_omni_token_to_kv_pool_allocators={
                    "talker": FakeKVAllocator(size=10, free=8),
                    "code_predictor": FakeKVAllocator(size=6, free=1),
                }
            )
        ),
        audio_turn_reqs={
            "active-a": SimpleNamespace(),
            "queued-b": SimpleNamespace(),
        },
        waiting_queue=[
            SimpleNamespace(rid="queued-b"),
            SimpleNamespace(rid="text-c"),
        ],
    )

    assert (
        SchedulerQwen3OmniMixin._qwen3_omni_component_kv_stats(scheduler)
        == "code_predictor:5/1,talker:2/8"
    )
    assert SchedulerQwen3OmniMixin._qwen3_omni_audio_turn_stats(scheduler) == (
        2,
        1,
    )


def test_audio_phase_queue_purge_filters_aborted_request():
    ready_states = [
        Qwen3OmniCodePredictorReady("turn-a", 0, 1.0),
        Qwen3OmniCodePredictorReady("turn-b", 1, 1.0),
        Qwen3OmniCodePredictorReady("turn-c", 2, 1.0),
    ]
    logits_output = SimpleNamespace(
        code_predictor_inputs=SimpleNamespace(
            request_indices=[0, 1, 2],
            input_embeds=torch.arange(12).view(3, 2, 2),
            codec_tokens=torch.tensor([11, 12, 13]),
        )
    )
    result = SimpleNamespace(logits_output=logits_output)
    scheduler = SimpleNamespace(
        qwen3_omni_audio_phase_queue=deque(
            [
                Qwen3OmniAudioPhaseItem(
                    logits_output=logits_output,
                    forward_batch=SimpleNamespace(),
                    next_token_ids=torch.tensor([1, 2, 3]),
                    reqs=[
                        SimpleNamespace(
                            rid="turn-a",
                            qwen3_omni_audio_ready_state=ready_states[0],
                            qwen3_omni_audio_ready_states=[ready_states[0]],
                        ),
                        SimpleNamespace(
                            rid="turn-b",
                            qwen3_omni_audio_ready_state=ready_states[1],
                            qwen3_omni_audio_ready_states=[ready_states[1]],
                        ),
                        SimpleNamespace(
                            rid="turn-c",
                            qwen3_omni_audio_ready_state=ready_states[2],
                            qwen3_omni_audio_ready_states=[ready_states[2]],
                        ),
                    ],
                    return_logprob=False,
                    phase="code_predictor",
                    request_indices=[0, 1, 2],
                    request_rids=["turn-a", "turn-b", "turn-c"],
                    payload=Qwen3OmniCodePredictorPayload(
                        request_indices=[0, 1, 2],
                        input_embeds=logits_output.code_predictor_inputs.input_embeds,
                        codec_tokens=logits_output.code_predictor_inputs.codec_tokens,
                    ),
                    ready_states=ready_states,
                )
            ]
        )
    )

    SchedulerQwen3OmniMixin._purge_qwen3_omni_audio_phase_queue(
        scheduler,
        SimpleNamespace(abort_all=False, rid="turn-b"),
    )

    [item] = scheduler.qwen3_omni_audio_phase_queue
    assert item.request_indices == [0, 2]
    assert item.request_rids == ["turn-a", "turn-c"]
    assert [state.request_rid for state in item.ready_states] == ["turn-a", "turn-c"]
    assert logits_output.code_predictor_inputs.request_indices == [0, 2]
    assert logits_output.code_predictor_inputs.codec_tokens.tolist() == [11, 13]
    assert item.payload.request_indices == [0, 2]


def test_audio_phase_queue_abort_all_consumes_owned_ready_states():
    ready_states = [
        Qwen3OmniCodePredictorReady("turn-a", 0, 1.0),
        Qwen3OmniCodePredictorReady("turn-b", 1, 1.0),
    ]
    reqs = [
        SimpleNamespace(
            rid="turn-a",
            qwen3_omni_audio_ready_state=ready_states[0],
            qwen3_omni_audio_ready_states=[ready_states[0]],
        ),
        SimpleNamespace(
            rid="turn-b",
            qwen3_omni_audio_ready_state=ready_states[1],
            qwen3_omni_audio_ready_states=[ready_states[1]],
        ),
    ]
    scheduler = SimpleNamespace(
        qwen3_omni_audio_phase_queue=deque(
            [
                Qwen3OmniAudioPhaseItem(
                    logits_output=SimpleNamespace(),
                    forward_batch=SimpleNamespace(),
                    next_token_ids=torch.tensor([1, 2]),
                    reqs=reqs,
                    return_logprob=False,
                    phase="code_predictor",
                    request_indices=[0, 1],
                    request_rids=["turn-a", "turn-b"],
                    payload=Qwen3OmniCodePredictorPayload(
                        request_indices=[0, 1],
                        input_embeds=torch.zeros(2, 4),
                        codec_tokens=torch.tensor([11, 12]),
                    ),
                    ready_states=ready_states,
                )
            ]
        )
    )

    SchedulerQwen3OmniMixin._purge_qwen3_omni_audio_phase_queue(
        scheduler,
        SimpleNamespace(abort_all=True, rid=""),
    )

    assert list(scheduler.qwen3_omni_audio_phase_queue) == []
    assert [req.qwen3_omni_audio_ready_state for req in reqs] == [None, None]
    assert [req.qwen3_omni_audio_ready_states for req in reqs] == [[], []]


def test_audio_phase_item_owns_forward_batch_and_next_tokens():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(code_predictor_inputs=None)
    next_token_ids = torch.tensor([17])
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        req_pool_indices=torch.tensor([10], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="DECODE"),
        positions=torch.tensor([1]),
        model_specific_states={
            "prev_codec_id": torch.tensor([11]),
            "prev_residual_codes": torch.tensor([[1, 2]]),
            "prev_codec_input_embeds": torch.zeros(1, 4),
            "talker_positions": torch.tensor([1]),
            "talker_kv_cache_locs_list": [torch.tensor([5])],
            "prev_sampled_thinker_token": torch.tensor([13]),
            "thinker_done": torch.tensor([False]),
            "talker_has_text_lag": torch.tensor([False]),
            "talker_can_decode": torch.tensor([True]),
            **_talker_sampler_state(1),
        },
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=next_token_ids,
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    reqs = [
        SimpleNamespace(
            rid="turn-a",
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
        )
    ]

    scheduler._enqueue_qwen3_omni_audio_phase_queue(result, False, reqs)
    assert isinstance(reqs[0].qwen3_omni_audio_ready_state, Qwen3OmniTalkerDecodeReady)
    assert reqs[0].qwen3_omni_audio_ready_state.request_rid == "turn-a"
    assert result.qwen3_omni_audio_forward_batch is None
    result.logits_output = SimpleNamespace(code_predictor_inputs="mutated")
    forward_batch.seq_lens[0] = 99
    forward_batch.req_pool_indices[0] = 99
    forward_batch.model_specific_states["prev_codec_id"] = torch.tensor([99])
    next_token_ids[0] = 99
    scheduler._drain_pending_qwen3_omni_audio_phase_queue()

    [
        (
            captured_forward_batch,
            captured_logits,
            captured_next_tokens,
            phase,
            indices,
            payload,
        )
    ] = (
        scheduler.tp_worker.calls
    )
    assert captured_forward_batch is not forward_batch
    assert captured_forward_batch.seq_lens.tolist() == [1]
    assert captured_forward_batch.req_pool_indices.tolist() == [10]
    assert captured_logits is logits_output
    assert captured_next_tokens.tolist() == [17]
    assert phase == "talker_decode"
    assert indices == [0]
    assert payload.prev_codec_id.tolist() == [11]
    assert payload.prev_sampled_thinker_token.tolist() == [17]


def test_audio_phase_item_fails_if_request_ready_state_changes_before_launch():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(code_predictor_inputs=None)
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        req_pool_indices=torch.tensor([10], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="DECODE"),
        positions=torch.tensor([1]),
        model_specific_states={
            "prev_codec_id": torch.tensor([11]),
            "prev_residual_codes": torch.tensor([[1, 2]]),
            "prev_codec_input_embeds": torch.zeros(1, 4),
            "talker_positions": torch.tensor([1]),
            "talker_kv_cache_locs_list": [torch.tensor([5])],
            "prev_sampled_thinker_token": torch.tensor([13]),
            "thinker_done": torch.tensor([False]),
            "talker_has_text_lag": torch.tensor([False]),
            "talker_can_decode": torch.tensor([True]),
            **_talker_sampler_state(1),
        },
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=torch.tensor([17]),
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    reqs = [
        SimpleNamespace(
            rid="turn-a",
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
        )
    ]

    scheduler._enqueue_qwen3_omni_audio_phase_queue(result, False, reqs)
    reqs[0].qwen3_omni_audio_ready_states.clear()

    with pytest.raises(RuntimeError, match="stale"):
        scheduler._drain_pending_qwen3_omni_audio_phase_queue()
    assert not scheduler.tp_worker.calls


def test_streaming_talker_prefill_owns_payload_after_live_ready_state_moves():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(
        rid="turn-a",
        audio_input_done=False,
        qwen3_omni_partial_talker_prefill_chunk_idx=0,
        finished=lambda: False,
    )
    ready_state = Qwen3OmniTalkerPrefillReady("turn-a", 0, 1.0, False, 1)
    req.qwen3_omni_audio_ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, 2.0)
    req.qwen3_omni_audio_ready_states = [ready_state]
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=SimpleNamespace(),
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="EXTEND")),
            next_token_ids=torch.tensor([17]),
            reqs=[req],
            return_logprob=False,
            phase="talker_prefill",
            request_indices=[0],
            request_rids=["turn-a"],
            payload=Qwen3OmniTalkerPrefillPayload(
                origin_input_ids=[torch.tensor([101, 102])],
                origin_input_ids_cpu=[[101, 102]],
                multimodal_inputs=None,
                prev_sampled_thinker_token=torch.tensor([17]),
                prompt_accept_hidden_states=None,
                tts_pad_embed=None,
                codec_eos_token_id=2150,
                talker_temperature=0.9,
                talker_top_k=50,
                talker_top_p=1.0,
                talker_repetition_penalty=1.05,
                talker_prev_codec_tokens=None,
                speakers=["chelsie"],
                talker_kv_cache_locs_list=[],
                talker_prefill_input_ids_list=[],
                sample_codec=False,
            ),
            ready_states=[ready_state],
        )
    )

    scheduler._drain_pending_qwen3_omni_audio_phase_queue()

    assert scheduler.tp_worker.calls[0][3] == "talker_prefill"


def test_uncommitted_talker_prefill_chunk_is_not_queued():
    scheduler = FakeAudioPhaseScheduler()
    origin_input_ids = torch.tensor([101, 102])
    accept_hidden = torch.ones(2, 3)
    sampled_token = torch.tensor([17])
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="EXTEND"),
        model_specific_states={
            "origin_input_ids": [origin_input_ids],
            "origin_input_ids_cpu": [origin_input_ids.tolist()],
        },
    )
    logits_output = SimpleNamespace(
        talker_needs_prefill=True,
        prompt_accept_hidden_states=[accept_hidden],
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=sampled_token,
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    req = SimpleNamespace(
        rid="turn-a",
        audio_input_done=False,
        _prefill_audio_done_snapshot=False,
        qwen3_omni_partial_talker_prefill_warmed=False,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        qwen3_omni_audio_phase=None,
        qwen3_omni_text_lag_tokens=[],
        qwen3_omni_audio_frame_lag=0,
        audio_input_chunk_idx=1,
        speaker="chelsie",
        is_chunked=0,
        talker_kv_cache_locs=None,
        talker_prefill_len=0,
        talker_needs_prefill=True,
        finished=lambda: False,
    )

    scheduler._drain_qwen3_omni_audio_phase_queue(result, False, [req])

    assert scheduler.qwen3_omni_audio_phase_queue == deque()
    assert req.qwen3_omni_audio_ready_states == []


def test_audio_lane_drains_dependent_code_predictor_tail():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    def run_audio_phase_item(
        forward_batch,
        logits_output,
        next_token_ids,
        phase,
        request_indices,
        payload,
    ):
        scheduler.tp_worker.calls.append(
            (
                forward_batch,
                logits_output,
                next_token_ids,
                phase,
                request_indices,
                payload,
            )
        )
        if phase == "talker_decode":
            return SimpleNamespace(
                code_predictor_inputs=Qwen3OmniCodePredictorInputs(
                    request_indices=[0],
                    input_embeds=torch.zeros(1, 2, 4),
                    codec_tokens=torch.tensor([11]),
                )
            )
        return SimpleNamespace(code_predictor_inputs=None)

    scheduler.tp_worker.run_qwen3_omni_audio_phase_item = run_audio_phase_item

    drained = scheduler._drain_pending_qwen3_omni_audio_phase_queue(
        max_items=1,
        drain_dependent_code_predictor=True,
    )

    assert drained == 2
    assert [call[3] for call in scheduler.tp_worker.calls] == ["talker_decode"]
    assert len(scheduler.code_predictor_batch_calls) == 1
    assert scheduler.code_predictor_batch_calls[0][1].codec_tokens.tolist() == [11]
    assert not scheduler.qwen3_omni_audio_phase_queue
    assert req.qwen3_omni_audio_ready_states == []


def _make_talker_decode_phase_item(
    req,
    ready_state,
    codec_id=11,
    residual_codes=(1, 2),
    next_token_id=17,
):
    return Qwen3OmniAudioPhaseItem(
        logits_output=SimpleNamespace(),
        forward_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(name="DECODE"),
            model_specific_states={},
            seq_lens=torch.tensor([1], dtype=torch.int32),
        ),
        next_token_ids=torch.tensor([next_token_id]),
        reqs=[req],
        return_logprob=False,
        phase="talker_decode",
        request_indices=[0],
        request_rids=[req.rid],
        payload=Qwen3OmniTalkerDecodePayload(
            prev_codec_id=torch.tensor([codec_id]),
            prev_residual_codes=torch.tensor([list(residual_codes)]),
            prev_codec_input_embeds=torch.zeros(1, 4),
            talker_positions=torch.tensor([1]),
            talker_kv_cache_locs_list=[torch.tensor([5])],
            thinker_done=torch.tensor([False]),
            talker_post_thinker_started=None,
            talker_has_text_lag=torch.tensor([False]),
            talker_needs_tts_eos=None,
            tts_pad_embed=None,
            prev_sampled_thinker_token=torch.tensor([next_token_id]),
            codec_eos_token_id=2150,
            talker_temperature=0.9,
            talker_top_k=50,
            talker_top_p=1.0,
            talker_repetition_penalty=1.05,
            talker_prev_codec_tokens=None,
            **_talker_sampler_state(1),
        ),
        ready_states=[ready_state],
    )


def test_audio_lane_batches_consecutive_talker_decode_items():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: False)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_a = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    ready_b = Qwen3OmniTalkerDecodeReady("turn-b", 0, time.perf_counter())
    set_audio_ready_state(req_a, ready_a)
    set_audio_ready_state(req_b, ready_b)
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_a, ready_a, codec_id=11)
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_b, ready_b, codec_id=12, next_token_id=19)
    )

    drained = scheduler._drain_pending_qwen3_omni_audio_phase_queue()

    assert drained == 2
    assert len(scheduler.tp_worker.calls) == 1
    _, _, next_token_ids, phase, request_indices, payload = scheduler.tp_worker.calls[0]
    assert phase == "talker_decode"
    assert request_indices == [0, 1]
    assert next_token_ids.tolist() == [17, 19]
    assert payload.prev_codec_id.tolist() == [11, 12]
    assert req_a.qwen3_omni_audio_ready_states == []
    assert req_b.qwen3_omni_audio_ready_states == []


def test_audio_lane_leaves_incompatible_talker_decode_for_next_launch():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: False)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_a = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    ready_b = Qwen3OmniTalkerDecodeReady("turn-b", 0, time.perf_counter())
    set_audio_ready_state(req_a, ready_a)
    set_audio_ready_state(req_b, ready_b)
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_a, ready_a)
    )
    item_b = _make_talker_decode_phase_item(req_b, ready_b, codec_id=12)
    item_b.payload = replace(item_b.payload, talker_post_thinker_started=torch.tensor([True]))
    scheduler.qwen3_omni_audio_phase_queue.append(item_b)

    drained = scheduler._drain_pending_qwen3_omni_audio_phase_queue(max_items=1)

    assert drained == 1
    assert len(scheduler.tp_worker.calls) == 1
    assert scheduler.tp_worker.calls[0][4] == [0]
    assert [item.request_rids for item in scheduler.qwen3_omni_audio_phase_queue] == [
        ["turn-b"]
    ]
    assert req_a.qwen3_omni_audio_ready_states == []
    assert req_b.qwen3_omni_audio_ready_states == [ready_b]


def test_audio_lane_batches_talker_decode_dependent_code_predictor_tail():
    scheduler = FakeAudioPhaseScheduler()
    reqs = [
        SimpleNamespace(rid="turn-a", finished=lambda: False),
        SimpleNamespace(rid="turn-b", finished=lambda: False),
    ]
    for req, codec_id in zip(reqs, (11, 12)):
        ready_state = Qwen3OmniTalkerDecodeReady(req.rid, 0, time.perf_counter())
        set_audio_ready_state(req, ready_state)
        scheduler.qwen3_omni_audio_phase_queue.append(
            _make_talker_decode_phase_item(req, ready_state, codec_id=codec_id)
        )

    def run_audio_phase_item(
        forward_batch,
        logits_output,
        next_token_ids,
        phase,
        request_indices,
        payload,
    ):
        scheduler.tp_worker.calls.append(
            (
                forward_batch,
                logits_output,
                next_token_ids,
                phase,
                request_indices,
                payload,
            )
        )
        return SimpleNamespace(
            code_predictor_inputs=Qwen3OmniCodePredictorInputs(
                request_indices=[0, 1],
                input_embeds=torch.zeros(2, 2, 4),
                codec_tokens=torch.tensor([11, 12]),
            )
        )

    scheduler.tp_worker.run_qwen3_omni_audio_phase_item = run_audio_phase_item

    drained = scheduler._drain_pending_qwen3_omni_audio_phase_queue(
        max_items=1,
        drain_dependent_code_predictor=True,
    )

    assert drained == 3
    assert [call[3] for call in scheduler.tp_worker.calls] == ["talker_decode"]
    assert len(scheduler.code_predictor_batch_calls) == 1
    assert scheduler.code_predictor_batch_calls[0][1].codec_tokens.tolist() == [11, 12]
    assert not scheduler.qwen3_omni_audio_phase_queue
    assert all(req.qwen3_omni_audio_ready_states == [] for req in reqs)


def _make_talker_prefill_phase_item(
    req,
    ready_state,
    token_id=17,
    sample_codec=True,
):
    return Qwen3OmniAudioPhaseItem(
        logits_output=SimpleNamespace(),
        forward_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND"),
            model_specific_states={},
            seq_lens=torch.tensor([1], dtype=torch.int32),
        ),
        next_token_ids=torch.tensor([token_id]),
        reqs=[req],
        return_logprob=False,
        phase="talker_prefill",
        request_indices=[0],
        request_rids=[req.rid],
        payload=Qwen3OmniTalkerPrefillPayload(
            origin_input_ids=[torch.tensor([101, 102])],
            origin_input_ids_cpu=[[101, 102]],
            multimodal_inputs=None,
            prev_sampled_thinker_token=torch.tensor([token_id]),
            prompt_accept_hidden_states=None,
            tts_pad_embed=None,
            codec_eos_token_id=2150,
            talker_temperature=0.9,
            talker_top_k=50,
            talker_top_p=1.0,
            talker_repetition_penalty=1.05,
            talker_prev_codec_tokens=None,
            speakers=["chelsie"],
            talker_kv_cache_locs_list=[],
            talker_prefill_input_ids_list=[],
            sample_codec=sample_codec,
        ),
        ready_states=[ready_state],
    )


def test_audio_lane_batches_consecutive_talker_prefill_items():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: False)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_a = Qwen3OmniTalkerPrefillReady("turn-a", 0, time.perf_counter(), True, 1)
    ready_b = Qwen3OmniTalkerPrefillReady("turn-b", 0, time.perf_counter(), True, 1)
    set_audio_ready_state(req_a, ready_a)
    set_audio_ready_state(req_b, ready_b)
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_prefill_phase_item(req_a, ready_a, token_id=17)
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_prefill_phase_item(req_b, ready_b, token_id=19)
    )

    drained = scheduler._drain_pending_qwen3_omni_audio_phase_queue()

    assert drained == 2
    assert len(scheduler.tp_worker.calls) == 1
    _, _, next_token_ids, phase, request_indices, payload = scheduler.tp_worker.calls[0]
    assert phase == "talker_prefill"
    assert request_indices == [0, 1]
    assert next_token_ids.tolist() == [17, 19]
    assert payload.prev_sampled_thinker_token.tolist() == [17, 19]
    assert [ids.tolist() for ids in payload.origin_input_ids] == [
        [101, 102],
        [101, 102],
    ]
    assert payload.speakers == ["chelsie", "chelsie"]
    assert payload.talker_kv_cache_locs_list == [None, None]
    assert req_a.qwen3_omni_audio_ready_states == []
    assert req_b.qwen3_omni_audio_ready_states == []


def test_audio_lane_keeps_streaming_and_committed_talker_prefill_separate():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: False)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_a = Qwen3OmniTalkerPrefillReady("turn-a", 0, time.perf_counter(), False, 1)
    ready_b = Qwen3OmniTalkerPrefillReady("turn-b", 0, time.perf_counter(), True, 1)
    set_audio_ready_state(req_a, ready_a)
    set_audio_ready_state(req_b, ready_b)
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_prefill_phase_item(req_a, ready_a, sample_codec=False)
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_prefill_phase_item(req_b, ready_b, sample_codec=True)
    )

    drained = scheduler._drain_pending_qwen3_omni_audio_phase_queue(max_items=1)

    assert drained == 1
    assert len(scheduler.tp_worker.calls) == 1
    assert scheduler.tp_worker.calls[0][3] == "talker_prefill"
    assert scheduler.tp_worker.calls[0][4] == [0]
    assert [item.request_rids for item in scheduler.qwen3_omni_audio_phase_queue] == [
        ["turn-b"]
    ]
    assert req_a.qwen3_omni_audio_ready_states == []
    assert req_b.qwen3_omni_audio_ready_states == [ready_b]


def test_scheduler_selects_thinker_when_audio_has_slack_and_independent_thinker_work_exists():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False, talker_pcm16_chunks=[b"x"])
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "thinker"


def test_scheduler_selects_first_pcm_audio_before_thinker_prefill_work():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    req.qwen3_omni_text_lag_tokens = deque([123])
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_selects_first_pcm_audio_before_thinker_decode_work():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    req.qwen3_omni_text_lag_tokens = deque([123])
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False,
        reqs=[SimpleNamespace(rid="thinker-a")],
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_delays_talker_decode_when_talker_has_no_text_lag():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: False, reqs=[req])
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    req.thinker_done = False
    req.qwen3_omni_tts_eos_pending = False

    assert scheduler._select_qwen3_omni_scheduler_action() == "thinker"


def test_scheduler_keeps_talker_decode_ordered_when_talker_has_text_lag():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    req.qwen3_omni_text_lag_tokens = [17]
    req.thinker_done = False
    req.qwen3_omni_tts_eos_pending = False
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: False, reqs=[req])
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_keeps_code_predictor_ordered_with_same_request_thinker_work():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniCodePredictorReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: False, reqs=[req])
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=SimpleNamespace(),
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
            next_token_ids=torch.tensor([17]),
            reqs=[req],
            return_logprob=False,
            phase="code_predictor",
            request_indices=[0],
            request_rids=["turn-a"],
            payload=Qwen3OmniCodePredictorPayload(
                request_indices=[0],
                input_embeds=torch.zeros(1, 2, 4),
                codec_tokens=torch.tensor([11]),
            ),
            ready_states=[ready_state],
        )
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_treats_streaming_talker_prefill_as_thinker_independent():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerPrefillReady("turn-a", 0, time.perf_counter(), False, 1)
    set_audio_ready_state(req, ready_state)
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: False, reqs=[req])
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_prefill_phase_item(req, ready_state, sample_codec=False)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "thinker"


def test_scheduler_keeps_first_pcm_talker_prefill_ordered_with_thinker_work():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerPrefillReady("turn-a", 0, time.perf_counter(), True, 1)
    set_audio_ready_state(req, ready_state)
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: False, reqs=[req])
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_prefill_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_selects_audio_when_oldest_audio_deadline_is_due():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady(
        "turn-a",
        0,
        time.perf_counter() - (AUDIO_BATCH_DELAY_MS + 1) / 1000,
    )
    set_audio_ready_state(req, ready_state)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_promotes_later_disjoint_audio_deadline():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_a = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req_a, ready_a)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_b = Qwen3OmniTalkerDecodeReady(
        "turn-b",
        0,
        time.perf_counter() - (AUDIO_BATCH_DELAY_MS + 1) / 1000,
    )
    set_audio_ready_state(req_b, ready_b)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_a, ready_a)
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_b, ready_b)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    assert scheduler.qwen3_omni_audio_phase_queue[0].request_rids == ["turn-b"]


def test_scheduler_promotes_later_disjoint_first_pcm_audio_item():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(
        rid="turn-a",
        finished=lambda: False,
        talker_pcm16_chunks=[b"pcm"],
    )
    ready_a = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req_a, ready_a)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_b = Qwen3OmniTalkerDecodeReady("turn-b", 0, time.perf_counter())
    set_audio_ready_state(req_b, ready_b)
    req_b.qwen3_omni_text_lag_tokens = deque([17])
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_a, ready_a)
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req_b, ready_b)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    assert scheduler.qwen3_omni_audio_phase_queue[0].request_rids == ["turn-b"]


def test_scheduler_preempts_phase_queue_for_due_code2wav_batch():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    code_req.return_logprob = False
    code_ready = (
        scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
            code_req,
            time.perf_counter() - (AUDIO_BATCH_DELAY_MS + 1) / 1000,
            flush=False,
        )
    )
    assert code_ready.codes_window == tuple(
        tuple(frame) for frame in code_req.talker_output_codes
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}

    phase_req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    phase_ready = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(phase_req, phase_ready)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(phase_req, phase_ready)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    assert scheduler._run_qwen3_omni_audio_lane_if_selected()

    assert scheduler.code2wav.calls[0].shape == (1, 2, 4)
    assert not scheduler.tp_worker.calls
    assert len(scheduler.qwen3_omni_audio_phase_queue) == 1


def test_scheduler_preempts_phase_queue_for_first_pcm_code2wav_batch():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    code_req.return_logprob = False
    scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}

    phase_req = SimpleNamespace(
        rid="turn-a",
        finished=lambda: False,
        talker_pcm16_chunks=[b"pcm"],
    )
    phase_ready = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(phase_req, phase_ready)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(phase_req, phase_ready)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    assert scheduler._run_qwen3_omni_audio_lane_if_selected()

    assert scheduler.code2wav.calls[0].shape == (1, 2, 4)
    assert not scheduler.tp_worker.calls
    assert len(scheduler.qwen3_omni_audio_phase_queue) == 1


def test_scheduler_runs_code2wav_with_slack_before_thinker_prefill_work():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    code_ready = scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}

    phase_req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    phase_ready = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(phase_req, phase_ready)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(phase_req, phase_ready)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_runs_code2wav_with_slack_before_thinker_prefill_without_phase_queue():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_completes_pending_code2wav_before_thinker_prefill_work():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    ready_state = scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.qwen3_omni_code2wav_pending_work = [
        SimpleNamespace(
            event=SimpleNamespace(query=lambda: False),
            launched_at=time.perf_counter(),
            items=[SimpleNamespace(req=code_req, ready_state=ready_state)],
        )
    ]
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_overlaps_nonurgent_pending_code2wav_with_thinker_prefill_work():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    code_req.talker_pcm16_chunks = [b"pcm"]
    code_req.qwen3_omni_audio_frame_lag = 0
    ready_state = scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.qwen3_omni_code2wav_pending_work = [
        SimpleNamespace(
            event=SimpleNamespace(query=lambda: False),
            launched_at=time.perf_counter(),
            items=[SimpleNamespace(req=code_req, ready_state=ready_state)],
        )
    ]
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]

    assert scheduler._select_qwen3_omni_scheduler_action() == "thinker"


def test_scheduler_keeps_code2wav_with_slack_behind_thinker_decode_work():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    code_req.talker_pcm16_chunks = [b"pcm"]
    scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}

    phase_req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    phase_ready = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(phase_req, phase_ready)
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False,
        reqs=[SimpleNamespace(rid="thinker-a")],
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(phase_req, phase_ready)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "thinker"


def test_scheduler_runs_first_pcm_code2wav_before_thinker_decode_work():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}

    phase_req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    phase_ready = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(phase_req, phase_ready)
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False,
        reqs=[SimpleNamespace(rid="thinker-a")],
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(phase_req, phase_ready)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_preempts_phase_queue_for_code2wav_lag_limit():
    scheduler = FakeAudioPhaseScheduler()
    code_req = make_req(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        rid="turn-code",
    )
    code_req.qwen3_omni_audio_frame_lag = AUDIO_MAX_AUDIO_FRAME_LAG
    code_ready = scheduler_output_processor_mixin.mark_qwen3_omni_code2wav_ready_state(
        code_req,
        time.perf_counter(),
        flush=False,
    )
    scheduler.audio_turn_reqs = {code_req.rid: code_req}

    phase_req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    phase_ready = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(phase_req, phase_ready)
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(phase_req, phase_ready)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_selects_audio_when_request_text_lag_is_at_limit():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    req.qwen3_omni_text_lag_tokens = [17] * AUDIO_MAX_TEXT_LAG
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"


def test_scheduler_logs_lane_decision_reason_deadline_and_lag(caplog):
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: False)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    req.qwen3_omni_text_lag_tokens = [17] * AUDIO_MAX_TEXT_LAG
    req.qwen3_omni_audio_frame_lag = 3
    scheduler.waiting_queue = [SimpleNamespace(rid="thinker-a")]
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    caplog.set_level(
        logging.INFO,
        logger="sglang.srt.managers.scheduler_qwen3_omni_mixin",
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"

    message = caplog.records[-1].getMessage()
    assert "reason=audio_lag_limit" in message
    assert f"max_text_lag_tokens={AUDIO_MAX_TEXT_LAG}" in message
    assert "max_audio_frame_lag=3" in message
    assert "oldest_audio_deadline_slack_ms=" in message


def test_scheduler_drops_finished_audio_phase_items_before_lane_selection():
    scheduler = FakeAudioPhaseScheduler()
    req = SimpleNamespace(rid="turn-a", finished=lambda: True)
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.qwen3_omni_audio_phase_queue.append(
        _make_talker_decode_phase_item(req, ready_state)
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "thinker"
    assert not scheduler.qwen3_omni_audio_phase_queue


def test_scheduler_slices_mixed_finished_talker_decode_item_before_lane_selection():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: True)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_a = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    ready_b = Qwen3OmniTalkerDecodeReady("turn-b", 1, time.perf_counter())
    set_audio_ready_state(req_a, ready_a)
    set_audio_ready_state(req_b, ready_b)
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=SimpleNamespace(),
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
            next_token_ids=torch.tensor([17, 19]),
            reqs=[req_a, req_b],
            return_logprob=False,
            phase="talker_decode",
            request_indices=[0, 1],
            request_rids=["turn-a", "turn-b"],
            payload=Qwen3OmniTalkerDecodePayload(
                prev_codec_id=torch.tensor([11, 12]),
                prev_residual_codes=torch.tensor([[1, 2], [3, 4]]),
                prev_codec_input_embeds=torch.zeros(2, 4),
                talker_positions=torch.tensor([1, 1]),
                talker_kv_cache_locs_list=[torch.tensor([5]), torch.tensor([6])],
                thinker_done=torch.tensor([False, False]),
                talker_post_thinker_started=None,
                talker_has_text_lag=torch.tensor([False, False]),
                talker_needs_tts_eos=None,
                tts_pad_embed=None,
                prev_sampled_thinker_token=torch.tensor([17, 19]),
                codec_eos_token_id=2150,
                talker_temperature=0.9,
                talker_top_k=50,
                talker_top_p=1.0,
                talker_repetition_penalty=1.05,
                talker_prev_codec_tokens=None,
                **_talker_sampler_state(2),
            ),
            ready_states=[ready_a, ready_b],
        )
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    [item] = scheduler.qwen3_omni_audio_phase_queue
    assert item.request_indices == [1]
    assert item.request_rids == ["turn-b"]
    assert item.ready_states == [ready_b]
    assert item.payload.prev_codec_id.tolist() == [11, 12]


def test_scheduler_slices_mixed_finished_code_predictor_item_before_lane_selection():
    scheduler = FakeAudioPhaseScheduler()
    req_a = SimpleNamespace(rid="turn-a", finished=lambda: True)
    req_b = SimpleNamespace(rid="turn-b", finished=lambda: False)
    ready_a = Qwen3OmniCodePredictorReady("turn-a", 0, time.perf_counter())
    ready_b = Qwen3OmniCodePredictorReady("turn-b", 1, time.perf_counter())
    set_audio_ready_state(req_a, ready_a)
    set_audio_ready_state(req_b, ready_b)
    code_predictor_inputs = Qwen3OmniCodePredictorInputs(
        request_indices=[0, 1],
        input_embeds=torch.arange(16, dtype=torch.float32).view(2, 2, 4),
        codec_tokens=torch.tensor([11, 12]),
    )
    payload = Qwen3OmniCodePredictorPayload(
        request_indices=[0, 1],
        input_embeds=code_predictor_inputs.input_embeds,
        codec_tokens=code_predictor_inputs.codec_tokens,
    )
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=SimpleNamespace(
                code_predictor_inputs=code_predictor_inputs
            ),
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
            next_token_ids=torch.tensor([17, 19]),
            reqs=[req_a, req_b],
            return_logprob=False,
            phase="code_predictor",
            request_indices=[0, 1],
            request_rids=["turn-a", "turn-b"],
            payload=payload,
            ready_states=[ready_a, ready_b],
        )
    )

    assert scheduler._select_qwen3_omni_scheduler_action() == "audio"
    [item] = scheduler.qwen3_omni_audio_phase_queue
    assert item.request_indices == [1]
    assert item.request_rids == ["turn-b"]
    assert item.ready_states == [ready_b]
    assert item.logits_output.code_predictor_inputs.request_indices == [1]
    assert item.logits_output.code_predictor_inputs.codec_tokens.tolist() == [12]
    assert item.payload.request_indices == [1]


def test_talker_decode_payload_keeps_text_lag_token_until_consumed():
    logits_output = SimpleNamespace(code_predictor_inputs=None)
    forward_batch = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        positions=torch.tensor([1]),
        model_specific_states={
            "prev_codec_id": torch.tensor([11]),
            "prev_residual_codes": torch.tensor([[1, 2]]),
            "prev_codec_input_embeds": torch.zeros(1, 4),
            "talker_positions": torch.tensor([1]),
            "talker_kv_cache_locs_list": [torch.tensor([5])],
            "prev_sampled_thinker_token": torch.tensor([13]),
            "thinker_done": torch.tensor([False]),
            "talker_has_text_lag": torch.tensor([True]),
            **_talker_sampler_state(1),
        },
    )

    payload = build_qwen3_omni_audio_phase_payload(
        phase="talker_decode",
        forward_batch=forward_batch,
        logits_output=logits_output,
        next_token_ids=torch.tensor([17]),
        request_indices=[0],
    )

    assert payload.prev_sampled_thinker_token.tolist() == [13]


def test_talker_decode_payload_owns_model_state_snapshots():
    prev_codec_id = torch.tensor([11])
    prev_residual_codes = torch.tensor([[1, 2]])
    prev_codec_input_embeds = torch.ones(1, 4)
    talker_positions = torch.tensor([1])
    kv_locs = torch.tensor([5])
    thinker_done = torch.tensor([False])
    talker_has_text_lag = torch.tensor([False])
    prev_sampled = torch.tensor([13])
    next_token_ids = torch.tensor([17])
    logits_output = SimpleNamespace(code_predictor_inputs=None)
    forward_batch = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        positions=torch.tensor([9]),
        model_specific_states={
            "prev_codec_id": prev_codec_id,
            "prev_residual_codes": prev_residual_codes,
            "prev_codec_input_embeds": prev_codec_input_embeds,
            "talker_positions": talker_positions,
            "talker_kv_cache_locs_list": [kv_locs],
            "prev_sampled_thinker_token": prev_sampled,
            "thinker_done": thinker_done,
            "talker_has_text_lag": talker_has_text_lag,
            "talker_prev_codec_tokens": [[1, 2]],
            **_talker_sampler_state(1),
        },
    )

    payload = build_qwen3_omni_audio_phase_payload(
        phase="talker_decode",
        forward_batch=forward_batch,
        logits_output=logits_output,
        next_token_ids=next_token_ids,
        request_indices=[0],
    )

    prev_codec_id[0] = 99
    prev_residual_codes[0, 0] = 99
    prev_codec_input_embeds[0, 0] = 99
    talker_positions[0] = 99
    kv_locs[0] = 99
    thinker_done[0] = True
    talker_has_text_lag[0] = True
    prev_sampled[0] = 99
    next_token_ids[0] = 99
    forward_batch.model_specific_states["talker_prev_codec_tokens"][0].append(99)

    assert payload.prev_codec_id.tolist() == [11]
    assert payload.prev_residual_codes.tolist() == [[1, 2]]
    assert payload.prev_codec_input_embeds[0, 0].item() == 1
    assert payload.talker_positions.tolist() == [1]
    assert payload.talker_kv_cache_locs_list[0].tolist() == [5]
    assert payload.thinker_done.tolist() == [False]
    assert payload.talker_has_text_lag.tolist() == [False]
    assert payload.prev_sampled_thinker_token.tolist() == [17]
    assert payload.talker_prev_codec_tokens == [[1, 2]]


def test_talker_prefill_payload_owns_model_state_snapshots():
    origin_input_ids = torch.tensor([101, 102])
    prev_sampled = torch.tensor([17])
    accept_hidden = torch.ones(2, 3)
    tts_pad = torch.ones(3)
    temperature = torch.tensor([0.8])
    kv_locs = torch.tensor([5, 6])
    prefill_input_ids = torch.tensor([101, 102, 21])
    codec_history = [[1, 2]]
    speakers = ["chelsie"]
    multimodal_inputs = [SimpleNamespace(tag="mm")]
    logits_output = SimpleNamespace(
        talker_needs_prefill=True,
        prompt_accept_hidden_states=[accept_hidden],
    )
    forward_batch = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        model_specific_states={
            "origin_input_ids": [origin_input_ids],
            "origin_input_ids_cpu": [origin_input_ids.tolist()],
            "multimodal_inputs": multimodal_inputs,
            "tts_pad_embed": tts_pad,
            "talker_temperature": temperature,
            "talker_prev_codec_tokens": codec_history,
            "speakers": speakers,
            "talker_kv_cache_locs_list": [kv_locs],
            "talker_prefill_input_ids_list": [prefill_input_ids],
        },
    )

    payload = build_qwen3_omni_audio_phase_payload(
        phase="talker_prefill",
        forward_batch=forward_batch,
        logits_output=logits_output,
        next_token_ids=prev_sampled,
        request_indices=[0],
    )

    origin_input_ids[0] = 999
    prev_sampled[0] = 999
    accept_hidden[0, 0] = 999
    tts_pad[0] = 999
    temperature[0] = 9.9
    kv_locs[0] = 999
    prefill_input_ids[0] = 999
    codec_history[0].append(999)
    speakers[0] = "ethan"
    multimodal_inputs.append(SimpleNamespace(tag="late"))

    assert payload.origin_input_ids[0].tolist() == [101, 102]
    assert payload.prev_sampled_thinker_token.tolist() == [17]
    assert payload.prompt_accept_hidden_states[0][0, 0].item() == 1
    assert payload.tts_pad_embed.tolist() == [1, 1, 1]
    assert payload.talker_temperature.item() == pytest.approx(0.8)
    assert payload.talker_kv_cache_locs_list[0].tolist() == [5, 6]
    assert payload.talker_prefill_input_ids_list[0].tolist() == [101, 102, 21]
    assert payload.talker_prev_codec_tokens == [[1, 2]]
    assert payload.speakers == ["chelsie"]
    assert len(payload.multimodal_inputs) == 1


def test_code_predictor_payload_owns_pending_inputs():
    pending = Qwen3OmniCodePredictorInputs(
        request_indices=[0],
        input_embeds=torch.ones(1, 2, 3),
        codec_tokens=torch.tensor([11]),
    )
    logits_output = SimpleNamespace(code_predictor_inputs=pending)
    forward_batch = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        model_specific_states={},
    )

    payload = build_qwen3_omni_audio_phase_payload(
        phase="code_predictor",
        forward_batch=forward_batch,
        logits_output=logits_output,
        next_token_ids=torch.tensor([17]),
        request_indices=[0],
    )

    pending.input_embeds[0, 0, 0] = 99
    pending.codec_tokens[0] = 99

    assert payload.input_embeds[0, 0, 0].item() == 1
    assert payload.codec_tokens.tolist() == [11]


def test_talker_prefill_phase_defers_to_scheduler_loop():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(talker_needs_prefill=True)
    next_token_ids = torch.tensor([17])
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="EXTEND"),
        model_specific_states={
            "origin_input_ids": [torch.tensor([101, 102])],
            "origin_input_ids_cpu": [[101, 102]],
        },
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=next_token_ids,
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    reqs = [
        SimpleNamespace(
            rid="turn-a",
            audio_input_done=True,
            speaker="chelsie",
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
            is_chunked=0,
        )
    ]

    scheduler._drain_qwen3_omni_audio_phase_queue(result, False, reqs)

    assert len(scheduler.qwen3_omni_audio_phase_queue) == 1
    assert not scheduler.tp_worker.calls
    [item] = scheduler.qwen3_omni_audio_phase_queue
    assert item.phase == "talker_prefill"
    assert item.return_logprob is False
    assert item.payload.speakers == ["chelsie"]
    assert isinstance(reqs[0].qwen3_omni_audio_ready_state, Qwen3OmniTalkerPrefillReady)
    assert reqs[0].qwen3_omni_audio_ready_state.committed_audio_input is True


def test_talker_prefill_phase_requires_speaker_state():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(talker_needs_prefill=True)
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        forward_mode=SimpleNamespace(name="EXTEND"),
        model_specific_states={
            "origin_input_ids": [torch.tensor([101, 102])],
            "origin_input_ids_cpu": [[101, 102]],
        },
    )
    result = SimpleNamespace(
        logits_output=logits_output,
        next_token_ids=torch.tensor([17]),
        qwen3_omni_audio_forward_batch=forward_batch,
    )
    reqs = [
        SimpleNamespace(
            rid="turn-a",
            audio_input_done=True,
            speaker=None,
            qwen3_omni_audio_ready_state=None,
            qwen3_omni_audio_ready_states=[],
            is_chunked=0,
        )
    ]

    with pytest.raises(RuntimeError, match="requires speaker state"):
        scheduler._drain_qwen3_omni_audio_phase_queue(result, False, reqs)


def test_deferred_audio_phase_applies_codec_frames_and_streams_audio():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(
        talker_needs_prefill=False,
        codec_frames=[[3, 13]],
        prev_codec_input_embeds_list=[torch.zeros(2)],
        tts_pad_embed=None,
        talker_out_cache_loc_list=None,
        updated_talker_kv_locs_list=None,
        code_predictor_inputs=None,
    )
    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=False,
        talker_output_codes=[[0, 10], [1, 11], [2, 12]],
        talker_last_decoded_frame=0,
        talker_pcm16_chunks=[],
        talker_pcm16_chunk_cap=1000,
        talker_codec_ids=[],
        talker_step=0,
        prev_residual_codes=None,
        qwen3_omni_text_lag_tokens=[101],
        qwen3_omni_audio_frame_lag=3,
        qwen3_omni_audio_phase=None,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        check_finished=lambda: None,
        finished=lambda: False,
    )
    ready_state = Qwen3OmniCodePredictorReady("turn-a", 0, 1.0)
    set_audio_ready_state(req, ready_state)
    scheduler.audio_turn_reqs[req.rid] = req
    item = Qwen3OmniAudioPhaseItem(
        logits_output=logits_output,
        forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
        next_token_ids=torch.tensor([17]),
        reqs=[req],
        return_logprob=False,
        phase="code_predictor",
        request_indices=[0],
        request_rids=["turn-a"],
        payload=Qwen3OmniCodePredictorPayload(
            request_indices=[0],
            input_embeds=torch.zeros(1, 2, 2),
            codec_tokens=torch.tensor([3]),
        ),
        ready_states=[ready_state],
    )
    scheduler.qwen3_omni_audio_phase_queue.append(item)

    scheduler._drain_pending_qwen3_omni_audio_phase_queue(apply_outputs=True)

    assert req.talker_codec_ids == [3]
    assert req.prev_residual_codes == [13]
    assert req.talker_last_decoded_frame == 4
    assert req.qwen3_omni_audio_frame_lag == 0
    assert scheduler.code2wav.calls[0].shape == (1, 2, 4)
    [(rids, return_logprob, audio_chunks)] = scheduler.streams
    assert rids == ["turn-a"]
    assert return_logprob is False
    assert len(audio_chunks[0]) == 1
    assert req.talker_pcm16_chunks == []


def test_deferred_audio_phase_finishes_when_talker_length_cap_is_hit():
    scheduler = FakeAudioPhaseScheduler()
    logits_output = SimpleNamespace(
        talker_needs_prefill=False,
        codec_frames=[[3, 13]],
        prev_codec_input_embeds_list=[torch.zeros(2)],
        tts_pad_embed=None,
        talker_out_cache_loc_list=None,
        updated_talker_kv_locs_list=None,
        code_predictor_inputs=None,
    )
    finished = {"value": False}

    def check_finished():
        finished["value"] = True

    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=True,
        qwen3_omni_talker_post_thinker_started=True,
        talker_output_codes=[[0, 10], [1, 11], [2, 12]],
        talker_last_decoded_frame=0,
        talker_pcm16_chunks=[],
        talker_pcm16_chunk_cap=1000,
        talker_codec_ids=[],
        talker_step=95,
        prev_residual_codes=None,
        qwen3_omni_text_lag_tokens=[],
        qwen3_omni_tts_eos_pending=False,
        qwen3_omni_audio_frame_lag=3,
        qwen3_omni_audio_phase=None,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        check_finished=check_finished,
        finished=lambda: finished["value"],
    )
    ready_state = Qwen3OmniCodePredictorReady("turn-a", 0, 1.0)
    set_audio_ready_state(req, ready_state)
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=logits_output,
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
            next_token_ids=torch.tensor([17]),
            reqs=[req],
            return_logprob=False,
            phase="code_predictor",
            request_indices=[0],
            request_rids=["turn-a"],
            payload=Qwen3OmniCodePredictorPayload(
                request_indices=[0],
                input_embeds=torch.zeros(1, 2, 2),
                codec_tokens=torch.tensor([3]),
            ),
            ready_states=[ready_state],
        )
    )

    scheduler._drain_pending_qwen3_omni_audio_phase_queue(apply_outputs=True)

    assert req.talker_step == 96
    assert scheduler.finished_audio_reqs == ["turn-a"]
    assert req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.DONE


def test_qwen3_omni_codec_eos_before_post_thinker_frame_does_not_finish():
    req = object.__new__(Req)
    req.codec_eos_token_id = 2150
    req.thinker_done = True
    req.qwen3_omni_talker_post_thinker_started = False
    req.talker_step = 12
    req.talker_codec_ids = [2150]
    req.sampling_params = SimpleNamespace(custom_params={}, max_new_tokens=999)
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.output_ids = [11, 12, 13]
    req.grammar = None

    req.check_finished()

    assert req.finished_reason is None


def test_qwen3_omni_codec_eos_after_post_thinker_frame_finishes():
    req = object.__new__(Req)
    req.codec_eos_token_id = 2150
    req.thinker_done = True
    req.qwen3_omni_talker_post_thinker_started = True
    req.talker_step = 13
    req.talker_codec_ids = [2150]
    req.sampling_params = SimpleNamespace(custom_params={}, max_new_tokens=999)
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.output_ids = [11, 12, 13]
    req.grammar = None

    req.check_finished()

    assert req.finished_reason is not None
    assert req.finished_len == len(req.output_ids)


def test_qwen3_omni_tokenizer_additional_stop_enters_talker_continuation():
    req = object.__new__(Req)
    req.codec_eos_token_id = 2150
    req.thinker_done = False
    req.qwen3_omni_talker_post_thinker_started = False
    req.talker_step = 0
    req.talker_codec_ids = []
    req.sampling_params = SimpleNamespace(
        custom_params={}, max_new_tokens=999, stop_token_ids=set()
    )
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.output_ids = [11, 151645]
    req.eos_token_ids = set()
    req.tokenizer = SimpleNamespace(
        eos_token_id=151643, additional_stop_token_ids={151645}
    )
    req.grammar = None

    req.check_finished()

    assert req.thinker_done is True
    assert req.finished_reason is None


def test_qwen3_omni_tokenizer_without_additional_stops_uses_eos():
    req = object.__new__(Req)
    req.codec_eos_token_id = 2150
    req.thinker_done = False
    req.qwen3_omni_talker_post_thinker_started = False
    req.talker_step = 0
    req.talker_codec_ids = []
    req.sampling_params = SimpleNamespace(
        custom_params={}, max_new_tokens=999, stop_token_ids=set()
    )
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.output_ids = [11, 151643]
    req.eos_token_ids = set()
    req.tokenizer = SimpleNamespace(
        eos_token_id=151643, additional_stop_token_ids=None
    )
    req.grammar = None

    req.check_finished()

    assert req.thinker_done is True
    assert req.finished_reason is None


def test_qwen3_omni_text_length_enters_talker_continuation():
    req = object.__new__(Req)
    req.codec_eos_token_id = 2150
    req.thinker_done = False
    req.qwen3_omni_tts_eos_pending = False
    req.qwen3_omni_talker_post_thinker_started = False
    req.talker_step = 0
    req.talker_codec_ids = []
    req.sampling_params = SimpleNamespace(
        custom_params={}, max_new_tokens=3, stop_token_ids=set()
    )
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    req.output_ids = [11, 12, 13]
    req.eos_token_ids = set()
    req.tokenizer = None
    req.grammar = None

    req.check_finished()

    assert req.thinker_done is True
    assert req.qwen3_omni_tts_eos_pending is True
    assert req.finished_reason is None

    req.check_finished()

    assert req.finished_reason is None


def test_qwen3_omni_post_thinker_frame_marks_talker_continuation_started():
    logits_output = SimpleNamespace(
        talker_needs_prefill=False,
        prompt_accept_hidden_states=None,
        codec_frames=[[7, 17]],
        prev_codec_input_embeds_list=[torch.zeros(2)],
        tts_pad_embed=None,
        talker_out_cache_loc_list=None,
        talker_mrope_position_delta_list=None,
        updated_talker_kv_locs_list=None,
    )
    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=True,
        qwen3_omni_talker_post_thinker_started=False,
        qwen3_omni_tts_eos_pending=True,
        qwen3_omni_audio_phase=None,
        talker_codec_ids=[],
        prev_residual_codes=None,
        prev_codec_input_embed=None,
        talker_output_codes=[],
        qwen3_omni_text_lag_tokens=[],
        qwen3_omni_audio_frame_lag=0,
        talker_step=0,
        talker_last_decoded_frame=0,
        talker_pcm16_chunk_cap=1000,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_code2wav_pending=False,
    )

    _update_qwen3_omni_state(req, logits_output, 0)

    assert req.qwen3_omni_talker_post_thinker_started is True
    assert req.qwen3_omni_tts_eos_pending is False
    assert req.talker_codec_ids == [7]


def test_deferred_audio_phase_finishes_on_talker_codec_eos_without_next_carry():
    scheduler = FakeAudioPhaseScheduler()
    finished = {"value": False}

    def check_finished():
        finished["value"] = True

    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=True,
        qwen3_omni_talker_post_thinker_started=True,
        qwen3_omni_audio_phase=None,
        talker_codec_ids=[],
        prev_residual_codes=None,
        prev_codec_input_embed=None,
        talker_output_codes=[[0, 10], [1, 11], [2, 12], [3, 13]],
        qwen3_omni_text_lag_tokens=[],
        qwen3_omni_audio_frame_lag=4,
        talker_step=8,
        talker_last_decoded_frame=0,
        talker_pcm16_chunks=[],
        talker_pcm16_chunk_cap=1000,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        check_finished=check_finished,
        finished=lambda: finished["value"],
    )
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=SimpleNamespace(
                talker_needs_prefill=False,
                prompt_accept_hidden_states=None,
                codec_frames=[[2150]],
                prev_codec_input_embeds_list=None,
                tts_pad_embed=None,
                talker_out_cache_loc_list=None,
                talker_mrope_position_delta_list=None,
                updated_talker_kv_locs_list=None,
            ),
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
            next_token_ids=torch.tensor([17]),
            reqs=[req],
            return_logprob=False,
            phase="talker_decode",
            request_indices=[0],
            request_rids=["turn-a"],
            payload=Qwen3OmniTalkerDecodePayload(
                prev_codec_id=torch.tensor([3]),
                prev_residual_codes=torch.tensor([[13]]),
                prev_codec_input_embeds=torch.zeros(1, 2),
                talker_positions=torch.tensor([4]),
                talker_kv_cache_locs_list=[torch.tensor([5])],
                thinker_done=torch.tensor([True]),
                talker_post_thinker_started=torch.tensor([True]),
                talker_has_text_lag=torch.tensor([False]),
                talker_needs_tts_eos=torch.tensor([False]),
                tts_pad_embed=torch.zeros(1, 2),
                prev_sampled_thinker_token=torch.tensor([17]),
                codec_eos_token_id=2150,
                talker_temperature=0.9,
                talker_top_k=50,
                talker_top_p=1.0,
                talker_repetition_penalty=1.05,
                talker_prev_codec_tokens=[[]],
                **_talker_sampler_state(1),
            ),
            ready_states=[ready_state],
        )
    )

    scheduler._drain_pending_qwen3_omni_audio_phase_queue(apply_outputs=True)

    assert req.talker_codec_ids == [2150]
    assert req.prev_residual_codes == []
    assert scheduler.finished_audio_reqs == ["turn-a"]
    assert req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.DONE
    assert scheduler.streams


def test_deferred_audio_phase_streams_terminal_without_new_pcm():
    scheduler = FakeAudioPhaseScheduler()
    finished = {"value": False}

    def check_finished():
        finished["value"] = True

    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=2150,
        thinker_done=True,
        qwen3_omni_talker_post_thinker_started=True,
        qwen3_omni_audio_phase=None,
        talker_codec_ids=[],
        prev_residual_codes=None,
        prev_codec_input_embed=None,
        talker_output_codes=[],
        qwen3_omni_text_lag_tokens=[],
        qwen3_omni_audio_frame_lag=0,
        talker_step=8,
        talker_last_decoded_frame=0,
        talker_pcm16_chunks=[],
        talker_pcm16_chunk_cap=1000,
        qwen3_omni_audio_ready_state=None,
        qwen3_omni_audio_ready_states=[],
        check_finished=check_finished,
        finished=lambda: finished["value"],
    )
    ready_state = Qwen3OmniTalkerDecodeReady("turn-a", 0, time.perf_counter())
    set_audio_ready_state(req, ready_state)
    scheduler.qwen3_omni_audio_phase_queue.append(
        Qwen3OmniAudioPhaseItem(
            logits_output=SimpleNamespace(
                talker_needs_prefill=False,
                prompt_accept_hidden_states=None,
                codec_frames=[[2150]],
                prev_codec_input_embeds_list=None,
                tts_pad_embed=None,
                talker_out_cache_loc_list=None,
                talker_mrope_position_delta_list=None,
                updated_talker_kv_locs_list=None,
            ),
            forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(name="DECODE")),
            next_token_ids=torch.tensor([17]),
            reqs=[req],
            return_logprob=False,
            phase="talker_decode",
            request_indices=[0],
            request_rids=["turn-a"],
            payload=Qwen3OmniTalkerDecodePayload(
                prev_codec_id=torch.tensor([3]),
                prev_residual_codes=torch.tensor([[13]]),
                prev_codec_input_embeds=torch.zeros(1, 2),
                talker_positions=torch.tensor([4]),
                talker_kv_cache_locs_list=[torch.tensor([5])],
                thinker_done=torch.tensor([True]),
                talker_post_thinker_started=torch.tensor([True]),
                talker_has_text_lag=torch.tensor([False]),
                talker_needs_tts_eos=torch.tensor([False]),
                tts_pad_embed=torch.zeros(1, 2),
                prev_sampled_thinker_token=torch.tensor([17]),
                codec_eos_token_id=2150,
                talker_temperature=0.9,
                talker_top_k=50,
                talker_top_p=1.0,
                talker_repetition_penalty=1.05,
                talker_prev_codec_tokens=[[]],
                **_talker_sampler_state(1),
            ),
            ready_states=[ready_state],
        )
    )

    scheduler._drain_pending_qwen3_omni_audio_phase_queue(apply_outputs=True)

    assert scheduler.finished_audio_reqs == ["turn-a"]
    assert req.qwen3_omni_audio_phase == Qwen3OmniAudioPhase.DONE
    assert scheduler.streams == [(["turn-a"], False, [[]])]


def test_qwen3_omni_masks_codec_eos_until_post_thinker_continuation():
    logits = torch.zeros(3, 8)
    logits[:, 5] = 10

    masked = _mask_codec_eos(
        logits, 5, torch.tensor([False, True, False])
    )

    assert torch.isneginf(masked[0, 5])
    assert masked[1, 5] == 10
    assert torch.isneginf(masked[2, 5])
    assert logits[0, 5] == 10


def test_deferred_audio_phase_batches_code2wav_after_queue_drain():
    scheduler = FakeAudioPhaseScheduler()
    reqs = []
    for rid in ("turn-a", "turn-b"):
        reqs.append(
            SimpleNamespace(
                rid=rid,
                codec_eos_token_id=2150,
                thinker_done=False,
                talker_output_codes=[[0, 10], [1, 11], [2, 12]],
                talker_last_decoded_frame=0,
                talker_pcm16_chunks=[],
                talker_pcm16_chunk_cap=1000,
                talker_codec_ids=[],
                talker_step=0,
                prev_residual_codes=None,
                qwen3_omni_text_lag_tokens=[101],
                qwen3_omni_audio_frame_lag=3,
                qwen3_omni_audio_phase=None,
                qwen3_omni_audio_ready_state=None,
                qwen3_omni_audio_ready_states=[],
                check_finished=lambda: None,
                finished=lambda: False,
            )
        )
    scheduler.audio_turn_reqs = {req.rid: req for req in reqs}

    for req, codec_token, residual_token in (
        (reqs[0], 3, 13),
        (reqs[1], 4, 14),
    ):
        logits_output = SimpleNamespace(
            talker_needs_prefill=False,
            codec_frames=[[codec_token, residual_token]],
            prev_codec_input_embeds_list=[torch.zeros(2)],
            tts_pad_embed=None,
            talker_out_cache_loc_list=None,
            updated_talker_kv_locs_list=None,
            code_predictor_inputs=None,
        )
        ready_state = Qwen3OmniCodePredictorReady(req.rid, 0, 1.0)
        set_audio_ready_state(req, ready_state)
        scheduler.qwen3_omni_audio_phase_queue.append(
            Qwen3OmniAudioPhaseItem(
                logits_output=logits_output,
                forward_batch=SimpleNamespace(
                    forward_mode=SimpleNamespace(name="DECODE")
                ),
                next_token_ids=torch.tensor([17]),
                reqs=[req],
                return_logprob=False,
                phase="code_predictor",
                request_indices=[0],
                request_rids=[req.rid],
                payload=Qwen3OmniCodePredictorPayload(
                    request_indices=[0],
                    input_embeds=torch.zeros(1, 2, 2),
                    codec_tokens=torch.tensor([codec_token]),
                ),
                ready_states=[ready_state],
            )
        )

    scheduler._drain_pending_qwen3_omni_audio_phase_queue(apply_outputs=True)

    assert len(scheduler.code_predictor_batch_calls) == 1
    assert scheduler.code_predictor_batch_calls[0][1].codec_tokens.tolist() == [3, 4]
    assert len(scheduler.code2wav.calls) == 1
    assert scheduler.code2wav.calls[0].shape == (2, 2, 4)
    [(rids, return_logprob, audio_chunks)] = scheduler.streams
    assert rids == ["turn-a", "turn-b"]
    assert return_logprob is False
    assert [len(chunks) for chunks in audio_chunks] == [1, 1]
    assert [req.talker_last_decoded_frame for req in reqs] == [4, 4]


def test_scheduler_not_idle_with_queued_audio_phase_work():
    scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(is_empty=lambda: True),
        chunked_req=None,
        dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
        last_batch=None,
        cur_batch=None,
        enable_overlap=False,
        result_queue=deque(),
        ps=SimpleNamespace(pp_size=1),
        pp_size=1,
        running_mbs=[],
        waiting_queue=[],
        qwen3_omni_audio_phase_queue=deque(["pending-audio-phase"]),
        audio_turn_reqs={},
    )

    assert not Scheduler.is_fully_idle(scheduler, for_health_check=True)
