"""Qwen3-Omni audio gRPC servicer."""

import asyncio
import json
import logging
import time
import uuid

import grpc

from sglang.srt.audio import StreamingAudioBuffer
from sglang.srt.grpc import qwen3_omni_audio_pb2 as audio_pb2
from sglang.srt.grpc import qwen3_omni_audio_pb2_grpc as audio_pb2_grpc
from sglang.srt.grpc.audio_request_manager import AudioStreamRequestManager
from sglang.srt.managers.io_struct import (
    AbortReq,
    AudioSessionRebuildReqInput,
    AudioSessionRebuildTurn,
    AudioInputChunkReqInput,
    AudioInputEndReqInput,
    AudioTurnStartReqInput,
    CloseAudioSessionReqInput,
    HeartbeatAudioSessionReqInput,
)
from sglang.srt.observability.qwen3_omni_audio_metrics import (
    Qwen3OmniAudioMetrics,
    observe_qwen3_omni_audio_tokens,
)
from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class Qwen3OmniAudioServicer(audio_pb2_grpc.Qwen3OmniAudioServicer):
    def __init__(
        self,
        request_manager,
        audio_manager: AudioStreamRequestManager,
        server_args=None,
        feature_extractor=None,
    ):
        self.request_manager = request_manager
        self.audio_manager = audio_manager
        self.server_args = server_args
        self.feature_extractor = feature_extractor
        if server_args is None:
            raise RuntimeError(
                "Qwen3-Omni audio gRPC requires server_args so realtime "
                "capacity is explicit"
            )
        self.max_concurrent_audio_turns = (
            server_args.qwen3_omni_max_concurrent_audio_turns
        )
        self.metrics = (
            Qwen3OmniAudioMetrics(server_args) if server_args.enable_metrics else None
        )
        assert (
            self.max_concurrent_audio_turns is None
            or self.max_concurrent_audio_turns >= 0
        )

    async def AudioTurn(self, request_iterator, context):
        rid = None
        audio_buffer = None
        output_started = False
        turn_start_time = time.monotonic()
        eoi_time = None

        try:
            async for request in request_iterator:
                event = request.WhichOneof("event")
                if event == "config":
                    rid = request.config.request_id or uuid.uuid4().hex
                    state = await self.audio_manager.try_register(
                        rid, self.max_concurrent_audio_turns
                    )
                    if state is None:
                        logger.info(
                            "Qwen3-Omni audio admission: admitted=False, "
                            "active_audio_turns=%s, queued_users=0, "
                            "rejected_users=1, max_concurrent_audio_turns=%s",
                            self.audio_manager.active_count(),
                            self.max_concurrent_audio_turns,
                        )
                        yield audio_pb2.AudioTurnEvent(
                            request_id=rid,
                            error=audio_pb2.AudioTurnError(
                                code="RESOURCE_EXHAUSTED",
                                message=(
                                    "Qwen3-Omni audio realtime capacity is full: "
                                    f"{self.max_concurrent_audio_turns} active turns"
                                ),
                            ),
                        )
                        self._observe_audio_turn(
                            "resource_exhausted",
                            time.monotonic() - turn_start_time,
                        )
                        return
                    logger.info(
                        "Qwen3-Omni audio admission: admitted=True, "
                        "active_audio_turns=%s, queued_users=0, "
                        "rejected_users=0, max_concurrent_audio_turns=%s",
                        self.audio_manager.active_count(),
                        self.max_concurrent_audio_turns,
                    )
                    audio_buffer = self._new_audio_buffer()
                    sampling_params = self._sampling_params_from_json(
                        request.config.sampling_params_json
                    )
                    await self.request_manager._send_to_scheduler(
                        AudioTurnStartReqInput(
                            rid=rid,
                            session_id=request.config.session_id or rid,
                            sampling_params=sampling_params,
                            parent_rid=request.config.parent_rid or None,
                            system_prompt=request.config.system_prompt or None,
                            speaker=request.config.speaker or None,
                        )
                    )
                elif event == "pcm16_audio":
                    self._require_config(rid)
                    chunks = await asyncio.to_thread(
                        audio_buffer.add_samples, request.pcm16_audio
                    )
                    for mel_features, num_frames in chunks:
                        await self.request_manager._send_to_scheduler(
                            AudioInputChunkReqInput(
                                rid=rid,
                                mel_features=mel_features["input_features"],
                                num_frames=num_frames,
                            )
                        )
                elif event == "end_of_input":
                    self._require_config(rid)
                    final_mel = None
                    final_num_frames = 0
                    flush_result = await asyncio.to_thread(audio_buffer.flush)
                    if flush_result is not None:
                        mel_features, final_num_frames = flush_result
                        final_mel = mel_features["input_features"]
                    await self.request_manager._send_to_scheduler(
                        AudioInputEndReqInput(
                            rid=rid,
                            final_mel_features=final_mel,
                            final_num_frames=final_num_frames,
                        )
                    )
                    output_started = True
                    eoi_time = time.monotonic()
                    async for output in self._stream_audio_output(
                        rid, context, turn_start_time, eoi_time
                    ):
                        yield output
                    break
                elif event == "abort_reason":
                    self._require_config(rid)
                    await self.request_manager._send_to_scheduler(AbortReq(rid=rid))
                    await self.audio_manager.abort_locally(
                        rid,
                        request.abort_reason or "Request aborted",
                    )
                    output_started = True
                    eoi_time = time.monotonic()
                    async for output in self._stream_audio_output(
                        rid, context, turn_start_time, eoi_time
                    ):
                        yield output
                    break
                else:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "AudioTurn first event must be config",
                    )

            if rid is not None and not output_started:
                await self.request_manager._send_to_scheduler(AbortReq(rid=rid))
                self.audio_manager.cleanup(rid)
                self._observe_audio_turn(
                    "client_closed", time.monotonic() - turn_start_time
                )
        except grpc.aio.AbortError:
            raise
        except Exception as e:
            logger.exception("Qwen3-Omni AudioTurn failed")
            if rid is not None:
                await self.request_manager._send_to_scheduler(AbortReq(rid=rid))
                self.audio_manager.cleanup(rid)
            self._observe_audio_turn(
                "internal_error", time.monotonic() - turn_start_time
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def _stream_audio_output(
        self, rid: str, context, turn_start_time: float, eoi_time: float
    ):
        state = self.audio_manager.rid_to_state.get(rid)
        if state is None:
            self._observe_audio_turn("aborted", time.monotonic() - turn_start_time)
            yield audio_pb2.AudioTurnEvent(
                request_id=rid,
                error=audio_pb2.AudioTurnError(
                    code="ABORTED",
                    message="Audio session was cleaned up before output streaming",
                ),
            )
            return
        first_audio_observed = False
        while True:
            if context.cancelled():
                await self.request_manager._send_to_scheduler(AbortReq(rid=rid))
                self._observe_audio_turn(
                    "cancelled", time.monotonic() - turn_start_time
                )
                break

            output = await state.out_queue.get()
            error = output.get("error")
            if error is not None:
                self._observe_audio_turn("aborted", time.monotonic() - turn_start_time)
                yield audio_pb2.AudioTurnEvent(
                    request_id=rid,
                    error=audio_pb2.AudioTurnError(
                        code="ABORTED",
                        message=error,
                    ),
                )
                break

            text_delta = output.get("decoded_text")
            if text_delta:
                yield audio_pb2.AudioTurnEvent(
                    request_id=rid,
                    text_delta=text_delta,
                )

            for chunk in output.get("audio_chunks", []):
                if not first_audio_observed:
                    first_audio_observed = True
                    self._observe_first_audio(time.monotonic() - eoi_time)
                yield audio_pb2.AudioTurnEvent(
                    request_id=rid,
                    pcm16_audio=chunk,
                )

            if output.get("finished"):
                meta = output.get("meta_info", {})
                self._observe_audio_tokens(meta)
                yield audio_pb2.AudioTurnEvent(
                    request_id=rid,
                    complete=audio_pb2.AudioTurnComplete(
                        turn_rid=rid,
                        full_text=state.cumulative_text,
                        prompt_tokens=meta.get("prompt_tokens", 0),
                        completion_tokens=meta.get("completion_tokens", 0),
                        thinker_token_count=meta.get("thinker_token_count", 0),
                        talker_token_count=meta.get("talker_token_count", 0),
                    ),
                )
                self._observe_audio_turn(
                    "finished", time.monotonic() - turn_start_time
                )
                break

        self.audio_manager.cleanup(rid)

    def _observe_audio_turn(self, status: str, latency_seconds: float):
        if self.metrics is not None:
            self.metrics.observe_turn(status, latency_seconds)

    def _observe_first_audio(self, latency_seconds: float):
        if self.metrics is not None:
            self.metrics.observe_first_audio(latency_seconds)

    def _observe_audio_tokens(self, meta: dict):
        observe_qwen3_omni_audio_tokens(
            self.server_args, "thinker", "prefill", meta["prompt_tokens"]
        )
        observe_qwen3_omni_audio_tokens(
            self.server_args,
            "thinker",
            "prefill_compute",
            meta["prompt_tokens"] - meta["cached_tokens"],
        )
        observe_qwen3_omni_audio_tokens(
            self.server_args, "thinker", "decode", meta["thinker_token_count"]
        )
        observe_qwen3_omni_audio_tokens(
            self.server_args, "talker", "decode", meta["talker_token_count"]
        )

    async def CloseAudioSession(self, request, context):
        rid = f"close-audio-session-{uuid.uuid4().hex}"
        future = self.audio_manager.expect_response(rid)
        await self.request_manager._send_to_scheduler(
            CloseAudioSessionReqInput(
                rid=rid,
                session_id=request.session_id,
                reason=request.reason or None,
            )
        )
        output = await asyncio.wait_for(future, timeout=30.0)
        return audio_pb2.CloseAudioSessionResponse(
            success=True,
            message=f"Audio session {request.session_id} closed",
            turn_count=output.turn_count,
            active_request_count=output.active_request_count,
        )

    async def RebuildAudioSession(self, request_iterator, context):
        config = None
        buffers = {}
        audio_chunks_by_turn = {}
        turns_by_index = {}
        future = None
        try:
            async for request in request_iterator:
                event = request.WhichOneof("event")
                if event == "config":
                    if config is not None:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession accepts exactly one config",
                        )
                    config = request.config
                    if not config.session_id:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession config.session_id is required",
                        )
                    if not config.system_prompt:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession config.system_prompt is required",
                        )
                    if not config.speaker:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession config.speaker is required",
                        )
                elif event == "user_audio":
                    if config is None:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession first event must be config",
                        )
                    turn_index = request.user_audio.turn_index
                    if turn_index < 0:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession turn_index must be non-negative",
                        )
                    if turn_index in turns_by_index:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"RebuildAudioSession turn {turn_index} already closed",
                        )
                    buffer = buffers.setdefault(turn_index, self._new_audio_buffer())
                    chunks = await asyncio.to_thread(
                        buffer.add_samples, request.user_audio.pcm16_audio
                    )
                    audio_chunks_by_turn.setdefault(turn_index, []).extend(
                        (mel_features["input_features"], num_frames)
                        for mel_features, num_frames in chunks
                    )
                elif event == "assistant_text":
                    if config is None:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession first event must be config",
                        )
                    text = request.assistant_text
                    turn_index = text.turn_index
                    if turn_index < 0:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "RebuildAudioSession turn_index must be non-negative",
                        )
                    if turn_index in turns_by_index:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"RebuildAudioSession turn {turn_index} is duplicated",
                        )
                    buffer = buffers.pop(turn_index, None)
                    if buffer is None:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"RebuildAudioSession turn {turn_index} has no user audio",
                        )
                    audio_chunks = audio_chunks_by_turn.pop(turn_index, [])
                    flush_result = await asyncio.to_thread(buffer.flush)
                    if flush_result is not None:
                        mel_features, final_num_frames = flush_result
                        audio_chunks.append(
                            (mel_features["input_features"], final_num_frames)
                        )
                    if not text.turn_rid:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            f"RebuildAudioSession turn {turn_index} missing turn_rid",
                        )
                    turns_by_index[turn_index] = AudioSessionRebuildTurn(
                        turn_index=turn_index,
                        turn_rid=text.turn_rid,
                        assistant_text=text.text,
                        audio_chunks=audio_chunks,
                    )
                else:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "unknown RebuildAudioSession event",
                    )
            if config is None:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "RebuildAudioSession stream closed before config",
                )
            if buffers:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "RebuildAudioSession stream closed with unfinished user audio",
                )
            turn_items = [turns_by_index[i] for i in sorted(turns_by_index)]
            for expected, turn in enumerate(turn_items):
                if turn.turn_index != expected:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "RebuildAudioSession turn indices must be contiguous",
                    )

            rid = config.request_id or f"rebuild-audio-session-{uuid.uuid4().hex}"
            future = self.audio_manager.expect_response(rid)
            await self.request_manager._send_to_scheduler(
                AudioSessionRebuildReqInput(
                    rid=rid,
                    session_id=config.session_id,
                    system_prompt=config.system_prompt,
                    speaker=config.speaker,
                    sampling_params=self._sampling_params_from_json(
                        config.sampling_params_json
                    ),
                    turns=turn_items,
                )
            )
            output = await asyncio.wait_for(future, timeout=300.0)
            return audio_pb2.RebuildAudioSessionResponse(
                success=True,
                message=f"Audio session {config.session_id} rebuilt",
                latest_turn_rid=output.latest_turn_rid,
                prefix_turn_count=output.prefix_turn_count,
                prompt_tokens=output.prompt_tokens,
            )
        except grpc.aio.AbortError:
            raise
        except Exception as e:
            if future is not None:
                future.cancel()
            logger.exception("Qwen3-Omni RebuildAudioSession failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def HeartbeatAudioSession(self, request, context):
        rid = f"heartbeat-audio-session-{uuid.uuid4().hex}"
        future = self.audio_manager.expect_response(rid)
        await self.request_manager._send_to_scheduler(
            HeartbeatAudioSessionReqInput(
                rid=rid,
                session_id=request.session_id,
                lease_token=request.lease_token,
                ttl_secs=request.ttl_secs,
            )
        )
        output = await asyncio.wait_for(future, timeout=30.0)
        return audio_pb2.HeartbeatAudioSessionResponse(
            success=output.success,
            message=output.message,
            lease_expires_at_unix=output.lease_expires_at_unix,
        )

    def _new_audio_buffer(self):
        if self.feature_extractor is None:
            if self.server_args is None:
                raise RuntimeError(
                    "Qwen3-Omni audio feature extractor is not initialized"
                )
            from sglang.srt.utils.hf_transformers_utils import get_processor

            processor = get_processor(
                self.server_args.tokenizer_path or self.server_args.model_path,
                tokenizer_mode=self.server_args.tokenizer_mode,
                trust_remote_code=self.server_args.trust_remote_code,
            )
            self.feature_extractor = processor.feature_extractor
        return StreamingAudioBuffer(self.feature_extractor)

    @staticmethod
    def _sampling_params_from_json(raw_json: str) -> SamplingParams:
        payload = json.loads(raw_json) if raw_json else {}
        if not isinstance(payload, dict):
            raise TypeError("sampling_params_json must encode an object")
        return SamplingParams(**payload)

    @staticmethod
    def _require_config(rid):
        if rid is None:
            raise ValueError("AudioTurn config must be sent before audio events")
