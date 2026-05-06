# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Request-state helper for Qwen3-Omni audio gRPC streams.

This module is intentionally transport-small. The gRPC API can evolve with the
async audio pipeline, while this manager owns only request registration, local
terminal aborts, and routing scheduler outputs into per-turn queues.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Dict

from sglang.srt.managers.io_struct import (
    AbortReq,
    AudioSessionRebuildReqOutput,
    BatchTokenIDOutput,
    CloseAudioSessionReqOutput,
    HeartbeatAudioSessionReqOutput,
)
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from sglang.utils import find_printable_text


@dataclasses.dataclass
class AudioStreamReqState:
    request_id: str
    out_queue: asyncio.Queue
    finished: bool
    event: asyncio.Event
    cumulative_text: str = ""
    decode_ids: list[int] = dataclasses.field(default_factory=list)
    surr_offset: int = 0
    read_offset: int = 0
    sent_offset: int = 0


class AudioStreamRequestManager:
    def __init__(self, server_args=None):
        self.rid_to_state: Dict[str, AudioStreamReqState] = {}
        self.rid_to_response_future: Dict[str, asyncio.Future] = {}
        self.register_lock = asyncio.Lock()
        self.tokenizer = None
        if server_args is not None:
            if server_args.skip_tokenizer_init:
                raise RuntimeError(
                    "Qwen3-Omni audio gRPC requires tokenizer initialization "
                    "to stream and validate text/audio alignment"
                )
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )

    def register(self, request_id: str) -> AudioStreamReqState:
        assert request_id not in self.rid_to_state, (
            f"Audio request {request_id} is already registered"
        )
        state = AudioStreamReqState(
            request_id=request_id,
            out_queue=asyncio.Queue(),
            finished=False,
            event=asyncio.Event(),
        )
        self.rid_to_state[request_id] = state
        return state

    async def try_register(
        self, request_id: str, max_active: int | None
    ) -> AudioStreamReqState | None:
        async with self.register_lock:
            if max_active is not None and len(self.rid_to_state) >= max_active:
                return None
            return self.register(request_id)

    def active_count(self) -> int:
        return len(self.rid_to_state)

    def cleanup(self, request_id: str):
        self.rid_to_state.pop(request_id, None)

    async def abort_locally(self, request_id: str, message: str = "Request aborted"):
        state = self.rid_to_state.get(request_id)
        if state is None:
            return False
        if not state.finished:
            state.finished = True
            await state.out_queue.put(
                {
                    "request_id": request_id,
                    "error": message,
                    "finished": True,
                    "meta_info": {
                        "id": request_id,
                        "finish_reason": {
                            "type": "abort",
                            "message": message,
                        },
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "thinker_token_count": 0,
                        "talker_token_count": 0,
                    },
                }
            )
            state.event.set()
        return True

    async def route_scheduler_output(self, output):
        if isinstance(
            output,
            (
                AudioSessionRebuildReqOutput,
                CloseAudioSessionReqOutput,
                HeartbeatAudioSessionReqOutput,
            ),
        ):
            future = self.rid_to_response_future.pop(output.rid, None)
            if future is not None and not future.done():
                future.set_result(output)
            return

        if isinstance(output, AbortReq):
            await self.abort_locally(output.rid)
            return

        if not isinstance(output, BatchTokenIDOutput):
            return

        audio_chunks = output.audio_chunks or [[] for _ in output.rids]
        for i, rid in enumerate(output.rids):
            state = self.rid_to_state.get(rid)
            if state is None:
                continue

            decoded_text = self._decode_text_delta(state, output, i)
            finished_reason = output.finished_reasons[i]
            finished = finished_reason is not None
            await state.out_queue.put(
                {
                    "request_id": rid,
                    "decoded_text": decoded_text,
                    "token_ids": output.output_ids[i] if output.output_ids else [],
                    "audio_chunks": audio_chunks[i],
                    "finished": finished,
                    "meta_info": {
                        "id": rid,
                        "finish_reason": finished_reason,
                        "prompt_tokens": output.prompt_tokens[i],
                        "completion_tokens": output.completion_tokens[i],
                        "thinker_token_count": output.completion_tokens[i],
                        "talker_token_count": output.talker_tokens[i],
                    },
                }
            )
            if finished:
                state.finished = True
                state.event.set()

    def _decode_text_delta(
        self,
        state: AudioStreamReqState,
        output: BatchTokenIDOutput,
        index: int,
    ) -> str:
        token_ids = output.output_ids[index] if output.output_ids else []

        decoded_texts = output.decoded_texts or []
        if index < len(decoded_texts) and decoded_texts[index]:
            state.cumulative_text += decoded_texts[index]
            state.sent_offset = len(state.cumulative_text)
            return decoded_texts[index]

        if self.tokenizer is None:
            return ""

        incoming_decode_ids = output.decode_ids[index] if output.decode_ids else []
        if not incoming_decode_ids:
            incoming_decode_ids = token_ids
        if incoming_decode_ids:
            if not state.decode_ids:
                state.read_offset = (
                    output.read_offsets[index] if output.read_offsets else 0
                )
            state.decode_ids.extend(incoming_decode_ids)

        if not state.decode_ids:
            return ""

        skip_special_tokens = output.skip_special_tokens[index]
        spaces_between_special_tokens = output.spaces_between_special_tokens[index]
        finished_reason = output.finished_reasons[index]

        read_ids = self._trim_matched_stop(
            state.decode_ids[state.surr_offset :],
            finished_reason,
            output.no_stop_trim[index],
        )
        surr_ids = state.decode_ids[state.surr_offset : state.read_offset]
        surr_text = self.tokenizer.decode(
            surr_ids,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        read_text = self.tokenizer.decode(
            read_ids,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = read_text[len(surr_text) :]

        if finished_reason is None:
            if new_text and not new_text.endswith("�"):
                state.cumulative_text += new_text
                state.surr_offset = state.read_offset
                state.read_offset = len(state.decode_ids)
                new_text = ""
            else:
                new_text = find_printable_text(new_text)

        output_text = self._trim_matched_stop(
            state.cumulative_text + new_text,
            finished_reason,
            output.no_stop_trim[index],
        )
        delta = output_text[state.sent_offset :]
        state.sent_offset = len(output_text)
        if finished_reason is not None:
            state.cumulative_text = output_text
        return delta

    @staticmethod
    def _trim_matched_stop(output: str | list[int], finished_reason, no_stop_trim: bool):
        if no_stop_trim or not finished_reason:
            return output
        matched = finished_reason.get("matched", None)
        if matched is None:
            return output
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output
        if isinstance(matched, int) and isinstance(output, list):
            assert output
            return output[:-1]
        return output

    def expect_response(self, rid: str) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self.rid_to_response_future[rid] = future
        # Pop the dict entry whenever the future settles (set_result, set_exception,
        # or cancel via asyncio.wait_for timeout). Without this, a never-responding
        # scheduler would leak rid entries indefinitely.
        future.add_done_callback(
            lambda _f, rid=rid: self.rid_to_response_future.pop(rid, None)
        )
        return future


def install_audio_output_routing(request_manager, audio_manager):
    old_handle_batch_output = request_manager._handle_batch_output
    old_handle_abort_req = request_manager._handle_abort_req

    async def handle_batch_output_with_audio(batch_out):
        await audio_manager.route_scheduler_output(batch_out)
        await old_handle_batch_output(batch_out)

    async def handle_abort_req_with_audio(abort_req):
        await audio_manager.route_scheduler_output(abort_req)
        await old_handle_abort_req(abort_req)

    request_manager._handle_batch_output = handle_batch_output_with_audio
    request_manager._handle_abort_req = handle_abort_req_with_audio
