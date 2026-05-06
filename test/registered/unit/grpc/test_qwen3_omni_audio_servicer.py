import asyncio
from types import SimpleNamespace

import numpy as np

from sglang.srt.grpc import qwen3_omni_audio_pb2 as audio_pb2
from sglang.srt.grpc.audio_request_manager import AudioStreamRequestManager
from sglang.srt.grpc.qwen3_omni_audio_servicer import Qwen3OmniAudioServicer
from sglang.srt.managers.io_struct import (
    AudioSessionRebuildReqInput,
    AudioSessionRebuildReqOutput,
    AudioInputEndReqInput,
    AudioTurnStartReqInput,
    BatchTokenIDOutput,
    CloseAudioSessionReqInput,
    CloseAudioSessionReqOutput,
    HeartbeatAudioSessionReqInput,
    HeartbeatAudioSessionReqOutput,
)


class FakeContext:
    def cancelled(self):
        return False

    async def abort(self, code, details):
        raise AssertionError(f"unexpected abort: {code} {details}")


class FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        return {"input_features": np.asarray(audio)[None, None, :]}


def _server_args(max_concurrent_audio_turns):
    return SimpleNamespace(
        qwen3_omni_max_concurrent_audio_turns=max_concurrent_audio_turns
    )


class FakeRequestManager:
    def __init__(self, audio_manager):
        self.audio_manager = audio_manager
        self.sent = []

    async def _send_to_scheduler(self, obj):
        self.sent.append(obj)
        if isinstance(obj, AudioInputEndReqInput):
            await self.audio_manager.route_scheduler_output(
                BatchTokenIDOutput(
                    rids=[obj.rid],
                    http_worker_ipcs=[None],
                    finished_reasons=[{"type": "stop"}],
                    decoded_texts=["hello"],
                    decode_ids=[[]],
                    read_offsets=[0],
                    output_ids=[[1]],
                    skip_special_tokens=[True],
                    spaces_between_special_tokens=[True],
                    no_stop_trim=[False],
                    prompt_tokens=[4],
                    reasoning_tokens=[0],
                    completion_tokens=[1],
                    talker_tokens=[16],
                    cached_tokens=[0],
                    input_token_logprobs_val=[],
                    input_token_logprobs_idx=[],
                    output_token_logprobs_val=[],
                    output_token_logprobs_idx=[],
                    input_top_logprobs_val=[],
                    input_top_logprobs_idx=[],
                    output_top_logprobs_val=[],
                    output_top_logprobs_idx=[],
                    input_token_ids_logprobs_val=[],
                    input_token_ids_logprobs_idx=[],
                    output_token_ids_logprobs_val=[],
                    output_token_ids_logprobs_idx=[],
                    output_token_entropy_val=[],
                    output_hidden_states=[],
                    routed_experts=[],
                    indexer_topk=[],
                    placeholder_tokens_idx=[],
                    placeholder_tokens_val=[],
                    retraction_counts=[0],
                    spec_verify_ct=[],
                    spec_num_correct_drafts=[],
                    spec_correct_drafts_histogram=[],
                )
            )
        elif isinstance(obj, CloseAudioSessionReqInput):
            await self.audio_manager.route_scheduler_output(
                CloseAudioSessionReqOutput(
                    rid=obj.rid,
                    session_id=obj.session_id,
                    turn_count=2,
                    active_request_count=0,
                )
            )
        elif isinstance(obj, HeartbeatAudioSessionReqInput):
            await self.audio_manager.route_scheduler_output(
                HeartbeatAudioSessionReqOutput(
                    rid=obj.rid,
                    session_id=obj.session_id,
                    success=True,
                    message="audio session lease extended",
                    lease_expires_at_unix=123.0,
                )
            )
        elif isinstance(obj, AudioSessionRebuildReqInput):
            await self.audio_manager.route_scheduler_output(
                AudioSessionRebuildReqOutput(
                    rid=obj.rid,
                    session_id=obj.session_id,
                    latest_turn_rid=obj.turns[-1].turn_rid if obj.turns else "",
                    prefix_turn_count=len(obj.turns),
                    prompt_tokens=123,
                )
            )


async def _request_stream_for(request_id):
    yield audio_pb2.AudioTurnRequest(
        config=audio_pb2.AudioTurnConfig(
            request_id=request_id,
            session_id="session-a",
            sampling_params_json='{"max_new_tokens": 1}',
            system_prompt="You are Lucy.",
            speaker="chelsie",
        )
    )
    yield audio_pb2.AudioTurnRequest(end_of_input=True)


async def _request_stream():
    async for request in _request_stream_for("turn-a"):
        yield request


def test_audio_turn_stream_reaches_scheduler_and_returns_terminal_event():
    asyncio.run(_test_audio_turn_stream_reaches_scheduler_and_returns_terminal_event())


async def _test_audio_turn_stream_reaches_scheduler_and_returns_terminal_event():
    audio_manager = AudioStreamRequestManager()
    request_manager = FakeRequestManager(audio_manager)
    servicer = Qwen3OmniAudioServicer(
        request_manager=request_manager,
        audio_manager=audio_manager,
        feature_extractor=FakeFeatureExtractor(),
        server_args=_server_args(4),
    )

    outputs = [
        output async for output in servicer.AudioTurn(_request_stream(), FakeContext())
    ]

    assert isinstance(request_manager.sent[0], AudioTurnStartReqInput)
    assert request_manager.sent[0].rid == "turn-a"
    assert isinstance(request_manager.sent[1], AudioInputEndReqInput)
    assert outputs[0].text_delta == "hello"
    assert outputs[1].complete.turn_rid == "turn-a"
    assert outputs[1].complete.full_text == "hello"
    assert outputs[1].complete.thinker_token_count == 1
    assert outputs[1].complete.talker_token_count == 16
    assert audio_manager.active_count() == 0


def test_audio_turn_completion_releases_realtime_capacity():
    asyncio.run(_test_audio_turn_completion_releases_realtime_capacity())


async def _test_audio_turn_completion_releases_realtime_capacity():
    audio_manager = AudioStreamRequestManager()
    request_manager = FakeRequestManager(audio_manager)

    servicer = Qwen3OmniAudioServicer(
        request_manager=request_manager,
        audio_manager=audio_manager,
        feature_extractor=FakeFeatureExtractor(),
        server_args=_server_args(1),
    )
    first_outputs = [
        output
        async for output in servicer.AudioTurn(
            _request_stream_for("turn-a"), FakeContext()
        )
    ]
    second_outputs = [
        output
        async for output in servicer.AudioTurn(
            _request_stream_for("turn-b"), FakeContext()
        )
    ]

    assert first_outputs[-1].complete.turn_rid == "turn-a"
    assert second_outputs[-1].complete.turn_rid == "turn-b"
    assert audio_manager.active_count() == 0
    started_turns = [
        req.rid
        for req in request_manager.sent
        if isinstance(req, AudioTurnStartReqInput)
    ]
    assert started_turns == ["turn-a", "turn-b"]


def test_audio_session_control_rpcs_route_scheduler_responses():
    asyncio.run(_test_audio_session_control_rpcs_route_scheduler_responses())


async def _test_audio_session_control_rpcs_route_scheduler_responses():
    audio_manager = AudioStreamRequestManager()
    request_manager = FakeRequestManager(audio_manager)
    servicer = Qwen3OmniAudioServicer(
        request_manager=request_manager,
        audio_manager=audio_manager,
        feature_extractor=FakeFeatureExtractor(),
        server_args=_server_args(4),
    )

    close = await servicer.CloseAudioSession(
        audio_pb2.CloseAudioSessionRequest(session_id="session-a"),
        FakeContext(),
    )
    heartbeat = await servicer.HeartbeatAudioSession(
        audio_pb2.HeartbeatAudioSessionRequest(
            session_id="session-a",
            lease_token="lease-a",
            ttl_secs=60,
        ),
        FakeContext(),
    )

    assert close.success
    assert close.turn_count == 2
    assert heartbeat.success
    assert heartbeat.lease_expires_at_unix == 123.0


async def _rebuild_request_stream():
    yield audio_pb2.RebuildAudioSessionRequest(
        config=audio_pb2.RebuildAudioSessionConfig(
            request_id="rebuild-a",
            session_id="session-a",
            system_prompt="You are Lucy.",
            speaker="chelsie",
            sampling_params_json='{"max_new_tokens": 1}',
        )
    )
    yield audio_pb2.RebuildAudioSessionRequest(
        user_audio=audio_pb2.RebuildAudioSessionUserAudio(
            turn_index=0,
            pcm16_audio=b"\0\0" * 160,
        )
    )
    yield audio_pb2.RebuildAudioSessionRequest(
        assistant_text=audio_pb2.RebuildAudioSessionAssistantText(
            turn_index=0,
            turn_rid="safe-rid",
            text="safe response",
        )
    )


def test_rebuild_audio_session_routes_full_prefix_to_scheduler():
    asyncio.run(_test_rebuild_audio_session_routes_full_prefix_to_scheduler())


async def _test_rebuild_audio_session_routes_full_prefix_to_scheduler():
    audio_manager = AudioStreamRequestManager()
    request_manager = FakeRequestManager(audio_manager)
    servicer = Qwen3OmniAudioServicer(
        request_manager=request_manager,
        audio_manager=audio_manager,
        feature_extractor=FakeFeatureExtractor(),
        server_args=_server_args(4),
    )

    response = await servicer.RebuildAudioSession(
        _rebuild_request_stream(), FakeContext()
    )

    rebuild_req = request_manager.sent[-1]
    assert isinstance(rebuild_req, AudioSessionRebuildReqInput)
    assert rebuild_req.rid == "rebuild-a"
    assert rebuild_req.session_id == "session-a"
    assert rebuild_req.system_prompt == "You are Lucy."
    assert rebuild_req.speaker == "chelsie"
    assert len(rebuild_req.turns) == 1
    assert rebuild_req.turns[0].turn_rid == "safe-rid"
    assert rebuild_req.turns[0].assistant_text == "safe response"
    assert len(rebuild_req.turns[0].audio_chunks) == 1
    assert response.latest_turn_rid == "safe-rid"
    assert response.prefix_turn_count == 1
    assert response.prompt_tokens == 123


async def _rejected_request_stream():
    yield audio_pb2.AudioTurnRequest(
        config=audio_pb2.AudioTurnConfig(
            request_id="turn-rejected",
            session_id="session-a",
            sampling_params_json='{"max_new_tokens": 1}',
        )
    )


def test_audio_turn_rejects_when_realtime_capacity_is_full():
    asyncio.run(_test_audio_turn_rejects_when_realtime_capacity_is_full())


async def _test_audio_turn_rejects_when_realtime_capacity_is_full():
    audio_manager = AudioStreamRequestManager()
    audio_manager.register("turn-active")
    request_manager = FakeRequestManager(audio_manager)

    servicer = Qwen3OmniAudioServicer(
        request_manager=request_manager,
        audio_manager=audio_manager,
        feature_extractor=FakeFeatureExtractor(),
        server_args=_server_args(1),
    )
    outputs = [
        output
        async for output in servicer.AudioTurn(
            _rejected_request_stream(), FakeContext()
        )
    ]

    assert len(outputs) == 1
    assert outputs[0].request_id == "turn-rejected"
    assert outputs[0].error.code == "RESOURCE_EXHAUSTED"
    assert request_manager.sent == []
    assert "turn-rejected" not in audio_manager.rid_to_state
