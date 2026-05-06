from types import SimpleNamespace

import torch

from sglang.srt.managers.io_struct import (
    AudioSessionRebuildReqInput,
    AudioSessionRebuildTurn,
    AudioTurnStartReqInput,
    HeartbeatAudioSessionReqInput,
)
from sglang.srt.managers.scheduler import (
    QWEN3_OMNI_TALKER_PREFIX_COMPONENT,
    AudioSessionState,
    Qwen3OmniTalkerPrefixState,
    Scheduler,
)
from sglang.srt.managers.scheduler_qwen3_omni_mixin import (
    release_qwen3_omni_post_thinker_kv,
)
from sglang.srt.session.session_controller import Session, SessionReqNode
from sglang.srt.sampling.sampling_params import SamplingParams


class FakeTreeCache:
    def supports_swa(self):
        return False

    def is_tree_cache(self):
        return False

    def supports_streaming_session(self):
        return False


class FakeStreamingTreeCache(FakeTreeCache):
    def supports_streaming_session(self):
        return True


class FakeSessionController:
    def __init__(self):
        self.sessions = {}
        self.closed = []

    def __contains__(self, session_id):
        return session_id in self.sessions

    def get(self, session_id):
        return self.sessions.get(session_id)

    def open(self, recv_req):
        self.sessions[recv_req.session_id] = Session(
            recv_req.capacity_of_str_len,
            recv_req.session_id,
            streaming=recv_req.streaming,
        )
        return SimpleNamespace(success=True)

    def close(self, recv_req):
        self.closed.append(recv_req.session_id)


class FakeKvAllocator:
    page_size = 64

    def __init__(self):
        self.freed = []

    def free(self, locs):
        self.freed.append(locs.detach().cpu().tolist())


class FakePagedTreeCache:
    page_size = 64

    def __init__(self):
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(512, dtype=torch.int32).reshape(1, 512)
        )
        self.token_to_kv_pool_allocator = FakeKvAllocator()


class FakeDetokenizerSender:
    def __init__(self):
        self.outputs = []

    def send_output(self, output, recv_req):
        self.outputs.append((output, recv_req))


class FakeSamplingParams:
    def normalize(self, tokenizer):
        self.tokenizer = tokenizer


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def new_scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler.audio_sessions = {}
    scheduler.audio_turn_reqs = {}
    scheduler.tree_cache = FakeStreamingTreeCache()
    scheduler.session_controller = FakeSessionController()
    scheduler.server_args = SimpleNamespace(enable_streaming_session=True)
    scheduler.send_to_detokenizer = FakeDetokenizerSender()
    scheduler.aborted_rids = []
    scheduler.tokenizer = FakeTokenizer()
    scheduler.model_config = SimpleNamespace(
        hf_eos_token_id=[151645],
        vocab_size=200000,
        hf_config=SimpleNamespace(
            talker_config=SimpleNamespace(codec_eos_token_id=2150)
        ),
    )
    scheduler.waiting_queue = []

    def abort_request(abort_req):
        scheduler.aborted_rids.append(abort_req.rid)

    scheduler.abort_request = abort_request
    scheduler.is_fully_idle = lambda: False
    return scheduler


def test_qwen3_omni_requires_streaming_session_at_startup():
    scheduler = object.__new__(Scheduler)
    scheduler.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_omni_moe")
    )
    scheduler.server_args = SimpleNamespace(enable_streaming_session=False)

    try:
        scheduler._validate_qwen3_omni_streaming_session_required()
    except RuntimeError as exc:
        assert "--enable-streaming-session" in str(exc)
    else:
        raise AssertionError("expected Qwen3-Omni startup to require streaming session")


def test_streaming_session_startup_gate_is_qwen3_omni_specific():
    scheduler = object.__new__(Scheduler)
    scheduler.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    scheduler.server_args = SimpleNamespace(enable_streaming_session=False)

    scheduler._validate_qwen3_omni_streaming_session_required()


def test_audio_session_heartbeat_sets_backend_lease():
    scheduler = new_scheduler()
    scheduler.audio_sessions["session-a"] = AudioSessionState()

    scheduler.heartbeat_audio_session(
        HeartbeatAudioSessionReqInput(
            rid="heartbeat-a",
            session_id="session-a",
            lease_token="lease-a",
            ttl_secs=120,
        )
    )

    session = scheduler.audio_sessions["session-a"]
    output, _ = scheduler.send_to_detokenizer.outputs[0]
    assert output.success
    assert output.session_id == "session-a"
    assert output.lease_expires_at_unix == session.lease_expires_at
    assert session.lease_token == "lease-a"
    assert session.last_backend_heartbeat_at is not None
    assert session.lease_expires_at > session.last_backend_heartbeat_at


def test_audio_session_lease_sweeper_aborts_active_requests():
    scheduler = new_scheduler()
    scheduler.session_controller.sessions["expired"] = Session(
        0, "expired", streaming=True
    )
    scheduler.audio_sessions["expired"] = AudioSessionState(
        completed_turn_count=1,
        lease_token="lease-expired",
        lease_expires_at=100.0,
    )
    scheduler.audio_sessions["live"] = AudioSessionState(
        lease_token="lease-live",
        lease_expires_at=200.0,
    )
    scheduler.audio_turn_reqs["active-rid"] = SimpleNamespace(
        rid="active-rid",
        audio_session_id="expired",
    )

    scheduler.cleanup_stale_audio_sessions(now=101.0)

    assert "expired" not in scheduler.audio_sessions
    assert "live" in scheduler.audio_sessions
    assert scheduler.aborted_rids == ["active-rid"]
    assert scheduler.session_controller.closed == ["expired"]


def test_completed_audio_turn_is_not_counted_as_active_on_close():
    scheduler = new_scheduler()
    scheduler.audio_sessions["session-a"] = AudioSessionState()
    req = SimpleNamespace(
        rid="turn-a",
        audio_session_id="session-a",
        finished_reason=None,
        output_ids=[2],
        origin_input_ids=[1],
        session=Session(0, "session-a", streaming=True),
        qwen3_omni_prompt_accept_hidden=None,
        talker_kv_cache_locs=None,
        talker_prefill_input_ids=None,
        talker_prefill_len=0,
        talker_user_prefill_len=0,
    )
    scheduler.audio_turn_reqs["turn-a"] = req

    scheduler._store_audio_turn_state(req)
    turn_count, active_request_count = scheduler._close_audio_session_state("session-a")

    assert turn_count == 1
    assert active_request_count == 0
    assert "turn-a" not in scheduler.audio_turn_reqs


def test_audio_turn_requires_streaming_session_ownership():
    scheduler = new_scheduler()
    scheduler.audio_sessions["session-a"] = AudioSessionState()
    req = SimpleNamespace(
        rid="turn-a",
        audio_session_id="session-a",
        finished_reason=None,
        output_ids=[2],
        origin_input_ids=[1],
        session=None,
        qwen3_omni_prompt_accept_hidden=None,
        talker_kv_cache_locs=None,
        talker_prefill_input_ids=None,
        talker_prefill_len=0,
        talker_user_prefill_len=0,
    )
    scheduler.audio_turn_reqs["turn-a"] = req

    try:
        scheduler._store_audio_turn_state(req)
    except RuntimeError as exc:
        assert "streaming session" in str(exc)
    else:
        raise AssertionError("expected _store_audio_turn_state to require streaming")

    req.session = Session(0, "session-a", streaming=True)
    scheduler._store_audio_turn_state(req)

    assert scheduler.audio_sessions["session-a"].completed_turn_count == 1


def test_audio_turn_start_rejects_non_latest_parent():
    scheduler = new_scheduler()
    scheduler.tree_cache = FakeStreamingTreeCache()
    scheduler.session_controller = FakeSessionController()
    scheduler.audio_sessions["session-a"] = AudioSessionState(
        speaker="chelsie",
        completed_turn_count=2,
    )
    latest_req = SimpleNamespace(
        rid="turn-b",
        origin_input_ids=[1, 2],
        output_ids=[3],
        qwen3_omni_prompt_accept_hidden=None,
    )
    streaming_session = Session(0, "session-a", streaming=True)
    streaming_session.req_nodes["turn-b"] = SessionReqNode(latest_req)
    streaming_session.set_component_kv(
        QWEN3_OMNI_TALKER_PREFIX_COMPONENT,
        Qwen3OmniTalkerPrefixState(
            kv_cache_locs=torch.tensor([10, 11]),
            prefill_input_ids=torch.tensor([1, 2]),
        ),
    )
    scheduler.session_controller.sessions["session-a"] = streaming_session

    scheduler.handle_audio_turn_start(
        AudioTurnStartReqInput(
            rid="turn-c",
            session_id="session-a",
            sampling_params=FakeSamplingParams(),
            parent_rid="turn-a",
        )
    )

    output, _ = scheduler.send_to_detokenizer.outputs[0]
    assert output.rid == "turn-c"
    assert output.finished_reason["type"] == "abort"
    assert output.finished_reason["status_code"] == 400
    assert "append-only" in output.finished_reason["message"]
    assert (
        streaming_session.get_component_kv(QWEN3_OMNI_TALKER_PREFIX_COMPONENT)
        is not None
    )


def test_post_thinker_kv_trim_preserves_partial_page_accounting():
    tree_cache = FakePagedTreeCache()
    req = SimpleNamespace(
        rid="turn-a",
        codec_eos_token_id=1,
        thinker_done=True,
        origin_input_ids=[0] * 100,
        output_ids=[0] * 100,
        cache_protected_len=0,
        kv_committed_len=224,
        kv_allocated_len=224,
        req_pool_idx=0,
    )

    assert release_qwen3_omni_post_thinker_kv(req, tree_cache)

    # Actual thinker length is 199, so a paged allocator cannot release
    # 199:224 without also releasing live tokens. Keep the partial page owned
    # and let the session/cache bookkeeping account for it.
    assert req.kv_committed_len == 199
    assert req.kv_allocated_len == 224
    assert tree_cache.token_to_kv_pool_allocator.freed == []


def test_audio_turn_start_reports_stale_parent_as_terminal_abort():
    scheduler = new_scheduler()

    scheduler.handle_audio_turn_start(
        AudioTurnStartReqInput(
            rid="turn-b",
            session_id="closed-session",
            sampling_params=FakeSamplingParams(),
            parent_rid="missing-parent",
        )
    )

    output, _ = scheduler.send_to_detokenizer.outputs[0]
    assert output.rid == "turn-b"
    assert output.finished_reason["type"] == "abort"
    assert output.finished_reason["status_code"] == 400
    assert "closed-session" in output.finished_reason["message"]


def test_audio_turn_start_requires_system_prompt_for_new_session():
    scheduler = new_scheduler()

    scheduler.handle_audio_turn_start(
        AudioTurnStartReqInput(
            rid="turn-a",
            session_id="session-a",
            sampling_params=FakeSamplingParams(),
            speaker="chelsie",
        )
    )

    output, _ = scheduler.send_to_detokenizer.outputs[0]
    assert output.rid == "turn-a"
    assert output.finished_reason["type"] == "abort"
    assert output.finished_reason["status_code"] == 400
    assert "system_prompt" in output.finished_reason["message"]
    assert "session-a" not in scheduler.audio_sessions


def test_audio_turn_start_requires_speaker_for_new_session():
    scheduler = new_scheduler()

    scheduler.handle_audio_turn_start(
        AudioTurnStartReqInput(
            rid="turn-a",
            session_id="session-a",
            sampling_params=FakeSamplingParams(),
            system_prompt="You are Lucy.",
        )
    )

    output, _ = scheduler.send_to_detokenizer.outputs[0]
    assert output.rid == "turn-a"
    assert output.finished_reason["type"] == "abort"
    assert output.finished_reason["status_code"] == 400
    assert "speaker" in output.finished_reason["message"]
    assert "session-a" not in scheduler.audio_sessions


def test_audio_session_rebuild_queues_full_prefix_prefill():
    scheduler = new_scheduler()
    scheduler.handle_audio_session_rebuild(
        AudioSessionRebuildReqInput(
            rid="rebuild-a",
            session_id="session-a",
            system_prompt="sys",
            speaker="chelsie",
            sampling_params=SamplingParams(max_new_tokens=1),
            turns=[
                AudioSessionRebuildTurn(
                    turn_index=0,
                    turn_rid="safe-rid",
                    assistant_text="hello",
                    audio_chunks=[(torch.ones(1, 80, 100), 100)],
                )
            ],
        )
    )

    assert scheduler.audio_sessions["session-a"].completed_turn_count == 1
    req = scheduler.waiting_queue[0]
    assert req.rid == "safe-rid"
    assert req.audio_session_id == "session-a"
    assert req.speaker == "chelsie"
    assert req.qwen3_omni_rebuild_request
    assert req.qwen3_omni_rebuild_response_rid == "rebuild-a"
    assert req.qwen3_omni_rebuild_turn_count == 1
    assert req.codec_eos_token_id == 2150
    assert req.session is scheduler.session_controller.get("session-a")
    assert req.session._inflight
    assert req.multimodal_inputs is not None
    assert len(req.multimodal_inputs.mm_items) == 1


def test_audio_session_rebuild_stores_talker_prefix_sidecar():
    scheduler = new_scheduler()
    scheduler.audio_sessions["session-a"] = AudioSessionState(speaker="chelsie")
    session = Session(0, "session-a", streaming=True)
    scheduler.session_controller.sessions["session-a"] = session
    req = SimpleNamespace(
        rid="safe-rid",
        audio_session_id="session-a",
        session=session,
        finished_reason=None,
        qwen3_omni_prompt_accept_hidden=None,
        qwen3_omni_rebuild_response_rid="rebuild-a",
        qwen3_omni_rebuild_turn_count=2,
        origin_input_ids=[1, 2, 3],
        talker_kv_cache_locs=torch.tensor([10, 11, 12]),
        talker_prefill_input_ids=torch.tensor([20, 21, 22]),
        talker_prefill_len=3,
        talker_user_prefill_len=3,
    )

    scheduler._store_audio_session_rebuild_state(req)

    talker_prefix = session.get_component_kv(QWEN3_OMNI_TALKER_PREFIX_COMPONENT)
    assert talker_prefix.kv_cache_locs.tolist() == [10, 11, 12]
    assert talker_prefix.prefill_input_ids.tolist() == [20, 21, 22]
    assert scheduler.audio_sessions["session-a"].completed_turn_count == 2
    output, _ = scheduler.send_to_detokenizer.outputs[0]
    assert output.rid == "rebuild-a"
    assert output.latest_turn_rid == "safe-rid"
    assert output.prefix_turn_count == 2
    assert output.prompt_tokens == 3
    assert req.talker_kv_cache_locs is None
    assert req.talker_prefill_input_ids is None
