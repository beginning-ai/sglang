from sglang.srt.session.session_controller import Session, SessionController


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id: str):
        self.released_sessions.append(session_id)


def test_deferred_close_waits_for_inflight_streaming_request_without_committed_node():
    tree_cache = FakeTreeCache()
    controller = SessionController(tree_cache)
    session = Session(capacity_of_str_len=4096, session_id="session-a", streaming=True)
    session._inflight = True
    session.close_on_finish = True
    controller.sessions[session.session_id] = session

    controller.maybe_reap(now=2.0, interval=0.0)

    assert "session-a" in controller.sessions
    assert controller.sessions["session-a"].close_on_finish
    assert tree_cache.released_sessions == []


def test_deferred_close_releases_after_streaming_abort_clears_inflight():
    tree_cache = FakeTreeCache()
    controller = SessionController(tree_cache)
    session = Session(capacity_of_str_len=4096, session_id="session-a", streaming=True)
    session._inflight = True
    session.close_on_finish = True
    controller.sessions[session.session_id] = session

    session.abort_req()
    controller.maybe_reap(now=2.0, interval=0.0)

    assert "session-a" not in controller.sessions
    assert tree_cache.released_sessions == ["session-a"]
