import asyncio

from sglang.srt.grpc.audio_request_manager import AudioStreamRequestManager
from sglang.srt.managers.io_struct import (
    AbortReq,
    AudioSessionRebuildReqOutput,
    BatchTokenIDOutput,
)


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        texts = {
            (): "",
            (1,): "Hello",
            (1, 2): "Hello world",
            (10,): "A",
            (10, 11): "AB",
            (10, 11, 12): "A revised",
            (11,): "B",
            (11, 12): "B revised",
        }
        return texts[tuple(token_ids)]


def test_abort_request_enqueues_terminal_audio_error():
    asyncio.run(_test_abort_request_enqueues_terminal_audio_error())


async def _test_abort_request_enqueues_terminal_audio_error():
    manager = AudioStreamRequestManager()
    state = manager.register("turn-a")

    assert await manager.abort_locally("turn-a")

    output = await asyncio.wait_for(state.out_queue.get(), timeout=0.1)
    assert output == {
        "request_id": "turn-a",
        "error": "Request aborted",
        "finished": True,
        "meta_info": {
            "id": "turn-a",
            "finish_reason": {
                "type": "abort",
                "message": "Request aborted",
            },
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinker_token_count": 0,
            "talker_token_count": 0,
        },
    }
    assert state.finished
    assert state.event.is_set()


def test_scheduler_abort_routes_to_audio_state():
    asyncio.run(_test_scheduler_abort_routes_to_audio_state())


async def _test_scheduler_abort_routes_to_audio_state():
    manager = AudioStreamRequestManager()
    state = manager.register("turn-b")

    await manager.route_scheduler_output(AbortReq(rid="turn-b"))

    output = await asyncio.wait_for(state.out_queue.get(), timeout=0.1)
    assert output["error"] == "Request aborted"
    assert output["finished"]
    assert state.finished


def test_audio_manager_decodes_text_from_token_ids():
    asyncio.run(_test_audio_manager_decodes_text_from_token_ids())


async def _test_audio_manager_decodes_text_from_token_ids():
    manager = AudioStreamRequestManager()
    manager.tokenizer = FakeTokenizer()
    state = manager.register("turn-c")

    await manager.route_scheduler_output(
        BatchTokenIDOutput(
            rids=["turn-c"],
            http_worker_ipcs=[None],
            finished_reasons=[None],
            decoded_texts=[""],
            decode_ids=[[1]],
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
    await manager.route_scheduler_output(
        BatchTokenIDOutput(
            rids=["turn-c"],
            http_worker_ipcs=[None],
            finished_reasons=[{"type": "stop"}],
            decoded_texts=[""],
            decode_ids=[[2]],
            read_offsets=[0],
            output_ids=[[2]],
            skip_special_tokens=[True],
            spaces_between_special_tokens=[True],
            no_stop_trim=[False],
            prompt_tokens=[4],
            reasoning_tokens=[0],
            completion_tokens=[2],
            talker_tokens=[32],
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

    first = await asyncio.wait_for(state.out_queue.get(), timeout=0.1)
    second = await asyncio.wait_for(state.out_queue.get(), timeout=0.1)
    assert first["decoded_text"] == "Hello"
    assert second["decoded_text"] == " world"
    assert second["meta_info"]["thinker_token_count"] == 2
    assert second["meta_info"]["talker_token_count"] == 32
    assert state.cumulative_text == "Hello world"


def test_audio_manager_uses_incremental_detokenization_for_revised_suffixes():
    asyncio.run(_test_audio_manager_uses_incremental_detokenization_for_revised_suffixes())


async def _test_audio_manager_uses_incremental_detokenization_for_revised_suffixes():
    manager = AudioStreamRequestManager()
    manager.tokenizer = FakeTokenizer()
    state = manager.register("turn-d")

    await manager.route_scheduler_output(
        BatchTokenIDOutput(
            rids=["turn-d"],
            http_worker_ipcs=[None],
            finished_reasons=[None],
            decoded_texts=[""],
            decode_ids=[[10, 11]],
            read_offsets=[1],
            output_ids=[[11]],
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
    await manager.route_scheduler_output(
        BatchTokenIDOutput(
            rids=["turn-d"],
            http_worker_ipcs=[None],
            finished_reasons=[{"type": "stop"}],
            decoded_texts=[""],
            decode_ids=[[12]],
            read_offsets=[1],
            output_ids=[[12]],
            skip_special_tokens=[True],
            spaces_between_special_tokens=[True],
            no_stop_trim=[False],
            prompt_tokens=[4],
            reasoning_tokens=[0],
            completion_tokens=[2],
            talker_tokens=[32],
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

    first = await asyncio.wait_for(state.out_queue.get(), timeout=0.1)
    second = await asyncio.wait_for(state.out_queue.get(), timeout=0.1)
    assert first["decoded_text"] == "B"
    assert second["decoded_text"] == " revised"
    assert state.cumulative_text == "B revised"


def test_audio_manager_routes_rebuild_response_future():
    asyncio.run(_test_audio_manager_routes_rebuild_response_future())


async def _test_audio_manager_routes_rebuild_response_future():
    manager = AudioStreamRequestManager()
    future = manager.expect_response("rebuild-a")

    await manager.route_scheduler_output(
        AudioSessionRebuildReqOutput(
            rid="rebuild-a",
            session_id="session-a",
            latest_turn_rid="safe-rid",
            prefix_turn_count=1,
            prompt_tokens=123,
        )
    )

    output = await asyncio.wait_for(future, timeout=0.1)
    assert output.latest_turn_rid == "safe-rid"
    assert output.prefix_turn_count == 1
    assert output.prompt_tokens == 123
