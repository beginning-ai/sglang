from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    RequestStage,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import get_global_server_args
from sglang.srt.tracing.trace import trace_slice, trace_slice_batch, trace_slice_end

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


def wav_to_pcm16(wav: torch.Tensor) -> bytes:
    """Convert float waveform tensor to PCM16 bytes.

    Args:
        wav: Float tensor in range [-1, 1], shape [batch, 1, samples] or [1, samples]

    Returns:
        PCM16 bytes (int16, little-endian)
    """
    # Squeeze to 1D
    wav = wav.squeeze()
    # Clamp and convert to int16
    wav = (wav.clamp(-1, 1) * 32767).to(torch.int16)
    return wav.cpu().numpy().tobytes()


def maybe_decode_code2wav_chunk(
    req: Req,
    code2wav,
    chunk_size: int = 10,
    left_context_size: int = 25,
) -> None:
    """Decode new codec frames to PCM16 if enough frames have accumulated."""
    if not req.talker_output_codes:
        return

    current_frame = len(req.talker_output_codes)
    new_frames = current_frame - req.talker_last_decoded_frame

    # Check if we have enough new frames to decode a chunk
    if new_frames < chunk_size:
        return

    # Check chunk cap to avoid unbounded memory
    if len(req.talker_pcm16_chunks) >= req.talker_pcm16_chunk_cap:
        return

    codes_list = req.talker_output_codes
    context_start = max(0, req.talker_last_decoded_frame - left_context_size)
    context_size = req.talker_last_decoded_frame - context_start

    codes_window = codes_list[context_start:current_frame]
    codes_tensor = torch.tensor(codes_window, dtype=torch.long, device=next(code2wav.parameters()).device)
    codes_tensor = codes_tensor.transpose(0, 1).unsqueeze(0)  # [1, num_quantizers, T]

    with torch.no_grad():
        wav_chunk = code2wav(codes_tensor)

    # Take last new_frames worth of samples to handle TransConv output length variation
    total_upsample = code2wav.total_upsample
    expected_new_samples = new_frames * total_upsample
    wav_new = wav_chunk[..., -expected_new_samples:]
    if wav_new.shape[-1] > 0:
        pcm16_bytes = wav_to_pcm16(wav_new)
        req.talker_pcm16_chunks.append(pcm16_bytes)

    # Update last decoded frame
    req.talker_last_decoded_frame = current_frame


def flush_code2wav_chunk(
    req: Req,
    code2wav,
    left_context_size: int = 25,
) -> None:
    """Decode any remaining frames when request finishes."""
    if not req.talker_output_codes:
        return

    current_frame = len(req.talker_output_codes)
    new_frames = current_frame - req.talker_last_decoded_frame

    if new_frames <= 0:
        return

    if len(req.talker_pcm16_chunks) >= req.talker_pcm16_chunk_cap:
        return

    codes_list = req.talker_output_codes
    context_start = max(0, req.talker_last_decoded_frame - left_context_size)

    codes_window = codes_list[context_start:current_frame]
    codes_tensor = torch.tensor(codes_window, dtype=torch.long, device=next(code2wav.parameters()).device)
    codes_tensor = codes_tensor.transpose(0, 1).unsqueeze(0)  # [1, num_quantizers, T]

    with torch.no_grad():
        wav_chunk = code2wav(codes_tensor)

    # Take last new_frames worth of samples, but skip final 20ms to remove end artifact
    total_upsample = code2wav.total_upsample
    expected_new_samples = new_frames * total_upsample
    end_artifact_samples = 480  # 20ms at 24kHz
    if wav_chunk.shape[-1] >= expected_new_samples + end_artifact_samples:
        wav_new = wav_chunk[..., -(expected_new_samples + end_artifact_samples):-end_artifact_samples]
    else:
        wav_new = wav_chunk[..., -expected_new_samples:]

    if wav_new.shape[-1] > 0:
        pcm16_bytes = wav_to_pcm16(wav_new)
        req.talker_pcm16_chunks.append(pcm16_bytes)

    req.talker_last_decoded_frame = current_frame


def save_pcm16_to_file(req: Req, code2wav=None) -> None:
    """Save all codec codes to a WAV file when request finishes.

    Decodes all accumulated codec codes using code2wav and writes to /tmp/<rid>.wav.
    """
    import wave

    # Check if we have codec codes to decode
    if not hasattr(req, "talker_output_codes") or not req.talker_output_codes:
        return

    # Get code2wav model - must be provided by caller
    if code2wav is None:
        return

    # Build codes tensor from all accumulated codes: [1, num_quantizers, frames]
    codes_list = req.talker_output_codes
    num_frames = len(codes_list)

    codes_tensor = torch.tensor(codes_list, dtype=torch.long, device=next(code2wav.parameters()).device)
    codes_tensor = codes_tensor.transpose(0, 1).unsqueeze(0)  # [1, num_quantizers, T]

    # Run code2wav (batch decode)
    with torch.no_grad():
        wav_output = code2wav(codes_tensor)

    # Convert to PCM16
    pcm16_bytes = wav_to_pcm16(wav_output)

    # Write WAV file
    output_path = f"/tmp/{req.rid}.wav"
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(24000)  # Qwen3-Omni uses 24kHz
        wav_file.writeframes(pcm16_bytes)

    print(f"[Code2Wav] Saved audio to {output_path} ({len(pcm16_bytes)} bytes, {num_frames} frames)")


def _update_qwen3_omni_state(
    req: Req,
    logits_output,
    req_idx: int,
    thinker_token_id: int = None,
    code2wav=None,
    chunk_size: int = 10,
    left_context_size: int = 25,
) -> None:
    """Update Qwen3-Omni talker state in Req after model forward.

    With talker-before-code-predictor design:
    - Model returns complete codec_frame (16 codes) instead of logits
    - No more sampling in scheduler, no more past_talker_hidden
    - Store prev_residual_codes for next decode step input

    With 1-step delay design:
    - At prefill: talker doesn't run, set talker_needs_prefill=True
    - At first decode: talker prefill runs, returns first complete frame
    - At subsequent decodes: talker uses prev_codec + prev_residual_codes as input
    """
    # Check if this is a ThinkerTalkerOutput
    if not hasattr(logits_output, "talker_needs_prefill"):
        return

    if logits_output.talker_needs_prefill:
        req.talker_needs_prefill = True

    if thinker_token_id is not None:
        req.prev_sampled_thinker_token = thinker_token_id

    # Handle codec frames
    frame = None
    if logits_output.codec_frames is not None and req_idx < len(logits_output.codec_frames):
        frame = logits_output.codec_frames[req_idx]

    if frame is not None:
        codec_token = frame[0]
        residual_codes = frame[1:]
        req.talker_codec_ids.append(codec_token)
        req.prev_residual_codes = residual_codes
        if codec_token != req.codec_eos_token_id:
            req.talker_output_codes.append(frame)

    if logits_output.tts_pad_embed is not None:
        if logits_output.tts_pad_embed.dim() == 2:
            req.tts_pad_embed = logits_output.tts_pad_embed[req_idx].detach()
        else:
            req.tts_pad_embed = logits_output.tts_pad_embed.detach()

    # Handle talker KV cache locations (prefill)
    if logits_output.talker_out_cache_loc_list is not None:
        if req_idx < len(logits_output.talker_out_cache_loc_list):
            kv_locs = logits_output.talker_out_cache_loc_list[req_idx]
            if kv_locs is not None:
                req.talker_kv_cache_locs = kv_locs
                req.talker_prefill_len = len(kv_locs)
                req.talker_needs_prefill = False

    # Handle talker KV cache locations (decode)
    if logits_output.updated_talker_kv_locs_list is not None:
        if req_idx < len(logits_output.updated_talker_kv_locs_list):
            kv_locs = logits_output.updated_talker_kv_locs_list[req_idx]
            if kv_locs is not None:
                req.talker_kv_cache_locs = kv_locs

    # Increment talker step when talker produces a frame (not during prefill)
    is_prefill = logits_output.talker_out_cache_loc_list is not None
    if frame is not None and not is_prefill:
        req.talker_step += 1

    # Trigger Code2Wav decode if model is available
    if code2wav is not None:
        maybe_decode_code2wav_chunk(req, code2wav, chunk_size, left_context_size)

    # NOTE: Do NOT clear pending state here!
    # With overlap scheduling, the same request can be in multiple batches.
    # Clearing here would remove state captured for Batch N while processing Batch N-1.
    # The pending state is naturally overwritten when new state is captured.


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _get_code2wav_model(self: "Scheduler"):
        """Get the code2wav model if available (for Qwen3-Omni)."""
        if getattr(self.model_config.hf_config, "model_type", None) != "qwen3_omni_moe":
            return None
        return self.tp_worker.model_runner.model.code2wav

    def process_batch_result_prebuilt(self: "Scheduler", batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        for req in batch.reqs:
            req.check_finished()
            if req.finished():
                req.time_stats.forward_entry_time = req.time_stats.completion_time = (
                    time.perf_counter()
                )
                trace_slice_end(
                    RequestStage.DECODE_QUICK_FINISH,
                    req.rid,
                    thread_finish_flag=True,
                )
                release_kv_cache(req, self.tree_cache)

        # Note: Logprobs should be handled on the prefill engine.
        trace_slice_batch(RequestStage.DECODE_FAKE_OUTPUT, batch.reqs)
        self.stream_output(batch.reqs, batch.return_logprob)

    def maybe_collect_routed_experts(self: Scheduler, req: Req):
        """Collect routed experts for a finished request."""
        req.routed_experts = get_global_experts_capturer().get_routed_experts(
            req_pool_idx=req.req_pool_idx,
            seqlen=req.seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def maybe_collect_customized_info(
        self: Scheduler, i: int, req: Req, logits_output: LogitsProcessorOutput
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                req.customized_info[k].append(v[i])

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            # Get code2wav model for Qwen3-Omni
            code2wav = self._get_code2wav_model()

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
                    continue

                # For streaming audio requests, don't generate output until audio_done
                # Use the snapshot from when prefill started to avoid race condition
                # where handle_streaming_audio_end sets audio_done=True during GPU forward
                prefill_audio_done = getattr(req, '_prefill_audio_done_snapshot', getattr(req, 'audio_done', True))
                if req.is_chunked <= 0 and prefill_audio_done:
                    if req.time_stats.prefill_finished_ts == 0.0:
                        req.time_stats.prefill_finished_ts = time.time()

                    # req output_ids are set here
                    # Skip dummy tokens when thinker is done (Qwen3-Omni talker-only mode)
                    if not req.thinker_done:
                        req.output_ids.append(next_token_id)

                    # Qwen3-Omni: Store talker state for next decode step
                    _update_qwen3_omni_state(req, logits_output, i, thinker_token_id=next_token_id, code2wav=code2wav)

                    req.check_finished()

                    if req.finished():
                        self.maybe_collect_routed_experts(req)
                        # Update req state from pending before freeing
                        # (pending state may have allocations/frames not yet stored by results processing)
                        pending = self.pending_talker_state.pop(req.rid, None)
                        if pending is not None:
                            if pending.get("kv_locs") is not None:
                                req.talker_kv_cache_locs = pending["kv_locs"]
                            # Store pending codec_frame if not already in talker_output_codes
                            pending_frame = pending.get("codec_frame")
                            if pending_frame is not None:
                                codec_token = pending_frame[0]
                                # Don't store EOS frames or duplicates
                                if codec_token != getattr(req, "codec_eos_token_id", None):
                                    # Check if already stored by results processing
                                    if not req.talker_output_codes or req.talker_output_codes[-1] != pending_frame:
                                        req.talker_output_codes.append(pending_frame)
                        # Skip radix cache insert for streaming audio to prevent cache pollution
                        is_insert = not getattr(req, 'disable_prefix_cache', False)
                        release_kv_cache(req, self.tree_cache, is_insert=is_insert)
                        req.time_stats.completion_time = time.perf_counter()
                        # Qwen3-Omni: Flush remaining audio frames for streaming
                        flush_code2wav_chunk(req, code2wav=code2wav)
                        # Qwen3-Omni: Save PCM16 audio to file
                        save_pcm16_to_file(req, code2wav=code2wav)
                        # Clean up streaming audio tracking
                        self.streaming_audio_reqs.pop(req.rid, None)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        # Skip for streaming audio to prevent cache pollution
                        if not getattr(req, 'disable_prefix_cache', False):
                            self.tree_cache.cache_unfinished_req(req)

                    self.maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]

                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )

                        if req.return_logprob:
                            self.add_logprob_return_values(
                                i,
                                req,
                                logprob_pt,
                                next_token_ids,
                                num_input_logprobs,
                                logits_output,
                            )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        # FIXME: this try-except block is for handling unexpected xgrammar issue.
                        try:
                            req.grammar.accept_token(next_token_id)
                        except ValueError as e:
                            # Grammar accept_token can raise ValueError if the token is not in the grammar.
                            # This can happen if the grammar is not set correctly or the token is invalid.
                            logger.error(
                                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                            )
                            self.abort_request(AbortReq(rid=req.rid))
                        req.grammar.finished = req.finished()

                    trace_slice(
                        RequestStage.PREFILL_FORWARD,
                        req.rid,
                        auto_next_anon=not req.finished(),
                        thread_finish_flag=req.finished(),
                    )

                elif req.is_chunked > 0:
                    # Regular chunked prefill - decrement counter
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = self._calculate_num_input_logprobs(
                                req, extend_input_len, extend_logprob_start_len
                            )
                            if req.return_logprob:
                                self.add_input_logprob_return_values(
                                    i,
                                    req,
                                    logits_output,
                                    logprob_pt,
                                    num_input_logprobs,
                                    last_prefill_chunk=False,
                                )
                            logprob_pt += num_input_logprobs

                    trace_slice(
                        RequestStage.PREFILL_CHUNKED_FORWARD,
                        req.rid,
                        auto_next_anon=True,
                    )
                else:
                    # Streaming audio intermediate prefill (audio_done=False, is_chunked<=0)
                    # Cache KV to radix tree and free req_pool_idx so next prefill
                    # can get a fresh allocation and reuse the cached KV

                    # Cache the KV we just computed - fill_ids was set by init_next_round_input
                    self.tree_cache.cache_unfinished_req(req, chunked=True)

                    # Free the req_pool_idx slot so next prefill allocates fresh
                    # The KV values at the cached cache_locs remain valid
                    self.req_to_token_pool.free(req.req_pool_idx)
                    req.req_pool_idx = None

                    # Don't stream output during intermediate prefills
                    skip_stream_req = req

        else:  # embedding or reward model
            if result.copy_done is not None:
                result.copy_done.synchronize()

            is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

            embeddings = result.embeddings

            if is_sparse:
                batch_ids, token_ids = embeddings.indices()
                values = embeddings.values()

                embeddings = [{} for _ in range(embeddings.size(0))]
                for i in range(batch_ids.shape[0]):
                    embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                        i
                    ].item()
            else:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                else:
                    embeddings = [tensor.tolist() for tensor in embeddings]

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

                trace_slice(
                    RequestStage.PREFILL_FORWARD,
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def _resolve_spec_overlap_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> List[List[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        result.accept_length_per_req_cpu = [x - 1 for x in accept_lens]

        predict_tokens = []
        stride = self.draft_worker.speculative_num_draft_tokens

        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_lens[i]
            predict_tokens.append(
                next_token_ids[i * stride : i * stride + accept_lens[i]]
            )
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_lens[i] - 1

        return predict_tokens

    def process_batch_result_idle(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        next_token_ids = result.next_token_ids.tolist()
        self.num_generated_tokens += len(next_token_ids)

        self.token_to_kv_pool_allocator.free_group_begin()

        assert len(batch.reqs) == 1, "batch size is currently expected to be 1"
        req = batch.reqs[0]

        for next_token_id in next_token_ids:
            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.finished():
                release_kv_cache(req, self.tree_cache)
                req.time_stats.completion_time = time.perf_counter()
                break

            self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        if batch.spec_algorithm.is_none():
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
        elif batch.is_spec_v2:
            next_token_ids = self._resolve_spec_overlap_token_ids(result, batch)

        self.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none():
            self.update_spec_metrics(batch.batch_size(), result.num_accepted_tokens)
        if self.enable_metrics:
            self.metrics_collector.increment_cuda_graph_pass(value=can_run_cuda_graph)

        self.token_to_kv_pool_allocator.free_group_begin()

        # NOTE: in any case, we should check finish here
        # if finished, also clean up committed kv cache and over-allocated kv cache here

        # Check finish condition
        # Get code2wav model for Qwen3-Omni
        code2wav = self._get_code2wav_model()

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req: Req

            if self.enable_overlap and (req.finished() or req.is_retracted):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # (currently not, e.g. Eagle V1 still check finish during forward)
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            new_accepted_len = 1
            # Skip dummy tokens when thinker is done (Qwen3-Omni talker-only mode)
            if not req.thinker_done:
                if batch.spec_algorithm.is_none():
                    req.output_ids.append(next_token_id)
                elif batch.is_spec_v2:
                    # Only spec v2's output_ids are updated here.
                    req.output_ids.extend(next_token_id)
                    new_accepted_len = len(next_token_id)

            # Update Mamba last track seqlen
            self._mamba_prefix_cache_update(req, batch, result, i)

            # Qwen3-Omni: Store talker state for next decode step
            _update_qwen3_omni_state(req, logits_output, i, thinker_token_id=next_token_id, code2wav=code2wav)

            req.check_finished(new_accepted_len)

            if req.finished():
                self.maybe_collect_routed_experts(req)

                # Update req state from pending before freeing
                # (pending state may have allocations/frames not yet stored by results processing)
                pending = self.pending_talker_state.pop(req.rid, None)
                if pending is not None:
                    if pending.get("kv_locs") is not None:
                        req.talker_kv_cache_locs = pending["kv_locs"]
                    # Store pending codec_frame if not already in talker_output_codes
                    pending_frame = pending.get("codec_frame")
                    if pending_frame is not None:
                        codec_token = pending_frame[0]
                        # Don't store EOS frames or duplicates
                        if codec_token != getattr(req, "codec_eos_token_id", None):
                            # Check if already stored by results processing
                            if not req.talker_output_codes or req.talker_output_codes[-1] != pending_frame:
                                req.talker_output_codes.append(pending_frame)

                # Skip radix cache insert for streaming audio to prevent cache pollution
                is_insert = not getattr(req, 'disable_prefix_cache', False)
                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                    if not self.decode_offload_manager.offload_kv_cache(req):
                        release_kv_cache(req, self.tree_cache, is_insert=is_insert)
                else:
                    release_kv_cache(req, self.tree_cache, is_insert=is_insert)

                req.time_stats.completion_time = time.perf_counter()
                # Qwen3-Omni: Flush remaining audio frames for streaming
                flush_code2wav_chunk(req, code2wav=code2wav)
                # Qwen3-Omni: Save PCM16 audio to file
                save_pcm16_to_file(req, code2wav=code2wav)

                # Clean up streaming audio tracking
                self.streaming_audio_reqs.pop(req.rid, None)

            self.maybe_collect_customized_info(i, req, logits_output)

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    if batch.spec_algorithm.is_none():
                        # Normal decode: single token
                        req.grammar.accept_token(next_token_id)
                    elif batch.is_spec_v2:
                        # Speculative decode: next_token_id is a list of accepted tokens
                        for token_id in next_token_id:
                            req.grammar.accept_token(token_id)
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                    )
                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.current_scheduler_metrics_enabled
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats(can_run_cuda_graph, running_batch=batch)
        if self.enable_metrics:
            self.log_decode_stats_every_iteration(
                batch, num_accepted_tokens=result.num_accepted_tokens
            )

    def _mamba_prefix_cache_update(
        self, req: Req, batch: ScheduleBatch, result: GenerationBatchResult, i: int
    ) -> None:
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        if req.mamba_ping_pong_track_buffer is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            if batch.spec_algorithm.is_none() and seq_len % mamba_track_interval == 0:
                # for non-spec decode, we update mamba_last_track_seqlen at the end of each track interval
                req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
                req.mamba_last_track_seqlen = seq_len
            elif (
                not batch.spec_algorithm.is_none()
                and result.accept_length_per_req_cpu is not None
            ):
                # for spec decode, update mamba_last_track_seqlen if this iteration crosses a track interval
                actual_seq_len = req.seqlen - 1
                if (
                    actual_seq_len // mamba_track_interval
                    != (actual_seq_len - result.accept_length_per_req_cpu[i])
                    // mamba_track_interval
                ):
                    req.mamba_last_track_seqlen = (
                        actual_seq_len // mamba_track_interval * mamba_track_interval
                    )

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # Multi-item scoring: only include delimiter token positions
            relevant_tokens = req.origin_input_ids[req.logprob_start_len :]
            input_token_logprobs_idx = [
                token_id
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            ]
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:
        """Process input top logprobs."""
        if req.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.input_top_logprobs_val.extend(val)
            req.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_top_logprobs_val.pop()
            req.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_token_ids_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_token_ids_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_token_ids_logprobs_val.pop()
            req.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)
        relevant_tokens = req.origin_input_ids[req.logprob_start_len :]

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens from logprob_start_len onwards
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens from logprob_start_len onwards
            return len(relevant_tokens)

    def _calculate_num_input_logprobs(
        self, req: Req, extend_input_len: int, extend_logprob_start_len: int
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens in the relevant portion
            relevant_tokens = req.origin_input_ids[
                extend_logprob_start_len:extend_input_len
            ]
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return req.is_prefill_only and self.server_args.multi_item_scoring_delimiter

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if (
            req.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.output_token_ids_logprobs_val.append(logprobs_val)
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.input_token_logprobs_val is None:
            req.input_token_logprobs_val = []
        if req.input_token_logprobs_idx is None:
            req.input_token_logprobs_idx = []
        if req.input_top_logprobs_val is None:
            req.input_top_logprobs_val = []
        if req.input_top_logprobs_idx is None:
            req.input_top_logprobs_idx = []
        if req.input_token_ids_logprobs_val is None:
            req.input_token_ids_logprobs_val = []
        if req.input_token_ids_logprobs_idx is None:
            req.input_token_ids_logprobs_idx = []

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        spec_accepted_tokens = []
        retraction_counts = []
        output_hidden_states = None
        load = self.get_load()
        output_routed_experts = None
        customized_info = {}

        # Streaming audio output (Qwen3-Omni)
        audio_chunks = []

        queue_times = []
        forward_entry_times = []
        prefill_launch_delays = []
        prefill_launch_latencies = []
        prefill_finished_timestamps = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_finish:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )

                    # origin stream_interval logic
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if not self.model_config.is_multimodal_gen
                        and stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )

                    if should_output:
                        # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                        should_output &= not req.check_match_stop_str_prefix()
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        if not self.model_config.is_multimodal_gen
                        else False
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)
                retraction_counts.append(req.retraction_count)

                # Extract audio chunks (Qwen3-Omni streaming audio output)
                if req.talker_pcm16_chunks:
                    audio_chunks.append(req.talker_pcm16_chunks.copy())
                    req.talker_pcm16_chunks.clear()
                else:
                    audio_chunks.append([])

                queue_times.append(req.time_stats.get_queueing_time())
                forward_entry_times.append(req.time_stats.forward_entry_time)

                prefill_launch_delays.append(req.time_stats.get_prefill_launch_delay())
                prefill_launch_latencies.append(
                    req.time_stats.get_prefill_launch_latency()
                )
                prefill_finished_timestamps.append(
                    req.time_stats.get_prefill_finished_ts()
                )

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if output_routed_experts is None:
                        output_routed_experts = []
                    output_routed_experts.append(req.routed_experts)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(v)

            if (
                req.finished()
                and self.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to detokenizer
        if reqs or is_idle_batch:
            if self.model_config.is_multimodal_gen:
                return

            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    queue_time=queue_times,
                    forward_entry_time=forward_entry_times,
                    prefill_launch_delay=prefill_launch_delays,
                    prefill_launch_latency=prefill_launch_latencies,
                    prefill_finished_ts=prefill_finished_timestamps,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    output_routed_experts=output_routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                    audio_chunks=audio_chunks if any(audio_chunks) else None,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        queue_times = []
        forward_entry_times = []
        prefill_launch_delays = []
        prefill_launch_latencies = []
        prefill_finished_timestamps = []
        retraction_counts = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                queue_times.append(req.time_stats.get_queueing_time())
                forward_entry_times.append(req.time_stats.forward_entry_time)

                prefill_launch_delays.append(req.time_stats.get_prefill_launch_delay())
                prefill_launch_latencies.append(
                    req.time_stats.get_prefill_launch_latency()
                )
                prefill_finished_timestamps.append(
                    req.time_stats.get_prefill_finished_ts()
                )
                retraction_counts.append(req.retraction_count)
        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                queue_time=queue_times,
                forward_entry_time=forward_entry_times,
                prefill_launch_delay=prefill_launch_delays,
                prefill_launch_latency=prefill_launch_latencies,
                prefill_finished_ts=prefill_finished_timestamps,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
            )
        )
