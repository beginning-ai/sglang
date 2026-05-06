from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class DecodeMultistepMetadata:
    seq_len: int
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    seq_lens: Optional[torch.Tensor]


def prepare_decode_multistep_metadata(
    capture_buffers: dict,
    cache_key: tuple,
    forward_batch: ForwardBatch,
    num_steps: int,
    base_seq_len: int,
    *,
    buffer_batch_size: int,
    kv_indices_dtype: torch.dtype,
    include_seq_lens: bool,
) -> Optional[list[DecodeMultistepMetadata]]:
    assert num_steps > 0
    assert base_seq_len > 0

    if not forward_batch.forward_mode.is_decode_or_idle():
        return None
    if forward_batch.spec_info is not None:
        return None

    override = forward_batch.override_req_to_token
    if override is None:
        return None

    max_step_seq_len = base_seq_len + num_steps - 1
    if override.shape[1] < max_step_seq_len:
        return None

    bs = forward_batch.batch_size
    if bs > buffer_batch_size:
        return None

    device = override.device
    seq_lens_dtype = forward_batch.seq_lens.dtype if include_seq_lens else None
    full_cache_key = (
        cache_key,
        buffer_batch_size,
        num_steps,
        base_seq_len,
        kv_indices_dtype,
        seq_lens_dtype,
    )

    if full_cache_key not in capture_buffers:
        indptrs = []
        seq_lens_buf = (
            torch.empty(
                (num_steps, buffer_batch_size),
                dtype=seq_lens_dtype,
                device=device,
            )
            if include_seq_lens
            else None
        )
        for step in range(num_steps):
            cur_seq_len = base_seq_len + step
            indptrs.append(
                torch.arange(
                    0,
                    (buffer_batch_size + 1) * cur_seq_len,
                    cur_seq_len,
                    dtype=torch.int32,
                    device=device,
                )
            )
            if seq_lens_buf is not None:
                seq_lens_buf[step].fill_(cur_seq_len)

        kv_indices_buf = torch.zeros(
            num_steps,
            buffer_batch_size * max_step_seq_len,
            dtype=kv_indices_dtype,
            device=device,
        )
        capture_buffers[full_cache_key] = (
            indptrs,
            seq_lens_buf,
            kv_indices_buf,
        )

    indptrs, seq_lens_buf, kv_indices_buf = capture_buffers[full_cache_key]

    metadata = []
    for step in range(num_steps):
        cur_seq_len = base_seq_len + step
        kv_indices = kv_indices_buf[step]
        kv_indices[: bs * cur_seq_len] = (
            override[:bs, :cur_seq_len].to(kv_indices_dtype).reshape(-1)
        )
        metadata.append(
            DecodeMultistepMetadata(
                seq_len=cur_seq_len,
                kv_indptr=indptrs[step][: bs + 1],
                kv_indices=kv_indices,
                seq_lens=seq_lens_buf[step, :bs] if seq_lens_buf is not None else None,
            )
        )

    return metadata
