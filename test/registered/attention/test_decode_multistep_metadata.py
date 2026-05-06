from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.multistep_metadata import (
    prepare_decode_multistep_metadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _batch(*, override_req_to_token):
    return SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.DECODE,
        override_req_to_token=override_req_to_token,
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        spec_info=None,
    )


def test_prepare_decode_multistep_metadata_materializes_steps():
    override = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=torch.int64,
    )
    capture_buffers = {}

    metadata = prepare_decode_multistep_metadata(
        capture_buffers,
        ("test",),
        _batch(override_req_to_token=override),
        num_steps=3,
        base_seq_len=3,
        buffer_batch_size=2,
        kv_indices_dtype=torch.int64,
        include_seq_lens=True,
    )

    assert metadata is not None
    assert [step.seq_len for step in metadata] == [3, 4, 5]
    assert metadata[0].kv_indptr.tolist() == [0, 3, 6]
    assert metadata[1].kv_indptr.tolist() == [0, 4, 8]
    assert metadata[2].kv_indptr.tolist() == [0, 5, 10]
    assert metadata[0].kv_indices[:6].tolist() == [10, 11, 12, 20, 21, 22]
    assert metadata[1].kv_indices[:8].tolist() == [
        10,
        11,
        12,
        13,
        20,
        21,
        22,
        23,
    ]
    assert metadata[2].kv_indices[:10].tolist() == [
        10,
        11,
        12,
        13,
        14,
        20,
        21,
        22,
        23,
        24,
    ]
    assert metadata[2].seq_lens.tolist() == [5, 5]


def test_prepare_decode_multistep_metadata_rejects_short_override():
    override = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=torch.int64,
    )

    metadata = prepare_decode_multistep_metadata(
        {},
        ("test",),
        _batch(override_req_to_token=override),
        num_steps=2,
        base_seq_len=3,
        buffer_batch_size=2,
        kv_indices_dtype=torch.int32,
        include_seq_lens=False,
    )

    assert metadata is None
