"""Row-level streaming for very large n-gram tables held on local storage.

Qwen3.8-Flash-Next's PLE table is 320M rows x 160 B = 47.7 GiB. A forward pass
touches only a handful of rows per token, so the table never needs to be
resident -- but on a discrete GPU a kernel cannot dereference pageable host
memory, and on a small-RAM host a fully pinned table does not fit.

What such a GPU *can* read is pinned host memory, over UVA. So each forward's
rows are read from storage into a small pinned ring, and the consuming gather
kernel gets a pointer into that ring plus row ids remapped to ring slots. The
gather kernel itself is unchanged. A captured CUDA graph cannot read ids back
to the host, so before each replay the model reads the step's rows here and the
graph gathers from the staged slots.

Rows are scattered across the table. Rows already in the page cache are copied
with a non-blocking read; the rest are queued as readahead hints first, so the
kernel fetches them in parallel instead of one storage round trip per row.
"""

from __future__ import annotations

import json
import logging
import os
import struct

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Tensor-name marker of the PLE n-gram table's shards in a Qwen4-Exp checkpoint.
PLE_SHARD_MARKER = "ngram_embedding.shard_"


def safetensors_segments(
    index_path: str, shard_marker: str, row_bytes: int, rows: int
) -> list[tuple[int, str, int]]:
    """``(first_row, path, byte_offset)`` per shard of a table stored as
    ``<...><shard_marker>N.weight`` tensors, in numeric shard order."""
    root = os.path.dirname(index_path)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = []
    for name, filename in weight_map.items():
        head, _, tail = name.rpartition(shard_marker)
        rank = tail.removesuffix(".weight")
        if head and tail.endswith(".weight") and rank.isdigit():
            shards.append((int(rank), name, os.path.join(root, filename)))
    if not shards:
        raise ValueError(f"no {shard_marker}* tensors in {index_path}")

    segments, first_row = [], 0
    for _, name, path in sorted(shards):
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            begin, end = json.loads(f.read(header_len))[name]["data_offsets"]
        if (end - begin) % row_bytes:
            raise ValueError(f"{name} is not a whole number of {row_bytes} B rows")
        segments.append((first_row, path, 8 + header_len + begin))
        first_row += (end - begin) // row_bytes
    if first_row != rows:
        raise ValueError(f"shards hold {first_row} rows, expected {rows}")
    return segments


class NgramRowStore:
    """Serves table rows from storage through a pinned ring.

    ``acquire(row_ids)`` reads the rows into the ring and returns their slots,
    which callers gather from in place of the global row ids.
    """

    def __init__(
        self,
        segments: list[tuple[int, str, int]],
        rows: int,
        ring_rows: int,
        dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.rows = rows
        self.capacity = ring_rows
        # Explicit devices: the PLE layer is built under a meta device context,
        # where meta tensors can be neither pinned nor moved.
        self.table = torch.empty(
            (ring_rows, dim), dtype=dtype, device="cpu", pin_memory=True
        )
        self._ring = self.table.view(torch.uint8).numpy()
        self.row_bytes = self._ring.shape[1]
        fds = {path: os.open(path, os.O_RDONLY) for _, path, _ in segments}
        self._first_row = np.array([s[0] for s in segments], dtype=np.int64)
        self._fd = np.array([fds[s[1]] for s in segments], dtype=np.int64)
        self._offset = np.array([s[2] for s in segments], dtype=np.int64)
        # Non-blocking buffered reads need kernel and filesystem support
        # (overlayfs, NFS or FUSE may refuse them); without it every row takes
        # the readahead-then-read path.
        self._nowait = True
        try:
            os.preadv(self._fd[0].item(), [self._ring[0]], 0, os.RWF_NOWAIT)
        except BlockingIOError:
            pass
        except OSError:
            self._nowait = False
        # Slots land in preallocated buffers: the decode graph reads them, so
        # no allocation may happen there, and the H2D source must be pinned.
        self._ids = torch.empty(
            ring_rows, dtype=torch.long, device=torch.cuda.current_device()
        )
        self._ids_host = torch.empty(
            ring_rows, dtype=torch.long, device="cpu", pin_memory=True
        )

    def acquire(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Read ``row_ids`` into the ring; return the slots holding them."""
        rows, slots = np.unique(row_ids.reshape(-1).cpu().numpy(), return_inverse=True)
        n = len(rows)
        if n > self.capacity:
            raise RuntimeError(
                f"n-gram row ring holds {self.capacity} rows but this step needs "
                f"{n}; lower --chunked-prefill-size"
            )
        if n and not 0 <= rows[0] <= rows[-1] < self.rows:
            raise IndexError(f"row ids outside [0, {self.rows})")
        seg = np.searchsorted(self._first_row, rows, side="right") - 1
        offsets = (self._offset[seg] + (rows - self._first_row[seg]) * self.row_bytes).tolist()
        fds = self._fd[seg].tolist()
        ring, row_bytes, preadv = self._ring, self.row_bytes, os.preadv

        misses = range(n)
        if self._nowait:
            misses = []
            for slot, (fd, off) in enumerate(zip(fds, offsets)):
                try:
                    if preadv(fd, [ring[slot]], off, os.RWF_NOWAIT) == row_bytes:
                        continue
                except BlockingIOError:
                    pass
                misses.append(slot)
        for slot in misses:
            os.posix_fadvise(fds[slot], offsets[slot], row_bytes, os.POSIX_FADV_WILLNEED)
        for slot in misses:
            if preadv(fds[slot], [ring[slot]], offsets[slot]) != row_bytes:
                raise OSError(f"short read of n-gram row {rows[slot]}")

        self._ids_host[: len(slots)].copy_(torch.from_numpy(slots))
        ids = self._ids[: len(slots)]
        ids.copy_(self._ids_host[: len(slots)], non_blocking=True)
        return ids.view(row_ids.shape)

    def capture_slots(self, shape: torch.Size) -> torch.Tensor:
        """The slot buffer, for use inside a CUDA graph capture."""
        return self._ids[: shape[0]].view(shape)


def open_ple_row_store(
    model_dir: str, rows: int, dim: int, dtype: torch.dtype, rows_per_token: int
) -> NgramRowStore:
    """The row store for a Qwen4-Exp checkpoint's PLE table, read in place.

    The ring covers any single forward: rows-per-token times the larger of
    SGLang's prefill ceiling and a full speculative decode batch.
    """
    from sglang.srt.runtime_context import (
        get_schedule,
        get_spec,
        max_prefill_buffer_tokens,
    )

    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise FileNotFoundError(
            f"the nvme PLE backend reads the table in place, but {index} does not "
            "exist; pass a local checkpoint directory as --model-path or "
            "--ple-offload-dir"
        )
    tokens = max(
        max_prefill_buffer_tokens(),
        (get_schedule().max_running_requests or 0)
        * (1 + (get_spec().speculative_num_draft_tokens or 0)),
    )
    row_bytes = dim * dtype.itemsize
    segments = safetensors_segments(index, PLE_SHARD_MARKER, row_bytes, rows)
    logger.info(
        "PLE table: streaming %d rows of %d B from %s (%d shards); pinned ring %.1f MiB",
        rows,
        row_bytes,
        model_dir,
        len(segments),
        rows_per_token * tokens * row_bytes / 2**20,
    )
    return NgramRowStore(segments, rows, rows_per_token * tokens, dim, dtype)
