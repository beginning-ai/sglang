"""Small-batch NVFP4 MoE for decode (``SGLANG_NVFP4_DECODE_MOE=1``).

At decode sizes FlashInfer's CUTLASS MoE spends ~10 kernels per layer on routing
bookkeeping (permute, prefix sums, strides, finalize), and its grouped GEMMs
run far below memory bandwidth. With a handful of tokens the work is one GEMV
per (token, expert) pair, so two Triton kernels do it on the CUTLASS weight
layout, nothing copied:

  fc1: one program per (pair, row block) -> silu(gate) * up in FP32
  fc2: one program per (pair, row block), atomically summed per token

Weights are packed E2M1 (two per byte, low nibble first) with E4M3 scales per
16 values in CUTLASS's 128x4 interleaved layout and a per-expert global scale.
Activations stay BF16 rather than being quantized to FP4.
"""

import torch
import triton
import triton.language as tl

MAX_TOKENS = 16


@triton.jit
def _dot_rows(x_ptr, w_ptr, s_ptr, rows, K, s_cols, BK: tl.constexpr, STAGES: tl.constexpr):
    """sum_k x[k] * W[rows, k] / 2^14 (callers fold 2^14 into their scale)."""
    BN: tl.constexpr = rows.shape[0]
    half = tl.arange(0, BK // 2)
    blk = tl.arange(0, BK // 16)
    # Scale (r, c) of the interleaved layout; see swizzle_blockscale.
    srow = (rows // 128 * (s_cols // 4) * 32 + rows % 32) * 16 + rows % 128 // 32 * 4
    acc = tl.zeros([BN, BK // 2], dtype=tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        xe = tl.load(x_ptr + k0 + 2 * half).to(tl.float32)
        xo = tl.load(x_ptr + k0 + 2 * half + 1).to(tl.float32)
        b = tl.load(w_ptr + rows[:, None] * (K // 2) + k0 // 2 + half[None, :])
        c = k0 // 16 + blk
        s = tl.load(s_ptr + srow[:, None] + (c // 4 * 512 + c % 4)[None, :]).to(tl.float32)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], [BN, BK // 16, 8]), [BN, BK // 2])
        # E2M1 -> FP16 bits: sign to bit 15, exponent and mantissa to bits 11..9,
        # which reads as the value times 2^-14.
        lo = (((b & 7).to(tl.uint16) << 9) | ((b & 8).to(tl.uint16) << 12)).to(tl.float16, bitcast=True)
        hi = (((b >> 4 & 7).to(tl.uint16) << 9) | ((b >> 4 & 8).to(tl.uint16) << 12)).to(tl.float16, bitcast=True)
        acc += (lo.to(tl.float32) * xe[None, :] + hi.to(tl.float32) * xo[None, :]) * s
    return tl.sum(acc, axis=1)


@triton.jit
def _global_scale(alpha_ptr, inv_in_ptr, e, inv_in_stride):
    # alpha = input_scale * weight_scale_2, so weight_scale_2 = alpha / input_scale.
    return tl.load(alpha_ptr + e) * tl.load(inv_in_ptr + e * inv_in_stride) * 16384.0


@triton.jit
def _fc1_kernel(x_ptr, ids_ptr, w_ptr, s_ptr, s_cols, alpha_ptr, inv_in_ptr, inv_in_stride,
                h_ptr, TOPK, K, I, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr):
    p = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    e = tl.load(ids_ptr + p).to(tl.int64)
    x = x_ptr + (p // TOPK) * K
    w = w_ptr + e * (2 * I) * (K // 2)
    s = s_ptr + e * tl.cdiv(2 * I, 128) * 128 * s_cols
    g = _global_scale(alpha_ptr, inv_in_ptr, e, inv_in_stride)
    up = _dot_rows(x, w, s, rows, K, s_cols, BK, STAGES) * g        # rows [0, I)
    gate = _dot_rows(x, w, s, I + rows, K, s_cols, BK, STAGES) * g  # rows [I, 2I)
    tl.store(h_ptr + p * I + rows, gate / (1 + tl.exp(-gate)) * up)


@triton.jit
def _fc2_kernel(h_ptr, ids_ptr, wt_ptr, w_ptr, s_ptr, s_cols, alpha_ptr, inv_in_ptr, inv_in_stride,
                out_ptr, TOPK, I, N, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr):
    p = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    e = tl.load(ids_ptr + p).to(tl.int64)
    w = w_ptr + e * N * (I // 2)
    s = s_ptr + e * tl.cdiv(N, 128) * 128 * s_cols
    g = _global_scale(alpha_ptr, inv_in_ptr, e, inv_in_stride) * tl.load(wt_ptr + p)
    y = _dot_rows(h_ptr + p * I, w, s, rows, I, s_cols, BK, STAGES) * g
    tl.atomic_add(out_ptr + (p // TOPK) * N + rows, y, sem="relaxed")


def nvfp4_decode_moe(x, topk_ids, topk_weights, w13, w2, quant_scales, out):
    """``out[t] = sum_k w[t, k] * W2_e(silu(gate_e(x_t)) * up_e(x_t))`` for the
    FlashInfer CUTLASS NVFP4 payload (``w13`` up rows first)."""
    inv_in13, s13, alpha13, inv_in2, s2, alpha2 = quant_scales
    T, K = x.shape
    topk = topk_ids.shape[1]
    I, N = w13.shape[1] // 2, w2.shape[1]
    ids = topk_ids.reshape(-1)
    h = torch.empty((T * topk, I), dtype=torch.float32, device=x.device)
    _fc1_kernel[(T * topk, I // 8)](
        x, ids, w13, s13, s13.shape[-1], alpha13, inv_in13, int(inv_in13.numel() > 1),
        h, topk, K, I, BN=8, BK=512, STAGES=3, num_warps=2,
    )
    acc = torch.zeros((T, N), dtype=torch.float32, device=x.device)
    _fc2_kernel[(T * topk, N // 64)](
        h, ids, topk_weights.reshape(-1).float(), w2, s2, s2.shape[-1], alpha2, inv_in2,
        int(inv_in2.numel() > 1), acc, topk, I, N, BN=64, BK=128, STAGES=2, num_warps=4,
    )
    out.copy_(acc)
    return out
