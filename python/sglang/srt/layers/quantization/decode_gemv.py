"""Decode-size GEMMs that read their weights once, near memory bandwidth.

With a handful of rows (one per speculative token) a GEMM is a read of the
whole weight, and library kernels tuned for large M leave most of the GPU idle
or spend more on launches than on bytes. Three cases, all Triton GEMVs:

* FP8 rows (``SGLANG_FP8_ROW_WEIGHTS=1``): decode-bound BF16 weights -- the
  output head (1.2 GiB for a 248k x 2560 head, read every step and for each
  draft step), hyper-connection mixes, and a speculative draft's linears --
  are stored as FP8 E4M3 with one FP32 scale per output row: half the bytes,
  activations stay BF16. The scale rides on the weight tensor as an attribute,
  so sharing the head with a draft, or gathering FR-Spec rows (``select_rows``),
  keeps values and scales together.
* 128x128 block-FP8 checkpoints: the weight is dequantized in registers, which
  skips the activation-quant kernel and keeps activations BF16.
* BF16 weights with few outputs (Gated DeltaNet's 96-wide b/a projection, MoE
  routers), where cuBLAS picks an 8-block kernel that takes ~10x longer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SCALE_ATTR = "fp8_row_scale"
_GEMV_MAX_ROWS = 64
_CHUNK_ROWS = 8192


def to_fp8_rows(weight: torch.Tensor) -> torch.nn.Parameter:
    """A BF16 [rows, cols] weight as FP8 E4M3 rows, each with one FP32 scale."""
    rows = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], _CHUNK_ROWS):
        block = weight.data[start : start + _CHUNK_ROWS].float()
        block_scale = block.abs().amax(dim=1).clamp_min(1e-12) / 448.0
        rows[start : start + _CHUNK_ROWS] = (block / block_scale[:, None]).to(rows.dtype)
        scale[start : start + _CHUNK_ROWS] = block_scale
    param = torch.nn.Parameter(rows, requires_grad=False)
    setattr(param, SCALE_ATTR, scale)
    return param


def dequantize_rows(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``weight`` in ``dtype``, applying its row scales if it has them."""
    scale = getattr(weight, SCALE_ATTR, None)
    if scale is None:
        return weight
    return (weight.float() * scale[:, None]).to(dtype)


def quantize_fp8_row_weights(model: torch.nn.Module, is_draft: bool) -> None:
    """Store the decode-bound BF16 weights as FP8 rows: the output head (unless
    tied to the input embedding; a draft gets the target's after loading) and
    hyper-connection mix projections. A speculative draft, whose quality only
    moves the accept rate, also gets its unquantized linear layers converted
    (except MoE routers, whose choices are discrete)."""
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    head = getattr(model, "lm_head", None)
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if (
        not is_draft
        and getattr(getattr(head, "weight", None), "dtype", None) == torch.bfloat16
        and head.weight is not getattr(embed, "weight", None)
    ):
        head.weight = to_fp8_rows(head.weight)
    for module in model.modules():
        for name in ("input_mix_weight_down", "input_mix_weight_up"):
            linear = getattr(module, name, None)
            if isinstance(linear, torch.nn.Linear) and linear.weight.dtype == torch.bfloat16:
                linear.weight = to_fp8_rows(linear.weight)
    if is_draft:
        for name, module in model.named_modules():
            if (
                isinstance(module, LinearBase)
                and isinstance(module.quant_method, UnquantizedLinearMethod)
                and module.weight.dtype == torch.bfloat16
                and module.weight.shape[0] >= 256
                and module.bias is None
                and not name.endswith(".gate")
            ):
                module.weight = to_fp8_rows(module.weight)
    # Hand the BF16 copies back, or the KV pool is sized without them.
    torch.cuda.empty_cache()


def select_rows(weight: torch.Tensor, rows: torch.Tensor) -> torch.nn.Parameter:
    """``weight[rows]`` as a new parameter, with the matching row scales."""
    selected = torch.nn.Parameter(weight.data[rows], requires_grad=False)
    scale = getattr(weight, SCALE_ATTR, None)
    if scale is not None:
        setattr(selected, SCALE_ATTR, scale[rows])
    return selected


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None] * K + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + cols[:, None] * K + k[None, :],
                    mask=(cols[:, None] < N) & (k[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    if HAS_SCALE:
        acc *= tl.load(scale_ptr + cols, mask=cols < N, other=0.0)[None, :]
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], out,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def fp8_row_linear(
    hidden_states: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``hidden_states @ (weight * scale).T`` for an FP8-row weight."""
    scale = getattr(weight, SCALE_ATTR)
    x = hidden_states.reshape(-1, weight.shape[1]).to(torch.bfloat16).contiguous()
    m, n = x.shape[0], weight.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m <= _GEMV_MAX_ROWS:
        # One pass over the weight; the few rows ride along in registers.
        _gemv_kernel[(triton.cdiv(n, 64),)](
            x, weight, scale, out, m, n, x.shape[1],
            BLOCK_M=max(16, triton.next_power_of_2(m)), BLOCK_N=64, BLOCK_K=128,
            HAS_SCALE=True, num_warps=4, num_stages=4,
        )
    else:
        # Prefill: dequantize a slice of rows at a time.
        for start in range(0, n, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, n)
            dense = (weight[start:stop].float() * scale[start:stop, None]).to(x.dtype)
            out[:, start:stop] = torch.mm(x, dense.T, out_dtype=out_dtype)
    return out.view(*hidden_states.shape[:-1], n)


def use_skinny_linear(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda and x.dim() == 2 and x.shape[0] <= 16 and weight.shape[0] <= 512
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous() and weight.is_contiguous()
        and not torch.compiler.is_compiling()
    )


def skinny_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _gemv_kernel[(triton.cdiv(weight.shape[0], 16),)](
        x, weight, None, out, x.shape[0], weight.shape[0], x.shape[1],
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=512, HAS_SCALE=False, num_warps=4,
    )
    return out


@triton.jit
def _block_fp8_gemv_kernel(
    x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, K_SPLIT,
    BLOCK_N: tl.constexpr, ATOMIC: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    k_start = tl.program_id(1) * K_SPLIT
    for k0 in range(k_start, k_start + K_SPLIT, 128):
        k = k0 + tl.arange(0, 128)
        x = tl.load(x_ptr + rows[:, None] * K + k[None, :], mask=rows[:, None] < M, other=0.0)
        w = tl.load(w_ptr + cols[:, None] * K + k[None, :])
        scale = tl.load(s_ptr + cols // 128 * (K // 128) + k0 // 128)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * scale[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptr + rows[:, None] * N + cols[None, :], acc, mask=rows[:, None] < M, sem="relaxed")
    else:
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), mask=rows[:, None] < M)


def use_block_fp8_gemv(x: torch.Tensor, weight: torch.Tensor, block_size) -> bool:
    return (
        x.shape[0] <= 16 and x.dtype == torch.bfloat16 and list(block_size) == [128, 128]
        and weight.shape[0] % 32 == 0 and weight.shape[1] % 128 == 0
        and not torch.compiler.is_compiling()
    )


def block_fp8_gemv(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """``x @ dequant(weight).T`` for ``weight`` [N, K] FP8 with [N/128, K/128] scales."""
    (m, k), n = x.shape, weight.shape[0]
    # Few output blocks leave SMs idle; split K across them and sum atomically.
    split = 3 if n <= 4096 and k % 384 == 0 else 1
    if split == 1:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    else:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    _block_fp8_gemv_kernel[(n // 32, split)](
        x, weight, weight_scale, out, m, n, k, k // split,
        BLOCK_N=32, ATOMIC=split > 1, num_warps=4, num_stages=4,
    )
    return out.to(x.dtype)
