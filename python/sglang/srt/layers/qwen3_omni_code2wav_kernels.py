from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _snakebeta_kernel(
    x,
    alpha_,
    beta_,
    y,
    n_elements,
    channels: tl.constexpr,
    seq_len: tl.constexpr,
    no_div_by_zero: tl.constexpr,
    block_size: tl.constexpr,
):
    # n_elements is runtime to avoid per-bs JIT specialization; bench-neutral.
    offs = tl.program_id(0) * block_size + tl.arange(0, block_size)
    channel = (offs // seq_len) % channels
    mask = offs < n_elements
    x_vals = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    alpha_vals = tl.exp(tl.load(alpha_ + channel, mask=mask, other=0.0).to(tl.float32))
    beta_vals = tl.exp(tl.load(beta_ + channel, mask=mask, other=0.0).to(tl.float32))
    sin_vals = tl.sin(x_vals * alpha_vals)
    y_vals = x_vals + (sin_vals * sin_vals) / (beta_vals + no_div_by_zero)
    tl.store(y + offs, y_vals, mask=mask)


def snakebeta(
    hidden: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    no_div_by_zero: float,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"expected [batch, channels, seq] hidden, got {hidden.shape}")
    if not hidden.is_contiguous():
        raise ValueError("Qwen3-Omni Code2Wav Triton SnakeBeta requires contiguous NCL")
    batch_size, channels, seq_len = hidden.shape
    if alpha.ndim != 1 or alpha.shape[0] != channels:
        raise ValueError(f"expected [{channels}] alpha, got {alpha.shape}")
    if beta.ndim != 1 or beta.shape[0] != channels:
        raise ValueError(f"expected [{channels}] beta, got {beta.shape}")
    if (
        hidden.device.type != "cuda"
        or alpha.device.type != "cuda"
        or beta.device.type != "cuda"
    ):
        raise ValueError(
            "Qwen3-Omni Code2Wav Triton SnakeBeta requires CUDA/HIP tensors"
        )
    if hidden.dtype != torch.bfloat16:
        raise ValueError(f"unsupported hidden dtype for Triton SnakeBeta: {hidden.dtype}")

    output = torch.empty_like(hidden)
    n_elements = batch_size * channels * seq_len
    block_size = 256
    _snakebeta_kernel[(triton.cdiv(n_elements, block_size),)](
        hidden,
        alpha,
        beta,
        output,
        n_elements,
        channels,
        seq_len,
        no_div_by_zero,
        block_size,
        num_warps=4,
    )
    return output


@triton.jit
def _dense_causal_conv1d_kernel7(
    x,
    w,
    bias_,
    y,
    batch_size,
    channels: tl.constexpr,
    seq_len: tl.constexpr,
    dilation_: tl.constexpr,
    block_m: tl.constexpr,
    block_o: tl.constexpr,
    block_k: tl.constexpr,
):
    # batch_size is runtime to avoid per-bs JIT specialization; bench-neutral.
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_o = pid_o * block_o + tl.arange(0, block_o)
    batch = offs_m // seq_len
    time_idx = offs_m - batch * seq_len
    bias_vals = tl.load(bias_ + offs_o, mask=offs_o < channels, other=0.0)
    acc = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]

    for k0 in range(0, channels * 7, block_k):
        offs_k = k0 + tl.arange(0, block_k)
        in_channel = offs_k // 7
        kernel_idx = offs_k - in_channel * 7
        source_time = time_idx[:, None] - (6 - kernel_idx[None, :]) * dilation_
        x_ptrs = (
            x
            + (batch[:, None] * channels + in_channel[None, :]) * seq_len
            + source_time
        )
        x_mask = (
            (offs_m[:, None] < batch_size * seq_len)
            & (offs_k[None, :] < channels * 7)
            & (source_time >= 0)
        )
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = (
            w
            + (offs_o[None, :] * channels + in_channel[:, None]) * 7
            + kernel_idx[:, None]
        )
        w_mask = (offs_o[None, :] < channels) & (offs_k[:, None] < channels * 7)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_vals, w_vals, input_precision="ieee")

    y_ptrs = (
        y
        + (batch[:, None] * channels + offs_o[None, :]) * seq_len
        + time_idx[:, None]
    )
    y_mask = (offs_m[:, None] < batch_size * seq_len) & (offs_o[None, :] < channels)
    tl.store(y_ptrs, acc, mask=y_mask)


def dense_causal_conv1d_kernel7(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dilation: int,
) -> torch.Tensor:

    if hidden.ndim != 3:
        raise ValueError(f"expected [batch, channels, seq] hidden, got {hidden.shape}")
    if weight.ndim != 3 or weight.shape[2] != 7:
        raise ValueError(f"expected [channels, channels, 7] weight, got {weight.shape}")
    batch_size, channels, seq_len = hidden.shape
    if weight.shape[0] != channels or weight.shape[1] != channels:
        raise ValueError(
            f"expected square dense weight for {channels} channels, got {weight.shape}"
        )
    if bias.ndim != 1 or bias.shape[0] != channels:
        raise ValueError(f"expected [{channels}] bias, got {bias.shape}")
    if (
        hidden.device.type != "cuda"
        or weight.device.type != "cuda"
        or bias.device.type != "cuda"
    ):
        raise ValueError("Qwen3-Omni Code2Wav Triton conv requires CUDA/HIP tensors")
    if hidden.dtype != torch.bfloat16:
        raise ValueError(
            f"Qwen3-Omni Code2Wav Triton conv requires bf16 input, got {hidden.dtype}"
        )

    output = torch.empty_like(hidden)
    # MI300X-oriented tiles measured on the actual Code2Wav replay shapes.
    # Keep this explicit so the custom-kernel loop tunes by shape instead of
    # hiding choices behind autotune noise.
    if channels == 96:
        block_m = 256
        block_o = 128
    elif channels == 192:
        block_m = 1024 if batch_size in (4, 5) else 512
        block_o = 64
    else:
        raise ValueError(f"unsupported Code2Wav dense conv channels: {channels}")
    block_k = 32
    grid = (triton.cdiv(batch_size * seq_len, block_m), triton.cdiv(channels, block_o))
    _dense_causal_conv1d_kernel7[grid](
        hidden,
        weight,
        bias,
        output,
        batch_size,
        channels,
        seq_len,
        dilation,
        block_m,
        block_o,
        block_k,
        num_warps=8,
    )
    return output

@triton.jit
def _causal_transconv1d_phase_inner_kernel_s3(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    batch_size,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_len: tl.constexpr,
    out_len: tl.constexpr,
    block_m: tl.constexpr,
    block_o: tl.constexpr,
    block_k: tl.constexpr,
):
    # stride=3 specialization. Matches the rejected 2-tap kernel's math:
    #   y[cropped_t] = bias + sum_in_ch (
    #       w[in_ch, out_ch, phase]          * x[in_ch, in_t + 1] +
    #       w[in_ch, out_ch, phase + stride] * x[in_ch, in_t]
    #   )
    # where cropped_t = in_t * 3 + phase and in_t = cropped_t // 3.
    # Materially different from the rejected probe: phase is an inner static
    # loop, so each x_left/x_right pair is loaded once and reused across all
    # three phase outputs, instead of being reloaded by stride independent
    # CTAs (one per phase). Three independent fp32 accumulators (one per
    # phase) live in registers; weight tiles are loaded once per phase.
    #
    # batch_size is a RUNTIME int — it varies across the split-decoder
    # (1/2/4/5/6/7/8) and we don't want a fresh Triton JIT per bs. Without
    # this, the first call at each new bs took 500-1000 ms (JIT compile),
    # which starved the scheduler's audio-frame-lag budget across the first
    # minute of the soak and crashed at ~7 min with "frame-producing audio
    # phase reached frame lag budget without pending Code2Wav work". The
    # other shape ints (in_channels=192, out_channels=96, in_len=18376) are
    # fixed by the shape-exact gate in the caller, so keeping them as
    # constexpr enables Triton specialization without JIT pressure.
    stride_: tl.constexpr = 3
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_o = pid_o * block_o + tl.arange(0, block_o)
    phase_len: tl.constexpr = in_len - 1
    batch = offs_m // phase_len
    in_t_left = offs_m - batch * phase_len
    in_t_right = in_t_left + 1

    m_mask = offs_m < batch_size * phase_len
    o_mask = offs_o < out_channels

    bias_vals = tl.load(bias_ptr + offs_o, mask=o_mask, other=0.0)

    acc_p0 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]
    acc_p1 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]
    acc_p2 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]

    for k0 in range(0, in_channels, block_k):
        offs_k = k0 + tl.arange(0, block_k)
        k_mask = offs_k < in_channels

        x_left_ptrs = (
            x_ptr
            + (batch[:, None] * in_channels + offs_k[None, :]) * in_len
            + in_t_left[:, None]
        )
        x_right_ptrs = x_left_ptrs + 1
        xy_mask = m_mask[:, None] & k_mask[None, :]
        x_left = tl.load(x_left_ptrs, mask=xy_mask, other=0.0)
        x_right = tl.load(
            x_right_ptrs,
            mask=xy_mask & (in_t_right[:, None] < in_len),
            other=0.0,
        )

        w_base = w_ptr + (offs_k[:, None] * out_channels + offs_o[None, :]) * (2 * stride_)
        w_mask = k_mask[:, None] & o_mask[None, :]

        w_p0_r = tl.load(w_base + 0, mask=w_mask, other=0.0)
        w_p0_l = tl.load(w_base + stride_, mask=w_mask, other=0.0)
        acc_p0 += tl.dot(x_right, w_p0_r, input_precision="ieee")
        acc_p0 += tl.dot(x_left, w_p0_l, input_precision="ieee")

        w_p1_r = tl.load(w_base + 1, mask=w_mask, other=0.0)
        w_p1_l = tl.load(w_base + (stride_ + 1), mask=w_mask, other=0.0)
        acc_p1 += tl.dot(x_right, w_p1_r, input_precision="ieee")
        acc_p1 += tl.dot(x_left, w_p1_l, input_precision="ieee")

        w_p2_r = tl.load(w_base + 2, mask=w_mask, other=0.0)
        w_p2_l = tl.load(w_base + (stride_ + 2), mask=w_mask, other=0.0)
        acc_p2 += tl.dot(x_right, w_p2_r, input_precision="ieee")
        acc_p2 += tl.dot(x_left, w_p2_l, input_precision="ieee")

    out_t_p0 = in_t_left * stride_ + 0
    out_t_p1 = in_t_left * stride_ + 1
    out_t_p2 = in_t_left * stride_ + 2
    y_base = y_ptr + (batch[:, None] * out_channels + offs_o[None, :]) * out_len
    y_mask = m_mask[:, None] & o_mask[None, :]
    tl.store(y_base + out_t_p0[:, None], acc_p0, mask=y_mask)
    tl.store(y_base + out_t_p1[:, None], acc_p1, mask=y_mask)
    tl.store(y_base + out_t_p2[:, None], acc_p2, mask=y_mask)


@triton.jit
def _causal_transconv1d_phase_inner_kernel_s4(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    batch_size,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_len: tl.constexpr,
    out_len: tl.constexpr,
    block_m: tl.constexpr,
    block_o: tl.constexpr,
    block_k: tl.constexpr,
):
    # Stride=4 phase-inner. Same algorithm as stride=3 but with 4 phase
    # accumulators (decoder.3.block.1: in_channels=384, out_channels=192,
    # in_len=4595). See stride=3 kernel for the math + design rationale.
    stride_: tl.constexpr = 4
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_o = pid_o * block_o + tl.arange(0, block_o)
    phase_len: tl.constexpr = in_len - 1
    batch = offs_m // phase_len
    in_t_left = offs_m - batch * phase_len
    in_t_right = in_t_left + 1

    m_mask = offs_m < batch_size * phase_len
    o_mask = offs_o < out_channels

    bias_vals = tl.load(bias_ptr + offs_o, mask=o_mask, other=0.0)

    acc_p0 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]
    acc_p1 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]
    acc_p2 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]
    acc_p3 = tl.zeros((block_m, block_o), tl.float32) + bias_vals[None, :]

    for k0 in range(0, in_channels, block_k):
        offs_k = k0 + tl.arange(0, block_k)
        k_mask = offs_k < in_channels

        x_left_ptrs = (
            x_ptr
            + (batch[:, None] * in_channels + offs_k[None, :]) * in_len
            + in_t_left[:, None]
        )
        x_right_ptrs = x_left_ptrs + 1
        xy_mask = m_mask[:, None] & k_mask[None, :]
        x_left = tl.load(x_left_ptrs, mask=xy_mask, other=0.0)
        x_right = tl.load(
            x_right_ptrs,
            mask=xy_mask & (in_t_right[:, None] < in_len),
            other=0.0,
        )

        w_base = w_ptr + (offs_k[:, None] * out_channels + offs_o[None, :]) * (2 * stride_)
        w_mask = k_mask[:, None] & o_mask[None, :]

        w_p0_r = tl.load(w_base + 0, mask=w_mask, other=0.0)
        w_p0_l = tl.load(w_base + stride_, mask=w_mask, other=0.0)
        acc_p0 += tl.dot(x_right, w_p0_r, input_precision="ieee")
        acc_p0 += tl.dot(x_left, w_p0_l, input_precision="ieee")

        w_p1_r = tl.load(w_base + 1, mask=w_mask, other=0.0)
        w_p1_l = tl.load(w_base + (stride_ + 1), mask=w_mask, other=0.0)
        acc_p1 += tl.dot(x_right, w_p1_r, input_precision="ieee")
        acc_p1 += tl.dot(x_left, w_p1_l, input_precision="ieee")

        w_p2_r = tl.load(w_base + 2, mask=w_mask, other=0.0)
        w_p2_l = tl.load(w_base + (stride_ + 2), mask=w_mask, other=0.0)
        acc_p2 += tl.dot(x_right, w_p2_r, input_precision="ieee")
        acc_p2 += tl.dot(x_left, w_p2_l, input_precision="ieee")

        w_p3_r = tl.load(w_base + 3, mask=w_mask, other=0.0)
        w_p3_l = tl.load(w_base + (stride_ + 3), mask=w_mask, other=0.0)
        acc_p3 += tl.dot(x_right, w_p3_r, input_precision="ieee")
        acc_p3 += tl.dot(x_left, w_p3_l, input_precision="ieee")

    out_t_p0 = in_t_left * stride_ + 0
    out_t_p1 = in_t_left * stride_ + 1
    out_t_p2 = in_t_left * stride_ + 2
    out_t_p3 = in_t_left * stride_ + 3
    y_base = y_ptr + (batch[:, None] * out_channels + offs_o[None, :]) * out_len
    y_mask = m_mask[:, None] & o_mask[None, :]
    tl.store(y_base + out_t_p0[:, None], acc_p0, mask=y_mask)
    tl.store(y_base + out_t_p1[:, None], acc_p1, mask=y_mask)
    tl.store(y_base + out_t_p2[:, None], acc_p2, mask=y_mask)
    tl.store(y_base + out_t_p3[:, None], acc_p3, mask=y_mask)


_PHASE_INNER_TILES = {
    # (stride, in_channels, out_channels): (block_m, block_o, block_k, num_warps, num_stages)
    (3, 192, 96): (128, 128, 32, 8, 3),
    (4, 384, 192): (128, 64, 32, 8, 3),
}


def causal_transconv1d_phase_inner(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"expected [batch, in_channels, seq] hidden, got {hidden.shape}")
    batch_size, in_channels, in_len = hidden.shape
    if weight.ndim != 3 or weight.shape[0] != in_channels:
        raise ValueError(
            f"expected [{in_channels}, out_channels, 2*stride] weight, got {weight.shape}"
        )
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    if kernel_size != 2 * stride:
        raise ValueError(
            f"expected kernel_size=2*stride={2 * stride}, got {kernel_size}"
        )
    if bias.ndim != 1 or bias.shape[0] != out_channels:
        raise ValueError(f"expected [{out_channels}] bias, got {bias.shape}")
    if (
        hidden.device.type != "cuda"
        or weight.device.type != "cuda"
        or bias.device.type != "cuda"
    ):
        raise ValueError(
            "Qwen3-Omni Code2Wav Triton transconv requires CUDA/HIP tensors"
        )
    if hidden.dtype != torch.bfloat16:
        raise ValueError(
            f"unsupported hidden dtype for Triton transconv: {hidden.dtype}"
        )
    kernels = {
        3: _causal_transconv1d_phase_inner_kernel_s3,
        4: _causal_transconv1d_phase_inner_kernel_s4,
    }
    if stride not in kernels:
        raise ValueError(
            f"Triton transconv phase-inner kernel specialized for stride in {sorted(kernels)}, got {stride}"
        )
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Qwen3-Omni Code2Wav Triton transconv requires contiguous inputs")

    out_len = (in_len - 1) * stride
    output = torch.empty(
        (batch_size, out_channels, out_len), device=hidden.device, dtype=hidden.dtype
    )

    tile_key = (stride, in_channels, out_channels)
    if tile_key not in _PHASE_INNER_TILES:
        raise ValueError(
            f"no MI300X-tuned tile for {tile_key}; supported tiles: "
            f"{sorted(_PHASE_INNER_TILES.keys())}"
        )
    block_m, block_o, block_k, num_warps, num_stages = _PHASE_INNER_TILES[tile_key]
    grid = (
        triton.cdiv(batch_size * (in_len - 1), block_m),
        triton.cdiv(out_channels, block_o),
    )
    kernels[stride][grid](
        hidden,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        in_len,
        out_len,
        block_m,
        block_o,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


@triton.jit
def _rmsnorm_bf16_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    hidden,
    eps,
    BLOCK_H: tl.constexpr,
):
    # bf16-native RMSNorm specialized for Code2Wav pre_transformer (hidden=1024).
    # Eliminates the fp32 cast + back round-trip the eager nn.Module path does,
    # at ~13-29% faster than eager on MI300X (`/tmp/bench_code2wav_rmsnorm.py`).
    # One CTA per row; row width is a power-of-2 covering `hidden`.
    row = tl.program_id(0)
    if row >= n_rows:
        return
    x_row = x_ptr + row * hidden
    out_row = out_ptr + row * hidden
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden
    x = tl.load(x_row + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    sq = xf * xf
    mean_sq = tl.sum(sq, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xf * rstd * w).to(tl.bfloat16)
    tl.store(out_row + offs, y, mask=mask)


def code2wav_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton bf16-native RMSNorm for Code2Wav pre_transformer.

    Replaces the eager `(x.float().pow(2).mean(-1, keepdim=True).rsqrt() * x).bf16()`
    path with a single fused kernel. Bench-confirmed 0.71-0.87x eager at
    Code2Wav pre_transformer shapes (`hidden=1024`, bs∈{1..30}, seq∈{4..29}).
    """
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError(
            f"Triton Code2Wav RMSNorm requires CUDA/HIP bf16 input, got "
            f"device={x.device.type} dtype={x.dtype}"
        )
    *lead, hidden = x.shape
    n_rows = 1
    for d in lead:
        n_rows *= int(d)
    x_flat = x.contiguous().view(n_rows, hidden)
    out = torch.empty_like(x_flat)
    block_h = triton.next_power_of_2(hidden)
    _rmsnorm_bf16_kernel[(n_rows,)](
        x_flat, weight, out, n_rows, hidden, eps, block_h,
    )
    return out.view(*lead, hidden)
