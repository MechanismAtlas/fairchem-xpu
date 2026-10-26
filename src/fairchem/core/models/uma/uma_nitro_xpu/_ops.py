"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _gemm(
    a,
    w,
    out,
    bias,
    scale,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SO: tl.constexpr,
    BIAS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
    FP32_A: tl.constexpr,
):
    # Group adjacent row tiles to share weight blocks through L2.
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    first = (pid // (4 * nn)) * 4
    group = tl.minimum(nm - first, 4)
    im = first + (pid % (4 * nn)) % group
    jn = (pid % (4 * nn)) // group
    ad = tl.make_tensor_descriptor(
        a, shape=(M, K), strides=(SA, 1), block_shape=(BM, BK)
    )
    wd = tl.make_tensor_descriptor(
        w, shape=(N, K), strides=(SW, 1), block_shape=(BN, BK)
    )
    total = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        av = ad.load([im * BM, k * BK])
        wv = wd.load([jn * BN, k * BK])
        if FP32_A:
            high = av.to(tl.float16)
            # Keep the rounded half explicit before forming the residual; the
            # first fused prototype loses its correction in native execution.
            bits = high.to(tl.uint16, bitcast=True).to(tl.uint32)
            sign = (bits & 0x8000) << 16
            exponent = (bits >> 10) & 31
            mantissa = bits & 1023
            normal_bits = sign | ((exponent + 112) << 23) | (mantissa << 13)
            full_bits = tl.where(
                exponent == 31, sign | 0x7F800000 | (mantissa << 13), normal_bits
            )
            normal = full_bits.to(tl.float32, bitcast=True)
            small = (
                mantissa.to(tl.float32) * (2.0**-24) * tl.where(sign == 0, 1.0, -1.0)
            )
            rounded = tl.where(exponent == 0, small, normal)
            # Exact power-of-two scaling protects residuals from half underflow.
            low = ((av - rounded) * 1024.0).to(tl.float16)
            total = tl.dot(high, wv.T, total, input_precision=PRECISION)
            correction = tl.dot(low, wv.T, input_precision=PRECISION)
            total += correction * (1.0 / 1024.0)
        else:
            total = tl.dot(av, wv.T, total, input_precision=PRECISION)
    rows = im * BM + tl.arange(0, BM)
    cols = jn * BN + tl.arange(0, BN)
    if SCALE:
        total *= tl.load(scale + rows, mask=rows < M, other=0)[:, None]
    if BIAS:
        total += tl.load(bias + cols, mask=cols < N, other=0)[None, :]
    cd = tl.make_tensor_descriptor(
        out, shape=(M, N), strides=(SO, 1), block_shape=(BM, BN)
    )
    cd.store([im * BM, jn * BN], total.to(out.dtype.element_ty))


@triton_op("uma_nitro_xpu::gemm", mutates_args={"out"})
def gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    rscale: torch.Tensor | None = None,
) -> None:
    assert w.dtype == torch.float16
    assert a.dtype in (torch.float16, torch.float32)
    assert a.dtype != torch.float32 or out.dtype == torch.float32
    assert out.dtype in (torch.float16, torch.float32)
    assert a.stride(1) == w.stride(1) == out.stride(1) == 1
    assert a.shape[1] == w.shape[1]
    assert out.shape == (a.shape[0], w.shape[0])
    if bias is not None:
        assert bias.shape == (w.shape[0],)
        assert bias.is_contiguous()
    if rscale is not None:
        assert rscale.shape == (a.shape[0],)
        assert rscale.is_contiguous()
    bm, bn, bk, warps, stages = 128, 128, 32, 16, 2
    wrap_triton(_gemm)[(triton.cdiv(a.shape[0], bm) * triton.cdiv(w.shape[0], bn),)](
        a,
        w,
        out,
        bias if bias is not None else out,
        rscale if rscale is not None else out,
        M=a.shape[0],
        N=w.shape[0],
        K=a.shape[1],
        SA=a.stride(0),
        SW=w.stride(0),
        SO=out.stride(0),
        BIAS=bias is not None,
        SCALE=rscale is not None,
        BM=bm,
        BN=bn,
        BK=bk,
        PRECISION="ieee",
        FP32_A=a.dtype == torch.float32,
        num_warps=warps,
        num_stages=stages,
        grf_mode="256",
        enable_fp_fusion=False,
    )


@triton.jit
def _scaled_scatter(
    src,
    index,
    scale,
    out,
    elements,
    ROW: tl.constexpr,
    B: tl.constexpr,
    SKIP_ZERO: tl.constexpr,
):
    flat = tl.program_id(0) * B + tl.arange(0, B)
    valid = flat < elements
    edge = flat // ROW
    column = flat % ROW
    node = tl.load(index + edge, mask=valid, other=0).to(tl.int64)
    factor = tl.load(scale + edge, mask=valid, other=0)
    value = tl.load(src + flat, mask=valid, other=0).to(tl.float32) * factor
    if SKIP_ZERO:
        # Padded edges contribute exactly zero, frequently to the same node.
        # Avoid contended no-op atomics without dropping any nonzero gradient.
        valid &= value != 0
    # No thread consumes partial sums. Kernel/graph dependencies order the
    # completed output before its consumers, so relaxed atomic order suffices.
    tl.atomic_add(out + node * ROW + column, value, mask=valid, sem="relaxed")


@triton_op("uma_nitro_xpu::scaled_scatter", mutates_args={"out"})
def scatter_scaled_fp16(
    src: torch.Tensor, index: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> None:
    assert src.is_contiguous()
    assert index.is_contiguous()
    assert scale.is_contiguous()
    assert out.is_contiguous()
    assert src.shape[1:] == out.shape[1:]
    block, warps = 256, 4
    wrap_triton(_scaled_scatter)[(triton.cdiv(src.numel(), block),)](
        src,
        index,
        scale,
        out,
        src.numel(),
        ROW=src.shape[1] * src.shape[2],
        B=block,
        SKIP_ZERO=True,
        num_warps=warps,
    )


@triton.jit
def _reduce_partials(
    partials, out, ROW: tl.constexpr, PARTS: tl.constexpr, B: tl.constexpr
):
    node = tl.program_id(0)
    column = tl.program_id(1) * B + tl.arange(0, B)
    part = tl.arange(0, PARTS)
    value = tl.load(
        partials + (node * PARTS + part[:, None]) * ROW + column[None, :],
        mask=column[None, :] < ROW,
        other=0,
    )
    tl.store(out + node * ROW + column, tl.sum(value, axis=0), mask=column < ROW)


@triton_op("uma_nitro_xpu::reduce_partials", mutates_args={"out"})
def reduce_partials(partials: torch.Tensor, out: torch.Tensor) -> None:
    assert partials.dtype == out.dtype == torch.float32
    assert partials.is_contiguous()
    assert out.is_contiguous()
    assert partials.shape[0] == out.shape[0]
    assert partials.shape[2:] == out.shape[1:]
    row = out[0].numel()
    wrap_triton(_reduce_partials)[(out.shape[0], triton.cdiv(row, 128))](
        partials, out, ROW=row, PARTS=partials.shape[1], B=128, num_warps=4
    )


@triton.jit
def _sort_buckets(flat, capacity: tl.constexpr, buckets, WIDTH: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, WIDTH)
    mask = (rows[:, None] < buckets) & (cols[None, :] < capacity)
    values = tl.load(flat + rows[:, None] * capacity + cols[None, :], mask, -1)
    sentinel = tl.full((), 0x7FFFFFFFFFFFFFFF, tl.int64)
    values = tl.sort(tl.where(values >= 0, values, sentinel), dim=1)
    tl.store(
        flat + rows[:, None] * capacity + cols[None, :],
        tl.where(values == sentinel, -1, values),
        mask,
    )


def canonicalize_buckets(flat, capacity, buckets):
    wrap_triton(_sort_buckets)[(triton.cdiv(buckets, 4),)](
        flat, capacity, buckets, WIDTH=triton.next_power_of_2(capacity), num_warps=4
    )


@triton.jit
def _radial_head_bias(
    h,
    table,
    numbers,
    edges,
    stride,
    M: tl.constexpr,
    N: tl.constexpr,
    Z: tl.constexpr,
    B: tl.constexpr,
):
    index = tl.program_id(0) * B + tl.arange(0, B)
    valid = index < M * N
    row, column = index // N, index % N
    source = tl.load(edges + row, mask=valid, other=0)
    target = tl.load(edges + stride + row, mask=valid, other=0)
    source_type = tl.load(numbers + source)
    target_type = tl.load(numbers + target)
    value = tl.load(h + index, mask=valid, other=0)
    value += tl.load(
        table + (source_type * Z + target_type) * N + column, mask=valid, other=0
    )
    tl.store(h + index, value, mask=valid)


@triton_op("uma_nitro_xpu::radial_head_bias", mutates_args={"out"})
def radial_head_bias(
    out: torch.Tensor, table: torch.Tensor, numbers: torch.Tensor, edges: torch.Tensor
) -> None:
    """Add the frozen ordered-type head contribution to the dynamic Gaussian projection."""
    assert table.is_contiguous()
    assert out.is_contiguous()
    assert table.shape[2] == out.shape[1]
    assert edges.shape[1] == out.shape[0]
    wrap_triton(_radial_head_bias)[(triton.cdiv(out.numel(), 1024),)](
        out,
        table,
        numbers,
        edges,
        edges.stride(0),
        M=out.shape[0],
        N=out.shape[1],
        Z=table.shape[1],
        B=1024,
        num_warps=4,
    )
