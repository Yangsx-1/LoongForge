# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""A-card (SM80) fused-DSA integration tests.

Validates the kernel-backend seam end to end at the autograd-function level:

* the indexer forward/backward agree between the ``sm80`` BF16 kernels and the
  ``torch_ref`` reference, including the ChunkPipe cases (``chunk_offset > 0`` and
  ``s_kv > s_q``);
* the sparse-MLA autograd function runs and produces finite gradients on SM80.

Run with::

    PYTHONPATH=third_party/Loong-Megatron:. python -m pytest tests/test_dsa_fused_sm80.py -q
"""

import os

import pytest
import torch

import loongforge.train  # noqa: F401 - breaks a circular import in loongforge.models

from loongforge.models.common.experimental_attention_variant import dsa_kernel_backend as bk
from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
    DSADotProductAttentionFunction,
    DSAIndexerKernelFunction,
)

NUM_HEADS = 32
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capability():
    return torch.cuda.get_device_capability()


def _use_backend(name):
    os.environ["LOONGFORGE_DSA_KERNEL_BACKEND"] = name
    bk.reset_dsa_backend()
    assert bk.get_dsa_backend().name == name


def _indexer_inputs(s_q, s_kv, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(s_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, NUM_HEADS, device="cuda", dtype=torch.bfloat16, generator=g).float()
    return q, k, w


def _run_indexer(backend, q, k, w, topk, chunk_offset):
    _use_backend(backend)
    qi = q.detach().clone().requires_grad_(True)
    ki = k.detach().clone().requires_grad_(True)
    wi = w.detach().clone().requires_grad_(True)
    scores, idx = DSAIndexerKernelFunction.apply(qi, ki, wi, topk, chunk_offset, None)
    finite = torch.isfinite(scores)
    # -inf entries must be dropped with `where`, not multiplied by a 0/1 mask:
    # -inf * 0 is NaN and would poison every gradient.
    torch.where(finite, scores, torch.zeros_like(scores)).sum().backward()
    return scores, idx, finite, qi.grad, ki.grad, wi.grad


def _rel(a, b):
    a, b = a.detach().float(), b.detach().float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
@pytest.mark.parametrize("chunk_offset", [0, 256, 768])
def test_indexer_sm80_matches_reference(chunk_offset):
    """sm80 BF16 kernels vs the PyTorch reference, ChunkPipe offsets included."""
    s_q, topk = 256, 64
    s_kv = chunk_offset + s_q
    q, k, w = _indexer_inputs(s_q, s_kv, seed=chunk_offset)

    ref_scores, ref_idx, ref_finite, *ref_grads = _run_indexer(
        "torch_ref", q, k, w, topk, chunk_offset)
    got_scores, got_idx, got_finite, *got_grads = _run_indexer(
        "sm80", q, k, w, topk, chunk_offset)

    # Early query tokens have fewer than `topk` unmasked keys, so the -inf slots
    # are filled in an arbitrary order that legitimately differs between
    # backends.  Only the finite selections are meaningful, and `top_k` is called
    # with sorted=False so even those come back in an unspecified order -- compare
    # the selected index *sets* per row.
    assert torch.equal(ref_finite, got_finite), "number of valid top-k slots differs"
    ref_sel = torch.where(ref_finite, ref_idx, torch.full_like(ref_idx, -1)).sort(-1).values
    got_sel = torch.where(got_finite, got_idx, torch.full_like(got_idx, -1)).sort(-1).values
    assert torch.equal(ref_sel, got_sel), "top-k index sets differ"
    assert _rel(got_scores[got_finite].sort().values,
                ref_scores[ref_finite].sort().values) < 1e-2

    for name, r, g in zip(("d_q", "d_k", "d_weights"), ref_grads, got_grads):
        assert g is not None, f"{name} gradient missing"
        assert torch.isfinite(g).all(), f"{name} has non-finite entries"
        assert _rel(g, r) < 1e-2, f"{name} relative error too large: {_rel(g, r):.3e}"



@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
def test_indexer_no_fp8_quantizer_on_sm80():
    """The TE FP8 blockwise quantizer needs cc >= 8.9; SM80 must not touch it."""
    _use_backend("sm80")
    assert not bk.get_dsa_backend().indexer_is_fp8
    assert DSAIndexerKernelFunction._quantizer is None
    q, k, w = _indexer_inputs(64, 64)
    DSAIndexerKernelFunction.apply(q, k, w, 32, 0, None)
    assert DSAIndexerKernelFunction._quantizer is None


@pytest.mark.parametrize("chunk_offset", [0, 64])
def test_sparse_mla_runs_on_sm80(chunk_offset):
    """Sparse MLA fwd+bwd must run and produce finite gradients on SM80.

    Shapes follow the GLM-5 fused path: h_q is padded to 128, d_qk = 576
    (kv_lora_rank 512 + qk_pos_emb_head_dim 64), d_v = 512, topk a multiple of 64.
    """
    _use_backend("torch_ref")
    q, kv, idx, s_q, topk, h_q, d_v = _mla_inputs(chunk_offset)
    out, p_out = DSADotProductAttentionFunction.apply(
        q, kv, idx, chunk_offset, 576**-0.5, d_v, True, None, None, None, 0, False
    )
    assert out.shape == (1, s_q, h_q, d_v)
    assert p_out.shape == (s_q, h_q, topk)
    assert torch.isfinite(out).all()
    out.float().square().sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()


def _mla_inputs(chunk_offset, s_q=32, topk=64, h_q=128, d_qk=576, d_v=512, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s_kv = chunk_offset + s_q
    q = (torch.randn(s_q, h_q, d_qk, device="cuda", dtype=torch.bfloat16, generator=g) * 0.25)
    kv = (torch.randn(s_kv, 1, d_qk, device="cuda", dtype=torch.bfloat16, generator=g) * 0.25)
    q.requires_grad_(True)
    kv.requires_grad_(True)
    limit = (torch.arange(s_q, device="cuda") + chunk_offset).clamp_max(s_kv - 1)
    rnd = torch.rand(s_q, 1, topk, device="cuda", generator=g)
    idx = (rnd * (limit + 1).view(s_q, 1, 1).float()).floor().int()
    return q, kv, idx, s_q, topk, h_q, d_v


@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
@pytest.mark.parametrize("chunk_offset", [0, 128])
def test_sparse_mla_sm80_matches_reference(chunk_offset):
    """The sm80 sparse MLA forward must agree with the reference through LoongForge."""
    q, kv, idx, s_q, topk, h_q, d_v = _mla_inputs(chunk_offset, seed=chunk_offset + 2)

    _use_backend("torch_ref")
    ref_out, ref_p = DSADotProductAttentionFunction.apply(
        q.detach(), kv.detach(), idx, chunk_offset, 576**-0.5, d_v, True, None, None, None, 0,
        False)
    _use_backend("sm80")
    got_out, got_p = DSADotProductAttentionFunction.apply(
        q.detach(), kv.detach(), idx, chunk_offset, 576**-0.5, d_v, True, None, None, None, 0,
        False)

    finite = torch.isfinite(ref_p)
    assert torch.equal(finite, torch.isfinite(got_p)), "p_out mask mismatch"
    # p_out is the unscaled qk and is computed in fp32, so it should be near exact;
    # out is looser because P is rounded to bf16 before the PV GEMM, as on SM100.
    assert _rel(got_p[finite], ref_p[finite]) < 1e-5
    assert _rel(got_out, ref_out) < 2e-2


@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
def test_sparse_mla_sm80_backward_matches_reference():
    """The sm80 sparse MLA backward must agree with the reference through LoongForge.

    The pre-existing TileLang `major != 10` fallback is not on this path: on A100 it
    requests 216 KB of dynamic shared memory against a 164 KB limit and cannot run.
    """
    q, kv, idx, s_q, topk, h_q, d_v = _mla_inputs(64)
    args = (idx, 64, 576**-0.5, d_v, True, None, None, None, 0, False)

    grads = {}
    for backend in ("torch_ref", "sm80"):
        _use_backend(backend)
        qd = q.detach().clone().requires_grad_(True)
        kd = kv.detach().clone().requires_grad_(True)
        out, _ = DSADotProductAttentionFunction.apply(qd, kd, *args)
        out.float().square().sum().backward()
        assert torch.isfinite(qd.grad).all() and torch.isfinite(kd.grad).all()
        grads[backend] = (qd.grad, kd.grad)

    ref_dq, ref_dkv = grads["torch_ref"]
    got_dq, got_dkv = grads["sm80"]
    assert ref_dq.abs().max() > 0 and ref_dkv.abs().max() > 0
    assert _rel(got_dq, ref_dq) < 5e-2
    assert _rel(got_dkv, ref_dkv) < 5e-2


@pytest.mark.parametrize(
    "num_heads, backend, expected",
    [(32, "sm80", 32), (40, "sm80", 48), (64, "sm80", 64), (8, "sm80", 16), (128, "sm80", 128),
     (32, "sm100", 128), (64, "torch_ref", 128), (160, "sm100", 160)],
)
def test_sparse_mla_padded_heads(num_heads, backend, expected):
    """SM80 pads to a multiple of 16; FlashMLA backends keep the 128-head minimum."""
    from loongforge.models.common.experimental_attention_variant.dsa_fused import (
        sparse_mla_padded_heads,
    )

    assert sparse_mla_padded_heads(num_heads, backend) == expected


@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
@pytest.mark.parametrize("num_heads", [32, 64])
def test_sparse_mla_sm80_head_padding_is_exact(num_heads):
    """Padding to 16 instead of 128 heads must not change the real heads.

    Zero query heads are independent of the real ones, so out / p_out / grad_q on the
    real heads and grad_kv must match the 128-head launch (grad_out is zero on padded
    heads, as in DSAttentionFused, which slices the output back to the real heads).
    """
    q, kv, idx, s_q, topk, _, d_v = _mla_inputs(64, s_q=64, h_q=num_heads, seed=num_heads)
    args = (idx, 64, 576**-0.5, d_v, True, None, None, None, 0, False)
    _use_backend("sm80")

    results = {}
    for pad_to in (128, num_heads):
        qd = q.detach().clone().requires_grad_(True)
        kd = kv.detach().clone().requires_grad_(True)
        qp = torch.nn.functional.pad(qd, [0, 0, 0, pad_to - num_heads]) if pad_to > num_heads else qd
        out, p_out = DSADotProductAttentionFunction.apply(qp, kd, *args)
        out, p_out = out[:, :, :num_heads], p_out[:, :num_heads]
        out.float().square().sum().backward()
        results[pad_to] = (out.detach(), p_out.detach(), qd.grad, kd.grad)

    for name, a, b in zip(("out", "p_out", "grad_q", "grad_kv"), results[128], results[num_heads]):
        if name == "p_out":
            finite = torch.isfinite(a)
            assert torch.equal(finite, torch.isfinite(b)), "p_out mask mismatch"
            a, b = a[finite], b[finite]
        assert _rel(b, a) < 1e-3, name


def _indexer_window(s_q, chunk_offset):
    ks = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    ke = torch.arange(s_q, dtype=torch.int32, device="cuda") + chunk_offset + 1
    return ks, ke


def _masked_rel(a, b):
    a, b = a.float(), b.float()
    finite = torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), finite), "masked positions differ"
    return ((a[finite] - b[finite]).abs().max() / b[finite].abs().max().clamp_min(1e-6)).item()


@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
@pytest.mark.parametrize("s_q, chunk_offset, topk", [(256, 0, 128), (512, 1024, 256)])
def test_indexer_sm80_64_heads_matches_reference(monkeypatch, s_q, chunk_offset, topk):
    """A 64-head indexer on the 32-head SM80 kernel (DeepSeek-V3.2) vs the reference.

    The forward sums two 32-head calls block by block over query rows; the block size is
    shrunk so several row blocks run.  The backward splits grad_q / grad_w by head group
    and sums grad_k.  Thresholds follow the lightning_indexer_sm80 kernel tests.
    """
    monkeypatch.setattr(bk, "_SM80_INDEXER_BLOCK_ELEMS", 1 << 16)
    heads, s_kv = 2 * bk._SM80_INDEXER_HEADS, s_q + chunk_offset
    g = torch.Generator(device="cuda").manual_seed(s_q + chunk_offset)
    q = torch.randn(s_q, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, heads, device="cuda", dtype=torch.float32, generator=g)
    ks, ke = _indexer_window(s_q, chunk_offset)
    backend = bk.DSAKernelBackend("sm80")

    logits = backend.mqa_logits(q, k, w, ks, ke)
    ref = bk._ref_bf16_mqa_logits(q, k, w, ks, ke)
    assert logits.shape == (s_q, s_kv)
    assert _masked_rel(logits, ref) < 1e-2

    idx = ref.topk(topk, dim=-1).indices.int()
    idx = torch.where(torch.gather(torch.isfinite(ref), 1, idx.long()), idx, torch.full_like(idx, -1))
    grad = torch.randn(s_q, topk, device="cuda", dtype=torch.float32, generator=g)
    got = backend.mqa_logits_bwd(grad, q, k, w, ks, ke, idx, topk)
    want = bk._ref_bf16_mqa_logits_bwd(grad, q, k, w, ks, ke, idx)
    assert got[0].shape == q.shape and got[2].shape == w.shape
    for name, a, b, tol in zip(("grad_q", "grad_k", "grad_w"), got, want, (1e-2, 1e-2, 1e-3)):
        assert _rel(a, b) < tol, name


@pytest.mark.skipif(_capability() != (8, 0), reason="sm80 backend test")
def test_sparse_mla_sm80_beyond_grid_y_limit():
    """s_q above CUDA's gridDim.y ceiling (65535) must launch and match per-row results.

    The kernels used to put one CTA per query token on gridDim.y and failed with
    `invalid configuration argument` past 65535 tokens per rank; they now stride over a
    clamped y.  Every output row (and grad_q row) depends only on its own query, so the
    first and last 1024 rows of the long launch must equal short launches of just those
    rows at the same q_start_index_s.
    """
    s_q, h_q, topk, block, d_v = 65535 + 1025, 16, 64, 1024, 512
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(s_q, h_q, 576, device="cuda", dtype=torch.bfloat16, generator=g) * 0.25
    kv = torch.randn(s_q, 1, 576, device="cuda", dtype=torch.bfloat16, generator=g) * 0.25
    limit = torch.arange(s_q, device="cuda")
    idx = (torch.rand(s_q, 1, topk, device="cuda", generator=g) * (limit + 1).view(s_q, 1, 1).float()).floor().int()
    d_out = torch.randn(s_q, h_q, d_v, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    backend, scale = bk.DSAKernelBackend("sm80"), 576**-0.5

    def run(lo, hi):
        out, _, lse, _ = backend.sparse_mla_fwd(q[lo:hi], kv, idx[lo:hi], scale, d_v, q_start_index_s=lo)
        grad_q, _ = backend.sparse_mla_bwd(q[lo:hi], kv, out, d_out[lo:hi], idx[lo:hi], lse,
                                           sm_scale=scale, q_start_index_s=lo)
        return out, lse, grad_q

    full = run(0, s_q)
    assert torch.isfinite(full[0]).all() and torch.isfinite(full[2]).all()
    for lo in (0, s_q - block):
        part = run(lo, lo + block)
        for name, a, b in zip(("out", "lse", "grad_q"), full, part):
            assert torch.equal(a[lo:lo + block], b), f"{name} rows {lo}..{lo + block}"


def teardown_module(module):  # noqa: ARG001
    os.environ.pop("LOONGFORGE_DSA_KERNEL_BACKEND", None)
    bk.reset_dsa_backend()
