"""对齐损失核心功能测试：正确性、gradcheck、分块不变性、边界与内存约束。"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from app.alignment import (
    HEAD_DIM,
    ChunkedAlignmentLoss,
    build_target_matrices,
    chunked_alignment_loss,
)
from app.alignment_reference import dense_alignment_loss_grads

torch.manual_seed(0)


def _make_qk(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    k = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    return q, k


def _doc_ids(doc_lengths: list[int]) -> torch.Tensor:
    return torch.cat(
        [torch.full((length,), i, dtype=torch.int64)
         for i, length in enumerate(doc_lengths)]
    )


def _run_pair(doc_lengths, pairs, chunk_size, s, seed=0):
    n = sum(doc_lengths)
    q, k = _make_qk(n, seed)
    docs = _doc_ids(doc_lengths)
    ti, tm = build_target_matrices(pairs)
    scale = s / 4.0

    qc, kc = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    loss_c = chunked_alignment_loss(qc, kc, docs, ti, tm, scale, chunk_size)
    gq_c, gk_c = torch.autograd.grad(loss_c, (qc, kc))

    loss_r, gq_r, gk_r, p_t, log_p_t = dense_alignment_loss_grads(
        q, k, docs, ti, tm, scale
    )
    return (float(loss_c.item()), gq_c, gk_c), (loss_r, gq_r, gk_r, p_t, log_p_t)


@pytest.mark.parametrize(
    "doc_lengths,pairs,chunk_size,s,seed",
    [
        ([4], [(0, [0])], 1, 1.0, 0),                       # 最小单 token 自指
        ([8], [(7, [0, 7])], 1, 1.0, 1),                    # 块长 1
        ([8], [(3, [0, 1, 3]), (7, [2, 7])], 4, 2.5, 2),    # 多查询多目标
        ([7, 11, 5], [(0, [0]), (6, [2, 6]), (17, [8, 17]), (22, [18, 22])], 8, 1.0, 42),
        ([16], [(15, [12, 13, 14, 15])], 4, 2.5, 7),        # 4 个目标 + 跨块
        ([33, 31], [(30, [0, 30]), (63, [33, 50, 63])], 32, 1000.0, 123),  # 大分数
        ([64], [(63, [0, 31, 63])], 7, 999.0, 2026),        # 长度上限、块长不整除
        ([1, 1], [(0, [0]), (1, [1])], 32, 0.5, 1),         # 两篇各 1 token
        ([3, 1, 9, 2], [(2, [0, 2]), (12, [4, 12]), (14, [14])], 5, 37.7, 5),
        # 8 个查询（上限），每处最多 4 个不同目标（上限）
        ([64], [
            (0, [0]),
            (1, [0, 1]),
            (2, [0, 1, 2]),
            *[
                (i, [i - 3, i - 2, i - 1, i])
                for i in range(3, 8)
            ],
        ], 16, 10.0, 99),
    ],
)
def test_loss_and_grad_match_dense(doc_lengths, pairs, chunk_size, s, seed):
    chunked, ref = _run_pair(doc_lengths, pairs, chunk_size, s, seed)
    loss_c, gq_c, gk_c = chunked
    loss_r, gq_r, gk_r, _, _ = ref
    for name, a, b in (
        ("loss", torch.tensor(loss_c), torch.tensor(loss_r)),
        ("grad_q", gq_c, gq_r),
        ("grad_k", gk_c, gk_r),
    ):
        assert torch.isfinite(a).all(), f"{name} 含非有限值"
        err = torch.max(torch.abs(a - b)).item()
        assert err <= 1e-8, f"{name} 误差 {err:.3e} 超过 1e-8"


def test_gradcheck_forward_backward_formulas():
    """有限差分核对自定义算子的前向损失与一阶反传公式。"""
    q, k = _make_qk(10, seed=11)
    docs = _doc_ids([6, 4])
    ti, tm = build_target_matrices([(2, [0, 2]), (5, [1, 5]), (9, [6, 9])])
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda qq, kk: ChunkedAlignmentLoss.apply(qq, kk, docs, ti, tm, 0.75, 4),
        (q, k),
        eps=1e-7,
        atol=1e-6,
        rtol=1e-4,
    )


def test_chunk_size_does_not_change_result():
    """同批数据在不同块长下损失与梯度必须一致。"""
    doc_lengths = [13, 17, 7]
    pairs = [(0, [0]), (5, [1, 5]), (20, [13, 20]), (26, [26])]
    n = sum(doc_lengths)
    q, k = _make_qk(n, seed=5)
    docs = _doc_ids(doc_lengths)
    ti, tm = build_target_matrices(pairs)
    scale = 3.0 / 4.0
    losses, gqs, gks = [], [], []
    for chunk in (1, 2, 3, 7, 16, 32):
        qc, kc = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        loss = chunked_alignment_loss(qc, kc, docs, ti, tm, scale, chunk)
        gq, gk = torch.autograd.grad(loss, (qc, kc))
        losses.append(float(loss.item()))
        gqs.append(gq.detach())
        gks.append(gk.detach())
    for other in losses[1:]:
        assert abs(losses[0] - other) <= 1e-12
    for base, other in zip(gqs[1:], gqs):
        assert torch.allclose(gqs[0], other, atol=1e-12, rtol=0)
    for other in gks[1:]:
        assert torch.allclose(gks[0], other, atol=1e-12, rtol=0)


def test_determinism_same_seed_same_result():
    a, _ = _run_pair([7, 5], [(3, [0, 3]), (11, [7, 11])], 4, 1.0, seed=42)
    b, _ = _run_pair([7, 5], [(3, [0, 3]), (11, [7, 11])], 4, 1.0, seed=42)
    assert a[0] == b[0]
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_loss_matches_manual_probability_definition():
    """直接按定义 -mean(log(sum 目标 P)) 手工计算，与算子结果一致。"""
    doc_lengths = [6, 4]
    pairs = [(2, [0, 2]), (9, [6, 7, 9])]
    n = sum(doc_lengths)
    q, k = _make_qk(n, seed=3)
    docs = _doc_ids(doc_lengths)
    ti, tm = build_target_matrices(pairs)
    scale = 0.25
    qc, kc = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    loss = chunked_alignment_loss(qc, kc, docs, ti, tm, scale, 3)

    pos = torch.arange(n)
    valid = (docs[:, None] == docs[None, :]) & (pos[None, :] <= pos[:, None])
    sc = (q @ k.T) * scale
    probs = torch.softmax(sc.masked_fill(~valid, float("-inf")), dim=1)
    q_idx = torch.nonzero(tm.any(dim=1)).flatten()
    manual = -torch.mean(
        torch.log(
            torch.stack(
                [probs[i, ti[i][tm[i]]].sum() for i in q_idx]
            )
        )
    )
    assert abs(float(loss.item()) - float(manual.item())) <= 1e-12


def test_non_query_rows_have_zero_q_gradient():
    """未作为查询的 token 不进入损失，其 Q 梯度必须严格为 0。"""
    doc_lengths = [8]
    pairs = [(7, [0, 7])]
    (_, gq, gk), _ = _run_pair(doc_lengths, pairs, 4, 1.0, seed=8)
    assert torch.count_nonzero(gq[:7]) == 0
    assert gq[7].abs().sum() > 0  # 查询行非零


def test_tiny_target_probability_still_finite_and_accurate():
    """目标概率小到 float64 下溢为 0 时，损失与梯度仍有限且与参考一致。"""
    n = 8
    q = torch.zeros(n, HEAD_DIM, dtype=torch.float64)
    q[:, 0] = 1.0
    k = torch.zeros(n, HEAD_DIM, dtype=torch.float64)
    scale = 250.0
    k[7, 0] = 1.0                       # 自身分数 250
    for i in range(7):                  # 其余分数 250-760=-510
        k[i, 0] = -760.0 / 250.0
    docs = torch.zeros(n, dtype=torch.int64)
    ti, tm = build_target_matrices([(7, [0])])

    loss_r, gq_r, gk_r, p_t, log_p_t = dense_alignment_loss_grads(
        q, k, docs, ti, tm, scale
    )
    assert p_t.item() == 0.0                       # 概率确实下溢
    assert torch.isfinite(torch.tensor(log_p_t.item())).item()

    for cs in (1, 4, 8):
        qc, kc = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        loss = chunked_alignment_loss(qc, kc, docs, ti, tm, scale, cs)
        gq, gk = torch.autograd.grad(loss, (qc, kc))
        assert torch.isfinite(loss)
        assert torch.isfinite(gq).all() and torch.isfinite(gk).all()
        assert abs(float(loss.item()) - loss_r) <= 1e-8
        assert torch.max(torch.abs(gq - gq_r)).item() <= 1e-8
        assert torch.max(torch.abs(gk - gk_r)).item() <= 1e-8


def test_saved_tensors_have_linear_footprint():
    """跨前后向保存量审计：不得出现 N x N（或更大）的二次规模张量。"""
    n = 64
    q, k = _make_qk(n, seed=9)
    docs = _doc_ids([40, 24])
    ti, tm = build_target_matrices([(10, [0, 10]), (50, [40, 50]), (63, [63])])
    saved: list[torch.Tensor] = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t), t)[1], lambda t: t
    ):
        qc = q.clone().requires_grad_(True)
        kc = k.clone().requires_grad_(True)
        loss = chunked_alignment_loss(qc, kc, docs, ti, tm, 0.5, 16)
        loss.backward()

    assert saved, "自定义算子未保存任何张量"
    cap = n * HEAD_DIM + n * 4 + 8
    for t in saved:
        assert t.numel() <= cap, (
            f"保存张量 {tuple(t.shape)} 规模 {t.numel()} 超过线性上限 {cap}"
        )
        assert not (len(t.shape) == 2 and t.shape[0] == n and t.shape[1] > HEAD_DIM)


def _attribute_names(tree: ast.AST) -> set[str]:
    """收集源码中实际出现的属性访问名（如 torch.softmax 的 softmax）。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_no_builtin_attention_or_softmax_in_source():
    """源码审计（AST 级，忽略注释/文档字符串）：不得调用现成注意力/softmax 原语。"""
    import app.alignment as alignment_mod

    source = inspect.getsource(alignment_mod)
    names = _attribute_names(ast.parse(source))
    forbidden = [
        "scaled_dot_product_attention",
        "softmax",
        "log_softmax",
        "logsumexp",
    ]
    for token in forbidden:
        assert token not in names, f"分块算子源码中出现禁用 API：{token}"


def test_dense_reference_is_independent_and_manual():
    """稠密参考同样不得使用内置 softmax/logsumexp，保持独立可审计。"""
    import app.alignment_reference as ref_mod

    names = _attribute_names(ast.parse(inspect.getsource(ref_mod)))
    for token in ("softmax", "log_softmax", "logsumexp",
                  "scaled_dot_product_attention"):
        assert token not in names
