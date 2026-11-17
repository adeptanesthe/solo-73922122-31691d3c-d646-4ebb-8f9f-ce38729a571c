"""对齐损失的数据生成、分块/稠密双路计算与误差核对。"""

from __future__ import annotations

import torch

from .alignment import HEAD_DIM, build_target_matrices, chunked_alignment_loss
from .alignment_reference import dense_alignment_loss_grads
from .schemas import AlignmentLossRequest, AlignmentLossResponse
from .verification import make_doc_ids

TOLERANCE = 1e-8


def generate_qk(req: AlignmentLossRequest) -> tuple[torch.Tensor, torch.Tensor]:
    """按种子确定性生成 float64 的 Q/K（16 维）。

    与 /verify 采用同一个 torch.Generator 与相同的抽取顺序
    （q、k 后本应还有 v、上游梯度），因此同 seed/文档长度下的
    Q/K 与 /verify 完全一致；本接口只需要 Q/K。
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(req.seed))
    n = sum(req.doc_lengths)
    q = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    k = torch.randn(n, HEAD_DIM, generator=g, dtype=torch.float64)
    return q, k


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def run_alignment_verification(
    req: AlignmentLossRequest, include_tensors: bool = True
) -> AlignmentLossResponse:
    """生成本批次 Q/K，分别走分块算子与稠密参考，比较损失与 Q/K 梯度。"""
    q, k = generate_qk(req)
    doc_ids = make_doc_ids(req.doc_lengths)
    scale = float(req.s) / 4.0

    pairs = [
        (item.query_position, list(item.target_positions))
        for item in req.query_targets
    ]
    target_idx, target_mask = build_target_matrices(pairs)

    # --- 分块自定义算子（前向 + 反向） ---
    q_c = q.detach().clone().requires_grad_(True)
    k_c = k.detach().clone().requires_grad_(True)
    loss_c = chunked_alignment_loss(
        q_c, k_c, doc_ids, target_idx, target_mask, scale, req.chunk_size
    )
    gq_c, gk_c = torch.autograd.grad(loss_c, (q_c, k_c))
    loss_value = float(loss_c.detach().item())

    # --- 独立稠密参考（前向 + 反向） ---
    loss_r, gq_r, gk_r, p_target, log_p_target = dense_alignment_loss_grads(
        q, k, doc_ids, target_idx, target_mask, scale
    )

    errors = {
        "loss_max_abs_err": abs(loss_value - loss_r),
        "grad_q_max_abs_err": _max_abs(gq_c, gq_r),
        "grad_k_max_abs_err": _max_abs(gk_c, gk_r),
    }
    finite_tensors = [loss_c.detach(), gq_c, gk_c]
    finite_ok = (
        all(torch.isfinite(t).all().item() for t in finite_tensors)
        and torch.isfinite(p_target).all().item()
        and torch.isfinite(log_p_target).all().item()
    )

    failure_reasons: list[str] = []
    for name, value in errors.items():
        if value > TOLERANCE:
            failure_reasons.append(f"{name}={value:.3e} 超过容差 {TOLERANCE:.0e}")
        if value != value:  # NaN
            failure_reasons.append(f"{name} 为 NaN")
    if not finite_ok:
        failure_reasons.append("分块实现的损失或梯度中存在 NaN/Inf")

    return AlignmentLossResponse(
        passed=not failure_reasons,
        tolerance=TOLERANCE,
        errors=errors,
        scale_used=scale,
        total_tokens=sum(req.doc_lengths),
        num_docs=len(req.doc_lengths),
        num_queries=len(req.query_targets),
        chunk_size=req.chunk_size,
        s=req.s,
        seed=req.seed,
        loss=loss_value,
        finite_ok=finite_ok,
        failure_reasons=failure_reasons,
        target_probabilities=p_target.tolist(),
        log_target_probabilities=log_p_target.tolist(),
        grad_q=gq_c.detach().tolist() if include_tensors else None,
        grad_k=gk_c.detach().tolist() if include_tensors else None,
    )
