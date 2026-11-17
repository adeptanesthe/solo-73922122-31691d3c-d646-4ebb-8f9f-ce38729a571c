"""对齐损失的独立稠密参考实现（不受 O(N + C^2) 工作张量限制）。

与分块算子完全独立：手工构造完整 N x N 掩码、注意力矩阵与目标概率，
不使用任何现成注意力/softmax/logsumexp 算子，仅使用基础张量运算，
并借助 autograd 反传 Q/K 梯度，用于核对分块实现的损失与梯度。

数值稳定性同样手工保证：log 概率全程在 log 空间（最大值平移后的
log-sum-exp）计算，目标概率即使小到 float64 下溢，损失仍为有限值。
"""

from __future__ import annotations

import torch

from .attention import NEG_INF  # 仅共享常量，不共享算法代码


def _manual_log_softmax_rows(scores: torch.Tensor) -> torch.Tensor:
    """逐行手工 log-softmax（不调用 torch.log_softmax/logsumexp）。"""
    row_max = torch.amax(scores, dim=1, keepdim=True)
    shifted = scores - row_max
    log_denom = row_max + torch.log(torch.sum(torch.exp(shifted), dim=1, keepdim=True))
    return scores - log_denom


def dense_alignment_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    doc_ids: torch.Tensor,
    target_idx: torch.Tensor,
    target_mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """稠密参考前向：返回 (标量损失, 每查询目标概率, 每查询 log 目标概率)。

    输入 q/k 需为带 autograd 历史的叶子张量，调用方可直接对损失反传。
    """
    n = q.shape[0]
    pos = torch.arange(n, dtype=torch.int64)
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    causal = pos[None, :] <= pos[:, None]
    valid = same_doc & causal

    scores = scale * (q @ k.T)
    scores = torch.where(valid, scores, NEG_INF)

    # 完整 N x N 注意力矩阵（参考实现不受内存约束），手工 softmax。
    row_max = torch.amax(scores, dim=1, keepdim=True)
    exp_scores = torch.where(valid, torch.exp(scores - row_max), 0.0)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

    q_idx = torch.nonzero(target_mask.any(dim=1), as_tuple=False).flatten()
    # 目标概率之和：直接在完整矩阵上按槽位收集。
    gathered = probs[q_idx.unsqueeze(1), target_idx[q_idx].clamp_min(0)]
    p_target = torch.sum(gathered * target_mask[q_idx].to(probs.dtype), dim=1)

    # log 概率走 log 空间的手工 log-softmax，避免极小概率下 log(0) = -inf。
    log_probs = _manual_log_softmax_rows(scores)
    log_gathered = log_probs[q_idx.unsqueeze(1), target_idx[q_idx].clamp_min(0)]
    # log(sum_j exp(log_p_ij))：以槽位内最大值平移，保持有限。
    log_gathered = torch.where(target_mask[q_idx], log_gathered, NEG_INF)
    t_max = torch.amax(log_gathered, dim=1, keepdim=True)
    log_p_target = (
        t_max.squeeze(1)
        + torch.log(
            torch.sum(
                torch.where(
                    target_mask[q_idx],
                    torch.exp(log_gathered - t_max),
                    0.0,
                ),
                dim=1,
            )
        )
    )

    loss = -torch.mean(log_p_target)
    return loss, p_target.detach(), log_p_target.detach()


def dense_alignment_loss_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    doc_ids: torch.Tensor,
    target_idx: torch.Tensor,
    target_mask: torch.Tensor,
    scale: float,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """稠密参考的损失与 Q/K 梯度（借助 autograd 反传）。"""
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    loss, p_target, log_p_target = dense_alignment_loss(
        q_ref, k_ref, doc_ids, target_idx, target_mask, scale
    )
    gq, gk = torch.autograd.grad(loss, (q_ref, k_ref))
    return float(loss.detach().item()), gq, gk, p_target, log_p_target


__all__ = ["dense_alignment_loss", "dense_alignment_loss_grads"]
