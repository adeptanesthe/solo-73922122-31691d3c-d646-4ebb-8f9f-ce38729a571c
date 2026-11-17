"""分块对齐损失（alignment loss）：自定义 autograd 算子。

长文档训练场景下，工程师为每个查询 token 标注了若干应当关注的证据
（目标 key）。本模块实现按块计算的对齐损失及其手写反向传播：

    L_i = -log( sum_{j in T_i} P_{ij} )          # T_i 为查询 i 的目标集合
    L   = (1 / |Q|) * sum_i L_i

其中 ``P = softmax(s * Q K^T / 4)`` 仍受“同篇文档、位置不晚于查询”的
文档内因果掩码约束。

约定
----
* 所有计算均在 CPU、float64 下完成。
* 前向、后向均按长度 C 的块逐块处理；单块工作张量为 O(C^2 + N) 量级，
  跨前后向保存的张量为 O(N) 量级（d=16 与每查询最多 4 个目标均为常数）。
* 不调用任何框架内置的注意力/softmax/logsumexp 原语，全部用
  最大值平移 + 逐点 exp/log 手写，保证可审计与数值稳定。
* 目标概率极小时（float64 下 exp 直接下溢为 0），前向以“目标集合自身的
  log-sum-exp”给出有限的 log 概率，后向以同样的稳定加权方式给出有限梯度。
"""

from __future__ import annotations

import torch

from .attention import HEAD_DIM, NEG_INF  # 仅共享常量，不共享算法代码

MAX_QUERY_SPANS = 8
MAX_TARGETS_PER_QUERY = 4


def _split(num: int, chunk: int) -> list[tuple[int, int]]:
    """返回左闭右开的分块区间列表。"""
    return [(b, min(b + chunk, num)) for b in range(0, num, chunk)]


def _block_valid(
    doc_ids: torch.Tensor, q0: int, q1: int, k0: int, k1: int
) -> torch.Tensor:
    """构造块 (q0:q1) x (k0:k1) 的文档内因果掩码。"""
    same_doc = doc_ids[q0:q1, None] == doc_ids[None, k0:k1]
    k_pos = torch.arange(k0, k1, dtype=torch.int64)
    q_pos = torch.arange(q0, q1, dtype=torch.int64)
    causal = k_pos[None, :] <= q_pos[:, None]
    return same_doc & causal


def build_target_matrices(
    pairs: list[tuple[int, list[int]]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (query 位置 -> 目标 key 位置列表) 展开为定长填充矩阵。

    返回
    ----
    target_idx : long tensor [N, 4]，空缺处填 -1
    target_mask: bool tensor [N, 4]，True 表示该槽位是有效目标
    """
    n = max((q for q, _ in pairs), default=-1) + 1
    target_idx = torch.full(
        (n, MAX_TARGETS_PER_QUERY), -1, dtype=torch.int64
    )
    target_mask = torch.zeros((n, MAX_TARGETS_PER_QUERY), dtype=torch.bool)
    for query_pos, keys in pairs:
        for slot, key_pos in enumerate(keys):
            target_idx[query_pos, slot] = key_pos
            target_mask[query_pos, slot] = True
    return target_idx, target_mask


class ChunkedAlignmentLoss(torch.autograd.Function):
    """分块对齐损失的自定义 autograd 算子（online softmax + 分块反向）。"""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        doc_ids: torch.Tensor,
        target_idx: torch.Tensor,
        target_mask: torch.Tensor,
        scale: float,
        chunk_size: int,
    ) -> torch.Tensor:
        n, d = q.shape
        blocks = _split(n, chunk_size)
        device = q.device

        # ---- 分块 online softmax：仅维护行最大值与归一化分母（O(N)） ----
        row_max = torch.full((n,), float("-inf"), dtype=torch.float64, device=device)
        row_denom = torch.zeros((n,), dtype=torch.float64, device=device)

        for k0, k1 in blocks:
            for q0, q1 in [(b0, b1) for (b0, b1) in blocks if k0 < b1]:
                valid = _block_valid(doc_ids, q0, q1, k0, k1)
                if not torch.any(valid):
                    # 整块被屏蔽：running 统计量保持不变，不会产生 NaN。
                    continue

                scores = (q[q0:q1] @ k[k0:k1].T) * scale
                scores = torch.where(valid, scores, NEG_INF)

                block_max = torch.amax(scores, dim=1)  # 完全屏蔽行为 -inf
                active = torch.isfinite(block_max)

                safe_prev = torch.where(active, row_max[q0:q1], 0.0)
                safe_block = torch.where(active, block_max, 0.0)
                new_max = torch.maximum(safe_prev, safe_block)
                new_max = torch.where(active, new_max, NEG_INF)

                prev_corr = torch.where(
                    active, torch.exp(safe_prev - new_max), 0.0
                )
                block_w = torch.where(
                    active.unsqueeze(1),
                    torch.exp(scores - new_max.unsqueeze(1)),
                    0.0,
                )
                block_denom = torch.sum(block_w, dim=1)
                row_denom[q0:q1] = row_denom[q0:q1] * prev_corr + block_denom
                row_max[q0:q1] = new_max

        log_denom = torch.log(row_denom) + row_max  # 每个查询行至少含自身，恒有限

        # ---- 目标集合的 log-sum-exp（以目标集合自身最大值平移） ----
        # 这样即使目标概率小到 exp 整体下溢，log 概率与损失仍是有限值。
        q_idx = torch.nonzero(target_mask.any(dim=1), as_tuple=False).flatten()
        t_idx = target_idx[q_idx]                       # [Nq, 4]
        t_mask = target_mask[q_idx]                     # [Nq, 4]
        raw_target_scores = scale * torch.sum(
            q[q_idx].unsqueeze(1) * k[t_idx.clamp_min(0)], dim=2
        )                                               # [Nq, 4]，槽位 -1 取行 0 占位
        target_scores = torch.where(t_mask, raw_target_scores, NEG_INF)
        target_max = torch.amax(target_scores, dim=1)  # 每查询至少 1 个目标，恒有限
        target_lse = target_max + torch.log(
            torch.sum(
                torch.where(
                    t_mask,
                    torch.exp(target_scores - target_max.unsqueeze(1)),
                    0.0,
                ),
                dim=1,
            )
        )
        log_p_target = target_lse - log_denom[q_idx]   # log P(T_i)，恒有限
        loss = -torch.mean(log_p_target)

        ctx.save_for_backward(
            q, k, doc_ids, row_max, row_denom,
            target_idx, target_mask, q_idx,
        )
        ctx.scale = float(scale)
        ctx.chunk_size = int(chunk_size)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (
            q,
            k,
            doc_ids,
            row_max,
            row_denom,
            target_idx,
            target_mask,
            q_idx,
        ) = ctx.saved_tensors
        n, d = q.shape
        scale: float = ctx.scale
        chunk_size: int = ctx.chunk_size
        blocks = _split(n, chunk_size)
        num_queries = int(q_idx.numel())
        # 标量损失的上游梯度（通常为 1），保留对一般 grad_output 的支持。
        g = float(grad_loss.item())
        inv_q = g / num_queries

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)

        # 每查询目标集合的稳定条件权重 w_ij = exp(z_ij - m_Ti) /
        # sum_{t in T_i} exp(z_it - m_Ti) = P_ij / P(T_i)，
        # 恒为有限值（即使目标概率整体下溢也不经过 P(T_i) 除法）。
        t_idx = target_idx[q_idx]
        t_mask = target_mask[q_idx]
        raw_target_scores = scale * torch.sum(
            q[q_idx].unsqueeze(1) * k[t_idx.clamp_min(0)], dim=2
        )
        target_scores = torch.where(t_mask, raw_target_scores, NEG_INF)
        target_max = torch.amax(target_scores, dim=1, keepdim=True)
        target_w = torch.where(
            t_mask,
            torch.exp(target_scores - target_max),
            0.0,
        )
        target_w = target_w / torch.sum(target_w, dim=1, keepdim=True)

        is_query = torch.zeros((n,), dtype=torch.bool)
        is_query[q_idx] = True
        # 全局 token 位置 -> q_idx 中的行序（target_w/t_idx 均按该行序排列）。
        query_order = torch.full((n,), -1, dtype=torch.int64)
        query_order[q_idx] = torch.arange(num_queries, dtype=torch.int64)

        # 解析梯度（z = scale * QK^T，行 a、列 b）：
        #   查询行: dL/dz_ab = (1/|Q|) * (P_ab - 1[b∈T_a] * w_ab)
        #   其余行: 0（softmax 雅可比行和为 0，且这些行没有目标）
        for k0, k1 in blocks:
            for q0, q1 in [(b0, b1) for (b0, b1) in blocks if k0 < b1]:
                valid = _block_valid(doc_ids, q0, q1, k0, k1)
                if not torch.any(valid):
                    continue

                # 用保存的行最大值/分母重算块内概率，块矩阵仅 O(C^2)。
                scores = (q[q0:q1] @ k[k0:k1].T) * scale
                prob = torch.where(
                    valid,
                    torch.exp(scores - row_max[q0:q1, None])
                    / row_denom[q0:q1, None],
                    0.0,
                )

                d_scores = torch.zeros_like(scores)
                local_q = torch.nonzero(is_query[q0:q1], as_tuple=False).flatten()
                if local_q.numel() > 0:
                    d_scores[local_q] = inv_q * prob[local_q]

                    # 目标项：减去 (1/|Q|) w_ij（只处理落在当前 key 块内的槽位）。
                    global_q = q0 + local_q
                    order = query_order[global_q]
                    slot_mask = (t_idx[order] >= k0) & (t_idx[order] < k1)
                    # sel_q 是 slot_mask 的行号，即 local_q 中的位置；
                    # 写入 d_scores 时需换算回块内局部行 local_q[sel_q]。
                    sel_q, sel_slot = torch.nonzero(slot_mask, as_tuple=True)
                    if sel_q.numel() > 0:
                        block_rows = local_q[sel_q]
                        ord_rows = order[sel_q]
                        gk_cols = t_idx[ord_rows, sel_slot] - k0
                        d_scores[block_rows, gk_cols] -= (
                            inv_q * target_w[ord_rows, sel_slot]
                        )

                d_scores = d_scores * scale
                grad_q[q0:q1] += d_scores @ k[k0:k1]
                grad_k[k0:k1] += d_scores.T @ q[q0:q1]

        return grad_q, grad_k, None, None, None, None, None


def chunked_alignment_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    doc_ids: torch.Tensor,
    target_idx: torch.Tensor,
    target_mask: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """分块对齐损失的函数式入口，返回标量损失（带 autograd 历史）。"""
    return ChunkedAlignmentLoss.apply(
        q, k, doc_ids, target_idx, target_mask, scale, chunk_size
    )


__all__ = [
    "HEAD_DIM",
    "MAX_QUERY_SPANS",
    "MAX_TARGETS_PER_QUERY",
    "build_target_matrices",
    "ChunkedAlignmentLoss",
    "chunked_alignment_loss",
]
