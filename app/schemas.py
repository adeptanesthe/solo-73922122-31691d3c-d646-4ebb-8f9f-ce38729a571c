"""请求/响应模型与输入校验。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_TOTAL_TOKENS = 512
ALIGNMENT_MAX_TOKENS = 64
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 1
MAX_S = 1000.0
HEAD_DIM = 16
MAX_QUERY_SPANS = 8
MAX_TARGETS_PER_QUERY = 4


class VerifyRequest(BaseModel):
    """/verify 请求体。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 512"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        total = sum(value)
        if total == 0 or total > MAX_TOTAL_TOKENS:
            raise ValueError(
                f"文档总长度必须在 1..{MAX_TOTAL_TOKENS} 之间，当前为 {total}"
            )
        return value

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value


class ErrorMetrics(BaseModel):
    output_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float
    grad_v_max_abs_err: float


class VerifyResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="所有输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None


class ScenarioSummary(BaseModel):
    name: str
    description: str
    request: VerifyRequest


class QueryTarget(BaseModel):
    """单个查询位置及其标注的目标 key 位置（1~4 个，互不相同）。"""

    query_position: int = Field(..., ge=0, description="查询 token 在批次中的位置")
    target_positions: list[int] = Field(
        ...,
        min_length=1,
        max_length=MAX_TARGETS_PER_QUERY,
        description="目标 key 位置列表，须同文档且不晚于查询位置",
    )

    @field_validator("target_positions")
    @classmethod
    def _check_targets_distinct_and_range(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x < 0 for x in value):
            raise ValueError("target_positions 必须为非负整数列表")
        if len(set(value)) != len(value):
            raise ValueError("同一查询的目标 key 位置必须互不相同")
        return value


class AlignmentLossRequest(BaseModel):
    """/alignment-loss 请求体：种子生成 16 维 Q/K（总长 ≤ 64）+ 证据标注。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数），与 /verify 同序生成 Q/K")
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 64"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")
    query_targets: list[QueryTarget] = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_SPANS,
        description="1~8 个不同查询位置，每处 1~4 个不同目标 key",
    )

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        total = sum(value)
        if total == 0 or total > ALIGNMENT_MAX_TOKENS:
            raise ValueError(
                f"文档总长度必须在 1..{ALIGNMENT_MAX_TOKENS} 之间，当前为 {total}"
            )
        return value

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value

    @model_validator(mode="after")
    def _check_query_target_semantics(self) -> "AlignmentLossRequest":
        n = sum(self.doc_lengths)
        # 每篇文档的 token 起始偏移，用于判定目标是否与查询同篇。
        offsets: list[int] = []
        cursor = 0
        for length in self.doc_lengths:
            offsets.append(cursor)
            cursor += length

        def doc_of(pos: int) -> int:
            # 二分查找所属文档（文档数很小，线性也可，保持直白）。
            for doc_id in range(len(offsets) - 1, -1, -1):
                if pos >= offsets[doc_id]:
                    return doc_id
            raise ValueError(f"位置 {pos} 超出批次范围")

        seen_queries: set[int] = set()
        for item in self.query_targets:
            q = item.query_position
            if q >= n:
                raise ValueError(f"query_position={q} 超出批次总长 {n}")
            if q in seen_queries:
                raise ValueError(f"查询位置 {q} 重复出现，查询位置必须互不相同")
            seen_queries.add(q)
            q_doc = doc_of(q)
            for t in item.target_positions:
                if t >= n:
                    raise ValueError(
                        f"目标位置 {t} 超出批次总长 {n}（查询位置 {q}）"
                    )
                if doc_of(t) != q_doc:
                    raise ValueError(
                        f"目标 key {t} 与查询 {q} 不属于同一篇文档，整单拒绝"
                    )
                if t > q:
                    raise ValueError(
                        f"目标 key {t} 晚于查询位置 {q}，违反因果约束，整单拒绝"
                    )
        return self


class AlignmentErrorMetrics(BaseModel):
    loss_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float


class AlignmentLossResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: AlignmentErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    num_queries: int
    chunk_size: int
    s: float
    seed: int
    loss: float = Field(..., description="分块实现计算的标量对齐损失（有限值）")
    finite_ok: bool = Field(..., description="损失与 Q/K 梯度是否均为有限值")
    failure_reasons: list[str]
    target_probabilities: list[float] = Field(
        ..., description="每个查询位置的目标概率之和（按请求顺序）"
    )
    log_target_probabilities: list[float] = Field(
        ..., description="每个查询位置目标概率之和的对数（极小概率下仍有限）"
    )
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
