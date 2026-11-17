"""/alignment-loss 端点测试：正常核对、整单拒绝、与 /verify 的数据一致性。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


VALID_BODY = {
    "seed": 42,
    "doc_lengths": [7, 5],
    "chunk_size": 4,
    "s": 1.0,
    "query_targets": [
        {"query_position": 0, "target_positions": [0]},
        {"query_position": 3, "target_positions": [0, 1, 3]},
        {"query_position": 6, "target_positions": [2, 6]},
        {"query_position": 11, "target_positions": [7, 11]},
    ],
}


def test_alignment_endpoint_in_openapi():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/alignment-loss" in paths


def test_alignment_basic_pass():
    resp = client.post("/alignment-loss", json=VALID_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["failure_reasons"] == []
    assert body["finite_ok"] is True
    assert body["num_queries"] == 4
    assert body["total_tokens"] == 12
    assert body["errors"]["loss_max_abs_err"] <= 1e-8
    assert body["errors"]["grad_q_max_abs_err"] <= 1e-8
    assert body["errors"]["grad_k_max_abs_err"] <= 1e-8
    assert len(body["grad_q"]) == 12 and len(body["grad_q"][0]) == 16
    assert len(body["grad_k"]) == 12 and len(body["grad_k"][0]) == 16
    assert len(body["target_probabilities"]) == 4
    assert len(body["log_target_probabilities"]) == 4
    # log(p) 与 p 自洽（p 不接近 0 的条目）
    import math

    for p, lp in zip(
        body["target_probabilities"], body["log_target_probabilities"]
    ):
        assert math.isfinite(lp)
        if p > 1e-12:
            assert abs(math.log(p) - lp) <= 1e-10


def test_alignment_without_tensors():
    body = client.post(
        "/alignment-loss?include_tensors=false",
        json={
            "seed": 1,
            "doc_lengths": [4],
            "chunk_size": 2,
            "s": 2.0,
            "query_targets": [
                {"query_position": 2, "target_positions": [0, 2]}
            ],
        },
    ).json()
    assert body["passed"] is True
    assert body["grad_q"] is None
    assert body["grad_k"] is None


def test_qk_match_verify_generation_order():
    """同 seed/doc_lengths 下，/alignment-loss 使用的 Q/K 与 /verify 同源同序。"""
    import torch

    from app.alignment_verification import generate_qk
    from app.verification import generate_inputs

    req_kw = {"seed": 123, "doc_lengths": [9, 7], "chunk_size": 4, "s": 1.5}
    from app.schemas import AlignmentLossRequest, VerifyRequest

    q_a, k_a = generate_qk(AlignmentLossRequest(
        **req_kw,
        query_targets=[{"query_position": 0, "target_positions": [0]}],
    ))
    data = generate_inputs(VerifyRequest(**req_kw))
    assert torch.equal(q_a, data.q)
    assert torch.equal(k_a, data.k)


def test_reject_cross_document_target():
    body = dict(VALID_BODY)
    body["query_targets"] = [
        {"query_position": 10, "target_positions": [0, 10]}  # 0 属另一篇文档
    ]
    resp = client.post("/alignment-loss", json=body)
    assert resp.status_code == 422
    assert "同一篇文档" in resp.text


def test_reject_future_target():
    body = dict(VALID_BODY)
    body["query_targets"] = [{"query_position": 2, "target_positions": [3]}]
    resp = client.post("/alignment-loss", json=body)
    assert resp.status_code == 422
    assert "因果" in resp.text


def test_reject_duplicate_query_positions():
    body = dict(VALID_BODY)
    body["query_targets"] = [
        {"query_position": 2, "target_positions": [2]},
        {"query_position": 2, "target_positions": [0]},
    ]
    resp = client.post("/alignment-loss", json=body)
    assert resp.status_code == 422
    assert "重复" in resp.text


def test_reject_duplicate_targets():
    body = dict(VALID_BODY)
    body["query_targets"] = [
        {"query_position": 3, "target_positions": [0, 0]}
    ]
    resp = client.post("/alignment-loss", json=body)
    assert resp.status_code == 422


def test_reject_out_of_range_positions():
    body = dict(VALID_BODY)
    body["query_targets"] = [
        {"query_position": 12, "target_positions": [0]}  # 总长 12，越界
    ]
    assert client.post("/alignment-loss", json=body).status_code == 422

    body2 = dict(VALID_BODY)
    body2["query_targets"] = [
        {"query_position": 3, "target_positions": [11]}  # 目标越界
    ]
    assert client.post("/alignment-loss", json=body2).status_code == 422


def test_reject_cardinality_limits():
    base = {"seed": 0, "doc_lengths": [64], "chunk_size": 4, "s": 1.0}

    # 总长 > 64
    resp = client.post(
        "/alignment-loss",
        json={**base, "doc_lengths": [65],
              "query_targets": [{"query_position": 0, "target_positions": [0]}]},
    )
    assert resp.status_code == 422

    # 查询位置 9 个（上限 8）
    resp = client.post(
        "/alignment-loss",
        json={
            **base,
            "query_targets": [
                {"query_position": i, "target_positions": [0]} for i in range(9)
            ],
        },
    )
    assert resp.status_code == 422

    # 目标 5 个（上限 4）
    resp = client.post(
        "/alignment-loss",
        json={
            **base,
            "query_targets": [
                {"query_position": 10, "target_positions": [0, 1, 2, 3, 4]}
            ],
        },
    )
    assert resp.status_code == 422

    # 空查询列表
    resp = client.post(
        "/alignment-loss", json={**base, "query_targets": []}
    )
    assert resp.status_code == 422


def test_reject_bad_scale_and_chunk():
    base = {
        "seed": 0,
        "doc_lengths": [8],
        "query_targets": [{"query_position": 0, "target_positions": [0]}],
    }
    for bad in (0.0, 1000.1):
        resp = client.post(
            "/alignment-loss", json={**base, "chunk_size": 4, "s": bad}
        )
        assert resp.status_code == 422
    resp = client.post(
        "/alignment-loss", json={**base, "chunk_size": 33, "s": 1.0}
    )
    assert resp.status_code == 422


def test_self_target_single_token_doc():
    """单 token 文档自指：目标概率恒为 1，损失为 0，梯度为 0。"""
    body = client.post(
        "/alignment-loss",
        json={
            "seed": 1,
            "doc_lengths": [1, 1],
            "chunk_size": 32,
            "s": 0.5,
            "query_targets": [
                {"query_position": 0, "target_positions": [0]},
                {"query_position": 1, "target_positions": [1]},
            ],
        },
    ).json()
    assert body["passed"] is True
    assert abs(body["loss"]) <= 1e-12
    assert body["target_probabilities"] == [1.0, 1.0]
    assert all(abs(v) <= 1e-12 for row in body["grad_q"] for v in row)
    assert all(abs(v) <= 1e-12 for row in body["grad_k"] for v in row)


def test_verify_endpoint_still_available():
    resp = client.post(
        "/verify",
        json={"seed": 42, "doc_lengths": [7, 11, 5], "chunk_size": 8, "s": 1.0},
    )
    assert resp.status_code == 200
    assert resp.json()["passed"] is True
    assert client.get("/docs").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
