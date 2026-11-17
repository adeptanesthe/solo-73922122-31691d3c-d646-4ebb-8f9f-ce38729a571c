# 分块文档内因果注意力核对 API

把多篇文档拼入同一批次（总长 N ≤ 512），在 **CPU + float64** 下核对一个
自定义分块注意力 autograd 算子与独立稠密参考实现的前向输出及 Q/K/V 梯度；
另提供 `/alignment-loss`，按工程师标注的证据 key 核对**分块对齐损失**及其
Q/K 梯度（该接口总长 N ≤ 64）。

- 每个 token 只能读取**同一篇文档中位置不晚于自己**的 token
- 输出：`O = softmax(s · Q Kᵀ / 4) · V`
- 对齐损失：`L = mean_i( -log Σ_{j∈T_i} P_ij )`，T_i 为查询 i 的证据目标集
- Q/K/V 与上游梯度均为 16 维，float64
- 自定义算子前向、后向**均分块**计算，块长 `chunk_size ∈ [1, 32]`
- 不调用任何框架内置注意力算子，也不依赖内置 softmax/logsumexp 原语
  （online softmax 与 log-sum-exp 均手写，仅用 exp/log/最大值平移）
- 工作张量与跨前后向保存量规模 **O(N + C²)**，C 为块长；独立稠密参考不受此限制
- 容差：最大绝对误差均 **≤ 1e-8**，否则 `passed=false` 并给出原因
- 对齐损失在目标概率小到 float64 下溢为 0 时，损失与梯度仍为**有限值**

## 目录结构

```
app/
  attention.py            # 自定义分块 autograd 算子（online softmax 前向 + 分块反向）
  reference.py            # 独立稠密参考（完整 N×N 掩码与注意力矩阵，不受内存约束）
  alignment.py            # 分块对齐损失 autograd 算子（分块前后向 + 稳定 log 概率）
  alignment_reference.py  # 对齐损失独立稠密参考（手工掩码/softmax/log-sum-exp）
  alignment_verification.py # 对齐损失的数据生成、双路计算与误差核对
  verification.py         # 注意力的数据生成、双路计算与误差核对
  schemas.py              # 请求/响应模型与输入校验
  scenarios.py            # 内置边界情形样例
  main.py                 # FastAPI 入口
tests/                    # pytest：正确性、gradcheck、边界情形、内存与源码审计、API
Dockerfile
docker-compose.yml
```

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# Swagger 文档：http://localhost:8000/docs
# OpenAPI：     http://localhost:8000/openapi.json
```

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt   # torch 使用 CPU wheel 源
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q                         # 运行自测
```

## 接口

### `POST /verify`

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成 |
| `doc_lengths` | int[] | 正整数列表，总长 1..512 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

查询参数 `include_tensors`（默认 `true`）：是否回传完整输出/梯度张量。

curl 示例：

```bash
curl -X POST "http://localhost:8000/verify?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,"s":1.0}'
```

响应关键字段：`passed`、`tolerance=1e-8`、`errors`（四项最大绝对误差）、
`finite_ok`、`failure_reasons`。核对失败是业务结果而非 HTTP 错误，仍返回 200，
通过 `passed=false` 与 `failure_reasons` 明确表达；参数非法返回 422。

### `POST /alignment-loss`

按工程师标注的证据 key 计算分块对齐损失，并与独立稠密参考核对损失与 Q/K 梯度。
Q/K 由与 `/verify` 相同的种子/生成顺序确定性产生（同 seed、同文档长度时
两端口的 Q、K 完全一致；本接口不需要 V），16 维、float64、总长 **≤ 64**。

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0 |
| `doc_lengths` | int[] | 正整数列表，总长 1..64 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000，进入 softmax 的系数为 s/4 |
| `query_targets` | 对象数组 | 1..8 项；查询位置互不相同 |
| `query_targets[].query_position` | int | 查询 token 的批次位置 |
| `query_targets[].target_positions` | int[] | 1..4 个**互不相同**的目标 key 位置 |

**整单拒绝（HTTP 422，不产生任何部分梯度）**：目标与查询不属于同一篇文档、
目标位置晚于查询、查询/目标位置越界、查询位置重复、目标重复、基数超限
（查询 >8 或每处目标 >4）、总长 >64、s/chunk_size 越界等。

损失定义（Q 为查询位置集合，T_i 为其目标集合，P 为文档内因果 softmax）：

```
L_i = -log( Σ_{j ∈ T_i} P_ij )          L = (1 / |Q|) Σ_i L_i
```

curl 示例：

```bash
curl -X POST "http://localhost:8000/alignment-loss?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,5],"chunk_size":4,"s":1.0,
       "query_targets":[
         {"query_position":3,"target_positions":[0,1,3]},
         {"query_position":11,"target_positions":[7,11]}]}'
```

响应关键字段：`passed`、`tolerance=1e-8`、
`errors`（`loss/grad_q/grad_k` 三项最大绝对误差）、`loss`、
`target_probabilities`（每查询的 Σ P_ij）、`log_target_probabilities`
（其对数，概率下溢时仍有限）、`finite_ok`、`failure_reasons`，
以及可选的完整 `grad_q`/`grad_k`（N×16）。非法标注返回 422；
核对失败仍返回 200 并以 `passed=false` 表达。

### `GET /scenarios` / `POST /scenarios/run`

列出或运行内置边界情形；`POST /scenarios/run?name=<名称>` 只跑单个样例。

## 内置边界情形（题目要求）

| 名称 | 配置 | 覆盖点 |
| --- | --- | --- |
| `doc_cross_chunk` | 文档 7/11/5，块长 8，s=1 | 文档边界落在块内部，切块不改变归一化 |
| `fully_masked_block` | 单文档 16，块长 4，s=2.5 | 整块被因果屏蔽时安全跳过、无 NaN |
| `large_score` | 文档 33/31，块长 32，s=1000 | 大分数下 online softmax 前后向无溢出 |
| `combined_stress` | 文档 100×4+112（N=512），块长 7，s=999 | 三类情形叠加 + 块长不整除 |
| `minimal_single_token` | 两篇各 1 token，块长 32 | 最小批次与退化分块循环 |

```bash
curl -X POST "http://localhost:8000/scenarios/run"
```

## 实现要点

1. **掩码**：`valid(i,j) = (doc_i == doc_j) ∧ (pos_j ≤ pos_i)`，按块即时构造 C×C 布尔掩码。
2. **前向 online softmax**：逐 key 块维护每行的运行最大值 `m`、归一化分母 `d`
   与归一化输出 `O`。遇到新块时以 `exp(m_old − m_new)` 修正旧累积量，
   全程只做最大值平移，不保存 N×N 矩阵。
3. **整块屏蔽**：块内无有效位置时直接跳过；行级 `-inf` 通过 `isfinite` 判定，
   以 0 参与指数运算，杜绝 `−inf − −inf`、`0/0` 产生的 NaN。
4. **后向**：用保存的行最大值/分母在每个块上**重算**块内概率 P（仅 C×C），
   利用 `dP = P ⊙ (dO·Vᵀ − Σ_j P_ij dO_iᵀV_j)` 与 `dS = scale·dP` 分块累加
   `dQ/dK/dV`；保存量仅为 Q/K/V、doc_ids、行最大值、行分母、输出，均为 O(N)。
5. **数值稳定**：float64 全程；s=1000 时原始分数很大，最大值平移保证
   `exp` 输入 ≤ 0，前后向均为有限值。
6. **确定性**：同一 `seed` 生成完全相同的输入与结果。
7. **对齐损失前向**：先分块 online softmax 得到行最大值/分母；再对每个查询
   的目标集合单独做最大值平移的 log-sum-exp，
   `log P(T_i) = LSE(目标分数) − (行最大值 + log 行分母)`。
   目标概率即使小到 `exp` 整体下溢为 0，log 概率与损失仍是有限值。
8. **对齐损失后向**：在每个块上重算块内概率 P（仅 C×C），按解析公式
   `dL/dz_ab = (1/|Q|)(P_ab − 1[b∈T_a]·w_ab)`（z=sQKᵀ/4）累加 Q/K 梯度，
   其中 `w_ab = exp(z_ab − m_T)/Σ_{t∈T}exp(z_at − m_T)` 用目标集合自身的
   最大值平移计算，不经过 `P(T)` 除法，因此极小概率下梯度仍有限；
   非查询行的 softmax 雅可比行和为 0，Q 梯度严格为 0。保存量仅 Q/K、
   doc_ids、行最大值/分母、定长目标矩阵（N×4）与查询索引，均为 O(N)。

## 自测结果

`pytest -q` 共 59 项全部通过。原有 27 项（11 组参数化前向+梯度核对、
`torch.autograd.gradcheck`、块长不变性、确定性、掩码语义、大分数有限性、
保存张量线性规模审计、禁用算子源码审计、全部 API 端点与参数校验）保持通过；
新增 32 项覆盖对齐损失：

- 10 组参数化损失+Q/K 梯度核对（误差均 ≤ 1e-8，实测 1e-14 ~ 1e-17 量级），
  含块长 1、文档跨块、8 查询×4 目标上限、N=64、s=1000 大分数等
- `torch.autograd.gradcheck` 有限差分核对对齐算子前后向公式
- 块长不变性、同种子确定性、与手工 `-mean(log ΣP)` 定义逐项对照
- 目标概率 **float64 严格下溢为 0** 时损失/梯度仍有限且与参考一致
- 非查询行 Q 梯度严格为 0、一般 `grad_output` 的线性缩放
- 保存张量 O(N) 规模审计、AST 级禁用算子源码审计（分块与参考两侧）
- `/alignment-loss` API：正常核对、整单 422（跨文档/未来 key/重复/越界/
  基数超限/s 与块长越界）、与 `/verify` 的 Q/K 同序一致性、
  `/verify` 与 `/docs` 保持可用
