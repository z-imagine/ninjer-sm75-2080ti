# Spec Delta

## Purpose

保证 27B W8A16 模型在 2080Ti 上端到端推理正确——GDN 三路径逐位一致、greedy 输出与参照一致、长上下文可用——并以独立部署形态提供服务，保留零成本回滚。

## ADDED Requirements

### Requirement: GDN 层三路径逐位一致
For GDN (gated delta net) linear-attention layers, prefill, decode, and verify paths SHALL produce bit-identical finite-precision transfer across chunked preparation and recurrence, so speculative-decoding replay remains numerically valid.

#### Scenario: GDN 回放测试通过
- **WHEN** 在 sm75 上运行测试套件中的 GDN 回放记录测试
- **THEN** 测试通过，prefill/decode/verify 三路径的有限精度转移逐位一致

### Requirement: 27B 权重加载
The sm75 build SHALL load the 27B W8A16 artifact (16.96 GiB) within 22 GB of device memory without out-of-memory, and report a successful engine start.

#### Scenario: 服务启动
- **WHEN** 以 sm75 构建启动 27B 服务
- **THEN** 启动日志依次出现端口绑定、引擎就绪（含加载摘要）、预热完成、服务就绪，且无显存溢出

### Requirement: greedy 输出与参照一致
With deterministic greedy decoding (temperature 0, exact argmax), the sm75 build SHALL emit a token sequence identical to the recorded reference (fork x99 baseline) for the same prompt.

#### Scenario: 固定 prompt 对比
- **WHEN** 以 greedy 模式对固定参照 prompt 运行 sm75 构建
- **THEN** 输出 token 序列与参照逐 token 一致

#### Scenario: prefill/decode 一致性
- **WHEN** 同一 prompt 分别经单 token decode 与长 prefill 路径运行
- **THEN** 两条路径的输出 token 序列一致

### Requirement: 投机解码运行时一致
When MTP speculative decoding is enabled, output SHALL remain token-identical to the non-speculative greedy output, and the draft acceptance report SHALL be produced normally.

#### Scenario: 开启 MTP 投机
- **WHEN** 以 greedy 模式加 MTP 投机（3 个 draft token）运行
- **THEN** 输出与非投机路径逐 token 一致，接受率报告正常

### Requirement: 长上下文可用性
At context lengths of 32k and 64k tokens, the service SHALL answer needle-in-a-haystack probes correctly, and a 1024-token generation SHALL complete without NaN, crash, or hang.

#### Scenario: 32k NIAH
- **WHEN** 在 32k 上下文中埋入 needle 并提问
- **THEN** 回答正确（含 needle 内容）

#### Scenario: 64k NIAH
- **WHEN** 在 64k 上下文中埋入 needle 并提问
- **THEN** 回答正确

#### Scenario: 长生成稳定
- **WHEN** 连续生成 1024 个 token
- **THEN** 生成完成，无 NaN、无崩溃、无挂起

### Requirement: 独立部署与零成本回滚
The sm75 build SHALL be deployed as an independent source/build tree and service port that does not modify the existing fork deployment; rollback SHALL be a client-side port switch with no data migration, and the v2 artifact SHALL coexist with the v3 artifact.

#### Scenario: 试跑不影响既有服务
- **WHEN** 在 x99 上以独立端口启动 sm75 服务
- **THEN** 既有 fork 服务在其端口上继续可用，两者互不影响

#### Scenario: 回滚
- **WHEN** 将客户端切回 fork 服务端口
- **THEN** 服务立即恢复，fork 构建与 v2 工件无需任何修改

### Requirement: 每次启动健康检查
Every service start SHALL be validated by three checks: the four startup log stages, the HTTP contract smoke (OpenAI, Anthropic, and streaming response surfaces), and one greedy prompt whose output matches the recorded reference.

#### Scenario: 健康检查通过
- **WHEN** 服务启动完成后执行健康检查
- **THEN** 启动日志、HTTP 契约 smoke、greedy 参照对比三项全部通过
