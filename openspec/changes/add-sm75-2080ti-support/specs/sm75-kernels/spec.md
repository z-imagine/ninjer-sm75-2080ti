# Spec Delta

## Purpose

在 2080Ti 上为缺失的硬件特性提供数值等价的替代实现——矩阵乘指令的等价必须逐位精确，异步拷贝与内核重叠以正确性优先的回退实现——并保证内核分发永不选中该架构不可用的 TMA 内核。

## ADDED Requirements

### Requirement: bf16 与 tf32 矩阵乘的逐位精确模拟
On sm75, bf16 and tf32 matrix-multiply instructions SHALL be emulated with floating-point FMAs such that results are bit-identical to a software reference for arbitrary inputs (bf16-to-fp32 conversion is lossless; tf32 inputs are rounded to tf32 before accumulation; accumulation order is preserved).

#### Scenario: 随机输入逐位一致
- **WHEN** 以随机矩阵输入运行 sm75 上的 bf16 与 tf32 矩阵乘模拟
- **THEN** 每个输出元素与软件参考实现逐位相等

#### Scenario: 27B 形状 GEMM 验证
- **WHEN** 运行覆盖 27B 模型实际权重形状（含 n14336_k5120、n5120_k17408 类）的 GEMM 单元测试
- **THEN** 输出与参考逐位一致

### Requirement: 异步拷贝回退
On sm75, asynchronous global-to-shared copies SHALL be replaced by synchronous plain copies that deliver identical data, so every kernel consuming them yields correct results.

#### Scenario: 经回退路径的内核正确
- **WHEN** 运行使用异步拷贝路径的 GEMM 与注意力单元测试
- **THEN** 结果与参考一致，无缺失数据或越界

### Requirement: 内核重叠特性回退为串行
On sm75, programmatic dependent-launch primitives SHALL be no-ops, and every kernel consuming them SHALL run serially with correct results.

#### Scenario: 消费内核重叠的内核正确
- **WHEN** 运行任何消费内核重叠原语的内核（GEMM、注意力、GDN 等）
- **THEN** 在 sm75 上串行执行且输出与参考一致

### Requirement: 分发路径不选中 TMA 内核
On sm75, kernel dispatch SHALL never select a TMA-based schedule. Large-batch bf16 GEMMs whose token count selects TMA on the baseline (T ≥ 64) SHALL be rerouted to an mma or sliced-k schedule with identical numeric results.

#### Scenario: 大批量 bf16 GEMM
- **WHEN** 在 sm75 上以不少于 64 个 token 运行 bf16 线性层（基线会选中 TMA 调度）
- **THEN** 实际执行 mma 或 sliced-k 调度，输出与参考逐位一致

#### Scenario: 小批量路径不变
- **WHEN** 在 sm75 上以少于 64 个 token 运行 bf16 线性层
- **THEN** 仍走基线的 GEMV/SIMT/mma 路径，输出与参考一致

### Requirement: 原生指令直接使用
Instructions already native on sm75 (ldmatrix, fp16 and int8 matrix-multiply m16n8k16/k32 shapes, dp4a) SHALL be used directly without emulation or fallback.

#### Scenario: int8 prefill 注意力原生执行
- **WHEN** 运行 int8 prefill 注意力（使用 int8 矩阵乘指令）
- **THEN** 在 sm75 上原生执行且结果正确
