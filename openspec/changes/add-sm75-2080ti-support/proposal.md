# Proposal

## Why

NInfer 基线锁死 sm_120a（`CMakeLists.txt:3-12` 拒绝其他架构），x99 上的 2080Ti 22G（sm75/TU102）缺 `cp.async`、`mma_bf16/tf32`、TMA、mbarrier、setmaxnreg、PDL 六项硬件特性。目标是在**不修改基线行为**的前提下，让 upstream 代码原样在 sm75 上编译、测试通过并服务 Qwen3.8-27B（W8A16 工件 16.96 GiB，BF16 装不进 22G）。

## What Changes

- **构建**：`CMAKE_CUDA_ARCHITECTURES` 允许列表 {75, 86, 120a}；sm75 构建定义全局宏 `NINFER_SM75`；CMake 层按家族排除 fp8/nvfp4 源文件、TMA 独占内核、`ninfer_nvfp4_non_rdc` 库（源清单逐家族显式，无 GLOB，排除是干净的）
- **内核回退层**（4 个共享头，`#if NINFER_SM75`）：`mma_bf16/mma_tf32` → FMA 模拟（**逐位精确**）；`cp.async` → 普通拷贝；PDL（`src/core/pdl.cuh`）→ no-op，16 个消费文件零改动；`mma_f16/mma_s8/ldmatrix` sm75 原生直接使用
- **分发适配**：bf16 per-shape select 的 TMA 分支（T≥64）在 sm75 改道 mma/sliced_k schedule；wrapper 35 文件的 fp8/nvfp4 if 链加 `#if !NINFER_SM75` 守卫
- **部署**：x99 上从源码构建树运行 `ninfer-serve` / `ninfer`（upstream 无 install 目标），独立端口试跑，既有 fork 构建保留为回滚

**不改变**：120a 基线的所有行为（所有改动都在 `#if NINFER_SM75` / 守卫之后，86/120a 构建产物不变）。

## Capabilities

### New Capabilities

- `sm75-build`: 构建系统对 sm75 的支持——架构允许列表、`NINFER_SM75` 宏、不支持格式（fp8/nvfp4）与 TMA 独占内核的构建期排除、运行时加载被排除格式工件时的显式拒绝
- `sm75-kernels`: 内核级等价层——mma_bf16/tf32 的逐位精确 FMA 模拟、cp.async 回退、PDL no-op、sm75 分发路径永不选中 TMA 内核
- `sm75-serving`: 2080Ti 上 27B W8A16 的端到端推理与部署——GDN replay 逐位一致、prefill/decode 一致性、greedy 输出与参照逐 token 一致、长上下文 NIAH、独立端口试跑与零成本回滚

### Modified Capabilities

<!-- 无：openspec/specs/ 为空，本变更只新增能力 -->

## Impact

- **代码**：`CMakeLists.txt`、`src/ops/common/{mma,memory,math}.cuh`、`src/core/pdl.cuh`、`src/ops/linear/{bf16,fp8,nvfp4}/**`、`src/ops/{attn_input_proj,gdn_input_proj,linear_add,linear_swiglu}/{fp8,nvfp4}/**`、`src/ops/wrapper/*.cpp`、`src/ops/softmax_attention/dense/causal_cache/{fp8,nvfp4}/**`
- **工件**：v2 工件需先用官方 `tools/upgrade_ninfer_v2_to_v3.py` 升 v3（`src/artifact/reader.cpp:31-34` 拒收 v2）；v2/v3 并存
- **依赖**：无新依赖；CUDA 工具链 13.1（upstream 已验证）
- **部署**：x99 新增独立源树 + build 树 + 服务端口；fork 构建与 v2 工件全程不动
- **范围外**（另行变更）：性能调优（f16 张量核 / W8A8 / 软件流水，P5 带实测数据再议）、q4/q5/q6 与 Kimi DA 的运行时启用（本次仅保证可编译）
