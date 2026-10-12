# Design

## Context

基线锁 sm_120a（`CMakeLists.txt:3-12`），upstream 是纯 sm_120a 代码库（动机见 proposal.md）。sm75（TU102）相对基线缺 6 项硬件特性，缺口面已逐文件实测：

| 特性 | 最低架构 | upstream 使用面 | 可回退性 |
|---|---|---|---|
| `cp.async` | sm80 | 55 文件 | 可（普通拷贝） |
| `mma_bf16` | sm80 | 37 文件 | 可（FMA 模拟，逐位精确） |
| `mma_tf32` | sm80 | 3 文件（GDN/Kimi recurrence） | 可（FMA 模拟） |
| **TMA** | sm90 | **47 文件** | **不可回退（无 no-op 形态），只能排除** |
| `mbarrier` | sm80 | 5 文件（全被 TMA 独占） | 随 TMA 排除 |
| `setmaxnreg` | sm90 | 2 文件（全 nvfp4） | 随 nvfp4 排除 |
| `mma_fp8` / `mma_nvfp4` | sm100 | 9 文件 | 格式物理不可用，排除 |
| PDL | sm90 | 16 文件（经 1 个共享头） | 可（no-op，消费方零改动） |

sm75 原生可用：`ldmatrix`（含 `_t`）、dp4a、`tanh.approx.f32`、`ex2.approx(.ftz).f32`、`cudaLaunchKernelEx`（PTX ISA / CUDA API 架构要求 ≤ sm75）。
**mma 原生面（task 2.1 核对结论，ptxas 12.4 逐一实测 + 2080Ti 运行验证）**：原生仅 `mma m16n8k8 f16`（f32 累加）与 `mma m8n8k16 s8`（s32 累加）两条，均运行验证正确。ptxas 拒绝 sm_75（"requires .target sm_80 or higher"）：`mma m16n8k16 f16`、`mma m16n8k16 bf16`、tf32 `m16n8k4`、tf32 `m16n8k8`、s8 `m16n8k16`、s8 `m16n8k32`。⇒ 本 Context 早期"mma_f16（m16n8k16）/mma_s8（m16n8k32）原生"的记载有误，已按实测更正；回退策略见 D2 与 D8。
硬件：68 SM、FP32 13.4 TFLOPS、FP16 张量核 53.8 TFLOPS、INT8 张量核 215.2 TOPS、带宽标称 616 GB/s（x99 底稿记 448，待实测）。
约束：upstream 无 install 目标，从源码构建树运行；27B 权重 W8A16（16.96 GiB），22G 显存下 KV 用 Int8Group64（33.8 KiB/tok）。

## Goals / Non-Goals

**Goals:**
- sm75 全树（含测试）编译通过、158 文件测试套件全绿（GDN 回放逐位一致是核心闸门）
- 27B W8A16 端到端正确：greedy 输出与参照逐 token 一致、NIAH 32k/64k 通过
- 所有 sm75 改动隔离在 `#if NINFER_SM75` / 守卫之后，86/120a 构建产物行为不变
- 独立部署 + 零成本回滚（fork 构建与 v2 工件不动）

**Non-Goals:**
- 性能调优（f16 张量核 / W8A8 / 软件流水）——另行变更，带实测数据再议
- q4/q5/q6 与 Kimi DA 的运行时启用（仅保证可编译）
- 多卡、并发 >1（本次 `--max-concurrency 1`）

## Decisions

**D1 替代代码落点：共享头内 `#if NINFER_SM75`（而非独立 .cu 文件或新目录）。**
4 个共享头（`src/ops/common/{mma,memory,math}.cuh`、`src/core/pdl.cuh`）是所有消费方的唯一入口，头内分支 = 消费文件零改动（PDL 的 16 个文件全部经 `if constexpr (JoinPdl)` 走头）。替代：新建 `sm75/` 目录用同名符号替换——需要改 60+ 消费方或加编译宏重定向，改动面大一个量级。fork 上同结构已在 x99 整树编译+运行验证过。

**D2 mma_bf16/tf32 默认路线：FMA 模拟（逐位精确），而非 f16 张量核。**
bf16→f32 无损、同序 FP32 累加 ⇒ 数学上输出必须与基线逐位一致，测试可机器验证。f16 张量核（13.4→53.8 TFLOPS）有数值差异，需要重跑全部数值测试重新立参照——正确性基线未立之前不切。P5 带实测数据再议。

**D3 TMA 处理：CMake 构建期排除，而非源码内分支。**
TMA 内核（47 文件）的 tensor map / 异步 tile 拷贝没有"关掉 TMA 还能编译"的 no-op 形态，头内 `#if` 只能挡掉引用、挡不掉内核本体。upstream 源清单逐家族显式（`sources.cmake`，无 GLOB）⇒ 排除 = 家族级条件，干净可审计。替代：给 TMA 内核写 sm75 双缓冲实现——工作量巨大且属性能范畴，归 P5 另议。

**D4 fp8/nvfp4 格式：排除。**
指令面 sm100+（`mma_fp8`/`mma_nvfp4`）+ TMA + setmaxnreg + 独立 non-RDC 库，物理不可用且无回退。27B 用 W8A16，运行时不经过这两家族。排除后 wrapper 的 if 链加 `#if !NINFER_SM75` 守卫，加载被排除格式工件时显式报错。

**D5 cp.async 默认：普通同步拷贝，而非手动预取流水。**
正确性优先，先立基线；双缓冲 overlap 损失的量级由 kernel 间隙实测决定（P5）。

**D6 工件：v2→v3 升级（官方脚本 `tools/upgrade_ninfer_v2_to_v3.py`）。**
`src/artifact/reader.cpp:31-34` 硬拒 v2，无兼容路径；脚本纯 stdlib、权重字节保留、写新文件。Mac 上执行，可与 P0–P1 并行。

**D7 分发适配：bf16 per-shape select 的 T≥64 TMA 分支 `#if !NINFER_SM75` 改道 mma/sliced_k。**
21 个 bf16 shape 文件 + 2 个 launch 头 + 35 个 wrapper。q8 家族（22 shape 文件）已核实无 TMA/fp8 引用，零改动。

**D8 mma_f16/mma_s8 回退：原生小 shape 分解，而非 FMA 模拟（task 2.1 核对落地）。**
核对结论（见 Context 更正）：sm75 原生 mma 只有 m16n8k8 f16 与 m8n8k16 s8。故 `mma_f16`（m16n8k16）= 2× 原生 m16n8k8：第一发 A={a0,a1}、B={b0} 累加 c0..c3，第二发 A={a2,a3}、B={b1} 继续累加——寄存器映射按 PTX ISA m16n8k16/m16n8k8 两个 fragment 表直接对应（m16n8k16 的 a2/a3 = m16n8k8 的 a0/a1 于 k 8-15，C/D 两 shape 布局相同），该分解已用随机输入探测在 2080Ti 上验证 maxerr=0。`mma_s8`（m16n8k32）= 4× 原生 m8n8k16：(a0,b0)→c0,c1（行 0-7 k 0-15）、(a1,b0)→c2,c3（行 8-15 k 0-15）、(a2,b1)→c0,c1（行 0-7 k 16-31）、(a3,b1)→c2,c3（行 8-15 k 16-31），整数运算与累加顺序无关，正确性由 L1 自测（2.2）验证。两条均走张量核，不经过 FMA 模拟。

## Risks / Trade-offs

- **[GDN 回放逐位一致测试红]** → 最大单点风险。预案：停在该闸门定位 F_fp 链条（recurrence/prepare 两阶段、回退实现是否统一、schedule 累加顺序）；短期修不动则先不开投机解码（投机是选项非必需，SpeculativeOptions 可空），主线部署不被卡。
- **[编译长尾超期（60–80 文件）]** → 保主线收敛：27B 运行时路径（q8 + bf16 非 TMA + GDN + attention + KV）必保；非 27B 家族（Kimi DA、q4/q5/q6）临时 CMake 排除，主线先通再逐个恢复。
- **[带宽偏低（实测 448 而非 616 GB/s）]** → decode 下限 38 ms/token（而非 27.6）；P5 提高 f16 张量核 / W8A8 选项优先级。用 `tools/hbm_bandwidth_probe.cu` 在 P0 后实测定局。
- **[SMEM opt-in 尺寸不匹配]** → upstream 内核按 99 KB opt-in SMEM 设计，sm75（Turing）上限更低 ⇒ 部分 tile 配置在 sm75 非法，需要 per-arch tile 参数（P2 长尾项）。
- **[单点机器（x99）]** → 源树在 Mac 可随时全量重建；fork 保留期间业务不中断。
- **Trade-off**：FMA 模拟使 sm75 的 bf16 GEMM 算力封顶 13.4 TFLOPS（f16 张量核 53.8、int8 215.2 未用）——为逐位精确性支付的已知代价，P5 用数据重审。

## Migration Plan

1. **工件**（Mac，并行）：`upgrade_ninfer_v2_to_v3.py` 升 v3；v2/v3 并存。
2. **试部署**（sm75 构建 + 测试全绿后）：rsync 独立源树 → x99 独立目录构建 → CLI greedy 参照对比 → 独立端口起 serve → HTTP 契约 smoke。fork 全程不动。
3. **正式切换**（长稳回归后）：客户端切新端口；fork 保留 1 周。
4. **回滚**：客户端切回 fork 端口，零数据迁移。

## Open Questions

- x99 实际带宽（616 vs 448 GB/s）——P0 后 `hbm_bandwidth_probe.cu` 实测；只影响 P5 优先级，不影响本变更 spec 与任务拆分。
- x99 上 fork 的进程管理方式（systemd 还是裸跑）——上机确认后正式部署照搬；不改变 spec。
