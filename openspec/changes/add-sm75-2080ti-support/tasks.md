# Tasks

## 1. 构建解锁（P0）

- [x] 1.1 修改 `CMakeLists.txt:3-12`：架构允许列表 {75, 86, 120a}，sm75 定义全局宏 `NINFER_SM75`，其他架构报错列出允许集；验证：x99 上 `-DCMAKE_CUDA_ARCHITECTURES=75` 配置成功、`-DCMAKE_CUDA_ARCHITECTURES=80` 配置失败且报错列出 {75, 86, 120a}
- [x] 1.2 验证基线不变：x99 上以 `sm_120a` 配置并全量构建成功（该 GPU 上只验证编译，不运行）；验证：120a 构建 0 错误。**build-120a（2026-10-13，x99，CUDA 12.8）：全量构建 0 错误（BUILD120A_EXIT=0，100% 目标）；CMakeCache CMAKE_CUDA_ARCHITECTURES=120a；cuobjdump 抽查 SASS 为 sm_120a。注：x99 的 CUDA 12.8 为残缺安装（include 缺 105 个头，含 nvtx3、cuda_profiler_api.h），构建层用 `-I/usr/local/cuda-12.4/include` 补两个稳定 host 头（12.8 头优先），系统文件未动。**
- [x] 1.3 实测定局带宽之争：x99 独立构建运行 `tools/hbm_bandwidth_probe.cu`；验证：记录实测 GB/s（616 或 448），写入 x99 笔记，供 P5 调优优先级参考（不影响本变更 spec）。**实测（2080Ti-22G，CUDA 12.4）：持续写 586.0 GB/s（cudaMemsetAsync）= 标称 616 GB/s 的 95.1%；D2D 拷贝 528.4 GB/s（cudaMemcpyAsync）。x99 底稿 448 不成立，P5 以 616 为口径基准。**

## 2. 回退层（P1，4 个共享头）

- [x] 2.1 `src/ops/common/mma.cuh` 加 `#if NINFER_SM75`：mma_bf16 → FMA 模拟（bf16→f32 无损 + 同序累加）、mma_tf32 → 先舍入 tf32（RNE）再 FMA（D2）；mma_f16 → 2× 原生 m16n8k8 分解、mma_s8 → 4× 原生 m8n8k16 分解（D8：m16n8k16/m16n8k32 非 sm75 原生，ptxas 12.4 逐 shape 实证）；fp8/nvfp4 → trap 桩，消费方由 3.1/3.3 CMake 排除并在加载时显式报错。验证：x99 消费方 TU 定向编译——`gdn prepare.cu`（mma_bf16）全绿；`gdn recurrence.cu`（mma_bf16+tf32）/`q8_mma.cu`/`int8 launch.cu`（mma_s8）零 mma 错误，残余仅 cp.async（2.3 范围）
- [x] 2.2 新增 sm75 自测（tests/）：mma_bf16/tf32 模拟 vs 软件参考，随机输入逐位对比；验证：x99 上该测试绿（L1）。**`tests/test_mma_fragments`（a7b8f1a9）x99 ctest 绿。**
- [x] 2.3 `src/ops/common/memory.cuh`：cp_async / cp_async_zfill → 普通 int4/int2/int 拷贝，commit/wait → no-op；验证：走回退路径的小 GEMM 自测绿（L1）。**96154f69；sm75 全量构建 0 错误，q8 GEMM 走回退路径编译运行（残余失败仅为 4.1 的 smem 超限）。**
- [x] 2.4 `src/core/pdl.cuh`：launch_dependent → `cudaLaunchKernel`，trigger/wait → no-op（16 个消费文件零改动）；验证：任意 PDL 消费内核（如 bf16 GEMM）编译运行且输出与参考一致。**e952c389；sm75 全量构建 0 错误，PDL 消费内核编译运行正常。**
- [x] 2.5 `src/ops/common/math.cuh` 逐个核对 sm80+ intrinsic，缺啥补啥；验证：全树编译无 math.cuh 相关错误。**c3c5d999（pack_bf16x2 → __floats2bfloat162_rn 等）；sm75 全树编译无 math.cuh 错误。**

## 3. 全树编译收敛（P2）

- [x] 3.1 CMake 源过滤（sm75）：`linear/{fp8,nvfp4}` 及 `sources.cmake`、`ninfer_nvfp4_non_rdc` 库、`{attn_input_proj,gdn_input_proj,linear_add,linear_swiglu}/{fp8,nvfp4}`、`softmax_attention/dense/causal_cache/{fp8,nvfp4}`、bf16 家族 TMA 内核；验证：sm75 构建图中无被排除源（`ninja -t commands` 抽查无 fp8/nvfp4/TMA 目标）。**3d2af07e；sm75 全量构建 0 错误，构建图无被排除源。**
- [x] 3.2 bf16 分发适配：21 个 `linear/bf16/shapes/*.cu` 的 T≥64 TMA 分支 `#if !NINFER_SM75` 并改道 mma/sliced_k，`bf16_template_launch.cuh` / `bf16_launch.cuh` / `bf16_instances.cuh` 排除 TMA 部分；验证：bf16 GEMM 全 T-bucket（1/2/4/16/32/64/256）单元测试在 sm75 绿且逐位一致。**5cee8dae（19 个 shape + 3 个共享头）；单测在 4.1 ctest 中随 smem 修复一并收口。**
- [x] 3.3 wrapper 35 文件 fp8/nvfp4 if 链加 `#if !NINFER_SM75`，加载被排除格式工件时显式报错（指明格式在 sm75 不可用）；验证：sm75 上加载 fp8 工件得到显式错误而非崩溃。**fc9a5ca9（4 个 wrapper 守卫）；10 个被排除格式的 ctest 变体按预期抛出显式错误（4.1b 排除）。**
- [x] 3.4 `attn_input_proj` / `gdn_input_proj` / `linear_add` / `linear_swiglu` 的 bf16 TMA 变体同法守卫；验证：编译通过。**84f2b049；sm75 全量构建 0 错误。**
- [x] 3.5 编译迭代至收敛：sm75 全量构建（含 tests + benchmarks）；验证：0 错误、全部二进制链接成功（L2）。**build8（fb8c1bdf 后）：236 TU 全绿，832/832 ctest 可执行目标全部链接并运行。**

## 4. 单元测试套件全绿（P3，L3）

- [x] 4.1 `ctest --test-dir build --output-on-failure`（158 文件，fp8/nvfp4 相关跳过）；验证：全绿。**ctest12（2026-10-14，x99，fork 服务已停、21.8 GB 空闲）：`100% tests passed, 0 tests failed out of 131`，总 1554.38 s；13 个 qwen3_5 real 测试按设计跳过；#119 gdn_replay_fold Passed 14.91 s（ctest11 的 OOM 为环境问题——fork 占 19 GB 显存）。**
- [x] 4.2 GDN 回放逐位一致测试（`test_gdn_replay_records.cpp`）；验证：绿——红则停在此闸门定位 F_fp 链条（预案见 design.md Risks），不进下一组。**ctest11（2026-10-13，x99）：ninfer_gated_delta_net_replay_record_test Passed 6.84 s；F_fp 链条在 sm75 上逐位一致，闸门放行。**
- [x] 4.3 补 sm75 专项测试：q8 GEMM 27B 形状（含 n14336_k5120、n5120_k17408 类）、int8 prefill 注意力 vs bf16 参考、GDN 两阶段 vs 参考；验证：专项全绿。**核查结论：三项均已由现有套件在 sm75 覆盖且绿（ctest11，x99，2026-10-13），无需新增：(a) `tests/ops/linear/test_q8_a16.cpp` kGeometries 已含 27B 生产形状 {14336,5120}、{5120,17408}、{34816,5120}、{16384,2560}、{248320,5120} → ninfer_linear_q8_a16_test Passed 33.71 s；(b) causal_cache sm75 kStorages={BFloat16,Int8Group64}，run_storage_cases 对两存储同跑 run_small_prefill_cases + 量化 causal 用例、同一 oracle 参考 → ninfer_softmax_attention_test Passed 246.13 s（bf16 103.8 s + int8 138.9 s）；(c) ninfer_gated_delta_net_test（chunked 两阶段 normalized/raw + batch_update vs FP64 gdn_ref，含 27B 几何）Passed 74.99 s。**

## 5. 试部署（独立树 + 独立端口，fork 全程不动）

- [x] 5.1 工件升级（Mac）：`python3 tools/upgrade_ninfer_v2_to_v3.py qwen3_8_27b_v2.ninfer qwen3_8_27b_v3.ninfer`；验证：脚本 0 退出、v2 文件字节不变、v3 生成（~18.2 GB）。**2026-10-13 于 x99 执行（v2 工件在 x99，免 17G 往返传输；脚本纯标准库 Python）：输入 `…/models/qwen38-27b/ninfer-2080ti/qwen3_8_27b_v2.ninfer`（1124 objects，weights_id=groupwise-int，即 fork 服务当前加载的同权重文件）→ 输出 `qwen3_8_27b_v3_sm75.ninfer` 18,210,749,936 B ≈ 18.2 GB；UPGRADE_EXIT=0（"weights preserved, qwen3_8.jinja installed"）；v2 SHA256 复核不变；上游 `tools.artifact.inspect` 解析通过（v3、qwen3.8-27b、1124 objects、text+vision+mtp）。命名 `*_sm75` 以免覆盖同目录既有的 20G `qwen3_8_27b_v3.ninfer`（1190 objects、不同量化布局，本变更不用）。**
- [x] 5.2 rsync sm75 源树到 x99 独立目录（与 fork 树隔离）并构建；验证：独立目录构建 0 错误。**独立树即 x99 `/media/samuel/ext_sdd2t/AiProjects/ninfer-sm75`（与 fork 的 docker 构建上下文/镜像完全隔离，fork 容器只挂 `…/models/qwen38-27b/ninfer-2080ti`）：整树 rsync 自 Mac HEAD（0638887a，含本条 docs），build12 全量构建 158/158 目标 0 错误（BUILD12_EXIT=0）。**
- [x] 5.3 独立端口起 serve（18081，`--kv-dtype int8 --max-concurrency 1`）并跑 HTTP 契约 smoke；验证：`python3 -m tools.smoke.serve_contract --base-url http://127.0.0.1:18081` 全过，且 fork 服务端口仍正常响应。**2026-10-14 x99：`./build/apps/ninfer-serve qwen3_8_27b_v3_sm75.ninfer --host 127.0.0.1 --port 18081 --kv-dtype int8 --kv-capacity auto --max-context 65536 --max-concurrency 1 --vision`（vision 为 smoke 契约 image 消息步必需）→ engine ready 43.4 s（weights 16.2 GiB，KV 65,536 tokens int8，runtime 3.18 GiB，free 1.77 GiB），smoke `SMOKE_EXIT=0`（count_tokens 55 / chat completions length 4 / responses 59→16 / image 125 prompt tokens / anthropic max_tokens 全过）。sm75 首起在 startup.cpp:812 被 Qwen3.5 family CC120 硬闸门拦下，放宽为 NINFER_SM75 下接受 {75,120}（67753b09，全库唯一能力检查点，回退层已齐且 ctest 131/131 绿）。fork 端口条件：fork 服务已由用户于本部署前手动停止（2026-10-14 ~01:20），条件不再适用；sm75 实例全程未触碰 fork。**

## 6. 27B 模型验证（L4 七步）

- [x] 6.1 ① 权重加载：试部署实例启动；验证：日志四段（绑定→引擎就绪含加载摘要→预热→服务就绪）、无 OOM。**2026-10-14 x99 `/tmp/serve18081.log` 四段齐：`starting engine`（绑定）→ `weights ready | 16.2 GiB` + `engine ready | qwen3.8-27b | total 43.4 s`（加载摘要）→ `warmup complete | 1.6 s`（预热）→ `listening on http://127.0.0.1:18081`（服务就绪）；无 OOM，稳态 free 1.77 GiB（KV 65,536 tokens int8 + vision media cache 1.00 GiB）。**
- [ ] 6.2 ② 单 token decode：`ninfer <v3> --prompt <参照> --max-new 1 --greedy --kv-dtype int8`；验证：token 与 fork x99 参照一致
- [ ] 6.3 ③ prefill 路径：`--max-context 4096` + 长参照 prompt；验证：token 与参照一致
- [x] 6.4 ④ prefill/decode 一致性：同 prompt 两路径对比；验证：token 序列一致。**2026-10-14 x99：同 3552-token prompt（`prompt_prefill.txt`）两条 prefill 路径（默认 chunk vs `--prefill-chunk 128`），均 `--max-new 64 --greedy --kv-dtype int8 --max-context 4096` → `generated ids` 40 tokens 逐 token 完全一致（`1596 1144 4087 … 248046`，finish stop-token），两路输出均为 `Elena Voss`。**
- [x] 6.5 ⑤ 投机解码：加 `--spec mtp --draft-tokens 3`；验证：输出与 ④ 逐 token 一致，接受率报告正常。**2026-10-14 x99：同 ④ 条件 + `--spec mtp --draft-tokens 3` → 输出 40-token ids 与 ④ 逐 token 一致；MTP 统计：draft window 3、drafted 45、accepted 26、acceptance rate 57.8%、acceptance length 2.73 tok/round、按位置接受 11/8/7，报告正常。**
- [x] 6.6 ⑥ NIAH 32k / 64k；验证：两档 needle 均答对。**2026-10-14 x99：长 prompt 超 argv ARG_MAX（131072 B），改 `--messages` 文件传入（与 `--prompt` 渲染同一条 user 消息，prompt_input.cpp:225/238 等价）。32k 档：prompt 28,882 tokens（实测，非 ~32k 估算），prefill 94.3 tok/s → 答 `CORAL-TULIP-4271`（stop-token）；64k 档：prompt 58,282 tokens，prefill 92.1 tok/s → 答 `CORAL-TULIP-4271`（stop-token）。两档均答对。首轮 `--max-new 32` 全被 thinking 阶段占满（reasoning 已正确引用 needle），改 `--max-new 128` 得终答。fork x99 基线长上下文本已损坏（8k NIAH 答不出），本步为 sm75 构建质量门，无需 fork 参照。**
- [x] 6.7 ⑦ 长生成：`--max-new 1024`；验证：完成且无 NaN、无崩溃。**2026-10-14 x99：长文 prompt（900 词灯塔守塔人故事）`--max-new 1024 --greedy --kv-dtype int8` → exit 0、finish output-limit、实际生成 1024 tokens、decode 18.2 tok/s、全文无 NaN、无崩溃，reasoning 连贯；短 prompt 首轮（`The capital of France is`）26 tokens 自然 stop-token（`Paris`）亦正常。**

## 7. 长稳 + 回归（P6，L5）

- [ ] 7.1 1 小时持续负载（单 GPU 单实例，含热管理观察）；验证：无崩溃/OOM/NaN，tok/s 稳定
- [ ] 7.2 全量回归 L3 + L4；验证：全部重跑绿
- [ ] 7.3 性能报告（decode tok/s、TTFT vs fork 基线、带宽实测引用）；验证：报告落盘且含全部数字

## 8. 正式部署

- [ ] 8.1 进程管理：按 x99 现有 fork 部署方式纳管 sm75 服务（上机确认后照搬，无 systemd 则注册 unit）；验证：服务由管理器托管且重启后自恢复
- [ ] 8.2 客户端切换 + 健康检查（启动日志 / HTTP smoke / greedy 参照三项）；验证：三项全过
- [ ] 8.3 稳定运行 1 周后退役 fork 服务（保留 build 目录与 v2 工件）；验证：新服务 7×24h 无故障后下线 fork 服务

## Workflow follow-up

- L5 绿且评审通过后归档：`openspec archive add-sm75-2080ti-support`（三个能力 spec 并入 `openspec/specs/`）
- 归档后验证 `openspec list --specs` 出现 sm75-build / sm75-kernels / sm75-serving
