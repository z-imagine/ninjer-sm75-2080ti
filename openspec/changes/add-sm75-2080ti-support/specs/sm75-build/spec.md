# Spec Delta

## Purpose

让构建系统在 2080Ti（sm75）上可配置、可编译，并显式排除该架构物理不支持的权重格式与内核，同时不影响既有 86/120a 构建行为。

## ADDED Requirements

### Requirement: 构建接受 sm75 架构
The build SHALL accept CUDA architecture 75 in its allowlist together with 86 and 120a. Configuring with any other architecture SHALL fail with an error that names the allowed set.

#### Scenario: sm75 配置成功
- **WHEN** 以 `-DCMAKE_CUDA_ARCHITECTURES=75` 执行项目配置
- **THEN** 配置成功并继续编译，不再因架构不在允许列表而失败

#### Scenario: 未授权架构被拒绝
- **WHEN** 以 75/86/120a 之外的架构值执行项目配置
- **THEN** 配置失败，错误信息列出允许的架构集合

### Requirement: sm75 构建排除不支持的格式
An sm75 build SHALL NOT compile fp8 or nvfp4 weight-format sources, TMA-only kernels, or the nvfp4 non-RDC library. q8, bf16 (non-TMA paths), GDN, and Kimi delta-attention sources SHALL remain compiled.

#### Scenario: 全量构建成功
- **WHEN** 在 sm75 目标上执行全量构建（含测试与基准程序）
- **THEN** 构建成功且可链接出全部应用与测试二进制

#### Scenario: 运行时加载被排除格式的工件
- **WHEN** 用 sm75 构建加载一个需要 fp8 或 nvfp4 权重的工件
- **THEN** 程序以显式错误拒绝并指明该格式在 sm75 不可用，而不是静默误路由或崩溃

### Requirement: 基线构建不受影响
Builds for architectures 86 and 120a SHALL keep behavior identical to the pre-change baseline: all weight formats and kernel paths remain available and no sm75 fallback code is visible to those builds.

#### Scenario: 120a 构建行为不变
- **WHEN** 以 sm_120a 配置并构建
- **THEN** 所有格式（含 fp8/nvfp4）与 TMA 内核路径照常可用，行为与变更前一致
