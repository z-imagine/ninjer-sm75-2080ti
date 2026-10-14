# sm75 (2080 Ti) Docker 部署

`-DCMAKE_CUDA_ARCHITECTURES=75`（CMake 自动加 `NINFER_SM75=1`）。模型工件不在镜像里，宿主只读挂载到 `/models`。

**构建基础**：ubuntu:24.04 + CUDA **12.4.131** deb（与 x99 验证过的宿主安装同版本：nvcc 12.4.131-1）。
不走 `nvidia/cuda` 官方镜像，原因：
- `nvidia/cuda:12.4.*` 只有 ubuntu22.04 标签，没有 ubuntu24.04；
- ubuntu22.04 自带 libcurl 7.81，低于 `cmake/Dependencies.cmake` 强制的 `>=7.85`。

CUDA deb 来自 x99 的离线本地仓库 `/var/cuda-repo-ubuntu2204-12-4-local`（12.4.131 runfile 生成），
通过 compose 的 `additional_contexts` 以命名上下文 `cuda-repo` 传入 Dockerfile。
**所以 `docker compose build` 只能在有该仓库的机器上跑（当前即 x99）。**

与 fork 服务（`ninfer-27b-gdnfix`）共用对外端口 **15026**，同一时间只能起一个：

| 服务 | model 名 | 起 | 停 |
|---|---|---|---|
| sm75 | `qwen3.8-27b-sm75` | `docker compose up -d` | `docker compose down` |
| fork | `qwen3.8-27b` | `docker start ninfer-27b-gdnfix` | `docker stop ninfer-27b-gdnfix` |

客户端固定打 `http://<host>:15026/v1`，`model` 字段按当前在跑的服务填。

## 构建（一次性）

```
docker compose build
```

## 运行

```
docker compose up -d
docker compose logs -f
```

日志四段：bind artifact → engine ready → warmup → listening on 8080。
27B 加载实测约 43 s，healthcheck `start_period` 给了 90 s。

## 验证

1. 四段日志齐全
2. `tools/smoke.serve_contract` 打 15026 → `SMOKE_EXIT=0`
3. 1-token greedy 请求 → token 760（L4 6.2 参照）
