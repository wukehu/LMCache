# vLLM / LMCache 定制源码管理

个人 Fork：
- https://github.com/wukehu/vllm
- https://github.com/wukehu/LMCache

## 分支职责

| 仓库 | 默认分支 | 上游 | 用途 |
|---|---|---|---|
| vLLM | main | vllm-project/vllm:main | 纯上游同步，不加入定制补丁 |
| LMCache | dev | LMCache/LMCache:dev | 纯上游同步，不人为改名为 main |

`origin` 指向个人 Fork，`upstream` 指向官方仓库。定制修改在 `feature-*` 分支。

## 补丁索引

| 仓库 | 分支 | 原始基线 / 内容 |
|---|---|---|
| vLLM | feature-metrics-preserved-20261010 | 87668ab69b3e；现有 metrics 修改 |
| vLLM | feature-v016-negative-counter-preserved-20261010 | 89a77b10846f；原 v0.16 回移工作区的 9 个源码/测试文件 |
| vLLM | feature-mla-w8a8-v026 | v0.26.0；MLA W8A8 context-prefill 保留浮点激活，避免转成 int8 后再次量化 |
| vLLM | feature-runtime-pp-exp30-preserved-20261010 | v0.26.0 + MLA 修复 + exp30 清单中 6 个 NIXL/PP/scheduler 文件 |
| LMCache | feature-mp-runtime-preserved-20261010 | 1a4b40b1d79b；原 MP 开发工作区 23 个源码/测试文件 |
| LMCache | feature-mp-pp-preserved-20261010 | 1a4b40b1d79b；原 PP 工作区 24 个源码/测试文件 |
| LMCache | feature-partial-prefix-fallback-v055 | 05a013b29da7 / 0.5.5；partial-prefix fallback 的 Tensor / data_ptr 参数兼容修复 |
| LMCache | feature-runtime-pp-exp30-preserved-20261010 | 05a013b29da7；exp30 安装清单中 44 个 LMCache 文件 |

两个 `feature-runtime-pp-exp30-preserved-20261010` 是配套源码快照。各补丁分支 `custom-patches/manifest.json` 记录基线、文件哈希和验证边界。

本次是源码保全和管理迁移，未重新构建镜像、运行 GPU 回归或宣称所有实验已验收。原 MP 工作区、PP 工作区和镜像 overlay 是不同发展状态，不能未经验证直接互相覆盖或合并。公开 PP 测试夹具的两个内部节点 IP 已替换为文档示例地址；原本地文件未改。

## 同步上游

先确保当前管理克隆干净，再执行（LMCache 将 main 换成 dev）：

```sh
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

只允许快进更新默认分支；有分叉时先调查，不使用 reset 或强推。补丁分支按其原始基线维护，不随默认分支更新自动重放。

## 后续定制与发版

1. 从目标发布 tag / SHA 建立版本化集成分支，例如 `feature-runtime-v026-lmc055`。
2. 从保留分支选取需要的修改，拆成可审阅的逻辑提交；先比较差异，避免重复应用同一修复。
3. 优先验证 MP 生命周期、PP > 1、partial-prefix fallback、NIXL P/D 与 MLA W8A8；记录拓扑、请求结果和错误/重启情况。
4. 验证通过后固定 vLLM SHA、LMCache SHA、Dynamo SHA、Torch/CUDA ABI 和基础镜像 digest，构建不可变镜像。
5. 发布记录同时保存源码 SHA、镜像 digest 与验收证据。源码已迁移不代表已有镜像自动包含新增提交。

不在纯上游默认分支积累定制修改；不把保留快照直接标记为生产发布。
