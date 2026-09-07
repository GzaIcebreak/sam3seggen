# SegviGen 微调改动报告（v1 → v6 + GeoSAM2 对照，截至 2026-09-06）

本文汇报我们在 SegviGen `full_seg_w_2d_map` 之上做的全部改动：LoRA 域适配（v1/v2）的结果与结论、
PartVerse 部件名重标注与补渲染、SAM3 概念库（Stage A/B）、图例 token + 双视角条件（Stage D）的设计，
与原论文损失函数/条件结构的逐项对照（第 6 节），显式颜色监督 v4 / 逐 token 注入 v5 的结果（第 7–8 节），
轨迹探针与 v6 轨迹训练（第 9 节），以及统一 3D 评测与 GeoSAM2 对照（第 10 节）。代码入口见同目录 `README.md`。
数据集与训练产物（`datasets/`、`finetune/runs/`、`wandb/`）不进 git。

---

## 0. SegviGen 是什么、现在卡在哪

### 0.1 原理与工作方式

SegviGen（arXiv 2603.16869）的思路是**把一个 3D 生成模型改造成分割器**，而不是训练一个分割网络。
本仓库的基座是 TRELLIS.2 的纹理生成 DiT `slat_flow_imgshape2tex_dit_1_3B_512_bf16`（1.3B，见 `finetune/model.py`）。

一次 `--two_d_map` 推理的数据流：

```
input.glb ─► 体素化 (o_voxel) ─► shape_slat / input_tex_slat      稀疏结构化 latent (SLAT)
                                          │
2D 引导色图 map.png ─► BiRefNet 去背 ─► DINOv3 ViT-L/16 ─► cond [1, ~1029, 1024]
                                          │
                          rectified flow 去噪 (v-pred) ─► output_tex_slat
                                          │
                     纹理解码 ─► 每 voxel 一个颜色 ─► 最近邻调色板 ─► 部件标签 ─► 拆 mesh
```

四个要点：

- **分割被表述成"给体素上色"**。模型的输出不是 logits 而是一张 3D 纹理，它要做的是把 2D 色图里的
  配色复制到 3D 表面上；之后取最近邻调色板色，才变成部件标签。
- **训练目标是 rectified flow 的速度回归**：`target = (1-σ_min)·noise − x_0`，逐 token MSE
  （`finetune/train.py` 的 `flow_loss`）。这是**重建损失，不是分类损失**——它直接约束几何/纹理的还原，
  对"颜色归属对不对"只有间接约束。这一条是第 1 节全部测量结果的根源。
- **条件只有一路**：`cond` 是 DINOv3 对**单张** 512×512 色图的 patch token，3D 侧 latent 靠 cross-attention
  去读它；`shape_slat` 走另一条 concat 通道喂几何（`structured_latent_flow.py`）。
- 本仓库在上游之上加的是**语义入口**：用 SAM3 按文本提示词分割渲染图，掩码上色后当 2D 引导图
  （`sam3_to_2dmap.py`）。所以"部件叫什么"只存在于 SAM3 那一步，进 SegviGen 之前就被压成了颜色。

### 0.2 优点

- **3D 一致性是天生的**。去噪发生在 3D latent 上，不是逐视角分割再融合，所以没有多视角标签冲突，
  也不需要跨视角对齐。
- **部件数严格可控**。提示词组（`body=head+face+hand`）加 `--unassigned_to`，输出部件数恒等于提示词数。
- **保留原始拓扑与贴图**。标签通过面心查体素回到原 mesh，UV 和 albedo 不动，不使用 SegviGen 自己的 remesh
  （`data_toolkit/interactive_partition.py`）。
- **对未见资产有泛化**。留出集 28 件里基座归属对 27 件（`eval_fidelity.py` 新口径 0.953），无需针对性训练。

### 0.3 缺点

1. **唯一的语义入口不传语义。** 色图配色是 `random_palette` 随机生成的。这个设计本身对——逼模型复制
   区域结构而不是背"红色=头"——但副作用是**颜色通道在设计上就不承载语义**。没有文本条件
   （TRELLIS.2 自带 `TextConditionedMixin`，没接）、没有相机 pose、没有 2D→3D 硬对应。
2. **所以"手贴着棍子"这类情况没有先验可用**，模型不知道该按哪条边界切。这是 monk 一类资产的主要失败模式。
3. **目标函数与要优化的东西错位。** 见 1.4：v-pred MSE 降了 8%，颜色归属反而从 27/28 变成 26/28。
4. **上游误差直接传导**。SAM3 侧 10.3% 的部件没被任何 prompt 绑定而变灰、边界粗，SegviGen 只会忠实复制。
5. **交互模式的掩码不构成划分**。monk 上 12 次查询各认领 4%–59% 的体素，85% 的体素被多于一个掩码认领。

### 0.4 对比 P3-SAM

P3-SAM（arXiv 2509.06784，腾讯混元）是**原生 3D** 的部件分割。**本地没有它的代码**，下面的机制描述来自论文：
点云 + 法向经 Sonata（自监督预训练的 PointTransformerV3）提逐点特征，单个正点提示送进两阶段多头
分割器，一次出 3 个不同粒度的候选 mask，再由 IoU 预测头自动选最好的一个；全自动分割则用 FPS 采提示点、
NMS 合并冗余 mask，最后把点级 mask 投影到面。训练数据约 370 万 artist mesh。

| 维度 | SegviGen（本仓库 + SAM3） | P3-SAM（论文说法） |
|---|---|---|
| 输入 | GLB → 体素化 + 单张渲染色图 | 从 mesh 采样的点云 + 法向 |
| 是否原生 3D | 否，语义要过 2D 渲染 | 是，不依赖任何 2D 基础模型 |
| 是否需要选视角 | 需要，且对朝向敏感（`--front_view` 专为此存在） | 不需要 |
| 提示形式 | 文本（经 SAM3 转成颜色） | 单个正点 |
| 语义 / 命名 | **有**，输出部件带名字 | **无**，class-agnostic，只给几何划分 |
| 粒度歧义 | 无机制，粒度由提示词决定 | 多头出 3 个粒度 + IoU 头自动选 |
| 输出是否构成划分 | 2D 引导模式是；交互模式否，需后处理 | 是（NMS 去重） |
| 训练数据 | TRELLIS.2 纹理预训练 + 本仓库 3513 variant 的 LoRA | 约 370 万 artist mesh |
| 推理成本 | 一次扩散采样 | 特征提取一次，之后每个 mask 毫秒级 |
| 主要失败模式 | 串色、碎片、细长件被邻件吞掉 | 无语义；粒度未必合期望 |

**P3-SAM 强在**：不经过渲染这一层，所以没有视角依赖、没有遮挡问题、没有 2D→3D 对应需要学；
对粒度歧义有显式机制；输出保证是划分；数据量大三个数量级。

**SegviGen 强在**：**有语义**。P3-SAM 能把 monk 干净地切成若干块，但不知道哪块是 staff，
也无法响应"把 head+face+hand 合成一个 body"这种要求。提示词组 + `--unassigned_to` 直接对应下游需求。

**值得借鉴的三点：**

1. （**已做**）掩码选择阶段。`data_toolkit/interactive_partition.py` 已经移植了 P3-SAM 的思路：
   稳定性筛选（阈值扰动下 mask 不变才保留）+ 重复合并 + 小件优先覆盖，把交互模式的重叠掩码变成划分。
2. **粒度歧义要显式建模**。v3 方案的图例 token 是"一个名字一个颜色"的硬绑定，没有处理"这个名字
   该覆盖多大范围"的歧义。P3-SAM 的多头 + IoU 选择是现成答案。
3. **原生 3D 特征值得作为第四个注入点**。`segvigen_finetune_notes.md` 里的三个方案都在 2D 侧
   （多视角、文本、投影）。`structured_latent_flow.py` 的 `concat_cond` 机制已经存在（现在喂 `shape_slat`），
   把 PTv3 一类的逐点特征拼进去在工程上是通的。

**但 P3-SAM 不能直接替代 SegviGen，因为它不给名字。** 更现实的组合是**用 P3-SAM 出干净划分、
再用 SAM3 或文本给这些块命名**——这条路绕开了 1.4 的核心矛盾（用重建损失去学语义归属）。
如果 Stage D 的 shuffle 对照失败，这是首选退路，优先级高于方案三。

---

## 1. 训练结果

### 1.1 三个模型在留出集上的表现

留出集：`pv_holdout_mix.txt`，35 个 PartVerse 对象 / 90 个 variant，固定 t 网格、固定噪声，步间可直接比较。
指标是 holdout v-pred MSE（越低越好）。

| 运行 | 配置 | 步数 | clean | corrupt | sam3 | 相对基座 |
|---|---|---|---|---|---|---|
| 基座 `full_seg_w_2d_map` | — | 0 | 0.1657 | 0.1541 | 0.0819 | — |
| pv_v1 | LoRA r16，batch 4，随机 t | 4000（完成） | 0.1531 | 0.1424 | 0.0754 | −7.6% / −7.6% / −8.0% |
| pv_v2 | LoRA r16，batch 16，分层 t | 1500 / 2000（手动停止于 1700） | 0.1530 | 0.1426 | 0.0754 | −7.7% / −7.5% / −7.9% |

**pv_v2 在 1500 步到达的位置和 pv_v1 4000 步完全一样。** 换句话说：LoRA 在 v-pred MSE 上能拿到的下降就是约 8%，
且五分之四在前 1000 步拿到。按样本数对齐后这一点更清楚：v2 不是每样本学得更快，只是每步看得更多（见 1.3）。

### 1.2 颜色保真度（真正关心的指标）

`eval_fidelity.py` 新口径（最近邻图例色归属，不设阈值），3 个留出对象 / 28 个部件，`sam3_az0` 条件：

| 模型 | fidelity | 归属正确 | 说明 |
|---|---|---|---|
| 基座 | 0.953 | 27 / 28 | 唯一错的是一把小钥匙，被判给邻件颜色 |
| pv_v1 合并后（`ckpt/full_seg_w_2d_map_sam3.ckpt`） | 0.906 | 26 / 28 | 略变差；`facial features` 的 margin 从 −63 掉到 −3.1，`gradient` 翻错 |
| pv_v2 | 未合并、未评估 | — | 1500 步的 MSE 与 v1 终点相同，评估不会有新信息，故直接停止 |

### 1.3 wandb 曲线

数据来自 wandb 0.29.0 的 `DataStore` 扫描 `.wandb` protobuf history（key 在 `nested_key`）：
`SegviGen/wandb/run-20260903_113544-pv_v1_full`（201 条，步 1–4000，约每 20 步）和
`run-20260903_150910-pv_v2`（169 条，步 1–1680，约每 10 步）。pv_v2 多出的 key 只有 `train/loss_step`；
分层 t 分桶指标未记录。

pv_v1 的 holdout 因 wandb 拒绝回写更小 step（`Tried to log to step 1000 that is less than the current step 4000`），
history 只留下步 4000；中间三次取自 `pv_train_v1.log` 的 `per-kind mean`（4 位小数）。
pv_v2 的 6 次 holdout 以 wandb history 为准，与 `pv_train_v2.log` 一致。
降采样取目标步最近的一条记录。标准差为样本标准差。最后 500 步：v1 为 3500–4000（n=26），v2 为 1180–1680（n=51）。

**pv_v1 train**（batch=4；250 / 750 / 1250 最近记录为 240 / 740 / 1240。`loss_clean` 第 1 步未记录，表内为步 20）

| step | 样本数 | loss | ema | clean | corrupt | sam3 | lr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.06789 | 0.06789 | 0.1385 | 0.08367 | 0.04461 | 2.000e-6 |
| 240 | 960 | 0.09603 | 0.08961 | 0.1228 | 0.1026 | 0.04715 | 9.971e-5 |
| 500 | 2000 | 0.07455 | 0.09146 | 0.08900 | 0.1264 | 0.07713 | 9.768e-5 |
| 740 | 2960 | 0.02616 | 0.08725 | 0.1941 | 0.1033 | 0.05450 | 9.415e-5 |
| 1000 | 4000 | 0.1166 | 0.08683 | 0.09367 | 0.09049 | 0.1103 | 8.868e-5 |
| 1500 | 6000 | 0.04511 | 0.08905 | 0.08030 | 0.09778 | 0.09921 | 7.429e-5 |
| 2000 | 8000 | 0.08603 | 0.07749 | 0.1327 | 0.09351 | 0.07463 | 5.681e-5 |
| 2500 | 10000 | 0.1042 | 0.08804 | 0.1043 | 0.1075 | 0.05364 | 3.904e-5 |
| 3000 | 12000 | 0.05738 | 0.09051 | 0.1090 | 0.1104 | 0.08439 | 2.383e-5 |
| 3500 | 14000 | 0.05884 | 0.09229 | 0.08200 | 0.1030 | 0.06989 | 1.360e-5 |
| 4000 | 16000 | 0.05054 | 0.07918 | 0.08232 | 0.1072 | 0.06048 | 1.000e-5 |

**pv_v2 train**（batch=16。`train/loss` 是窗口平滑量；单步噪声在 `train/loss_step`。v1 没有 `loss_step`）

| step | 样本数 | loss | ema | clean | corrupt | sam3 | lr | loss_step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.09452 | 0.07649 | 0.1444 | 0.09875 | 0.04575 | 2.000e-6 | 0.07649 |
| 200 | 3200 | 0.09410 | 0.08470 | 0.1332 | 0.08169 | 0.07161 | 9.939e-5 | 0.08398 |
| 400 | 6400 | 0.09980 | 0.08760 | 0.1129 | 0.1054 | 0.06836 | 9.458e-5 | 0.1545 |
| 600 | 9600 | 0.09255 | 0.09181 | 0.1222 | 0.09442 | 0.05359 | 8.548e-5 | 0.08830 |
| 800 | 12800 | 0.09995 | 0.08682 | 0.09811 | 0.1139 | 0.06255 | 7.308e-5 | 0.08366 |
| 1000 | 16000 | 0.09500 | 0.08404 | 0.08079 | 0.1008 | 0.08678 | 5.872e-5 | 0.04984 |
| 1200 | 19200 | 0.08743 | 0.09034 | 0.09749 | 0.08634 | 0.07916 | 4.395e-5 | 0.06810 |
| 1400 | 22400 | 0.09103 | 0.09106 | 0.09963 | 0.09419 | 0.07445 | 3.039e-5 | 0.1539 |
| 1600 | 25600 | 0.09463 | 0.08641 | 0.1094 | 0.09867 | 0.06598 | 1.949e-5 | 0.1110 |
| 1680 | 26880 | 0.1027 | 0.09324 | 0.1261 | 0.1063 | 0.07397 | 1.615e-5 | 0.07025 |

**holdout**（固定 t 网格 + 固定噪声，步间可比）

| run | step | 样本数 | clean | corrupt | sam3 | 来源 |
|---|---:|---:|---:|---:|---:|---|
| 基座 | 0 | 0 | 0.1657 | 0.1541 | 0.0819 | 文档给定 |
| pv_v1 | 1000 | 4000 | 0.1552 | 0.1445 | 0.0765 | v1 训练日志 |
| pv_v1 | 2000 | 8000 | 0.1540 | 0.1433 | 0.0758 | v1 训练日志 |
| pv_v1 | 3000 | 12000 | 0.1534 | 0.1427 | 0.0755 | v1 训练日志 |
| pv_v1 | 4000 | 16000 | 0.1531 | 0.1424 | 0.07535 | wandb history |
| pv_v2 | 250 | 4000 | 0.1561 | 0.1453 | 0.07705 | wandb history |
| pv_v2 | 500 | 8000 | 0.1549 | 0.1442 | 0.07625 | wandb history |
| pv_v2 | 750 | 12000 | 0.1543 | 0.1435 | 0.07587 | wandb history |
| pv_v2 | 1000 | 16000 | 0.1538 | 0.1432 | 0.07560 | wandb history |
| pv_v2 | 1250 | 20000 | 0.1531 | 0.1428 | 0.07562 | wandb history |
| pv_v2 | 1500 | 24000 | 0.1530 | 0.1426 | 0.07544 | wandb history |

v2 在 1500 之后到 1680 没有再记 holdout。相对基座，v1@4000 降幅 **7.61% / 7.60% / 7.99%**，
v2@1500 降幅 **7.67% / 7.44% / 7.89%**。v1 总降幅的约 83% 发生在步 1000 之前、约 93% 在步 2000 之前；
v2 约 75% 在步 250、约 85% 在步 500、约 94% 在步 1000。

同样本数对比：4000 样本时 v1@1000 = 0.1552 / 0.1445 / 0.0765，v2@250 = 0.1561 / 0.1453 / 0.0771；
16000 样本时 v1@4000 = 0.1531 / 0.1424 / 0.0754，v2@1000 = 0.1538 / 0.1432 / 0.0756。
**对齐样本后 v1 略好，v2 没有领先。**

关键统计（末 500 步 = v1 3500–4000 / v2 1180–1680）：

| run | metric | 起点 (step) | 终点 (step) | 最小 (step) | 末 500 均值 | 末 500 std | 末 500 CV |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | train/loss | 0.06789 (1) | 0.05054 (4000) | 0.01893 (3100) | 0.07472 | 0.03848 | 0.515 |
| v1 | train/ema | 0.06789 (1) | 0.07918 (4000) | 0.06789 (1) | 0.08518 | 0.00425 | 0.050 |
| v1 | train/loss_clean | 0.1385 (20) | 0.08232 (4000) | 0.05191 (3940) | 0.1021 | 0.02629 | 0.258 |
| v1 | train/loss_corrupt | 0.08367 (1) | 0.1072 (4000) | 0.07037 (520) | 0.09645 | 0.01370 | 0.142 |
| v1 | train/loss_sam3 | 0.04461 (1) | 0.06048 (4000) | 0.03696 (1580) | 0.06564 | 0.01091 | 0.166 |
| v2 | train/loss | 0.09452 (1) | 0.1027 (1680) | 0.08294 (1520) | 0.09494 | 0.00603 | 0.063 |
| v2 | train/ema | 0.07649 (1) | 0.09324 (1680) | 0.07649 (1) | 0.08858 | 0.00363 | 0.041 |
| v2 | train/loss_step | 0.07649 (1) | 0.07025 (1680) | 0.02211 (1320) | 0.09058 | 0.04459 | 0.492 |

v1 `train/loss` 全程斜率为正（+1.3e-6 /step），是因为步 1 碰巧偏低，不是后期在升。
v2 `train/loss` 斜率 −1.5e-6 /step，对应 1680 步只降约 0.0025，可忽略。
train EMA 从第一个 500 步窗口起就在 0.085–0.089 横盘，**没有对应 holdout 那 8% 的下降**——两条信号脱耦。

按样本数分箱，v1 / v2 的 EMA 均值差不超过 0.006，曲线重合。v2 的 `train/loss` 看起来更平，
是因为它是平滑量，不能和 v1 的单步 `train/loss` 叠在一起比高低。
直接比日志里的 `train/loss` std（0.03848 → 0.00603）会误判"方差被压低"；
v2 的单步量 `loss_step` 末 500 std 是 **0.04459**，和 v1 的 0.03848、蒙特卡洛 0.0437 / 0.0452 同量级。
**单步方差几乎没降；被压低的是平滑后的日志尺度。**

holdout 基本单调。唯一回升是 v2 `holdout/sam3` 在 1000→1250：0.07560→0.07562（+1.7e-5），随后再降到 0.07544。
VRAM 阶梯上升后平台（v1 8.38 → 11.73 GiB，v2 8.38 → 11.71 GiB），不是抖动泄漏。
没有过拟合或不稳定迹象。三条 holdout 相对基座几乎锁步（7.4–8.0%），没有一条单独拉动。

完整曲线和校验：`pv_v1_report.html`、`pv_v2_report.html`；wandb `segvigen-sam3 / pv_v1, pv_v2`。
计划任务 `SegviGenTrainPV2` 已禁用，GPU 空闲。

### 1.4 结论

1. **"损失降了 8%"成立，但指的是 holdout 相对基座，不是 train 曲线。**
   v1@4000 为 7.61% / 7.60% / 7.99%，v2@1500 为 7.67% / 7.44% / 7.89%。
   train EMA 从暖机后就在 0.085–0.089，全程斜率接近 0。颜色归属从 27/28 变成 26/28
   → **损失里主导的是几何/纹理重建，不是颜色归属**。继续在这个目标函数上加步数、调超参没有意义。
2. **"震荡是采样噪声，不是不稳定"成立。**
   v1 末 500 步 `train/loss` std=0.03848；v2 单步 `loss_step` std=0.04459，
   与蒙特卡洛 0.0437 / 0.0452 同量级。holdout 单调、EMA 平稳、VRAM 平台。
   唯一无噪声的信号是 holdout 校验。
3. **需要修正"v2 更快 / 分层 t + 大 batch 压低了方差"。**
   按步数更快（1500 vs 4000），按样本数没有更快，同样本 holdout 还略差。
   平滑后的 `train/loss` std 确实降到约 1/6；单步 `loss_step` 没有降。变好看的是日志尺度，不是优化更稳。
4. **两套配方落到同一 holdout 天花板（约 0.153 / 0.143 / 0.075）。**
   五分之四的降幅在前 1000 步（v1）或前 500–750 步（v2），残差斜率约 1e-4 / 250 步。
   三条 holdout 锁步下降，只说明学到的是条件无关的重建残差。
5. 模型只有一个语义入口（DINOv3 对单张色图的 patch token），**没有文本、没有相机、没有 2D→3D 对应**。
   想让它知道"这块红色是 staff"，必须另开一路。这是 v3 方案的出发点。
6. 留出集太简单（基座已 27/28），测不出改进。v3 评估必须换成能复现痛点的硬样本。

---

## 2. 数据修复

### 2.1 部件名重标注（已完成并经人工审阅）

原名字是从 PartVerse caption 启发式抽取的，17.6% 是 `cylindrical component` 这类无语义占位词。
用 grok 4.6 读 caption 重标了全部对象，然后按 `relabel/HANDOVER.md` 第 2 节的口径审阅修复：
抽样 12 对象（种子 42）+ 全库"主导部件"筛查（316 个候选全部过掩码图目检），共改 157 处 / 156 个部件，
主要是把整体被当作部件时的局部名（`head`、`seat`）改成 `body` 一类。修复后 merge + apply 全部通过。

当前写回数据集的状态（9-04 实测，`merge_relabel.py` 与 `pv/*/names.json` 逐条一致）：

| 指标 | 首轮（9-03 修复前） | 当前 |
|---|---|---|
| 对象 / 部件 | 2000 / 14303 | 2000 / 14303 |
| 名字相对旧版改动 | 13049 (91.2%) | 13055 (91.3%) |
| 唯一名字数 | 1563 | 1558（683 个只出现 1 次；≥ 8 次的 302 个） |
| `uncertain` 部件 | 1061 (7.4%) | 1054 (7.4%) |
| 自动校验 problems / 坏批次 | 0 / 0 | 0 / 0 |
| 泛称/外观词命中 | 1（`orange` 是水果，误报） | 1（同上） |

写回位置：每个对象的 `names.json`（新）、`names_v1.json`（旧名备份）、`names_meta.json`（整体物体名 + 逐部件 `uncertain`）。
汇总产物 `relabel/names_v2.json`。回滚方法见 HANDOVER 3.6。

"≥ 8 次的名字只有 302 个"对 v3 方案的 Stage B 有直接影响：逐名字概念向量覆盖约 300 个名字，
但这 300 个名字占了 **81.8% 的部件实例**（11703 / 14303；前 15 个是 leg 678、body 622、head 535、wheel 405、
arm 396、torso 364、base 358、handle 281、boot 212、lid 187 …）。其余 1256 个长尾名字走共享的 `E_0`。
这是设计里预期的，不需要改方案。

**尚未做**：SAM3 mask 和 `prompts.json` 还是旧名字生成的，`variants/sam3_*` 还是旧结果。这一步排在 v3 方案的 Stage C，
要等概念向量库（Stage B）做完再重跑，否则要跑两遍。

### 2.2 补渲染（前提性动作，9-04 执行）

v3 方案的 Stage B（概念向量库）和 Stage C（扩大 Path A）都需要每个对象有 `render.png` + `ids.npy`，
之前只有 343 个 Path A 对象有渲染。新写了 `finetune/render_views.py`：只渲染 + 光栅化 GT 部件 id，
**不走 `prepare_object`，所以不会碰 `o_voxel`**；可续跑，`--chunk` 每 N 个对象换一个子进程。

```
finetune\run_ft.bat render_views.py --dataset_root E:\AI_New\ModelGen\datasets\pv --azimuths 0,135 --chunk 50
```

先在 2 个对象上验证：render 前景与 ids 前景 IoU 0.959 / 0.968（差的是轮廓抗锯齿），相机一致。
全量 1654 个对象约 0.85 s/对象，23 分钟跑完，0 失败。结果见第 5 节。

---

## 3. 数据集现状

`E:\AI_New\ModelGen\datasets\pv\`，2000 个 PartVerse 对象。**注意分层**：

| 层 | 对象数 | 有什么 | 缺什么 |
|---|---|---|---|
| 全部 | 2000 | `input.glb`、`parts/*.glb`、`captions.json`、新 `names.json`、渲染 az0/az135 + `ids.npy`（补渲染后） | — |
| 已 prepare（体素化 + latent） | 694 | `shape_slat.pth`、`input_tex_slat.pth`、`voxel_part.npy`、`ids_meta.json`，以及 variants | — |
| Path A（有 SAM3 variant） | 343 | `variants/sam3_az0/az135`，`views/*/sam3_masks.npz` | SAM3 结果是旧名字的 |
| 只有 Path B | ~350 | `clean_*`、`corrupt_*` variants | 无 SAM3 |
| 未 prepare | 1306 | 只有几何、名字、渲染 | 没有 latent，**不能直接进 SegviGen 训练** |

variants 合计 3513：clean 708 / corrupt 2124 / sam3 681（训练用 3354 + 留出 90 + 少量被排除）。

这意味着 v3 方案里"Path A 扩到 1000 个对象"**不只是重跑 SAM3**，还要先把 657 个对象过一遍 `prepare_object`
（体素化 + 编码，`o_voxel` 会崩，必须用 `run_batch.py` 驱动）。这是 Stage C 里最耗时的一段，估算按 pv 首轮的速度
（`pv_run_batch_console.log`）：每对象 1–2 分钟 → 657 个约 12–20 小时 GPU 后台。

---

## 4. 下一步要执行的任务

按阶段顺序，这里只列**可直接开工的动作**和验收标准。下面的清单是 9-04 上午的计划原文；截至 9-04 下午的执行状态：

| Stage | 状态 | 结果 |
|---|---|---|
| A 基座/模板 | 完成 | `bench_sam3.py`；裸 `{name}` 最好（mIoU 0.311、FP 8.8%）；加 `{object}` 绑定率升但 FP 到 27%；"body/torso" 是主要歧义词 |
| B 概念库 | v1/v2 完成，v3 在跑 | `concept_bank.py`；等误检率下 v1@τ0.5 mIoU 0.428（基线 0.323）、硬 FP 41%（基线 29%）；v2（硬负例）@τ0.5 mIoU 0.374、硬 FP 20%；v3 = 混合负例 + `lr_e0` 5e-4。`--concept_bank` 已接入 `sam3_to_2dmap.py` / `sam3_masks.py` |
| C 重跑 Path A | 待 B 选型后启动 | `pv_hard.txt`（20 个硬样本）与 `pv_holdout_v3.txt` 已选；`pv_list_a_v3.txt` 1028 个对象 |
| D v3 代码 | 完成，待 GPU 冒烟 | `dataset.py` / `model.py::LegendEncoder` / `train.py` / `merge_lora.py` / `inference_full.py` / `eval_fidelity.py` 全部接通 |
| D v3 训练 | 未开始 | 依赖 C |
| E 指标 | 完成 | `fragments`、`boundary_f1`、`--shuffle_legend`、`--swap_names`、`pick_hard.py` |

### Stage A：文本编码器基座选择（零训练，1 天）

- [ ] 写 `finetune/bench_sam3.py`：对全部 2000 对象 × 2 视角，逐唯一名字构造 `gt_union`，加 5 个难负例
- [ ] 跑三组：原生 SAM3 × 模板 `{name}` / `{object} {name}` / `{name} of {object}`
- [ ] 指标：绑定率（IoU ≥ 0.5）、灰像素率、部件 mIoU、难负例误检率、5 px 边界 F1
- [ ] 输出 `datasets/bench_sam3/<candidate>.json` + 汇总表；按绑定率倒序导出歧义词清单
- [ ] 可选：装 SAM3-I（<https://github.com/debby-0527/SAM3-I>），能跑就加入对比；部件 mIoU 或边界 F1 高 ≥ 3 分才换基座

验收：拿到默认模板；歧义词清单落盘。

### Stage B：概念向量库（2–3 天）

- [ ] `finetune/concept_bank.py`：`text_embeds = pooler_output + E_0 + E_name`，主干冻结，损失 = 并集 BCE + Dice + 0.5·presence BCE
- [ ] 数据：全部 2000 对象 × 2 视角；`uncertain == false` 才作正样本；按对象 10% 留出，**必须包含 `pv_holdout_mix.txt` 的 35 个**
- [ ] `E_name` 只给出现 ≥ 8 次的名字；其余走 `E_0`
- [ ] 外部资产回归：monk / dwarf / mushroom / `1.glb` 上只加 `E_0` 不能比不加差
- [ ] 产物：`datasets/concept_bank/bank.pt`、`text_cache.pt`（全部名字 + 对象名的 256 维向量）
- [ ] `sam3_to_2dmap.py` / `sam3_masks.py` 加 `--concept_bank`

验收：留出集绑定率 ≥ 基座 +10；误检率不升；灰像素率 < 5%。

### Stage C：重跑并扩大 Path A（GPU 后台 1–2 天）

- [ ] 确认无训练进程；`apply_relabel.py --reset_sam3 --dry` → 去掉 `--dry`
- [ ] 选 657 个未 prepare 的对象（部件数 ≥ 5 优先）写 `pv_list_a2.txt`
- [ ] `run_batch.py --jobs b a --objects_b pv_list_a2.txt --objects_a <343 + 657>` 让 Path B 先 prepare、再跑 A；`--args_a` 带 `--concept_bank`
- [ ] `make_samples_a.py` 的 `meta.json` 同时记模板串和原名

验收：新 `meta.json` 汇总 `unbound` 比例和灰像素率达到 Stage B 目标；sam3 variant 占比 ≈ 40%。

### Stage D：SegviGen v3 代码与训练（代码 2 天，可与 C 并行；训练每轮 8–10 小时）

- [ ] `dataset.py`：从 `meta.json` + `names.json` + `text_cache.pt` 构造图例；返回双视角 cond
- [ ] `model.py`：`LegendEncoder`（`W_t` 256→1024、`MLP` 3→256→1024、`e_view0/1`、`e_legend`）；cond 拼接；`p_drop_legend 0.2`、`p_drop_view1 0.3`
- [ ] `train.py`：新模块参数组 lr 1e-3；flag `--shuffle_legend` / `--no_legend` / `--single_view`；保存新模块权重
- [ ] `merge_lora.py` / `inference_full.py`：一并导出和加载 `LegendEncoder`
- [ ] 先跑 `pv_v3` 与 `pv_v3_shuffle` 各 1000 步看趋势，再决定是否跑满 3000 步和 `pv_v3_noleg`

验收：**`pv_v3` 必须明显优于 `pv_v3_shuffle`**，否则说明 cross-attn 没读图例，转方案三。

### Stage E：评估集与指标（与 D 并行准备）

- [ ] 硬样本集：monk / dwarf / mushroom / `1.glb` + 20 个 PartVerse 对象（部件 ≥ 6，细长部件贴大部件，用 `voxel_part.npy` 接触面积筛）
- [ ] `eval_fidelity.py` 加碎片数（每 GT 部件的连通分量数 − 1）、边界 IoU、语义冲突测试（互换图例名字看边界是否移动）

### 先后依赖

```
补渲染(9-04 完成) ─► A ─► B ─► C(GPU 后台) ─┐
                          D 代码(并行) ─────┴─► D 训练 v3 / shuffle ─► E
```

---

## 5. 补渲染结果（9-04 10:03 完成）

| 项 | 结果 |
|---|---|
| 处理对象 | 1654（此前缺渲染的全部对象），失败 0 |
| 耗时 | 23.4 分钟，约 0.85 s/对象（2 视角渲染 + 2 张 GT id 光栅） |
| 现在有 `render.png` + `ids.npy`（az0、az135）的对象 | **2000 / 2000** |
| 前景为空或近空（< 200 px）的视角 | 0 |
| 渲染 alpha 前景 vs `ids.npy` 前景 IoU（随机 6 对象 × 2 视角） | 最低 0.924，平均 0.975；差值全在轮廓抗锯齿 |

日志里反复出现的 `ERROR (bke.lib_id_delete): Deleting IMRender Result which still has 1 users` 是 Blender 在
`init_scene` 清空图像时的已知噪音，不影响输出，全部 render.png 都已落盘。

新增文件（每对象）：`views/az0/{render.png, ids.npy, ids_preview.png}`、`views/az135/{同上}`。
状态文件 `pv/render_views_state.json`，日志 `pv_render_views.log`、`pv_render_views_console.log`。

概念向量库（Stage B）的输入现在齐了：4000 张图、28606 个部件实例（14303 × 2 视角，去掉各视角不可见的会少一些）。

---

## 6. 损失函数与条件结构：原版 SegviGen vs 我们的改动（9-04 整理）

### 6.1 原论文损失函数设计（arXiv 2603.16869，Sec. 3.1–3.3）

论文**写了**损失函数，就是标准的条件 flow matching，没有任何针对颜色/语义的额外项：

- 噪声插值（Eq. 3）：\(y_t = (1-t)\,y + t\,\epsilon\)，\(\epsilon\sim\mathcal N(0,I)\)，\(t\sim\mathcal U(0,1)\)；\(y\) 是"部件着色后的资产"经冻结 SC-VAE 编码得到的目标 latent
- 网络（Eq. 4）：\(\hat v_\theta = f_\theta(y_t,\ z,\ C,\ e_\tau,\ t)\)，\(z\) 是输入资产的几何 latent，\(C\) 是任务条件，\(e_\tau\) 是任务嵌入
- 目标（Eq. 5）：\(\mathcal L(\theta)=\mathbb E\big[w(t)\,\|\hat v_\theta-(\epsilon-y)\|_2^2\big]\)，\(w(t)\) 为"可选的时间步加权"，论文没给具体形式

其余训练细节：

| 项 | 论文设定 |
|---|---|
| 可训练部分 | **整个 Tex-SLAT flow model 全量微调**，SC-VAE 冻结 |
| 优化器 | AdamW，lr 1e-4；8×A800，8 小时 |
| 全分割目标 | 每部件随机取色，正确性"允许颜色置换"；每个形状 **K=10 套独立调色板** |
| 2D 引导任务 | 先给 3D 部件着色 → 渲染出 2D 分色图作为条件 → 训练模型生成与 2D 颜色一致的 3D 体素颜色（**全部是干净渲染图**，没有 2D 分割器噪声） |
| 条件注入 | 2D 图经图像编码器 \(g_\phi\) 得到 token \(p\)，走交叉注意力（Eq. 8–9）；同时保留 10 个全零的点 token 占位 |
| 任务嵌入 | \(e_\tau=\mathrm{MLP}(\mathrm{PE}(\tau))\)，与时间步信号一起注入（Eq. 10–11） |
| 推理 | 12 步 |

论文**没说**的：\(w(t)\) 的形式、CFG 的 dropout 概率、batch、数据增强、颜色是否有任何显式监督（没有——颜色只通过 latent MSE 隐式学习，这正是"SAM3 颜色进不了 SegviGen"的结构性原因）。

### 6.2 我们复现的损失（`finetune/train.py::flow_loss`）

与 Eq. 3–5 一致，只多了 TRELLIS.2 采样器里的 \(\sigma_{min}=10^{-5}\)：

\[
x_t=(1-t)\,x_0+(\sigma_{min}+(1-\sigma_{min})t)\,\epsilon,\qquad
v^\*=(1-\sigma_{min})\,\epsilon-x_0,\qquad
\mathcal L=\operatorname{mean}_{\text{token},\,\text{dim}}\|f_\theta(x_t,t,c)-v^\*\|^2
\]

有意与论文不同的地方：

| 项 | 论文 | 我们 | 原因 |
|---|---|---|---|
| \(t\) 采样 | \(\mathcal U(0,1)\) | logit-normal(0,1) + **batch 内分层**（`sample_t`） | 损失随 \(t\) 变化数倍，独立采样导致曲线剧烈震荡；分层保持边缘分布不变、降低方差 |
| \(w(t)\) | 可选、未给出 | 1 | 无依据加权 |
| 可训练参数 | 全量微调 | **LoRA r=16 α=32**（self/cross attention），基座冻结 | 单卡；且只有 2000 个对象，全量微调会遗忘先验 |
| CFG dropout | 未说明 | `p_uncond=0.1`，无条件分支 = 全零 token（与 `neg_cond` 一致） | 保持推理时 CFG 可用 |
| 留出评测 | — | 固定 \(t\in\{0.2,0.5,0.8\}\) + 每样本固定噪声（`check_loss`） | 跨 run/checkpoint 可比 |
| 其他 | — | grad clip 1.0、bf16、梯度检查点、有效 batch 4 | 显存 |

### 6.3 条件结构：单路图像 → 多 token 上下文（v3）

原版：\(c=g_\phi(I_{guide})\)，即 DINOv3 对 2D 分色图的 \([T,1024]\) token，仅此一路。

v3（`finetune/model.py::LegendEncoder`）把每个样本的上下文改成变长序列：

\[
c=\big[\ \underbrace{g_\phi(I_0)+e_{view0}}_{\text{主视角}}\ ;\ \underbrace{g_\phi(I_1)+e_{view1}}_{\text{第二视角，可缺}}\ ;\ \underbrace{\mathrm{LN}(W_t s_{obj}+e_{obj})}_{\text{物体名}}\ ;\ \underbrace{\{\mathrm{LN}(W_t s_g+\mathrm{MLP}(rgb_g)+e_{legend})\}_{g=1..G}}_{\text{图例：每个颜色组一个}}\ \big]
\]

- \(s_g\in\mathbb R^{256}\)：第 \(g\) 组部件名经 **SAM3 文本编码器 + 概念库偏移**后的池化向量——就是 SAM3 生成这块掩码时用的那个向量，所以 2D 图上的颜色和 token 的语义同源
- \(rgb_g\)：该组在 2D 图上的颜色，3→256→1024 MLP
- \(e_{view0/1},e_{legend},e_{obj}\)：类型嵌入；新 token 过 LayerNorm 与 DINO token 同 rms 尺度
- DiT 交叉注意力原生接受 per-sample 变长上下文列表，**DiT 本体不改**，无需 padding
- 训练时三级独立 dropout：全零 0.1、只丢图例 0.2、丢第二视角 0.3；推理可对图例单独 CFG
- 新模块从头训练，单独 lr 1e-3（`--new_lr`）；与 LoRA 参数在同一优化器的两个 param group
- 对照：`--legend_shuffle` 把物体内各组名字随机置换；若 fidelity 与正常 run 相同，说明模型没读文本

### 6.4 数据/目标的改动

| 项 | 原版 | 我们 |
|---|---|---|
| 3D 目标 | 部件着色 → SC-VAE 编码 | 同，但按颜色组重着色体素，**未绑定部件涂灰**；同一对象两个视角共享一个 3D 目标（`target_from`） |
| 2D 条件 | 全部干净渲染 | 三类：`clean`（part id 精确光栅）、`corrupt`（clean + 合成扰动）、`sam3`（真实 SAM3 + 概念库输出，Path A）——把推理噪声分布放进训练分布 |
| 调色板 | K=10 套/形状 | 每 variant 随机一套；同对象多 variant |
| 部件名 | 未用 | Grok 4.6 重标注 + 人工审阅的 `names_v2.json`，进图例 token |

### 6.5 SAM3 侧的概念库损失（Stage B，`finetune/concept_bank.py`）

SAM3 冻结，只学两个 256 维偏移加到文本池化向量：\(s'=s+E_0+E_{name}\)（\(E_{name}\) 仅训练集出现 ≥8 次的名字）。

\[
\mathcal L=\underbrace{\mathrm{BCE}+\mathrm{Dice}}_{\text{正样本：软联合掩码 vs 面积下采样 GT}}
+0.5\,\underbrace{\mathrm{BCE}(\hat m,0)}_{\text{负样本掩码归零}}
+0.5\,\underbrace{\mathrm{BCE}(\text{presence})}_{\text{存在性头}}
(+\lambda\|E_{name}\|^2)
\]

v1/v2/v3 只差负样本来源（随机 / 硬 / 混合）与 \(E_0\) 学习率（1e-3 → 5e-4）。留出集等误检率对比：v1@τ0.5 mIoU 0.428（基线 0.323）、硬 FP 41%（基线 29%）；v2@τ0.5 mIoU 0.374、硬 FP 20%；v3 在跑。

### 6.6 局限

主干侧的语义监督仍是**隐式**的：图例 token 进了条件，但损失还是 latent 上的 MSE，梯度不知道哪个 token 对应哪块颜色。Stage D 的 shuffle 对照就是检验这一点；若 1000 步后无差异，第二阶段加显式项（解码后对目标调色板的最近色一致性损失，或按边界加权的 MSE）。

---

## 7. v3 结果与 v4 显式监督方案（9-05）

### 7.1 v3 的结论：图例 token 没有进入输出

pv_v3 与 pv_v3_shuffle 各训 1000 步（bs 4×4，lr 1e-4 / 图例编码器 1e-3）：

| 证据 | 数值 | 含义 |
|---|---|---|
| 留出 MSE（250/500/750/1000 步） | 两条 run 到小数点后 4 位完全相同 | 输出对图例 token 不敏感 |
| LoRA 权重相对距离 | 5.5%（300 个张量无一相同） | 两个模型确实不同，不是没训 |
| 图例编码器 | `text_proj` 相差 54%、`e_obj` 150%、`e_legend` 117% | 编码器收到了梯度 |
| 硬样本 fidelity（基座 → v3，8 个对象） | 0.501→0.362、0.981→0.713、0.300→0.210、0.460→0.473、0.645→0.508、0.000→0.483、0.294→0.595、0.822→0.745 | 与 v1/v2 一样，LoRA+MSE 让颜色归属整体变差 |

机制层面的原因：
- **损失不需要文本。** 对可见部件 2D 图已完整给出颜色；探针统计显示 gt 策略下只有 2.3% 的 latent cell 属于隐藏部件，即"只能靠图例知道颜色"的体素几乎没有。
- **注意力稀释。** ≤10 个图例 token 与约 1000 个 DINO token 同在一个 softmax 里，经冻结 K/V 投影后是分布外向量，权重接近 1/1000。
- **目标错位。** v-pred MSE 被大块几何/纹理重建主导，细部件的颜色错误对损失几乎没有贡献，这与 fidelity 关心的东西正好相反。

### 7.2 可行性探针（40 对象，CPU）

| 问题 | 结果 |
|---|---|
| latent cell ↔ 精细体素对齐 | 因子 16，100% 的 cell 都能映射到体素块 |
| cell 的部件纯度 ≥0.9 的比例 | 91.3%（边界 cell 可单独降权/忽略） |
| 对象内最近原型分类（latent 空间） | 97.5% |
| 跨对象线性探针 latent→RGB（留出对象） | R² 0.906，最近调色板 96.5%（正式 400 对象含 GREY 类：R² 0.862 / 90.9%） |
| 隐藏部件 cell 占比 | 2.3% → 必须造"只能靠图例"的样本 |

### 7.3 v4 三项改动

**(a) 显式颜色损失** `finetune/cells.py`、`train.py::color_loss`

\[
\hat{x}_0=\frac{x_t-a\hat v}{1-t+a},\quad a=\frac{\sigma_{min}+(1-\sigma_{min})t}{1-\sigma_{min}};\qquad
\hat c_i = W\hat{x}_{0,i}+b\ (\text{冻结探针})
\]
\[
L_{color}=\frac{1}{B}\sum_{\text{sample}}\ \sum_{i\in\text{pure cells}} w_i\,\mathrm{CE}\!\left(\mathrm{softmax}_j\!\left(-\|\hat c_i-c_j\|^2/\tau\right),\ g_i\right),\qquad
w_i=\frac{1}{|\{\text{class}(g_i)\}|\cdot\#\text{classes}}
\]

- 类别 = 变体调色板的颜色组 + GREY；cell 标签来自 `cell_labels.py` 预计算的 `cell_part.npz`（每个 latent cell 的多数部件与纯度）经变体 meta 映射。
- 探针在**目标** latent 上就读错的 cell（约 9%）不计入，损失不会要求 VAE latent 表达不了的东西。
- 对 t 不加权：高噪声段是采样起点，也是细部件第一次被染成主体颜色的地方。
- 总损失 = v-pred MSE + 0.3·L_color，τ = 0.03（RGB 0–1 尺度；调色板最小间距 60/255 ⇒ 边界处 logit 差 ≈ 1.8）。

**(b) partial 变体** `make_samples_partial.py`（kind=`partial`，978 个 Path B 对象 × 2 视角 × 2 ≈ 3.9k 变体）

干净 2D 图上随机把 1–3 个可见部件（≥0.5% 前景像素）抹成 GREY，3D 目标直接复用 `clean_*` 的 `output_tex_slat.pth`，图例仍列出全部"名字 + RGB"。`write_variant` 新增 `mask_2d`（只影响 2D 绘制，不进 `grey_parts`），meta 记 `masked_parts`。被抹部件的 cell 上 L_color 只能靠读图例降下来；图例被 dropout 掉的样本上这些 cell 不计损失。

**(c) 解耦图例注意力** `model.py::LegendCrossAttention`

每个 block：\(h = \mathrm{Attn}(q, K_{img}, V_{img}) + W_{out}^{leg}\,\mathrm{Attn}(q, K_{leg}, V_{leg})\)，\(K_{leg},V_{leg}\) 来自图例 token 经本模块自己的 `to_kv`（从图像 `to_kv` 初始化，fp32 主权重，lr 1e-4），读出矩阵 \(W_{out}^{leg}\)（1536×1536）**零初始化**（ControlNet 式 zero-linear，lr 3e-4）。30 个 block 共 94.5M K/V + 70.8M 读出参数；第 0 步 DiT 就是基座。LoRA 键名因此变为 `…cross_attn.inner.…`，`merge_lora.py` 合并后解包，图例注意力状态与图例编码器一起存入 `*_legend.pt`，`inference_full.py` 通过 `--legend_ckpt` 自动装回，CFG 的负分支不带图例。

*第一版用的是零初始化标量门控 \(\tanh(g)\)，与图像路径共用 `to_out`（`runs/pv_v4_gate0`，跑到 690 步停）。结果门控根本打不开：250 步时 |tanh(g)| 均值 0.013、500 步 0.009（还在缩），30 个 block 没有一个超过 0.05；留出 partial 的 masked_acc(t=0.95) 虽然从基座 0.6% 升到 42%，但 `--check_shuffle` 的打乱图例组给出一模一样的 40.2%/42.0%，说明这 42% 全部来自 LoRA 学到的"抹灰处别输出 GREY"启发式，与文本无关。原因是鸡生蛋问题：标量门控只有在原始注意力输出与所需方向平均相关时才会移动，而门控为 0 又切断了上游 `to_kv`/图例编码器的全部梯度。换成零矩阵后，读出层从第 1 步起就能学任意有用的线性组合并把梯度传回上游（冒烟 6 步后各 block 读出增益已 0.025）。*

### 7.4 读数与判据

- `masked_acc`：partial 变体被抹 cell 的最近调色板准确率；在 `check_ts` 最大 t（0.95）处基座为 **0.6%**（它把抹灰部件预测成 GREY），随机水平约 10%。**pv_v4 在此指标上显著高于 pv_v4_shuffle** 是"文本进了模型"的唯一判据；shuffle 组的图例名字被打乱，颜色-名字对应错误，读图例只会把它带向错的颜色。
- `color_acc`/`color`：全体 cell 的准确率与均衡 CE，基座在留出集上约 0.98 / 0.5–0.7（均衡 CE 高说明有整块细部件全错，正是 fidelity 看到的失败）。
- 留出 MSE（clean/corrupt/sam3）不应比 v3 明显变差，否则 λ 过大。
- 最终仍以硬样本集 `eval_fidelity`（fidelity、boundary F1、碎片数）为准。

### 7.5 运行

pv_v4 与 pv_v4_shuffle **顺序**跑（同跑时 VRAM 30.7/32.6 GB 触发 Windows 换页，单步从 10 s 涨到 90 min），1500 步，每 250 步留出检查 100 个变体（含 partial），同时用 `--check_shuffle` 在同一次运行内给出打乱图例的对照读数，wandb 项目 `segvigen-sam3`。基座在留出 partial 上的参考值：MSE 0.220，color 4.25，color_acc 0.849，masked_acc 0.543（t=0.95：0.006）。

### 7.6 pv_v4（零矩阵读出版）结果：颜色损失有效，图例仍未被读

留出 partial 变体（n=25，t∈{0.5,0.8,0.95}）：

| 步数 | color CE | color_acc | masked_acc(t=0.95) 真图例 | 打乱名字 | 读出增益 |
|---|---|---|---|---|---|
| 基座 | 4.25 | 0.849 | 0.006 | – | – |
| 250 | 1.096 | 0.895 | 0.178 | 0.179 | 0.086 |
| 500 | 1.127 | 0.891 | 0.140 | 0.142 | 0.135 |
| 750 | 1.044 | 0.915 | 0.305 | 0.305 | – |

读出矩阵这次确实在增长（500 步 0.135，是门控版的 15 倍），梯度流进了图例路径；但真图例与打乱名字的读数在三次检查里都一致到小数点后三位——**名字的语义内容没有被使用**。

**消融（`--check_only --check_no_legend`，step-500 权重，同一批 100 个留出变体）**——把图例整个拿掉：

| partial | 有图例 | 无图例 |
|---|---|---|
| MSE | 0.214 | **0.198** |
| color CE | 1.127 | **0.842** |
| color_acc | 0.891 | **0.947** |
| masked_acc(t=0.95) | 0.140 | **0.175** |

clean/corrupt/sam3 同样是无图例更好（MSE 低约 10%）。训练集上 color_acc 0.998、masked_acc 0.997，留出只有 0.89–0.95：图例路径学到的是**对 ~1000 个训练对象的记忆**（调色板颜色集合当对象指纹），到留出集上是噪声。名字之所以用不上，是因为按全局图例给被抹部件上色要求模型先**从几何认出这块是什么**再去图例查色——从 1000 个对象学一个几何→语义识别器数据量差两个量级，结果只能背答案。

同时确认：显式颜色损失本身有效——无图例条件下 partial 的 color CE 4.25 → 0.84、color_acc 0.849 → 0.947，训练 CE 降到 0.07。"抹灰处别输出 GREY、用调色板里的颜色"学会了，只是不知道该用**哪个**。因此 pv_v4_shuffle 对照取消（运行内对照已给出零差距），pv_v4 跑完后用无图例模式在硬样本集上跑 `eval_fidelity`，看纯颜色损失对 fidelity / 碎片数的实际提升。

---

## 8. v5：逐 token 语义注入（把名字送到像素上）

### 8.1 动机

v3/v4 都把名字放在**全局**图例 token 里，模型要用它就得先识别几何。v5 把每个部件的名字直接加到**它所在的 DINO patch token** 上：颜色和名字绑在同一位置，被抹部件的 token 带着名字但颜色是灰，模型只需做 token 局部的"按名查色"（同名的图例 token 里有颜色），不需要认几何；对正常上色的区域，名字则是"这一片是同一个部件"的语义先验。这也正是部署时能拿到的信息——SAM3 的每个掩码本来就带着提示词。

### 8.2 实现

- **`token_labels.py`**：为全部 13,683 个变体生成 `<variant>/tokens.npz`（99 s，CPU）。复现 `img_to_cond.preprocess_image` 的裁切几何（用色图非白掩码代替 rembg alpha，12 个样本对比偏差 ≤2 px，一个 patch 约 10 px），在 32×32 patch 网格上按像素多数投票出部件名（前景 <20% 的 patch 无名）。clean/corrupt/partial 用 GT 光栅 `ids.npy → names.json`；sam3 变体用覆盖该像素的 SAM3 提示词掩码（分数高者优先），与部署一致。全体 patch 的 33.9% 有名字（≈前景占比）。
- **`dataset.py`**：`token_text=True` 时给出 `token_text` [1029, 256]（CLS/register/无名 patch 为零行）及配对视角的对应张量；`legend_shuffle` 改为对该变体全部不同名字做一个随机置换，**同时**作用于图例和逐 token 名字，对照组的名字一致地错。
- **`model.py::LegendEncoder`**：新增 `tok_proj = Linear(256→1024, bias=False)`，**零初始化**：无名 token 永远不被触碰，第 0 步等于基座；`tok_gain()` 报告其增长。旧 v3/v4 权重加载时自动补零。
- **`train.py --token_text`**：逐 token 名字与图例一起被 `p_drop_legend` 丢弃（一个"无文本"模式）；wandb 记 `train/tok_gain`。**不用** v4 的解耦注意力（7.6 显示它只学到记忆），图例按 v3 方式拼进上下文，与 patch token 同处一个 K/V 集合，供"按名查色"。
- **`inference_full.py`**：v5 编码器（`tok_gain>0`）下，从预处理后的色图按最近图例颜色（容差 0.12）反推每个 patch 的名字（`token_text_from_map`），无需额外输入。
- 判据不变：`--check_shuffle` 的真名字 vs 乱名字差距；`masked_acc(t=0.95)`；硬样本 `eval_fidelity`。

### 8.3 运行

pv_v4 结束 → `merge_lora` → 硬样本评估（base / v4 / v4 无图例）→ v5 6 步冒烟 → `pv_v5`（1500 步，与 v4 同超参，`--token_text`，无 `--legend_attn`），链式脚本自动执行，wandb `segvigen-sam3/pv_v5_r2`。GPU 同时只跑一个作业：本机还有 Blender/浏览器等常驻约 9 GB，训练 16 GB 之上再加任何检查都会把 32 GB 顶满换页。

### 8.4 pv_v4 硬样本评估：颜色损失没有转化成端到端收益

20 个硬对象、`sam3_az0` 变体、完整 25 步采样 + CFG 后的 `eval_fidelity`：

| 条件 | mean_fidelity | 正确部件数 | boundary F1 | purity |
|---|---|---|---|---|
| base | **0.617** | **6.95** | **0.751** | **0.884** |
| v4 无图例 | 0.605 | 6.65 | 0.683 | 0.852 |
| v4 有图例 | 0.567 | 6.15 | 0.711 | 0.871 |

逐对象 fidelity：v4 优于 base 7 个、差 12 个、平 1 个（两种模式相同）。留出集上单步 x0 估计的颜色 CE 从 4.25 降到 0.84，走完采样轨迹后 mesh 上的部件归属却没有变好，边界 F1 还略降：训练时的单步颜色监督与推理时的完整采样之间存在错位，LoRA + 这套损失没能闭合它。

### 8.5 pv_v5 结果：逐 token 名字只被当成"这里该上色"的开关

pv_v5 跑完 1500 步（206 min，峰值 11.7 GB），`tok_gain` 单调升到 0.222。留出检查（n=25/类，t∈{0.5,0.8,0.95}）：

| 1500 步 | clean MSE | partial color_acc | masked_acc(t=0.95) 真名字 | 乱名字 | sam3 color CE |
|---|---|---|---|---|---|
| v5 有文本 | 0.210 | 0.899 | 0.566 | 0.557 | 0.661 |
| v5 无文本（`--check_no_legend`） | **0.171** | **0.956** | 0.158 | – | **0.459** |
| v4 有图例（对照） | 0.173 | 0.921 | 0.345 | 0.345 | 0.358 |

三点结论：

1. **名字的内容仍未被使用**：六次检查真名字与乱名字差距 ≤0.01，与 v3/v4 一样。
2. **名字的"存在"被当成开关**：带文本时 masked_acc(t=0.95) 0.57，拿掉文本只有 0.16——因为在 partial/corrupt 变体里"灰色 patch 上有名字"恰好等价于"3D 目标这里有颜色"，而 sam3 的未绑定灰区没有名字。模型学到的是这个捷径，而不是"按名查色"。
3. **文本注入反而伤害重建**：同一组权重，去掉文本后 clean MSE 0.210 → 0.171、color_acc 0.90 → 0.96、sam3 color CE 0.66 → 0.46。零初始化的 `tok_proj` 长到 0.22 后，对 DINO token 的扰动已足以让图像条件变差，而 LoRA 没有把它补回来。

结构原因：DiT 的 cross-attention 里条件 token 之间不交互，"按名查色"需要 3D latent 先从 patch 读到名字、再在后面的 block 里以该名字为 query 去图例 token 取颜色——两跳、跨 block、只有 partial 变体的被抹 cell 提供监督，1500 步 LoRA 学不出来。三个版本（全局图例 / 解耦注意力 / 逐 token）一致的零差距说明，在 1k 对象 + LoRA 的量级上，**这个 DiT 不会自发利用部件名字的语义**；名字目前证明有效的位置只有 SAM3 那一步（概念库，Stage B）。

## 9. 轨迹探针、v5 硬样本结果与 v6 轨迹训练（9-06）

### 9.1 pv_v5 硬样本评估（20 对象，完整）

与 8.4 同一批 20 个硬对象、`sam3_az0` 变体、`eval_fidelity`：

| 条件 | mean_fidelity | 正确部件数 | boundary F1 | purity | 逐对象 vs base |
|---|---|---|---|---|---|
| base | **0.617** | **6.95** | **0.751** | 0.884 | – |
| v4 无图例 | 0.605 | 6.65 | 0.683 | 0.852 | 7 优 / 12 差 |
| v4 有图例 | 0.567 | 6.15 | 0.711 | 0.871 | 7 优 / 12 差 |
| v5 有文本 | 0.185 | 2.05 | 0.568 | **0.886** | 2 优 / 18 差 |
| v5 无文本 | 0.598 | 6.60 | 0.642 | 0.865 | 5 优 / 15 差 |

v5 带文本部署几乎不可用（fidelity 0.19），去掉文本后回到 base 附近但仍略差；purity 反而最高——
"整块颜色一致"与"整块归错部件"同时发生，purity 不能作为验收指标。四个微调版本在硬样本上**没有一个**超过基座。

### 9.2 轨迹探针：颜色在第一步就定了，单步监督在"抄答案"

`finetune/trajectory_probe.py` 用推理采样器本身（12 步 Euler，`rescale_t=3`，t = 1, .97, .94, .90, .86, .81, .75, .68, .60, .50, .38, .21, 0）
在 20 个留出对象上逐步解码 x̂₀ 并读颜色（探针 `color_probe.pt` 的调色板准确率），比较两条曲线：
**traj**（真实自由轨迹）与 **forced**（在同一 t 用 GT 构造 x_t 的 teacher forcing，即训练时的输入）：

| 模型 / 变体 | 自由轨迹 acc：t=1 → 中段 → 终点 | teacher-forced acc：t=1 → 中段 → 终点 |
|---|---|---|
| base clean | 0.72 → 0.74 → **0.73** | 0.72 → 0.98 → 0.99 |
| base sam3 | 0.50 → 0.51 → **0.52** | 0.50 → 0.98 → 1.00 |
| base partial（被抹 cell） | – → 0.001 → **0.002** | – → – → 1.00 |
| v4 clean | 0.75 → 0.76 → **0.74** | 0.75 → 0.99 → 1.00 |
| v4 sam3 | 0.58 → 0.61 → **0.60** | 0.58 → 1.00 → 1.00 |
| v4 partial（被抹 cell） | – → 0.006 → **0.006** | – → – → 1.00 |

三个事实：

1. **自由轨迹的颜色准确率从第一步（t=1，输入是纯噪声）起就几乎不再变化**：base clean 0.72 → 0.73，sam3 0.50 → 0.52。
   颜色布局由前 1–3 步决定，后面 9 步只在精修纹理，不会改归属。
2. **teacher-forced 曲线在 t≤0.9 就已 ≥0.92**，因为 x_t=(1-t)x₀+σε 里带着答案。训练（含 v4 的颜色 CE）看到的正是这条曲线：
   留出 color_acc 0.9+ 是在"抄答案"，与推理无关。这解释了 8.4：颜色 CE 从 4.25 降到 0.84，端到端却没有变好。
3. **partial 变体的被抹 cell 在自由轨迹上准确率 ≈ 0**（base 0.002，v4 0.006）——图例/名字对推理输出的贡献实测为零，与 7.6/8.5 的留出读数一致。

结论：在这个采样器里，**唯一诚实的监督点是 t=1 附近的自由轨迹**；任何在 t<1 teacher-forced 的损失都在测量模型抄写 x_t 的能力。

### 9.3 v6：Path A 轨迹训练（运行中）

基于 9.2，`train.py` 增加 `--p_traj/--traj_steps`（`rollout()`）：以概率 p_traj 把样本从纯噪声用真实采样器（同一 cond、无 CFG）
走 k∈{1..traj_steps} 步得到 x_t，再在该 x_t 上算 flow loss + 颜色 CE（`--color_t_min 0.8` 只在 t≥0.8 计颜色项，即采样器实际决定颜色的区间）。
其余样本仍走 teacher forcing 作为约束。不带图例、不带逐 token 文本，回到 v2 的纯图像条件 + LoRA r16：

```
finetune\train_loop.bat 3 finetune\runs\pv_v6 --kinds clean corrupt sam3 --color_probe finetune\color_probe.pt
    --color_weight 0.3 --color_tau 0.03 --color_t_min 0.8 --p_traj 0.5 --traj_steps 3
    --batch_size 4 --grad_accum 4 --max_steps 1500 --check_ts "0.5,0.95,1.0" --wandb
```

9465 个变体（clean 1900 / corrupt 5700 / sam3 1865），留出 99。注意 `--check_ts` 在 `cmd /c` 里必须加引号，
否则逗号被 cmd 当作分隔符（首次启动即因此失败三次）。读数看 `color_acc_traj`（rollout 样本上的颜色准确率，
这是与推理同分布的数字）与 t=1.0 的留出检查。wandb run `pv_v6`，1500 步 4.9 h（前 40 min 与 GeoSAM2 评测争显存，
两边一度卡死 10 min：训练进程缓存 19–22 GB + GeoSAM2 自动分割 ~10 GB 撑满 32 GB，之后改为分时运行）。

### 9.4 v6 结果：自由轨迹上的颜色准确率首次明显上升

留出检查（n=33/类，t∈{0.5,0.95,1.0}，teacher-forced）：

| step | clean MSE / color_acc_hi | corrupt MSE / color_acc_hi | sam3 MSE / color_acc_hi |
|---|---|---|---|
| 250 | 0.304 / 0.645 | 0.254 / 0.659 | 0.266 / 0.622 |
| 500 | 0.279 / 0.749 | 0.239 / 0.743 | 0.269 / 0.706 |
| 1000 | 0.278 / 0.755 | 0.232 / 0.754 | 0.268 / 0.702 |
| 1500 | 0.276 / **0.766** | 0.229 / **0.761** | 0.263 / **0.718** |

训练窗口 `color_acc_traj`（rollout 样本）从 0.45–0.64 升到 0.77；轨迹 flow loss（`per_kind.traj`）0.46 → 0.29。

**轨迹探针**（与 9.2 同一 20 个留出对象、同一采样器；这是唯一与推理同分布的读数）：

| 模型 | clean 自由轨迹终点 acc | sam3 自由轨迹终点 acc |
|---|---|---|
| base | 0.733 | 0.520 |
| v4 | 0.744 | 0.603 |
| **v6** | **0.850** | **0.621** |

v6 是第一个把自由轨迹准确率明显推高的版本（clean +11.7、sam3 +10.1 个点），而 v4 在同一读数上只有 +1/+8。
这验证了 9.2 的判断：把监督放到采样器真正经过的 x_t 上，颜色决定才会改变。端到端硬样本（`eval_fidelity` + `eval_parts`）见 10.3。

## 10. 统一 3D 评测与 GeoSAM2 对照（9-06）

### 10.1 `finetune/eval_parts.py`：按独立 GT 计分，不再只测"配色遵从"

`eval_fidelity` 只回答"输出是否保持了 2D 图的配色"，隐藏部件不计分、未覆盖部件被跳过、SAM3 漏绑的部件在目标里本来就是灰色。
新评测把预测当作**表面的一个划分**（切割器真正拿到的东西），对**所有** GT 部件计分：

- 在 GT 部件上按面积均匀采 6 万点（每部件 ≥40），在预测表面最近点读标签：外部方法读面标签，SegviGen 读纹理最近调色板色（灰 = 未标）。
- `miou`：每个 GT 部件与任一预测段的最佳 IoU 均值（未标段也算一段：一块恰好盖住某部件的未命名区域仍是正确的切口）；
  `miou_matched`：匈牙利一对一匹配后的均值，两个部件被合成一段要付两次代价。
- `sem_miou`：按**唯一名字**计——同名部件（两只 boot）合成一个 GT 区域，与所有带该名字的预测段的并集求 IoU。
  这是语义分割视角，也是"按名字提示"的管线能公平比较的口径；无名预测计背景。
- `name_acc`：GT 部件的多数段带正确名字的比例；`small_part_recall`：面积 <1% 的部件 IoU≥0.5 的比例；
  `over_seg_parts` / `under_seg_segments`：被拆到 ≥2 段（各 ≥20%）的部件数 / 跨 ≥2 个部件（各 ≥20%）的段数；
  `boundary_f1`：类无关点边界 F1；`unlabelled_share`：灰/未标/未覆盖的采样比例。

对 SegviGen 输出，颜色→名字来自变体 meta（Path A 分组）或 `sam3_to_2dmap` 的 legend（部署路径）；
对外部方法，来自 `labels.json` 的 id→名字。两者在同一 GT、同一采样点上计分。

### 10.2 GeoSAM2 接入（Windows）

- 仓库 `E:\AI_New\ModelGen\GeoSAM2`，独立 venv `.venv_geosam2`（torch CUDA、`opencv-python==4.14`——5.0 无 OpenEXR，深度图读成 None），
  权重 `ckpt/geosam2.pt`。`utils/mode_ext.py` 的 C++/OpenMP 扩展在无 MSVC 时不可编译，加了 torch `scatter_add` 回退（结果一致）。
- 渲染：`finetune/geosam2_render.py` 用 bpy 模块复用其 `geosam2_render.py` 的相机/归一化/深度法向输出（12 视角、1024²、EEVEE），
  修 Blender 4.1+ 去掉的 `use_auto_smooth`，并把 view transform 设为 Standard（AgX 会把示例的中灰渲成深灰）。
  hard set 20 + 外部资产 4 个（狗/椅/机器人/人体），每个 8–11 s。
- SAM3 掩码：`finetune/geosam2_masks.py` 在其渲染上跑 SAM3 + 概念库 v3（与 Path A 同一套），每视角写标签图
  （小掩码先画、先占先得，与 `bind_masks` 一致）和 `summary.json`（各视角找到的部件数、前景覆盖率、`best_view`）。
  20 个 hard 对象在最佳视角平均找到 ≈85% 的名字、覆盖 0.6–0.99 的前景。
- 驱动两种：
  - `geosam2_run.py`（**single**，GeoSAM2 默认用法）：一个视角的 SAM3 标签图做 mask prompt，对面视角跑其 SAM2 自动分割补全，
    `--enable-postprocess --pa 0.02`。提示视角选 `match`（与 SegviGen az0 轮廓 IoU 最高的视角，两方法同一输入视角）或 `best`。
  - `geosam2_dual.py`（**dual**）：v 与 v+6 两张 SAM3 标签图都做 prompt，同名同 id，不跑自动分割。
  - 两者都另存 `*_filled.npy`：把 GeoSAM2 留下的未标面（0/999）按最近已标面质心填充——SegviGen 从不留空，这才是同口径；raw 版展示它留了多少空。
- 输出：`datasets/geosam2/results/<obj>/<config>/`，`labels.json` 记 id→名字、提示视角、耗时。
  `finetune/geosam2_to_glb.py` 把面标签转成每部件一个平色材质子网格并转回原资产坐标系（其导出是 Z-up + 顶点色，bpy 渲出来是白的），
  可用 `render_cond_view.py` 与 SegviGen 输出同机位渲染，也可直接拆件。

首个对象（c35ddc…，16 个 GT 部件、11 个名字）的观察：GeoSAM2 single 只给 SAM3 命中的区域上标签，
对面视角的自动分割在该对象上**没有补进任何段**（其 `filter_mask_area(alpha=1)` 要求掩码在其它视角不比锚视角大，
侧视角做锚时几乎全被过滤），47% 的面留白；dual 把留白降到 28%，填充后为 0。
单对象 mIoU：SegviGen base 0.29 / GeoSAM2 single 0.31（raw）→ 0.23（filled） / dual 0.26 → 0.21。
SAM3 在 1024² 灰材质渲染上的 2D 质量本身是主要瓶颈（chest pack 吞掉 torso，两只 boot 一个掩码）。

### 10.3 hard set 20 对象同表对比（`eval_parts`，同一 GT、同一采样点）

三类 SegviGen 条件：**base**（基座，Path A 的 `sam3_az0` 变体图：SAM3 掩码经 GT 绑定，未绑定的为灰）、
**v6**（同一张图，9.3 的 LoRA）、**raw**（部署路径：`sam3_to_2dmap` 直接在 az0 渲染上出图，颜色→名字来自 legend，无任何 GT 信息）。
GeoSAM2 四种：single/dual × match/best（10.2），raw 与 filled 两种口径。

| 配置 | mIoU | mIoU 一对一 | 语义 mIoU | 命名准确率 | 小件召回 | 边界 F1 | 留白 | 段数 | 过分割件 | 欠分割段 |
|---|---|---|---|---|---|---|---|---|---|---|
| SegviGen base | 0.278 | 0.208 | 0.265 | 0.588 | 0.013 | 0.666 | 0.126 | 5.6 | 2.95 | 2.80 |
| **SegviGen v6** | **0.296** | 0.229 | **0.274** | **0.620** | 0.037 | **0.704** | 0.129 | 5.7 | 3.10 | 2.70 |
| SegviGen raw（部署） | **0.304** | 0.239 | 0.219 | 0.354 | 0.047 | 0.668 | 0.142 | 6.4 | **2.65** | 2.95 |
| GeoSAM2 single match | 0.284 | 0.238 | 0.187 | 0.342 | 0.070 | 0.616 | 0.174 | 7.3 | 4.55 | 3.25 |
| GeoSAM2 single match filled | 0.266 | 0.219 | 0.185 | 0.362 | 0.062 | 0.549 | 0 | 7.4 | 4.05 | 3.15 |
| GeoSAM2 single best | 0.294 | **0.250** | 0.194 | 0.327 | 0.067 | 0.604 | 0.150 | 8.5 | 4.70 | 3.90 |
| GeoSAM2 single best filled | 0.281 | 0.236 | 0.193 | 0.363 | **0.080** | 0.564 | 0 | 8.5 | 4.30 | 4.00 |
| GeoSAM2 dual match | 0.276 | 0.227 | 0.213 | 0.424 | 0.037 | 0.628 | 0.106 | 7.1 | 4.35 | 3.65 |
| GeoSAM2 dual match filled | 0.264 | 0.213 | 0.211 | 0.447 | 0.037 | 0.599 | 0 | 7.1 | 4.05 | 3.45 |
| GeoSAM2 dual best | 0.283 | 0.237 | 0.212 | 0.364 | 0.049 | 0.625 | 0.090 | 7.8 | 5.10 | 3.85 |
| GeoSAM2 dual best filled | 0.275 | 0.227 | 0.211 | 0.386 | 0.046 | 0.617 | 0 | 7.8 | 4.60 | 3.70 |

逐对象 mIoU 对 base：v6 **14 优 / 5 差**；raw 11/8；GeoSAM2 single best 12/8，dual best 11/9，其余 ≤ 10/10。
平均每对象 13.6 个 GT 部件、8.8 个唯一名字，112 个 <1% 面积小件。耗时（不含渲染与 SAM3）：GeoSAM2 single 中位 ~30 s、dual ~20 s；
SegviGen 推理 ~45 s。

**怎么读这张表**

1. **所有方法都在 0.26–0.30 的 mIoU 区间**，一个部件 IoU≥0.5 的只有 2–2.7 个/13.6 个。瓶颈在上游：
   SAM3 在这批硬对象上平均只命中 ≈85% 的名字，命中的掩码又常吞并邻件（chest pack 吞 torso、两只 boot 一个掩码），
   两条 2D→3D 的路径（SegviGen 生成式上色 / GeoSAM2 几何传播）都只能忠实地把这个 2D 结果搬到 3D。
   要把 mIoU 拉出这个区间，改 2D 掩码质量比换 3D 传播方式更有效。
2. **v6 是唯一在所有主指标上都优于 base 的 SegviGen 版本**（mIoU +0.018、语义 mIoU +0.009、命名 +3.2 点、边界 F1 +3.8 点、14/5），
   与 9.4 的探针和 `eval_fidelity`（0.617 → 0.663，11/8）三个口径一致。
3. **命名准确率上 base 有 GT 帮忙**：`sam3_az0` 变体的颜色→名字是 GT 绑定的，raw 一去掉绑定就从 0.59 掉到 0.35。
   与 GeoSAM2 公平比较应看 raw 这一行：语义 mIoU raw 0.219 vs GeoSAM2 0.19–0.21，命名 raw 0.354 vs GeoSAM2 0.33–0.45（dual 更高，
   因为两个视角都带名字）。**在同一套 SAM3 掩码下，两条路径的语义指标基本打平**。
4. **SegviGen 的优势在边界和完整性**：边界 F1 0.67–0.70 vs GeoSAM2 0.55–0.63，过分割件 2.7–3.1 vs 4.1–5.1，段数 5.6–6.4 vs 7.1–8.5。
   GeoSAM2 逐面投票 + 后处理留下更多碎块，这正是用户最初反对"2D 投票到 3D 会碎"的现象；填充留白后边界还会再降（single 0.616 → 0.549）。
5. **GeoSAM2 的优势在小件召回**（0.06–0.08 vs SegviGen 0.01–0.05）和一对一 mIoU（single best 0.250）：几何传播不会像生成式上色那样把
   细小部件涂成邻件颜色。它的默认用法（single + 自动分割）在名字提示下留白 15–17%，dual 降到 9–11%。
6. **GeoSAM2 对超大网格不实用**：外部资产椅/人体/机器人为 ~200 万面，其逐面采样 + Python 后处理 500–960 s/对象，机器人两次都因
   CPU 内存（182 GB 申请）失败；SegviGen 在体素上工作，与面数无关。接 GeoSAM2 前需先减面。

外部资产（狗、椅，同机位；SegviGen 已转回原坐标系）：`datasets/geosam2/ext_vis/ext_compare.png`。
两者在这类简单资产上给出几乎相同的切分（头/耳/身/腿；背/座/腿），GeoSAM2 把椅子横撑单独分出，SegviGen 则严格跟随 2D 图的配色。

**结论与下一步**

- GeoSAM2 作为替代方案：在同一 SAM3 掩码下语义指标与 SegviGen 部署路径打平，边界更碎、留白需填、超大网格不可用；
  它值得保留的是"两视角提示 + 几何传播"这一思路对小件的保真，可作为 SegviGen 灰区（未上色部件）的补充而不是替代。
- SegviGen 这条线：v6 证明了**监督放到采样轨迹上就有端到端收益**，且没有任何文本注入。下一步按收益排序：
  (a) 提升 SAM3 2D 掩码（概念库已 +8 点 mIoU；再做难负例/多视角一致性）——这是两条路径共同的上限；
  (b) v6 加大 `p_traj`/`traj_steps`、在 sam3 变体上单独加权，看 sam3 自由轨迹 acc 能否从 0.62 继续升；
  (c) 用 `eval_parts` 固定 20 硬 + 35 回归对象作为选型门槛（mIoU +5 点、回归 −1 点以内），不再用 fidelity 做验收。

## 11. 外部资产（`3D拆件.zip`，10 个模型）三路对照（9-06）

`finetune/ext_bench.py`，全部结果与图在 `datasets/ext_bench/REPORT.md`（总览 `compare_front/back.png`，每资产 512 分辩率正/背面 `detail/<key>.png`，
可拆件的 GLB 在 `<key>/`）。没有 GT，指标是结构统计与方法间一致性。五列：SegviGen 原生 `full_seg`（无 SAM3）、base+SAM3、**v6+SAM3**、
GeoSAM2 single（同视角）、GeoSAM2 dual（同视角，填充）；SAM3+概念库 v3 对前视图 51 个提示词命中 48 个。

| 方法（10 资产均值） | 段数 | 碎片/段 | 留白 | 边界密度 | 与 v6 一致性（类无关 / 按名） | 秒 |
|---|---|---|---|---|---|---|
| SegviGen 原生 | 5.5 | **1.20** | 0 | **0.018** | 0.51 / – | 65 |
| SegviGen base + SAM3 | 4.4 | 1.94 | 0.04 | 0.034 | 0.86 / 0.75 | 52 |
| v6 + SAM3 | 4.7 | 2.11 | 0.03 | 0.040 | – | 49 |
| GeoSAM2 single（raw） | 5.1 | 1.57 | 0.08 | 0.051 | 0.72 / 0.55 | 75 |
| GeoSAM2 dual（填充） | 4.7 | 1.83 | 0 | 0.037 | 0.71 / 0.56 | 89 |

- SegviGen 原生是另一套口径：有机体极粗（小狗 3 段、跑车 2 段）、人造物按几何拆细（椅子 10 段），无名字；块最整但不受控。
- v6 相对 base：多找回小件（米老鼠 ear/foot、飞机 propeller）、留白更少（机器人 0.08 → 0.02），但碎片略多，跑车/米老鼠背面未观测的轮子、后脑更容易变色——
  硬样本上的边界 F1 优势在这批资产上没有复现。
- GeoSAM2 与 v6 正面一致、背面不同：重复部件（4 个轮子）颜色一致是几何传播的优势，但后脑、后腿有异色补丁；raw 留白 7–8%（飞机 32%）。
  200 万–500 万面网格用 `decimate_glb.py` 减到 20 万面后 75–89 s/资产。
- 对初衷（保留 SAM3 的语义 + SegviGen 的整块）：两条 2D→3D 路径语义命中相同，差在未观测面。若继续走 SegviGen 线，训练或推理端需要加"未观测面整块"约束
  （图割 / 连通块合并）；务实组合是 v6 出主分割、GeoSAM2 dual 只校正重复件一致性与灰区。

## 12. 路线总结与下一步（9-07，GeoSAM2 消融之后）

GeoSAM2 消融（`REPORT_geosam2_ablation.md`）把提升类路线的上限摸清了：所有传播实现 mIoU 0.24–0.28、边界 F1 ≤0.56，v6 仍是 hard 集所有主指标最好的方法。回到 SegviGen 线。

### 12.1 现有路线

```
渲染正面图 ─► SAM3 + 概念库（E_0 + E_name 偏移，2D mIoU 0.288→0.368）─► 掩码上色成 2D 引导图 + legend（颜色→名字）
        ─► BiRefNet 去背 ─► DINOv3 ViT-L/16 token（冻结）─► TRELLIS.2 纹理 DiT 1.3B + LoRA r16（v6）
        ─► 25 步 Euler + CFG 去噪 output_tex_slat ─► 解码为逐体素颜色 ─► 最近调色板 ─► 面标签 ─► 拆 mesh（原拓扑/UV 不动）
```

我们在上游之上加的四层：**语义入口**（SAM3 + 概念库 + legend）、**数据**（PartVerse 14 303 物体 Grok 重标注 + 人工复核 2000 条；clean / corrupt / sam3 三类变体，把推理噪声放进训练分布）、**训练**（LoRA + 轨迹监督）、**评测**（`eval_parts`：独立 GT、全部部件计分，取代只测配色遵从的 `eval_fidelity`）。

### 12.2 六轮微调学到的

| 版本 | 改动 | 结果（hard 20 / 探针） | 结论 |
|---|---|---|---|
| v1 / v2 | LoRA + v-pred MSE，PartVerse 变体，双视角 | MSE −8 %，颜色归属 27/28 → 26/28 | 重建损失不约束归属 |
| v3 | 概念库 + 图例 token + 物体名 token | 图例 shuffle 前后留出 MSE 到小数点后 4 位相同 | 输出对文本 token 不敏感（注意力稀释 + 损失不需要文本）；概念库本身有效，保留 |
| v4 | 显式颜色 CE（单步 x̂₀）+ partial 变体 + 解耦图例注意力 | 颜色 CE 4.25 → 0.84，fidelity 0.617 → 0.605 | teacher-forced 的 x_t 带着答案，单步监督在抄写 |
| v5 | 名字直接加到 DINO patch token | 带文本 fidelity 0.19，去文本 0.598 | 名字只被当"这里该上色"的开关 |
| **v6** | **rollout 1–3 步的自由轨迹 + t≥0.8 颜色 CE**，纯图像条件 | 自由轨迹 acc clean 0.733 → **0.850**、sam3 0.520 → **0.621**；mIoU 0.278 → **0.296**、边界 F1 0.666 → **0.704**、命名 0.588 → 0.620，逐物体 14/5 | **唯一全面优于基座的版本**：监督必须放在采样器真正经过的 x_t 上 |

两个结构性事实决定了后面怎么走：(1) 颜色布局在采样前 1–3 步（t ≥ 0.8）就定了，后 9 步只修纹理；(2) 四次文本注入全部失败，原因不在实现而在损失——2D 图已把可见部件的颜色给全了，模型没有理由读文本。语义只能从两处进入：2D 图本身的质量，和轨迹上的显式颜色监督。

### 12.3 现在的位置

| | v6 + SAM3 | 最好的提升类（SAM3 tracker p2 + GeoSAM2 提升） |
|---|---|---|
| mIoU / 边界 F1 / 命名 | **0.296 / 0.704 / 0.620** | 0.276 / 0.555 / 0.415 |
| 小件召回 | 0.037 | **0.074** |
| 外部资产 | 与提升类类无关一致性 0.8；背面轮子 / 后脑易变色，碎片 2.1/段 | 重复件一致，薄件碎 |
| 时间 | ~45–60 s | ~10 s |

共同上限：所有方法 mIoU 都在 0.2–0.3——SAM3 在 hard 集平均只命中 ≈85 % 的名字、掩码常吞邻件，两条路径都只是把这个 2D 结果搬到 3D。

v6 线自己的短板：(a) sam3 条件下自由轨迹 acc 只有 0.62（clean 0.85），说明模型对真实 SAM3 图（边界粗、灰区、错绑）还没适应；(b) 部署路径命名 0.354 vs GT 绑定的 0.588——差距来自 legend 的颜色→名字绑定错误，训练时从未见过；(c) 小件被涂成邻件颜色；(d) 单张正面图，背面完全靠先验；(e) 逐体素最近调色板是硬判决，没有任何空间正则，碎片 2.1/段。

### 12.4 下一步（按收益 / 成本）

| # | 动作 | 针对 | 判据（hard 20 `eval_parts` + 20 留出探针） | 成本 |
|---|---|---|---|---|
| 1 | **v7 = v6 加码**：`p_traj` 0.5 → 0.8、`traj_steps` 3 → 6（覆盖到 t≈0.6）、sam3 变体损失权重 ×2（推理采样器 `guidance_strength` 为 1.0，即无 CFG，rollout 与推理已一致） | 12.3(a) | sam3 自由轨迹 acc 0.62 → 0.70+，mIoU +0.02 | 一轮 5–8 h，零代码风险 |
| 2 | **训练条件换成部署分布**：sam3 变体的 2D 图改用 `sam3_to_2dmap` 原样输出（legend 绑定、含错绑与灰区），3D 目标仍按 GT 部件着色——让模型学"纠正 2D 错误"而不是"复制 2D" | 12.3(b) | 部署路径命名 0.354 → 0.45+，raw 行与 base 行差距收窄 | 数据重生成 1 天 + 一轮训练 |
| 3 | **推理端面图正则化**（零训练）：解码出的逐体素颜色对调色板的距离做 unary，面邻接图上二面角 / 凹性做 pairwise，graph cut 一次；同色小连通块并入邻块 | 12.3(e)、外部资产碎片 | 碎片 2.1 → ≈1.2，边界 F1 +0.02–0.05；同一套代码也可给提升路线用 | 1–2 天 |
| 4 | **第二张条件图 = 背面**：用 SAM3 tracker 把正面掩码传播到背面视角（消融里已跑通，10 s），两视角 DINO token 拼接作条件。v2/v3 的双视角在 MSE 训练下无效，但那时监督根本不动颜色决策，在 v6 的轨迹监督下值得重测 | 12.3(d) | 背面部件 mIoU、外部资产背面变色 | 数据已有双视角字段；一轮训练 |
| 5 | **上游 2D**：概念库继续扩名（难负例、多物体上下文）；tracker 传播回投正面做多视角一致性修正；细杆部件高分辨率 tile | 共同上限 | 2D mIoU 每 +0.05 → 3D 约 +0.03 | 持续 |
| 6 | **小件补丁**：v6 主分割 + 提升路线（tracker p2）在 v6 的块内做面级名字投票，小件按提升路线的段覆盖 | 12.3(c) | 小件召回 0.037 → 0.06+，边界不降 | 1 天 |
| 7 | 不再做：文本 / 图例注入（四次证伪）、DINOv3 微调、GeoSAM2 重训 | | | |

顺序：1 与 2 可合并成一轮（v7 数据 + 超参），3 与之并行；4 在 v7 结果出来后决定；5 长期；6 作为工程化时的兜底。每一步都以 `eval_parts` hard 20 的 mIoU / 边界 F1 / 碎片和 35 个回归对象（−1 点以内）做门槛，不再用 fidelity 验收。
