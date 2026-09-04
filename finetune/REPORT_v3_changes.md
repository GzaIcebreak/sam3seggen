# SegviGen 微调改动报告（v1 → v3，截至 2026-09-04）

本文汇报我们在 SegviGen `full_seg_w_2d_map` 之上做的全部改动：LoRA 域适配（v1/v2）的结果与结论、
PartVerse 部件名重标注与补渲染、SAM3 概念库（Stage A/B）、图例 token + 双视角条件（Stage D）的设计，
以及与原论文损失函数/条件结构的逐项对照（第 6 节）。代码入口见同目录 `README.md`。
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
