# Mask RankGNN（候选级排序 + 掩码级叠涂）评测报告

日期：2026-09-09。接 `REPORT_concept_bank_v5_eval.md` §7–8 的结论：v3 冻结解码器的 200 个候选里已有正确掩码（像素上限 0.92），
错在 SAM3 自己的分数把它排在第 38 位；v5 的逐像素竞争把「选掩码」变成「选像素」，碎片化是算子的定义。
本轮把两件事分开做：**排序器**（谁是好候选）和**画法**（被选中的硬掩码整块叠涂，不再逐像素决定）。

代码：`finetune/mask_rank.py`（特征 / 模型 / 训练）、`mask_rank_feats.py`（离线抽特征）、`mask_rank_paint.py`（重绘评测，含新画法与块指标）。
数据：`runs/mask_rank_v3/feats2_{train,holdout,mw}`（v3 bank，每名字 top-32 候选，3600 / 400 / 168 张）。

---

## 0. 技术路线

设计原文：`docs/superpowers/specs/2026-09-09-mask-rankgnn-design.md`。v5 §7 的 oracle 把问题收成一句：冻结 SAM3 的候选里**已经有**正确掩码（像素上限 0.92），错在分数把正确的排到第 38 位；v5 的逐像素 argmax 把「选掩码」变成「选像素」，碎片化是算子定义。本轮只做后置，不碰 SAM3、不碰 v3 bank。

### 总路线：候选级排序 + 掩码级叠涂

**缺口。** SAM3 每个 (名字, query) 独立打分，不看同图其他候选——不知道 `arm` 被 `body` 包住、不知道两个候选高度重叠。v4 改绝对校准，v5 改像素竞争，候选之间的相对可信度从没被建模。部署要的是整块颜色，不能再按像素投票。

**做法。** 两件事拆开。排序器给每个候选一个 `keep ∈ [0, 1]`；画法只叠涂被选中的硬掩码（sigmoid > 0.5，全分辨率），不再经过 `paint_argmax`。失败时 `keep` 不用、退回 v3 的 score ≥ 0.4 叠涂。

**实测 / 判定。** 画法过关（无噪点）；排序器同分布有效、跨分布失效，**不部署**。上数据前又试了两条顶会配方（Venice-H1、EASE-DETR），结论见下。

### 画法：`paint_select` / `paint_hybrid`

**缺口。** v5 的碎斑来自「每个像素独立 argmax」。即便排序器完美，只要还走 `paint_argmax`，块也不会干净。

**做法。** `paint_masks`：被选中的候选取硬掩码，按 `small`（每名字并集、小的先涂 = v3）或 `keep`（按权重降序、不覆盖）叠上剪影。`select`：`keep` 或 `score` ≥ κ 的留下，可选每名字保底一个（`+fb`）。`hybrid`：从 v3 集合（score ≥ 0.4）出发，排序器有把握坏的删（keep < d）、有把握好的加（keep ≥ a），拿不准的地方 v3 说了算——Venice-H1 的失败门控用到集合上，而不是 top-1。过关加块指标：`块比` = 涂出连通块 / GT 块，`碎块` = 小于剪影 0.5% 的块数。

**实测。** 留出集肉眼 select / hybrid 与 v3 一样整块，argmax 对照全是噪点。hybrid d0.2 a0.6 等覆盖下涂对 +3.7、涂错 −3.0，块数少 1/4。`small` 一律比 `keep` 好 1–2 个点。

**判定。** 画法过关，已进 `sam3_to_2dmap --assign rank`。剩下的全是排序器对不对。

### MaskRankGNN（线性边，主排序器）

**缺口。** SAM3 自己的分数在 top-32 里对「precision ≥ 0.7」的 AUC 是 0.50——头部就没有信息。需要一个看着候选集合、输出相对序的后置模块。

**做法。** 节点 = 15 标量（分数 / 面积 / 框 / 质心 / 质量）+ 概念向量 256 + 掩码内 FPN 均值 256 + 多尺度网格签名 737（Venice-H1 的 4/8/16 avg-max 池化 + 边界能量 + 出剪影比）。边 = IoU / 双向包含 / 中心距 / 同名 / log 分数比 / 排名差（7 通道，线性投影成注意力偏置）。3 层多头消息传递，0.42–0.52 M。目标 `g = prec · √rec`（偏准不偏大）；损失 = 0.2 MSE + 0.8 成对 hinge + BCE(prec ≥ 0.8)。成对后来改成只在同名内比（`--rank_scope name`）。最佳：`grid_name_lr` ep 11。

**实测。** val good-AUC 0.787、kept 精度 0.737；4–5 个 epoch 后 val 到平台，train 继续涨到 0.91——在记物体。概念向量是关键通道（去掉 −0.05），FPN 几乎不贡献，网格 +0.01。留出集 hybrid 对 v3 Pareto 改进；mw 涂错 +4.6～7.7，只加/只删两步都翻。3600 张 / 1800 物体 / 280 名字不够。

**判定。** 未上线。特征不是瓶颈，数据量是。7 通道线性边里的几何通道给记忆开了后门（见 EASE）。

### Venice-H1（`--arch venice`，上数据前对照）

**缺口。** 论文（arXiv 2606.22546）是后置、集合级 Transformer + 失败门控：默认选择 82–93% 是对的，只在有把握 Query 0 错时才换，有害切换 < 0.5%。问的是：同样配方在我们「top-1 经常错」的候选集上还成不成。

**做法。** 每名字 32 个分数排序的候选为一集。2 层 MLP → 3 层 pre-norm Transformer（8 头、GELU、**无边特征**）。Gain 头回归 `IoU_i − IoU_0`，Failure Gate 用 `[mean h; max h; h_0; scores; areas]` 判「score-top-1 不是最好」。损失 = focal BCE(γ=2, 自动正样本权重) + 5 × smooth-L1。推理 `p_fail > τ` 才换成 argmax gain，否则留默认。他们的 query 向量换成概念向量（dump 没有 SAM3 解码器输出）。`venice_s`：d=128，wd 0.1。

**实测。** 默认 top-1 IoU 0.384，best-of-K 0.635，门控后最好 0.405（+2.1），gate AUC 0.66。有害切换占切换的 37%。两个前提不成立：我们 81% 的名字 top-1 都不是最好，「失败门控」退化成「总是换」；IoU-gain 偏爱大掩码。重绘：涂错 0.222 → 0.304，块比 4.1 → 9.6；mw 两项都更差。ext_bench 上米老鼠的头被 ear 吃掉、菠萝叶子换色。

**判定。** 不进下一轮。hybrid 只借了它「拿不准就不动基线」的集合版思路，不用 gate + IoU-gain 配方。

### EASE-DETR（`--edge_mode ease`，当前首选边）

**缺口。** EASE-DETR（CVPR 2024）的诊断是 DETR query 之间恶性竞争：每个 query 都想当领先者，收敛慢、重复框多。它不另训一个排序头，只改注意力里**谁可以看谁**——用相对排名 × IoU 做成对衰减，让落后且重叠的 query 少打扰领先者。我们的 7 通道线性边（IoU / 包含 / 中心距 / 同名 / 分数比 / 排名差）在 1800 个物体上 train AUC 0.91、mw 失效，怀疑几何通道在给「这个物体长什么样」开后门。问：把边收窄到论文那一个标量，排序器还能否工作、跨分布会不会稳一点。

**做法。** 其余与 `grid_name_lr` 完全相同（同名内排序、lr 1e-3、bce 1.0、网格签名、g = prec · √rec）。唯一改动在 `RankMessagePassing`：不再把 7 维边线性投成注意力偏置，只取

```
lead_{ij} = −sign(rank_i − rank_j)          # +1：行 i 比列 j 领先（分数更高 / 排名更前）
bias_{ij} = log σ( MLP( lead_{ij} · IoU_{ij} ) )
```

`bias` 加到注意力 logit 上，等价于把注意力权重乘一个 (0, 1) 的 decay。MLP 是 `1 → 16 → heads`，每层每头一套衰减。领先且重叠的邻居衰减小（可以说话），落后且重叠的衰减大（少打扰）。没有包含比、没有中心距、没有同名 flag——这些要从节点特征和「同名内成对」的损失里自己长出来。权重：`runs/mask_rank_v3/ease/rank_epoch8.pt`（val good-AUC 最高的 epoch）。部署仍走 `paint_hybrid`（不是 Venice 的 top-1）：`--assign rank --rank_drop 0.1 --rank_add 0.9 --rank_model .../ease/rank_epoch8.pt`。

**实测。** 离线与线性边持平（AUC 0.783 vs 0.787，kept 精度 0.718 vs 0.737）。重绘上更好：留出集 hybrid d0.2 a0.75 为 0.631 / 0.192 / 0.177（线性边最佳 hybrid 0.612 / 0.188 / 0.200）；mw 上同涂错（0.189 vs 0.194）多涂对 5 个点、少 4.5 个点灰（0.669 / 0.189 / 0.141 vs 0.619 / 0.194 / 0.186）。ext_bench 肉眼与 RankGNN 列几乎不可分，跑车车身更接近 v3 的蓝、剑同样会丢掉部分 guard。对 v3 的 mw 涂错 0.140，EASE 仍高 4.9 个点，加进来的像素对错约 1.2 : 1。

**判定。** 边收窄反而泛化更好，证实线性边的几何通道在记忆。作为**下一轮扩数据时的默认边**（`--edge_mode ease`），不是部署产物——mw 涂错仍不可接受。不用 Venice 的目标，保留 `g = prec · √rec`；hybrid 阈值等新数据的留出集再扫。

---

## 1. 画法：`paint_masks`

路线说明见 §0。

- `select`：keep ≥ κ 的候选取硬掩码（sigmoid > 0.5，全分辨率，与 `sam3_to_2dmap --threshold` 同一路），
  `small` = 每名字并集、小的先涂（v3 规则，`--select score --kappa 0.4` 精确复现 v3）；`keep` = 单掩码按 keep 降序涂、不覆盖。
  `+fb` = 每名字至少保留 keep 最高的一个。
- `hybrid`（Venice-H1 的失败门控思路用到集合上）：从 v3 的集合（score ≥ 0.4）出发，
  剔掉排序器有把握是坏的（keep < d），补上有把握是好的（keep ≥ a）。排序器拿不准的地方 v3 说了算。
- `argmax`：v5 算子，作对照。
- 块指标：`块比` = 涂出的 4 连通块数 / GT 块数（1 = 和标注一样整），`碎块` = 小于剪影 0.5 % 的块数/图。`clean` = `clean_components(0.005)`。

肉眼（`runs/mask_rank_v3/paint_grid/dump_holdout/*.png`）：select / hybrid 与 v3 一样是整块，argmax 列在巴士车身、房子墙面上全是噪点。
**画法这一层的碎片问题已经解决**，剩下的全是排序器对不对。

## 2. 排序器

路线说明见 §0。`MaskRankGNN`：节点 = 15 个标量 + 概念向量 256 + 掩码内 FPN 均值 256 (+ 网格签名 737)，边 = IoU / 双向包含 / 中心距 / 同名 (+ log 分数比 / 排名差)，
3 层带边偏置的多头注意力消息传递，0.42–0.52 M 参数。目标 g = prec · √rec；损失 = 0.2 MSE + 0.8 成对排序 hinge + BCE(prec ≥ 0.8)。

离线代理（feats 留出集，400 张）。`good` = precision ≥ 0.7（§7 oracle 的「保留集」）；`kept@0.5` = keep ≥ 0.5 被选中的候选：

| run | 改动 | val good-AUC | kept 精度 | good 召回 | train good-AUC |
|---|---|---:|---:|---:|---:|
| SAM3 自己的分数 | – | **0.502** | – | – | 0.468 |
| `rank.pt` | 原方案（跨名成对） | 0.765 | 0.693 | 0.48 | 0.915 |
| `name` | 成对只在同名内 | 0.775 | 0.725 | 0.44 | 0.915 |
| `name_lr` | + lr 1e-3, bce 1.0 | 0.766 | 0.662 | 0.54 | 0.943 |
| `grid_name` | + 网格签名 + 排名边 | 0.784 | 0.696 | 0.51 | 0.914 |
| `grid_name_lr` (ep 11) | 同上 + lr 1e-3, bce 1.0 | **0.787** | **0.737** | 0.45 | 0.913 |
| `abl_novis` | grid 去 FPN 视觉向量 | 0.773 | 0.719 | 0.44 | 0.851 |
| `abl_notext` | grid 去概念向量 | 0.738 | 0.725 | 0.32 | 0.780 |
| `abl_geom` | 只留标量 + 网格 + 边 | 0.733 | 0.715 | 0.32 | 0.777 |

读法：
- **SAM3 的分数在自己的 top-32 里对「precision ≥ 0.7」的 AUC 是 0.50**——完全随机。§7 说的「正确掩码排第 38」不是尾部现象，是分数在头部就没有信息。
- 排序器 4–5 个 epoch 后 val 就到 0.775–0.787 的平台，之后只有 train 在涨（0.91）：**在记物体，不是在学**。3600 张图只有约 1800 个物体、280 个名字。
- 概念向量是关键通道（去掉 −0.05 AUC），FPN 视觉向量几乎不贡献，网格签名 +0.01。特征不是瓶颈，数据量是。
- `rank-top1 IoU` 这个代理对精度导向的目标不公平（挑出的是小而准的掩码），本报告改看 AUC / kept 精度。

## 3. 重绘评测（真正的裁判）

`grid_name_lr/rank_epoch11.pt`，`runs/mask_rank_v3/paint_grid/`。留出集 240 张（与 v5 报告同一集合）：

| 画法 | 涂对 | 涂错 | 未涂 | 块比 | 碎块 |
|---|---:|---:|---:|---:|---:|
| v3 叠涂 @0.4 | 0.582 | 0.218 | 0.200 | 4.69 | 28.2 |
| v3 + clean | 0.587 | 0.221 | 0.192 | 0.77 | 0 |
| v5 算子 argmax（v3 bank） | 0.650 | 0.304 | 0.045 | 17.1 | 135 |
| select κ0.5 small +fb | 0.582 | 0.193 | 0.225 | 5.06 | 33.0 |
| select κ0.7 small | 0.437 | **0.112** | 0.450 | 1.99 | 10.0 |
| **hybrid d0.2 a0.6 small** | **0.619** | **0.188** | 0.193 | 3.59 | 20.0 |
| hybrid d0.2 a0.6 small + clean | 0.624 | 0.191 | 0.186 | 0.61 | 0 |
| hybrid d0.2 a0.75 small | 0.602 | 0.179 | 0.219 | 2.93 | 14.8 |

留出集上 hybrid 是对 v3 的 Pareto 改进：等覆盖下涂对 +3.7、涂错 −3.0，块数还少 1/4。`keep` 顺序一律比 `small` 差 1–2 个点——
面积小先涂这条 v3 规则比排序器的置信度更可靠。hard 子集（14 张）噪声大，不列。

mw 外部集 168 张（makerworld，与训练不同分布）：

| 画法 | 涂对 | 涂错 | 未涂 | 块比 |
|---|---:|---:|---:|---:|
| v3 叠涂 @0.4 | 0.610 | **0.140** | 0.250 | 3.98 |
| argmax | 0.741 | 0.179 | 0.080 | 17.5 |
| select κ0.5 small | 0.605 | 0.189 | 0.206 | 10.7 |
| hybrid d0.2 a0.6 small | 0.627 | 0.217 | 0.157 | 9.96 |
| hybrid d0.35 a0.75 small（最保守） | 0.604 | 0.186 | 0.210 | 7.59 |

**外部集上全部变差**：涂错 +4.6～7.7 个点，块数翻倍。排序器学到的是 pv 上「这个名字长什么样」，换到 makerworld 就成了噪声。
这与 §2 的过拟合判断一致。只删/只加的拆解见 §3.1。

### 3.1 只删 / 只加（`paint_grid2`）

把 hybrid 拆成两步单独看（`d0 a1.01` 恰好复现 v3，作为一致性检验）：

| 步骤 | 留出集 涂对 / 涂错 | 相对 v3 | mw 涂对 / 涂错 | 相对 v3 |
|---|---:|---:|---:|---:|
| v3 | 0.582 / 0.218 | – | 0.610 / 0.140 | – |
| 只加 keep ≥ 0.9 | 0.622 / 0.226 | +4.0 / +0.8 | 0.647 / 0.193 | +3.7 / **+5.3** |
| 只加 keep ≥ 0.75 | 0.634 / 0.242 | +5.2 / +2.4 | 0.677 / 0.209 | +6.7 / **+6.9** |
| 只删 keep < 0.1 | 0.566 / 0.183 | −1.6 / **−3.5** | 0.576 / 0.136 | **−3.4** / −0.4 |
| 只删 keep < 0.2 | 0.534 / 0.151 | −4.8 / −6.7 | 0.540 / 0.105 | −7.0 / −3.5 |
| 删 < 0.1 + 加 ≥ 0.9 | 0.612 / 0.188 | **+3.0 / −3.0**（未涂持平） | 0.619 / 0.194 | +0.9 / +5.4 |

留出集上两步都在做对的事：加进来的像素 5 对 1 错，删掉的 2 错 1 对。mw 上两步都翻了：加进来的对错各半，删掉的 8 对 1 错。
排序器在 mw 上对 v3 集合的判断只略好于随机——没有一个阈值组合能在 mw 上同时不涨涂错、不掉涂对。

### 3.2 外部资产渲染（`assets/ext_bench/compare_{front,back}.png` 第 14 列）

`sam3_to_2dmap.py --assign rank --rank_model ... --rank_drop 0.1 --rank_add 0.9` 已接进部署路径（`rank_fields` / `rank_parts`，
特征路径与 `mask_rank_feats.py` 同一份代码），`ext_bench.py` 加了 `map_rank` 变体，作为新列追加到原 13 列右侧（原列未动）。
肉眼：无噪点、块与 v3 同级；多数资产与 v3 几乎一致；剑的 guard / pommel 被删成灰（`dropped=1 / 2`），狗、机器人、跑车有增补。
和 mw 的结论一致——排序器在训练分布之外的判断不可信。

之后按同一格式又加了 4 列（`map_ease` / `lift_ease` / `map_venice` / `lift_venice`，共 19 列）：EASE 边 hybrid 与 Venice 门控 top-1，
各带 SegviGen 3D 提升（`lift_maps_segvigen.py --tags ease,venice`）。肉眼：EASE 列与 RankGNN 列几乎不可分——同一 v3 集合上的小改动，
剑的 guard / pommel 同样被删；Venice 列每名字只留一个掩码，米老鼠正面头部被 ear 的大掩码吃掉、菠萝背面叶子整片丢失、
跑车 body 被 window 覆盖——正是 §4.1 说的「IoU-gain 偏爱大掩码」在 3D 上的样子。

第 15 列是它的 SegviGen 3D 提升（`lift_maps_segvigen.py --tags rank`）；第 16–19 列是 §4.1 两个变体的 2D 图 + 3D 提升
（`map_ease` / `lift_ease`、`map_venice` / `lift_venice`，`--tags ease,venice`）。肉眼：EASE 列最接近 v3（跑车车身回到 v3 的蓝、
剑保住了 guard），Venice 每名字只留一个掩码，米老鼠的头被 ear 的大掩码吃掉、菠萝叶子换色——和 §4.1 的「偏大掩码」一致。

## 4. 近期顶会里的候选级排序器（关注 DETR 一线）

按对我们（**冻结 SAM3、后置、每名字 ≤ 200 个候选**）的可用性排：

| 工作 | 场合 | 做法 | 对我们 |
|---|---|---|---|
| **Venice-H1**（arXiv 2606.22546, 2026） | RIS，DeRIS 的 N 个 query 后置重排 | 冻结骨干，每候选 [query 向量; 分数; 掩码统计; **4/8/16 网格签名 675 维**] → 3 层 Transformer 集合级重排；**失败门控**只在有把握 Query 0 错时才换，有害切换率 < 0.5 % | 与本方案同构。网格签名已采纳（§2 +0.01 AUC）；失败门控 = 本报告的 `hybrid`。他们的经验：重排只在 7–18 % 失败样本上有收益，其余保持基线——我们在 mw 上的失败正是没守住这条 |
| **iFAN / APMR**（arXiv 2608.03216, 2026） | Mask2Former 类 plain mask transformer | 诊断与我们完全一致：「分数最高的 query 不是掩码最好的」（低 IoU 赢家 29 %）；加 soft-IoU 质量头 + **匹配 query 对其目标区域内的困难负 query 的成对排序** | 需要训解码器，冻结下用不上；但「困难负样本对」的采样可直接搬进我们的 rank loss（现在是同名内所有 gap > 0.1 的对） |
| **MDS-DETR**（arXiv 2605.23507, 2026） | DETR 去重 | 按置信度排序 + 因果自注意力（低分只能看高分）= 可学习的并行 NMS | **不适用**：它假设高分可靠、让低分让路；我们的失败模式恰是 top-1 错、正确的在第 38 位，需要高分被低分「反驳」。因果掩码会切断这条路 |
| **EASE-DETR**（CVPR 2024） | DETR 收敛 | 相对排名 × IoU 作为注意力偏置，偏向 leading query | 排名关系边已采纳（log 分数比、排名差） |
| **Route-DETR**（arXiv 2512.13876） | DETR | 用 [相似度, 置信度, 面积] 成对门控出「抑制/分工」两类注意力偏置 | 训练期技巧，冻结下无用 |
| **Rea2Seg**（CVPR 2026） | 推理分割 | 候选掩码分组交给 MLLM 打分重排 | 思路（比较式打分）对；代价（每图跑 MLLM）不适合批量拆件 |
| **SAMRefiner++**（ICLR 2025） | SAM 掩码精修 | 只在 IoU 头加 LoRA + 成对排序损失，不碰掩码生成 | 与本方案精神一致（冻结生成、只学选择）；SAM3 无独立 IoU 头，我们的排序器就是那个头 |
| LTR / Rank-NMS（ICCV 2019）、RANK++LETR（NeurIPS 2025） | 检测 | 连续质量标签 + 排序损失 + 困难对采样 | 损失形式的先例；困难对采样待加 |

结论：**没有一个能绕过数据量**。这些工作里后置重排器的训练集都是 RefCOCO / COCO 量级；Venice-H1 在 RIS 上也只敢在 7–18 % 的样本上动手。
架构上我们已经是集合级 Transformer + 关系边 + 网格签名，剩下可搬的只有困难对采样和更严格的门控。

## 4.1 上数据前先试：Venice-H1 与 EASE-DETR 在同一数据上的实测

路线说明见 §0（Venice / EASE 两条）。两者都实现成 `mask_rank.py` 的可选项，同一份 `feats2_*`、同一评测。

**Venice-H1**（`--arch venice`）：每名字的 32 个候选为一个集合，2 层 MLP 编码 → 3 层 pre-norm Transformer（8 头、GELU、无边特征）
→ Gain 头回归 IoU_i − IoU_0、Failure Gate 判「score-top-1 不是最好」；损失 = focal BCE(γ=2, 自动正样本权重) + 5 × smooth-L1；
推理 p_fail > τ 才换成 argmax gain。他们的 query 向量换成概念向量（dump 里没有 SAM3 解码器输出）。`venice_s`：d=128，wd 0.1。

| | 默认 top-1 IoU | 门控后 IoU | best-of-K | gate AUC | switch | 有害 switch |
|---|---:|---:|---:|---:|---:|---:|
| venice (d256) ep 8, τ 0.5 | 0.384 | 0.403 | 0.635 | 0.643 | 0.48 | 0.19 |
| venice_s (d128) ep 16, τ 0.5 | 0.384 | **0.405** | 0.635 | 0.664 | 0.54 | 0.20 |

τ 从 0 到 0.4 结果都一样（+1.7～2.0），0.6 以上门就关了。IoU 25 个点的余量只拿回 2 个；有害切换占切换的 37%（论文 < 0.5%）。
两个前提在我们这里不成立：(1) 他们的默认选择 82–93% 是对的，门控只需抓少数失败；我们 **81% 的名字 top-1 都不是最好**，
「失败门控」退化成「总是换」；(2) IoU-gain 目标偏爱大掩码，涂到 3D 上就是出血。重绘（`paint_venice`）：

| 每名字一个掩码 | 留出集 涂对 / 涂错 / 未涂 | 块比 | mw 涂对 / 涂错 |
|---|---:|---:|---:|
| SAM3 score top-1 | 0.554 / 0.222 / 0.224 | 4.1 | 0.670 / 0.222 |
| Venice 门控 top-1 | 0.564 / **0.304** / 0.132 | **9.6** | 0.614 / 0.257 |

IoU 涨了、涂错涨 8 个点、块数翻倍。**Venice 的 gate + IoU-gain 配方不适合本任务**，不进下一轮。

**EASE-DETR**（`--edge_mode ease`）：注意力里唯一的关系是「谁领先 × IoU」：decay = sigmoid(MLP(sign(rank_j − rank_i) · IoU_ij))，
以 log 形式加到注意力 logit（= 乘性衰减）。其余与 `grid_name_lr` 完全相同（同名内排序、lr 1e-3、bce 1.0、网格签名）。

| | val good-AUC | kept 精度 | 留出集 hybrid | mw hybrid d0.1 a0.9 |
|---|---:|---:|---|---|
| 7 通道线性边（`grid_name_lr` ep 11） | 0.787 | 0.737 | d0.1 a0.9: 0.612 / 0.188 / 0.200 | 0.619 / 0.194 / 0.186 |
| EASE 边（`ease` ep 8） | 0.783 | 0.718 | d0.2 a0.75: **0.631 / 0.192** / 0.177 | **0.669 / 0.189** / 0.141 |
| v3 | 0.502 | – | 0.582 / 0.218 / 0.200 | 0.610 / 0.140 / 0.250 |

离线指标持平，重绘上 EASE 更好，**尤其在 mw 上**：同样的涂错（0.189 vs 0.194）多涂对 5 个点、少 4.5 个点灰。
把边特征从 7 个通道收窄到「排名 × IoU」一个标量，泛化反而更好——线性边里的中心距、包含比等通道是在给记忆开后门。
但 EASE 在 mw 上涂错仍比 v3 高 4.9 个点，加进来的像素对错约 1.2 : 1，**还是不能上线**。

上数据前的决定：下一轮排序器用 `--edge_mode ease`（其余沿 `grid_name_lr`），不用 Venice 的门控/IoU-gain；
保留精度导向的目标 g = prec · √rec；hybrid 阈值在新数据的留出集上重扫。

## 5. 结论

1. **画法问题解了**：掩码级叠涂（select / hybrid）在块数上与 v3 打平或更少，没有 argmax 的噪点。这部分可以直接进 `sam3_to_2dmap`。
2. **排序器在同分布上有效、跨分布失效**：留出集 +3.7 / −3.0，mw 上涂错 +4.6～7.7。原因是记忆而非泛化（train AUC 0.91 vs val 0.78）。
3. **部署仍是 v3**。排序器不能上线，除非训练集覆盖到目标分布。
4. 数据是唯一没试的杠杆：`segvigen-pv-2view` / `relabel-work` / `pv-raw` 三个仓库若能把物体数从 1800 提到万级、名字覆盖 makerworld 的词表，
   排序器的 val 平台才可能抬起来。抽特征 3600 张 18 分钟、训练 20 分钟，扩数据的成本很低。
5. 上数据前的架构测试（§4.1）：Venice-H1 的门控 + IoU-gain 不适合（前提反、目标偏大掩码，涂错 +8）；EASE-DETR 的「排名 × IoU」
   单标量边比 7 通道线性边跨分布更稳（mw 同涂错下涂对 +5），**下一轮用 `--edge_mode ease`**。

## 6. 复现

```bash
# 特征（v3 bank，top-32，含网格签名）
python finetune/mask_rank_feats.py --dataset_root /root/autodl-tmp/datasets/pv --split_file runs/v5_ce_lora/split.json \
    --part train --resume datasets/concept_bank_v3/bank.pt --out runs/mask_rank_v3/feats2_train
# 排序器
python finetune/mask_rank.py --feats runs/mask_rank_v3/feats2_train --val_feats runs/mask_rank_v3/feats2_holdout \
    --out runs/mask_rank_v3/grid_name_lr --topk 32 --epochs 30 --rank_scope name --lr 1e-3 --bce_w 1.0
# 排序器变体（§4.1）
python finetune/mask_rank.py ... --out runs/mask_rank_v3/ease --rank_scope name --lr 1e-3 --bce_w 1.0 --edge_mode ease
python finetune/mask_rank.py ... --out runs/mask_rank_v3/venice_s --arch venice --dim 128 --heads 8 --weight_decay 0.1
# 重绘评测
python finetune/mask_rank_paint.py ... --rank_model runs/mask_rank_v3/grid_name_lr/rank_epoch11.pt \
    --extra_eval mw=... --min_comp 0.005 --dump 12
# 部署 / 外部资产
python sam3_to_2dmap.py --image render.png --out map.png --concept_bank datasets/concept_bank_v3/bank.pt --threshold 0.4 \
    --assign rank --rank_model runs/mask_rank_v3/grid_name_lr/rank_epoch11.pt --rank_drop 0.1 --rank_add 0.9 --prompts ...
python finetune/ext_bench.py sam3 --maps map_rank && python finetune/ext_bench.py montage --columns map_rank --extend assets/ext_bench
```
