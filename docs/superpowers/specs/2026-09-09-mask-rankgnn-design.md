# Mask RankGNN：SAM3 候选掩码的联合排序（含 oracle 前置实验）

日期：2026-09-09。机器：AutoDL RTX 5090，`/root/autodl-tmp`。环境：`.venv_holo`（SAM3，`transformers>=5`）。

## 1. 要解决的问题

部署路径（`sam3_to_2dmap.py` → `inference_full.py`）在 hard 20 上语义 mIoU 0.219、命名正确率 0.354，而用 GT 绑定条件图是 0.274 / 0.588（`HANDOVER_cloud_segvigen.md` §4 E2）。差距来自 2D 图本身：`colorize` 把每个提示词的实例并集按面积从小到大"先到先得"地叠涂，名字之间没有竞争、分数没有参与、几何关系没有参与。典型错误是 `arm` 掩码在阈值下没出来（或出来了但更大），`body` 掩码盖过去，3D 照抄。

概念库 v4 的 A–H 六个批次（`REPORT_concept_bank_v4_eval.md`）和 v5 的像素级 assignment CE（`runs/v5_ce_bank`，epoch 1 argmax：涂对 0.609 / 涂错 0.294，对 v3 0.541 / 0.205）都在同一条"更敢涂 ↔ 更多错"的权衡线上。它们改的是单个提示词的**绝对分数校准**；`colorize` 真正需要的是同一张图内候选之间的**相对排序**。

SAMV-DUSt3R（`/root/autodl-tmp/icme2025_template_anonymized.pdf`）的 Spatial RankGNN 正是这个形状：节点 = 候选，注意力式消息传递，全局读出，Score Rank Loss（MSE + margin ranking）学相对序而非绝对值。本设计把它迁到 SAM3 候选掩码上。

## 2. 两个阶段

| 阶段 | 产物 | 回答的问题 | 成本 |
|---|---|---|---|
| 0 oracle | `finetune/oracle_candidates.py`，`runs/oracle_v3/` | SAM3 的候选里**有没有**正确答案？错误是"选错"还是"漏检"？ | 一次 eval 前向，< 30 min |
| 1 Mask RankGNN | `finetune/mask_rank.py`，`runs/mask_rank_v1/` | 一个排序模块能否把可救的错误救回来 | 数据抽取 1–2 h + 训练 < 1 h |

阶段 0 的判定不过，阶段 1 不做（转 E2 / 多视角）。

## 3. 阶段 0：oracle 上限与错误分解

### 3.1 数据与对照

与 v3 评测逐字一致：留出 `concept_bank_v3/split.json` 200 物体 / 240 张图 / 1068 提示；hard 切片 `pv_hard.txt` 落在留出里的 14 张图；MakerWorld `concept_bank_clean_v2/objects` 168 张图。复用 `concept_bank.py` 的 `ImageSample` 与样本加载；前向用 `sam3_bank.bank_forward`，bank = v3 `bank.pt`（同时跑一遍无 bank 的裸 SAM3 作对照）。

### 3.2 候选

对每张图的全部 GT 名字（`s.gts.keys()`）批量前向一次。候选 = 每个 (名字 n, query q) 中 `batch_scores` > 0.05 的项；掩码 = `sigmoid(pred_masks[n, q]) > 0.5`，双线性上采样到原图，裁到轮廓 `ids >= 0`；面积 < 16 像素的丢掉。每个候选记录：

- `score`、面积、包围盒、质心（归一化到 [0,1]）
- 对 GT 的 `precision`（掩码内像素属于名字 n 的比例）、`recall`（名字 n 的 GT 像素被覆盖比例）、`iou`
- 与同图其他候选的 IoU 矩阵（阶段 1 的边要用）

预期每个名字 ≤ 10 个候选（SAM3 DETR 200 个 query，0.05 以上很少）。

### 3.3 三种上色（同一套 F 指标）

F 指标沿用 `concept_bank.evaluate`：`pixel_acc`（涂对）/ `pixel_wrong`（涂错）/ `pixel_unassigned`（未涂）/ `painted_part_rate`，按 GT 部件像素归一。

1. **基线 v3 @0.5**：`score > 0.5` 的候选按名字取并集，`sam3_bank.paint`（小先大后）。必须复现 `f_base_v3` 的 0.541 / 0.205 / 0.254（±0.005），否则脚本有错。
2. **Oracle-select(p)**：只保留 `precision >= p` 的候选（p ∈ {0.6, 0.7, 0.8, 0.9}），其余与 1 相同。这是"完美选择器 + 现有上色规则"的上限，即阶段 1 的可达上限。
3. **Oracle-pixel**：像素被任一同名候选覆盖就算涂对；涂错 = 0。任何选择/分配方法的绝对上限。

### 3.4 错误分解

对基线 1 的每个 GT 像素归类：

| 类 | 定义 | 含义 |
|---|---|---|
| 涂对 | painted == gt | – |
| 涂错-可救 | painted ≠ gt，且存在 gt 名字的候选覆盖该像素 | 选择/竞争错误，阶段 1 能修 |
| 涂错-漏检 | painted ≠ gt，无 gt 名字候选覆盖 | 检出问题，阶段 1 修不了 |
| 未涂-可救 / 未涂-漏检 | painted == -1，同上两分 | 同上 |

另报：每个名字 IoU 最高的候选，其 `score` 落在 [0.05, 0.5) 的比例（"降阈值 + 选择"这个杠杆的长度）；按名字统计"可救"错误最多的前 20 个名字对（被谁抢 → 应是谁）。

### 3.5 输出与判定

`runs/oracle_v3/{holdout,hard,mw}.json` + 一张汇总表（stdout）。裸 SAM3 与 v3 bank 各一份。

**判定（提前定）**：
- 通过 → 进阶段 1：Oracle-select(0.8) 涂对 ≥ 0.65，且（涂错-可救 + 未涂-可救）≥ 0.15（即 v3 涂错 0.205 + 未涂 0.254 中至少三分之一可救）。
- 不通过：Oracle-pixel ≤ 0.60，或可救总量 < 0.08。转 E2 / 多视角，本设计到此结束。
- 中间：把 p 降到 0.7 再看；仍不到就按不通过处理。

## 4. 阶段 1：Mask RankGNN

### 4.1 输入图

每张图一个图。节点 = §3.2 的候选（阈值 0.05 上限 12 个/名字，不足不补）。

节点特征（拼接后线性投影到 d=128）：
- SAM3 `score`（1）、`log(面积/轮廓面积)`（1）、包围盒 + 质心（6）
- 概念库 `text_vec`（256；v3 bank 的 `E_0 + E[name]` 已加，`sam3_masks.py` 写法一致）
- 掩码区域内 SAM3 vision embedding 的均值池化（256；`encode_image` 的输出按掩码池化，不重跑骨干）
- 名字在本图是"同名多实例"的标记（1；四条腿）

边：全连接（一张图 ≤ 100 个节点，O(N²) 可忽略），边特征 = [IoU, 包含度 A∩B/|A|, 包含度 A∩B/|B|, 质心距离, 是否同名]（5），经线性层加到注意力 logit 上（相当于带边偏置的多头自注意力，与论文 `RankMessagePassing` 的 QKV + softmax 一致，多了边偏置）。

### 4.2 网络

`RankMessagePassing` × L=3，d=128，4 头，残差 + LayerNorm + FFN，与论文相同。读出：每节点 MLP → `keep ∈ [0,1]`。不做图级池化（论文是图级排序，这里是节点级）。参数量 ≈ 0.3 M。

### 4.3 目标与损失

GT 分 `g_i = precision_i · recall_i^{0.5}`（偏向精度：`colorize` 里错涂比漏涂贵——错涂会被 3D 照抄，漏涂只是灰），Min-Max 归一到本图 [0,1]（同论文：分数只在同一集合内有意义）。

Score Rank Loss（论文式 1）：`L = α·MSE(keep, g) + (1−α)·MarginRanking`，α=0.2，margin ranking 在同图所有候选对 (i, j) 上、`|g_i − g_j| > 0.1` 的对才计入，margin 0.1。再加一项 `BCE(keep_i, [precision_i ≥ 0.8])` 权重 0.5，让 `keep` 的绝对值可以直接当阈值用。

### 4.4 推理算子（替换 `colorize` 的排序）

1. 候选 `keep ≥ τ`（τ 在留出集上扫，取涂错占比不高于 v3 的最大涂对）。
2. 每个像素归给覆盖它的、`keep` 最高的候选的名字；不再按面积。
3. 同名候选并集。轮廓内无人覆盖 → 灰。
4. 输出 `map.png` + `legend.json`，格式与 `sam3_to_2dmap.colorize` 一致，下游零改动。

作为对照同时报"`keep ≥ τ` 后仍用小先大后"的版本，看第 2 步（按分归属）单独贡献多少。

### 4.5 数据

训练：`split.json` 的 1800 个训练物体 × az0/az135 = 3600 张图；候选与特征在阶段 0 的脚本上加 `--dump` 一次抽出来（`runs/mask_rank_v1/feats/<obj>_<az>.npz`），之后训练不再碰 SAM3。留出 240 张 / hard 14 张 / MW 168 张与阶段 0 相同。

### 4.6 训练

AdamW lr 3e-4、weight decay 1e-2、batch 32 张图、20 epoch、余弦到 0；一张图的所有节点一起过网络。单卡几分钟一个 epoch。

### 4.7 评测与验收

F 指标三档：留出 / hard / MW，阈值 τ 扫 {0.3, 0.4, 0.5, 0.6}。对照：v3 @0.5（0.541 / 0.205）、v3 @0.4（0.583 / 0.217）、Oracle-select(0.8)。

**通过**：留出集上存在一个 τ 使涂对 ≥ v3@0.4 且涂错 ≤ v3@0.5；hard 切片涂错 ≤ 0.25（v3 0.310）；MW 涂对 ≥ 0.58 不退。**止损**：留出涂对到不了 Oracle-select(0.8) 的 60%，停，不扫超参。

**两个对照，必须跑**：
- **shuffle 对照**：把节点的 `text_vec` 在图内随机置换再评测——若指标不变，说明模型没用名字语义，只学了"大掩码不可信"这种几何先验；这仍可能有用，但要如实写。
- **消融**：去掉边特征（纯 self-attention）、去掉 vision 池化特征，各一跑，看结构与外观各贡献多少。

**下游验证**（通过后，需要 3D 数据的机器）：hard 20 部署路径重跑（`HANDOVER_cloud_segvigen.md` §6 第 2 步），`sam3_to_2dmap.py` 加 `--rank_model` 开关；看语义 mIoU（0.219 →）与命名正确率（0.354 →）。这一步不在本设计的验收里，但它才是最终指标。

## 5. 文件

| 文件 | 阶段 | 说明 |
|---|---|---|
| `finetune/oracle_candidates.py` | 0 | 候选抽取 + 三种上色 + 错误分解；`--dump` 写训练特征 |
| `finetune/mask_rank.py` | 1 | `RankMessagePassing`、`MaskRankGNN`、训练/评测入口、`rank_paint()` 推理算子 |
| `sam3_to_2dmap.py` | 1 | `--rank_model <pt>`：有则用 `rank_paint` 替换 `colorize` 的排序，无则行为不变 |
| `runs/oracle_v3/`、`runs/mask_rank_v1/` | 0/1 | 结果、特征缓存、`rank.pt` |

不改 `concept_bank.py` / `sam3_bank.py` 的训练路径；只 import 它们的加载和前向函数。

## 6. 不做的

- 不在 3D latent 上建图（信息已在 2D 侧丢失，见讨论）。
- 不把名字 token 注入 DiT（v3–v5 已否定）。
- 不改 SAM3 权重、不动 bank。RankGNN 是纯后处理，产物是一个 < 2 MB 的 `rank.pt`。
- 不做多视角（本机无 `parts/`）。View RankGNN 另开设计。
