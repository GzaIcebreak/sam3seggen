# Mask RankGNN：SAM3 候选掩码的联合排序（含 oracle 前置实验）

日期：2026-09-09（v5 跑完后修订）。机器：AutoDL RTX 5090，`/root/autodl-tmp`。
环境：`/root/autodl-tmp/envs/sam3/bin/python`（SAM3，`transformers>=5`）。SAM3 权重：`sam3seggen/weights/facebook/sam3`。

## 1. 要解决的问题

部署路径（2D 图 → `inference_full.py`）在 hard 20 上语义 mIoU 0.219、命名正确率 0.354，而用 GT 绑定条件图是 0.274 / 0.588（`HANDOVER_cloud_segvigen.md` §4 E2）。差距来自 2D 图本身。

### 1.1 v5 已经解决的部分

v3 时代 `sam3_to_2dmap.colorize` 按面积从小到大叠涂，名字之间完全无竞争。v5（`REPORT_concept_bank_v5_eval.md`）换了目标和算子：per-pixel assignment CE + `paint_argmax`（每个像素在所有名字的 `name_logit_maps` 上取胜者）。**像素级竞争已经有了**，效果是 v4 权衡线第一次被同时改善：

| | 涂对 | 涂错 | 未涂 | MW 涂对 |
|---|---:|---:|---:|---:|
| v3 叠涂 @0.5 | 0.541 | 0.205 | 0.254 | 0.581 |
| `v5_ce_lora` ep2 argmax τ=0.5 | **0.626** | **0.159** | 0.215 | **0.730** |
| `v5_ce_lora_full` ep2 argmax τ=0.5 | 0.633 | 0.162 | 0.205 | 0.729 |

### 1.2 剩下的部分：候选级结构

`v5_ce_lora_full` 在点积两端都加了 LoRA（`mask,text,embed,proj`），结果与只有 `mask,text` 在噪声内。**容量不再是瓶颈**。剩余误差要么是信息不在图里（漏检），要么是结构没被利用。

没被利用的结构在这个式子里（`sam3_bank.name_logit_maps`）：

```
L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} ]
```

每个 (名字 n, query q) 候选掩码以自己的 `score_{n,q}` 为权重进入像素竞争。这个分数是 SAM3 对该提示词独立打的，**不看同一张图里的其他候选**：不知道 `arm` 的候选被 `body` 的候选包住了，不知道两个候选高度重叠，不知道一个候选横跨了三个 GT 部件。概念库 v4 的 A–H 与 v5 都在改这个分数的**绝对校准**；候选之间的**相对可信度**从没被建模过。

SAMV-DUSt3R（`/root/autodl-tmp/icme2025_template_anonymized.pdf`）的 Spatial RankGNN 正是这个形状：节点 = 候选，注意力式消息传递，Score Rank Loss（MSE + margin ranking）学相对序而非绝对值，GT 分只在同一集合内有意义。本设计把它迁到 SAM3 的候选掩码上：

```
L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} + λ · log keep_{n,q} ]
```

`keep_{n,q}` 由 RankGNN 给出。纯后处理：不碰 SAM3 权重、不碰 bank、不改 `paint_argmax`，失败时把 λ 设为 0 即完全回退。

## 2. 两个阶段

| 阶段 | 产物 | 回答的问题 | 成本 |
|---|---|---|---|
| 0 oracle | `finetune/oracle_candidates.py`，`runs/oracle_v5/` | 候选里**有没有**正确答案？误差是"权重错"还是"漏检"？ | 一次 eval 前向，< 40 min |
| 1 Mask RankGNN | `finetune/mask_rank.py`，`runs/mask_rank_v1/` | 一个候选级排序模块能否把可救的误差救回来 | 特征抽取 1–2 h + 训练 < 1 h |

阶段 0 判定不过，阶段 1 不做（转 E2 / 多视角）。

## 3. 阶段 0：oracle 上限与误差分解

### 3.1 模型与数据

**基线模型 = 当前部署最佳**：`runs/v5_ce_lora/bank_epoch2.pt` + `decoder_lora_epoch2.pt`（`--decoder_lora 8 --lora_scope mask,text`），`paint_argmax`、`assign_temp` 1.0、τ 扫 {0, 0.3, 0.5}。同时跑 v3 `bank.pt`（无 LoRA）作历史对照。

数据与 v5 评测逐字一致，直接读 `runs/v5_ce_lora/split.json` 的 holdout 列表以保证同一留出集：

- 留出：200 物体，`list_images(root, hold_objs, [az0, az135])[:240]` = 240 张图 / 1068 提示
- hard 切片：`pv_hard.txt` ∩ 留出，打到 14 张图
- 外部：MakerWorld `concept_bank_clean_v2/objects`，168 张图 / 375 提示

复用 `concept_bank.py` 的 `ImageSample` / `list_images` / `read_id_file` 与 `sam3_bank` 的前向函数，不复制逻辑。

### 3.2 候选

对每张图的全部 GT 名字批量前向一次（`sb.bank_forward`）。候选 = 每个 (名字 n, query q) 中 `batch_scores` > 0.05 的项。掩码 = `sigmoid(pred_masks[n,q]) > 0.5`，上采样到原图，裁到轮廓 `ids >= 0`；面积 < 16 像素丢弃。每个候选记录：

- `score`、面积占轮廓比、包围盒、质心（归一化）
- `precision`（掩码内像素属于名字 n 的比例）、`recall`（名字 n 的 GT 像素被覆盖比例）、`iou`
- 与同图其他候选的 IoU / 包含度（阶段 1 的边）

先按分数过滤再上采样，避免 `[N, Q, H, W]` 的显存。

### 3.3 三种上色（同一套 F 指标）

F 指标沿用 `concept_bank.evaluate`：`pixel_acc`（涂对）/ `pixel_wrong`（涂错）/ `pixel_unassigned`（未涂）/ `painted_part_rate`，按 GT 部件像素归一。

1. **基线**：全部候选按原 `score` 权重做 `name_logit_maps` → `paint_argmax`。**必须复现 `v5_ce_lora` ep2 的 0.626 / 0.159 / 0.215（±0.005）**，否则脚本或权重加载有错，停下来查。
2. **Oracle-select(p)**：只保留 `precision >= p` 的候选（p ∈ {0.6, 0.7, 0.8, 0.9}），其余从 logsumexp 里剔除，重算 `name_logit_maps` → `paint_argmax`。这是"完美的 per-candidate 权重 + 现有像素算子"的上限，即阶段 1 的可达上限。
3. **Oracle-pixel**：像素被正确名字的任一候选覆盖就算涂对，涂错 = 0。任何候选级方法的绝对上限。

叠涂（`sb.paint`）列同一次前向顺带报一份，作与 v3 的历史对照。

### 3.4 误差分解

对基线 argmax 上色的每个 GT 像素归类：

| 类 | 定义 | 含义 |
|---|---|---|
| 涂对 | painted == gt | – |
| 涂错-可救 | painted ≠ gt，且存在 gt 名字的候选覆盖该像素 | 权重/竞争错误，阶段 1 能修 |
| 涂错-漏检 | painted ≠ gt，无 gt 名字的候选覆盖 | 检出问题，阶段 1 修不了 |
| 未涂-可救 / 未涂-漏检 | painted == -1，同上两分 | 同上 |

另报：每个名字 IoU 最高的候选，其 `score` 落在 [0.05, 0.5) 的比例（"降阈值 + 重排"这个杠杆的长度）；"可救"误差最多的前 20 个名字对（被谁抢 → 应是谁）；每张图的候选数分布。

### 3.5 输出与判定

`runs/oracle_v5/{holdout,hard,mw}.json` + stdout 汇总表。v5_ce_lora ep2 与 v3 各一份。

**判定（提前定）**：
- **通过** → 进阶段 1：Oracle-select(0.8) 涂对 ≥ 0.72，且（涂错-可救 + 未涂-可救）≥ 0.12（基线涂错 0.159 + 未涂 0.215 = 0.374 中至少三分之一可救）。
- **不通过**：Oracle-pixel ≤ 0.70，或可救总量 < 0.06。转 E2 / 多视角，本设计到此结束。
- **中间**：p 降到 0.7 再看；仍不到按不通过处理。

## 4. 阶段 1：Mask RankGNN

### 4.1 输入图

每张图一个图。节点 = §3.2 的候选（每名字上限 12 个）。

节点特征（拼接后线性投影到 d=128）：
- SAM3 `score`（1）、`log(面积/轮廓面积)`（1）、包围盒 + 质心（6）
- 概念库 `text_vec`（256，即 `E_0 + E[name]` 后的池化文本向量）
- 掩码区域内 SAM3 vision embedding 的均值池化（256，`encode_image` 的输出按掩码池化，不重跑骨干）
- 本名字在本图有多个候选的标记（1，四条腿）

边：全连接（一张图 ≤ 100 节点，O(N²) 可忽略），边特征 = [IoU, A∩B/|A|, A∩B/|B|, 质心距离, 是否同名]（5），经线性层加到注意力 logit 上（带边偏置的多头自注意力；论文的 `RankMessagePassing` 是无边偏置版本，边偏置作为消融项单独验证）。

### 4.2 网络

`RankMessagePassing` × L=3，d=128，4 头，残差 + LayerNorm + FFN，与论文相同。读出：每节点 MLP → `keep ∈ [0,1]`。节点级输出，不做图级池化。参数 ≈ 0.3 M。

### 4.3 目标与损失

GT 分 `g = precision · recall^0.5`（偏向精度：错涂被 3D 照抄，漏涂只是灰），Min-Max 归一到本图 [0,1]（同论文：分数只在同一集合内有意义）。

Score Rank Loss（论文式 1）：`L = α·MSE(keep, g) + (1−α)·MarginRanking`，α=0.2；margin ranking 在同图 `|g_i − g_j| > 0.1` 的候选对上，margin 0.1。加 `0.5·BCE(keep, [precision ≥ 0.8])`，让 `keep` 的绝对值可直接作阈值。

### 4.4 推理算子

改 `sam3_bank.name_logit_maps`，加可选的 per-candidate 权重：

```
L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} + λ · log keep_{n,q} ]
```

λ 在留出集上扫 {0.5, 1, 2}，λ=0 即现状。`paint_argmax` 与 τ 不变，下游 `legend.json` 格式不变。

对照：硬门限版（`keep < τ_k` 的候选直接剔除）单独报一份，看软加权与硬剔除哪个好。

### 4.5 数据

训练：`split.json` 的 1800 个训练物体 × az0/az135 = 3600 张图。候选与特征用阶段 0 的脚本加 `--dump` 一次抽出（`runs/mask_rank_v1/feats/<obj>_<az>.npz`），之后训练不再碰 SAM3。留出 / hard / MW 与阶段 0 相同。

### 4.6 训练

AdamW lr 3e-4、weight decay 1e-2、batch 32 张图、20 epoch、余弦到 0。一张图的所有节点一起过网络。单卡几分钟一个 epoch。

### 4.7 评测与验收

F 指标三档（留出 / hard / MW），τ ∈ {0, 0.3, 0.5} × λ ∈ {0.5, 1, 2}。对照：`v5_ce_lora` ep2（0.626 / 0.159）、Oracle-select(0.8)。

**通过**：留出集上存在一组 (λ, τ) 使涂对 ≥ 0.66 且涂错 ≤ 0.159；MW 涂对 ≥ 0.73 不退；hard 切片涂错不比基线差。
**止损**：涂对到不了 Oracle-select(0.8) 的 60%，停，不扫超参。

**两个对照，必须跑**：
- **shuffle 对照**：图内随机置换节点的 `text_vec` 再评测。指标不变 = 模型只学了几何先验（"大掩码不可信"），没用名字语义。仍可能有用，但要如实写。
- **消融**：去掉边特征（纯 self-attention）、去掉 vision 池化特征，各一跑。

**下游验证**（通过后，需要 3D 数据的机器）：hard 20 部署路径重跑（`HANDOVER_cloud_segvigen.md` §6 第 2 步），看语义 mIoU（0.219 →）与命名正确率（0.354 →）。不在本设计验收内，但它才是最终指标。

## 5. 文件

| 文件 | 阶段 | 说明 |
|---|---|---|
| `finetune/oracle_candidates.py` | 0 | 候选抽取 + 三种上色 + 误差分解；`--dump` 写训练特征 |
| `finetune/mask_rank.py` | 1 | `RankMessagePassing`、`MaskRankGNN`、训练/评测、`keep` 推理 |
| `finetune/sam3_bank.py` | 1 | `name_logit_maps` 加可选 `keep` 权重参数（默认 None = 现状） |
| `sam3_to_2dmap.py` | 1 | `--rank_model <pt>`：加载 RankGNN 参与 argmax；无则行为不变 |
| `runs/oracle_v5/`、`runs/mask_rank_v1/` | 0/1 | 结果、特征缓存、`rank.pt` |

不改 `concept_bank.py` 的训练路径；只 import 它的加载函数。

## 6. 不做的

- 不在 3D latent 上建图（语义信息在 2D 侧就已丢失）。
- 不把名字 token 注入 DiT（v3–v5 已否定）。
- 不改 SAM3 权重、不重训 bank。产物是一个 < 2 MB 的 `rank.pt`。
- 不做多视角（本机无 `parts/`）。View RankGNN 另开设计。
