# 概念库 v5：换目标——逐像素归属，而不是逐名字二分类

配套：v4 评测 `REPORT_concept_bank_v4_eval.md`（A–H 全部未过），v4 方案 `PLAN_concept_bank_v4.md`，
数据扩容 `PLAN_data_expansion.md`。代码：`finetune/concept_bank.py`、`finetune/sam3_bank.py`、`sam3_to_2dmap.py`。

---

## 0. v4 留下的结论，v5 必须回答的问题

v4 在 2000 个物体上把「只改文本输入」（A/B/C/E/G）和「给解码器加容量但不换目标」（H）都试了，
每一次都在同一条权衡线上滑：推检出则难负例 / 涂错涨，压难负例则灰 / 涂对掉。三条事实：

1. **目标和部署不是一回事。** 训练是每个名字独立的 BCE+Dice（绝对分数），部署 `colorize` 是逐像素归属
   （谁盖住这个像素谁说话）。BCE+Dice 没有任何一项在问「这个像素是 arm 还是 body」。
2. **A 加了竞争项也没用，原因有三个，缺一个都会失败。**
   - 竞争的对象是 `logit(1 − ∏(1 − m_q·s_q))`——先把 Q 个掩码乘分数、做概率并、再取 logit，梯度被两次 sigmoid 和乘积挤扁；
   - 可训参数只有 `E_0 + E[name]`：所有名字共用一张图像特征，一次平移改不了像素级排序；
   - **推理仍然是阈值 0.5 + 小掩码优先叠涂**，训练学的排序在部署里根本没被用到。
3. **H 证明容量是够的。** decoder LoRA 是唯一把 loss 压到 v3 地板以下的跑次（0.85 → 0.75），但在 BCE+Dice 上它学到的是重校准
   （0.5 过严、0.4 又像 v3），不是分辨。容量给对了，目标给错了。

v5 一次把三件事换掉：**目标换成逐像素多类 CE，容量给到掩码点积两端，推理换成 argmax 归属。三件事同一个算子，训练和部署对齐。**

---

## 1. 目标函数：逐像素归属

### 1.1 名字级像素 logit

SAM3 的掩码就是点积：`pred_masks[q] = ⟨mask_embed(query_q), instance_embed(p)⟩`（`modeling_sam3.py` L2113）。
对一个名字 n，它的 Q 个查询给 Q 张 logit 图和 Q 个检出分数 `s_q = σ(logit_q)·σ(presence)`。定义名字级像素 logit：

```
L_n(p) = logsumexp_q [ M_{n,q}(p) + log s_{n,q} ]          # M = pred_masks（原始 logit，不过 sigmoid）
```

- 用原始 `M` 而不是 `σ(M)`，用 `logsumexp` 而不是概率并——梯度直接回到点积两端。
- `log s` 项让「没检出」的查询自然退出竞争，但仍是可微的；不再有 `s > 0.5` 这种硬门。
- 多实例（四条 `leg`）由 logsumexp 天然合并，不需要 union。

### 1.2 背景 / 未归属

再加一个背景 logit `L_bg(p) = b_bg`（可学习标量，初值 0）。它就是「这里不属于任何提示词、留灰」的阈值，
由 CE 自己校准，代替 v3 手调的 0.5。

### 1.3 损失

对图像里的全部正名 `P` 和采样的负名 `N`（共现 / 文本相近，`--syn_cos` 跳过近义词）一起做 softmax：

```
p_n(p) = softmax_{n ∈ P ∪ N ∪ {bg}} ( L_n(p) / T )
loss_assign = − Σ_p w(p) · log p_{y(p)}(p)                 # y(p) = ids.npy 对应的正名索引
```

- **标签**：`y(p)` 来自 `ids.npy` → `names.json`。前景外像素忽略。`uncertain` 部件、空名字部件的像素**忽略**（不当背景）。
- **负名的监督是隐式的**：它们不是任何像素的标签，CE 把它们压到每个像素的败方。这正是「认错」的直接惩罚——`body` 在 `torso` 的像素上会被扣分，而 v3 的 BCE 只要求 `body` 的整体掩码为空。
- **近义词遮罩**：`torso` vs `body`、`head` vs `helmet` 这种标注口径差（v4 §2 已确认是涂错来源），对文本余弦 > `--syn_cos` 的名字对不互相惩罚：计算 CE 时把近义名的 logit 合并到标签名里（`logsumexp`），而不是让模型在噪声上学。
- **类平衡** `w(p)`：按部件面积的 `1/sqrt(area)`（A3E 用过，比 `1/area` 稳）。
- **辅助项**：`0.5 · presence_BCE`（负名 presence → 0，正名 → 1）保留，因为单提示词部署时没有竞争对手，检出仍要靠 presence。
  **BCE+Dice 默认权重 0**，只作消融，不再是主项。

### 1.4 和 A 的差别一句话

A：`CE( logit(union_prob) , ids )`，参数只有文本偏移，推理仍是阈值叠涂。
v5：`CE( logsumexp(mask_logit + log s) , ids )`，参数在点积两端，推理用同一个 softmax 的 argmax。

---

## 2. 容量：把可训参数放在点积两端

| 位置 | 作用 | 参数量 | 备注 |
|---|---|---|---|
| `E_0`, `E[name]`（v3 bank） | 文本侧先验，从 v3 热启动 | 72 K | 保留 |
| LoRA r=8 @ `detr_decoder.layers[*].text_cross_attn` | 查询怎么读文本 → 决定 `mask_embed` | 6 层 × 4 × 2·8·256 | H 已有 |
| LoRA r=8 @ `mask_decoder.prompt_cross_attn` | 像素特征怎么被文本调制 → 决定 `instance_embed` | 4 × 2·8·256 | H 已有 |
| **LoRA r=8 @ `mask_decoder.mask_embedder`（MLP）** | 点积左端 | 新增 | `--lora_scope` 加 `embed` |
| **`mask_decoder.instance_projection`（1×1 conv, 256→256）全量或 LoRA** | 点积右端 | 65 K | `--lora_scope` 加 `proj` |
| `b_bg` | 背景阈值 | 1 | |

图像编码器、DETR encoder、pixel decoder 的 FPN 卷积**全部冻结**。总可训参数约 0.3 M，产物 `bank.pt`（287 KB）+ `decoder_lora.pt`（约 1 MB）。

先跑 `mask,text`（= H 的范围）+ CE，看目标换了之后 H 的容量够不够；不够再加 `embed,proj`。

---

## 3. 推理：argmax 归属

`sam3_to_2dmap.colorize` 现在是「每个提示词一张硬掩码，小的先涂，不覆盖」。v5 增加 `--assign argmax`：

```
maps  = [L_n(p) for n in prompts] + [b_bg]        # 一次前向，N 个提示词分块跑，只保留 [N, h, w]
label = argmax_n softmax(maps)(p)
grey  = (label == bg) or (max prob < τ)          # τ 默认 0，用 b_bg 自己的校准；扫 0.3/0.5 看稳定性
```

- 训练里 `evaluate` 的 F 指标（涂对 / 涂错 / 未涂 / 难负例）同时用**两个画法**算：旧的叠涂（和 v3 可比）和 argmax（v5 的部署方式）。
- 难负例 FP 在 argmax 下定义为「负名在前景里赢了任何像素」。
- 存 `[N, h, w]` 而不是 `[N, Q, h, w]`，显存反而比现在低。

**先做一个零训练实验**：v3 权重 + argmax 画法。如果只换画法就已经降涂错，说明排序信息 v3 里本来就有一部分，只是被叠涂丢了；这一档作为 v5 的新对照。

---

## 4. 实施

### 4.1 代码（约 1.5 天）

| 文件 | 改动 |
|---|---|
| `sam3_bank.py` | `name_logit_maps(outputs, bias) -> [N, h, w]`（§1.1）；`paint_argmax(maps, fg, b_bg, tau)`；`inject_decoder_lora` 增加 `embed` / `proj` scope |
| `concept_bank.py` | `--assign_ce W --ce_temp T --syn_cos`：`assign_loss(maps, label, syn_groups, weights)`；`--bce_weight` 默认改 0 当开 `--assign_ce`；`b_bg` 进 bank；`evaluate` 双画法输出 `*_argmax` 列；`--split_file` 锁定 v3 的 200 留出（为 7826 物体准备）；`--dataset_root` 多目录 |
| `sam3_to_2dmap.py` | `segment_prompts` 返回 maps 的路径；`colorize(..., assign="paint"|"argmax")`；`load_sam3` 后按 bank.meta 自动注入并加载 `decoder_lora.pt` |
| `cb_table.py` | 加 argmax 列 |

### 4.2 实验（2000 物体，每跑约 70 min；显存 < 20 GB）

| 跑次 | 配置 | 回答什么 |
|---|---|---|
| `v5_paint` | v3 权重，`--eval_only`，argmax 画法 | 换画法本身值多少（零训练对照） |
| `v5_ce_bank` | `--assign_ce 1`，只训 bank（无 LoRA） | 目标对了但没容量会怎样（预期像 A：学不动） |
| **`v5_ce_lora`** | `--assign_ce 1 --decoder_lora 8 --lora_scope mask,text`，从 v3 热启 | **主实验** |
| `v5_ce_lora_full` | 同上 + `embed,proj` | 容量是否还是瓶颈 |
| `v5_ce_lora_bce` | 主实验 + `--bce_weight 0.3` | 绝对校准要不要留一点 |
| （已有）`v4_H` | LoRA + BCE+Dice | 容量对、目标错的对照 |

2×2（目标 × 容量）用 `v4_H` / `v5_ce_bank` / `v5_ce_lora` / v3 四格填满，能把「到底是目标还是容量」这个问题一次说清。

### 4.3 判据（阈值口径变了，用等覆盖比）

留出集仍是 v3 的 200 个物体。argmax 画法下没有 0.5 这个阈值，所以**在「未涂比例」相等的点上比**：

| 指标 | v3（叠涂 @0.5） | v5 过关线 |
|---|---:|---|
| 未涂 `pixel_unassigned` | 0.254 | 扫 τ 到 ≈ 0.25 |
| 涂对 `pixel_acc` | 0.541 | ≥ 0.56 |
| 涂错 `pixel_wrong` | 0.205 | **≤ 0.16**（涂错占比 27.5 % → ≤ 22 %） |
| 难负例 FP（argmax 定义） | 待 `v5_paint` 给出 | 不高于 `v5_paint` |
| MakerWorld 涂对 | 0.581 | ≥ 0.58 |
| hard 切片涂错 | 0.310 | ≤ 0.26 |

涂错是主指标——v5 就是为它设计的。mIoU 只作参考（argmax 下名字级 IoU 的定义和叠涂不同）。
MakerWorld 是过拟合的哨兵：H 在那里掉了 0.22，v5 掉 > 0.03 就停。

---

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| decoder 过拟合灰模风格（H 的 MW 掉分） | LR 1e-4、r=8、3 epoch、按 MW 涂对早停；只训 cross-attn 和点积两端，pixel decoder 卷积冻结 |
| 标注口径噪声（`body/torso`）被 CE 当成真错误 | §1.3 近义词遮罩；`uncertain` 像素忽略 |
| 单提示词场景没有竞争对手 | 保留 presence BCE；`b_bg` 提供最低门槛；`v5_paint` 验证单名时 argmax 退化为「和 bg 比」是否合理 |
| 大部件主导 CE | `1/sqrt(area)` 平衡；看 hard 切片（细杆、多部件）单列 |
| 训练把「多查询合并」学成「一个查询包全身」 | logsumexp 保留多实例；如出现，加 `--topk_q` 限制每名字参与的查询数 |
| 和 v3 的历史指标不可比 | 双画法同时报；叠涂列继续和 v3 对齐 |

---

## 6. 和数据扩容的关系

v5 不等数据。在 2000 个物体上先把 2×2 跑完，能不能看见「认错」下降是目标层面的问题，
不需要更多物体来回答。5826 个新物体清洗完成后（`PLAN_data_expansion.md` 路线 A），
用 `--split_file concept_bank_v3/split.json` 锁定同一个留出集重跑主实验：
新数据解决的是词表覆盖（MakerWorld 45 个 OOV），v5 解决的是排序，两者正交，应该分别报收益。

## 7. 顺序

1. `v5_paint`（半天，含画法代码）→ 定 argmax 下的新对照
2. `assign_loss` + `v5_ce_bank` / `v5_ce_lora`（1 天代码 + 2 跑）→ 填 2×2
3. 按结果决定 `full` / `bce` 消融
4. 过关则接 `sam3_to_2dmap` 部署路径，产物 = `bank.pt` + `decoder_lora.pt`；不过关，v5 停在这里，等 PartNeXt 那种人工部件名的数据，不再在 PartVerse 上叠方案
