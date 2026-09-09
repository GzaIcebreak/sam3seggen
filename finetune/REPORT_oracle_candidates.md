# 候选掩码 oracle 上限（Mask RankGNN 阶段 0）

日期：2026-09-09。机器：AutoDL RTX 5090。方案：`docs/superpowers/specs/2026-09-09-mask-rankgnn-design.md` §3。
代码：`finetune/oracle_candidates.py`。跑次：`runs/oracle_v5/`（v5_ce_lora ep2）、`runs/oracle_v3/`（v3 bank）。
留出集与 v5 逐字一致（`runs/v5_ce_lora/split.json`，240 张图 / 1068 提示；hard 切片 20 物体；MakerWorld 168 张图）。

---

## 0. 一页结论

**判定：通过，且大幅通过。** 阶段 1（Mask RankGNN）可以做。

| 判定线 | 要求 | 实测（v5，留出 τ=0.5） |
|---|---|---|
| Oracle-select(0.8) 涂对 | ≥ 0.72 | **0.783** |
| 可救误差（涂错 + 未涂） | ≥ 0.12 | **0.331**（0.141 + 0.190） |
| Oracle-pixel | > 0.70 | **0.945** |

三件事：

1. **基线逐位复现**。`base_all` 在 v5_ce_lora ep2 上是 0.626 / 0.160 / 0.215，在 v3 上是 0.618 / 0.258 / 0.124；τ=0 / 0.3 / 0.5 六个数字全部对上 `REPORT_concept_bank_v5_eval.md`。脚本重算的 `name_logit_maps` 与 SAM3 原路径逐位相同（`maps_max_dev = 0.0`），所以 oracle 与基线站在同一个前向上。
2. **误差几乎全部可救**。v5 留出集上涂错 0.160 里 0.141（88%）、未涂 0.215 里 0.190（88%）的像素，都**存在一个正确名字的候选掩码覆盖它**，只是这个候选在 `logsumexp_q [mask_logit + log score]` 里没赢。真正的漏检只有 0.044。
3. **v5 移动了工作点，没有抬高天花板**。v3 与 v5 的 Oracle-select(0.8) 涂对是 0.777 与 0.783，几乎相同。v5 的 CE + LoRA 把实际工作点从 0.618 / 0.258 推到 0.626 / 0.160（主要是压涂错），但可达上限一直是 ~0.78。

**给阶段 1 的一个非预期结论：排序器应该建在 v3 bank 上，不是 v5。** 见 §4。

---

## 1. 协议

对每张图的全部 GT 名字前向一次，取 SAM3 的**全部 200 个 query**作候选（不做分数阈值：实测最低分 3.2e-3，log 域 −5.7，没有一个 query 能凭分数被忽略；早期用 0.05 阈值的版本只留 42% 的 query，涂对就掉了 0.05，说明阈值本身会污染 oracle）。

四种上色，同一个 `paint_argmax`、同一次前向：

| 名称 | 含义 |
|---|---|
| `base_all` | 全部 query、原始 score。部署现状 |
| `base_cand` | 候选池上色。与 `base_all` 按构造相同，作自检 |
| `oracle_sel(p)` | 只保留 GT 精度 ≥ p 的候选，重算 logsumexp。**完美 per-candidate 权重 + 现有像素算子**的上限 |
| `oracle_pixel` | 像素被正确名字的任一候选覆盖即算涂对。任何候选级方法的绝对上限 |

指标沿用 F：涂对 `pixel_acc` / 涂错 `pixel_wrong` / 未涂 `pixel_unassigned`，按 GT 部件像素归一。

---

## 2. 留出集（v5_ce_lora ep2）

| 上色 | τ=0 涂对/涂错/未涂 | τ=0.3 | τ=0.5 |
|---|---|---|---|
| `base_all`（部署现状） | 0.716 / 0.281 / 0.003 | 0.710 / 0.264 / 0.026 | **0.626 / 0.160 / 0.215** |
| `oracle_sel(0.6)` | 0.846 / 0.109 / 0.045 | 0.845 / 0.109 / 0.046 | 0.827 / 0.080 / 0.093 |
| `oracle_sel(0.7)` | 0.842 / 0.094 / 0.065 | 0.841 / 0.093 / 0.065 | 0.821 / 0.065 / 0.115 |
| `oracle_sel(0.8)` | 0.807 / 0.083 / 0.110 | 0.807 / 0.083 / 0.110 | **0.783 / 0.057 / 0.160** |
| `oracle_sel(0.9)` | 0.745 / 0.072 / 0.184 | 0.745 / 0.071 / 0.184 | 0.724 / 0.047 / 0.229 |
| `oracle_pixel` | – | – | **0.945 / 0 / 0.055** |

误差分解（τ=0.5，按 GT 部件像素）：

| | 占比 |
|---|---:|
| 涂对 | 0.626 |
| 涂错-可救（存在正确名字的候选覆盖） | **0.141** |
| 涂错-漏检 | 0.019 |
| 未涂-可救 | **0.190** |
| 未涂-漏检 | 0.025 |

`oracle_sel(0.9)` 的涂对（0.724）低于 `oracle_sel(0.8)`（0.783）不是矛盾：精度门限越高，越多名字一个候选都留不下，整行变 −inf 再也赢不到像素，换成未涂。只有 75.7% 的名字有至少一个精度 ≥ 0.8 的候选（≥ 0.6 时 83.4%），这就是 `oracle_sel(0.8)` 未涂 0.160 的地板。

---

## 3. 分数排序有多弱

对每个 (图, 名字) 统计它 200 个候选的 IoU（留出集，v5）：

| 取谁 | 平均 IoU |
|---|---:|
| 随机一个候选 | 0.264 |
| **SAM3 分数最高的那个** | **0.428** |
| 分数前 5 里最好的 | 0.553 |
| 200 个里最好的（oracle） | 0.702 |

分数**有**信息（0.264 → 0.428），但离最优差 0.274。关键的一条是**"分数前 5 里最好的"已经到 0.553**——一个重排器不需要在 200 个里大海捞针，只要在头部几个里挑对，就能吃掉一半以上的差距。

排名统计（留出集）：

| bank | 最优候选恰好是分数第一 | 最优候选的分数排名中位数 |
|---|---:|---:|
| v3 | 11.0% | 38 / 200 |
| v5_ce_lora ep2 | 3.7% | 63 / 200 |

分数若完全无信息，中位排名应是 100。两个 bank 都只在弱信息区。**v5 的排名比 v3 更差**：v5 的 assign CE 优化的是像素级 logsumexp 的结果，不是 per-query 分数的序，它是把权重重新分配，不是把候选排得更准。这正是"候选级排序从没被建模过"的直接证据。

（这里有选择偏差：200 个里取 max 本身会抬高 oracle 那一行。所以判定用的是 `oracle_sel(p)`——一个对全体候选统一施加的阈值规则，不是逐名字挑冠军。）

---

## 4. v3 与 v5 的 oracle 前沿：排序器应该建在 v3 上

留出集 τ=0.5：

| bank | 现状 涂对/涂错/未涂 | `oracle_sel(0.8)` | `oracle_sel(0.6)` | `oracle_pixel` |
|---|---|---|---|---|
| v3 | 0.618 / 0.258 / 0.124 | **0.777 / 0.010 / 0.213** | 0.808 / 0.024 / 0.168 | 0.924 |
| v5_ce_lora ep2 | 0.626 / 0.160 / 0.215 | 0.783 / 0.057 / 0.160 | 0.827 / 0.080 / 0.093 | 0.945 |

涂对的天花板两者相同（0.777 / 0.783），但**涂错差 5.7 倍**（0.010 / 0.057）。v3 的候选在被完美筛选后几乎不再抢别人的像素；v5 的候选即使精度 ≥ 0.8 仍会外溢。

对下游这不是平局：`REPORT_concept_bank_v4_eval.md` §2 已经确定，涂错会被 SegviGen 照抄进 3D，未涂只是留灰。v3 的前沿（0.777 / 0.010 / 0.213）在这个偏好下严格优于 v5 的（0.783 / 0.057 / 0.160）。

**所以阶段 1 的第一个跑次用 v3 bank 作底座**，v5_ce_lora ep2 作第二个对照。这与设计文档 §4 写的"基线 = v5"相反，按本节结果修订。注意这条结论只说明"哪个底座的可达前沿更好"，不说明学出来的排序器能走到那里——那是阶段 1 要测的。

---

## 5. hard 切片与 MakerWorld

hard 切片（20 物体 / 14 张图，v5，τ=0.5）：

| 上色 | 涂对 | 涂错 | 未涂 |
|---|---:|---:|---:|
| `base_all` | 0.419 | 0.140 | 0.441 |
| `oracle_sel(0.6)` | 0.669 | 0.129 | 0.201 |
| `oracle_sel(0.8)` | 0.493 | 0.095 | 0.412 |
| `oracle_pixel` | 0.909 | 0 | 0.091 |

hard 上 `oracle_sel(0.8)` 只到 0.493，而 `oracle_sel(0.6)` 到 0.669：难物体的部件小、边界糊，高精度候选稀少，0.8 的门限会把太多名字整行删掉。阶段 1 在 hard 上要用更宽的目标，或者让 `keep` 是软权重而不是硬门限（设计文档 §4.4 的 λ 加权版本正是为此）。误差分解：可救 0.126 + 0.377 = 0.503，漏检只有 0.078。

MakerWorld（168 张图，v5，τ=0.5）：`base_all` 0.730 / 0.165 / 0.105 → `oracle_sel(0.8)` 0.802 / 0.036 / 0.162。外部集上涂对空间小（+0.07），压涂错空间大（0.165 → 0.036）。可救 0.152 + 0.071 = 0.223。

三个集合的共同点：**漏检都很小**（留出 0.044、hard 0.078、MW 0.047）。SAM3 几乎总是在某个 query 里画对了，只是那个 query 没被选中。

---

## 6. 最常见的可救混淆

留出 hard 切片（v5）：`body→rim`、`shell→chin strap`、`door→side panel`、`connector→sleeve`、`wing→forewing`、`rim→body`、`side panel→door`、`shelf→counter`。

MakerWorld：`tube→collar`、`panel→box`、`base→box`、`eraser→holder`、`clip→handle`、`box→lid`、`latch→plate`、`body→cap`。

两类：**包含关系**（`wing`/`forewing`、`box`/`lid`、`rim`/`body`）和**同级邻接**（`door`/`side panel`、`shelf`/`counter`）。两类都是候选之间的**几何关系**，恰好是图模型的边特征能表达而单个候选的分数无法表达的东西。

---

## 7. 复现命令

```bash
cd /root/autodl-tmp/sam3seggen
unset http_proxy https_proxy
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_HUB_OFFLINE=1
PY=/root/autodl-tmp/envs/sam3/bin/python
COMMON="--dataset_root /root/autodl-tmp/datasets/pv \
  --model /root/autodl-tmp/sam3seggen/weights/facebook/sam3 \
  --split_file /root/autodl-tmp/runs/v5_ce_lora/split.json \
  --hard_file /root/autodl-tmp/datasets/pv_hard.txt \
  --extra_eval mw=/root/autodl-tmp/datasets/makerworld/concept_bank_clean_v2/objects"

# v5_ce_lora ep2（当前部署最佳），--dump 顺带缓存阶段 1 的候选特征
$PY finetune/oracle_candidates.py $COMMON --out /root/autodl-tmp/runs/oracle_v5 --dump \
  --resume /root/autodl-tmp/runs/v5_ce_lora/bank_epoch2.pt \
  --decoder_lora 8 --lora_scope mask,text \
  --lora_file /root/autodl-tmp/runs/v5_ce_lora/decoder_lora_epoch2.pt

# v3 对照
$PY finetune/oracle_candidates.py $COMMON --out /root/autodl-tmp/runs/oracle_v3 \
  --resume /root/autodl-tmp/datasets/concept_bank_v3/bank.pt
```

每跑约 90 s（240 + 168 张图，含模型加载）。`--dump` 写 `feats/<obj>_<az>.npz`（每候选的 score / precision / recall / IoU / 面积），阶段 1 的训练特征从同一脚本对训练集再跑一次拿到。

---

## 8. 下一步

1. **阶段 1 建在 v3 bank 上**（§4），v5_ce_lora ep2 作第二对照。
2. 目标用软权重 `λ·log keep` 而不是硬门限（hard 切片的 `oracle_sel(0.8)` 塌掉说明硬门限在难物体上代价太大）。
3. 训练集特征：`oracle_candidates.py --split_file` 换成对 1800 个训练物体跑一遍（约 3600 张图 ≈ 12 min）。
4. 排序器只需要在头部候选里挑对（§3 的 0.553），可以先试一个"只对分数前 16 个候选重排"的轻量版本，训练更快、也更接近部署时的算力预算。
