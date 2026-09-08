# 扩大数据集：可选路线与实测成本

2026-09-08。所有速率与产出率都是本机实测（见 §2），不是估算。
相关：概念库数据格式 `DATA_concept_bank.md`、概念库改进 `PLAN_concept_bank_v4.md`、总览 `REPORT_overview.md`。

---

## 1. 结论先说

**最便宜的一步不需要下载任何东西。** 本地 `datasets/partverse/` 已经是**完整的 PartVerse 12 030 个物体**（含 `anno_infos` 面标签、`normalized_glbs`、`text_captions.json`，MIT 许可，[arXiv 2507.08772](https://arxiv.org/abs/2507.08772)），而 `datasets/pv/` 只用了其中 **2 000** 个。剩下 10 030 个已经躺在盘上。

| 路线 | 新增可用物体 | 下载 | 代码改动 | 计算 | 磁盘 |
|---|---|---|---|---|---|
| **A. 用完本地 PartVerse** | +5 800（10 030 × 58 % 产出率） | 0 | 0 | 6.4 h | 39 GB |
| **B. PartVerse-XL** | +16 000（净新增约 28 000 × 58 %） | ~106 GB | 0（格式相同） | 18 h | 107 GB |
| **C. PartNeXt** | +23 500（**自带人工部件名**） | 待查 | 新 importer 约 300 行 | 26 h | 160 GB |
| **D. 挖已有物体**（视角 / 层级 / 补变体） | 0（同样物体，更多样本） | 0 | 小 | 11–31 h | 18–110 GB |

E 盘剩余 **1.1 TB**，磁盘不是概念库数据的约束；只有 SegviGen LoRA 的 `variants/` 才吃盘（见 §5）。

建议顺序：**A → D → B → C**。A 零成本零风险；D 提升的是「同样物体的信息利用率」，而 v1–v6 的教训正是模型在过拟合 PartVerse 的风格；B 是纯线性放大同一分布；C 分布互补且省掉命名环节，但要写新 importer。

---

## 2. 实测数据（本机，2026-09-08）

在 40 个从未处理过的 PartVerse 物体上跑 `import_partverse.py --dry_run`，再对其中 7 个跑真实导入 + 渲染：

| 项 | 实测 |
|---|---|
| 导入产出率 | **58 %**（40 个 → 23 保留 / 15 被部件数过滤 / 2 因 `segmented.glb` 与整网格对不齐失败）。真实导入 12 → 7，同样 58 % |
| `import_partverse.py` | 0.17 s/物体（dry-run）、约 0.4 s/物体（含导出 `parts/*.glb`） |
| `render_views.py --azimuths 0,135` | **3.4 s/物体**（Cycles 32 samples 的 `render.png` + nvdiffrast 的 `ids.npy`，两视角） |
| 磁盘：概念库最小集 | **1.46 MB/物体**（`names.json` + 2 × (`render.png` + `ids.npy`)） |
| 磁盘：整个物体目录 | **6.7 MB/物体**（再加 `parts/*.glb` + `input.glb` + `captions.json`） |
| 产出校验 | 7/7 通过：`ids.npy` 是 int16 512×512、与 `render.png` 严格对齐、id 全部 < `len(names)` |

**启发式名字必须重标。** 探针里 PartVerse caption 抽出来的名字长这样：`detailed three-dimensional model`、`cylindrical component`、`texture`、`part`、`representing the ground`、`highlighted component`。这正是 `relabel/HANDOVER.md` 记录的 17.6 % 垃圾名问题，2 000 个物体当初就是靠 Grok 4.6 重标 + 人工复核解决的（改动了 92 % 的名字）。新物体走同一条路。

现成清单：`datasets/pv_fresh_all.txt`，10 030 个尚未处理的 id。

---

## 3. 路线 A：用完本地 PartVerse（+5 800，建议先做）

三步，全部有现成脚本：

```bat
REM 1. 导入：切 parts、写占位 names.json / captions.json（约 40 min）
finetune\run_ft.bat import_partverse.py --partverse E:\AI_New\ModelGen\datasets\partverse ^
  --out E:\AI_New\ModelGen\datasets\pv --min_parts 3 --max_parts 24 ^
  --ids @E:\AI_New\ModelGen\datasets\pv_fresh_all.txt

REM 2. 重标名字：prep_relabel.py 出批 → Grok 4.6 逐批 → merge → apply（见 relabel\HANDOVER.md §5、§6）
REM    5 800 个物体 = 116 批 × 50。并行 6–8 个 subagent，每批 1–3 min

REM 3. 渲染两视角（5 800 × 3.4 s ≈ 5.5 h，可断点续跑）
finetune\run_ft.bat render_views.py --dataset_root E:\AI_New\ModelGen\datasets\pv --azimuths 0,135 --chunk 25
```

做完概念库训练集从 **2 000 → 约 7 800（3.9×）**，打包体积从 730 MB → 约 2.8 GB。

**风险与注意：**

- **名字质量是唯一真实风险。** 2 000 个物体的质量是靠人工复核撑起来的（`review_md.py` 抽样 + `screen_dominant.py` 全库主导部件筛查 + uncertain 清零三轮）。5 800 个没法同等力度人工过。建议：自动校验（`review_relabel.py` 的泛称词 / 词数 / 覆盖检查）全量必过，**人工只抽 5 %**，并把 `screen_dominant.py`（占比 ≥ 45 % 却给了局部名的部件）作为强制筛查——那是当初 139 处错误的来源，也是最伤 SAM3 的一类错。
- **旧的 2 000 个要单独成一层。** 它们经过人工复核，新的 5 800 个没有。训练时按 `--min_count` 统计词表没问题，但**留出集必须继续只从旧 2 000 里选**，否则 v3 的 0.368 就没法比了。建议把新物体记进 `pv_list_d_new.txt`，`names_meta.json` 的 `source` 字段标明来源。
- 那 2 个 `segmented.glb` 对不齐的物体（`419b01d4`、`794f3801`）是数据本身的问题，`max_far_frac` 已经把它们挡掉了，不用管。
- 想再多要一些物体，可以把 `--max_parts` 从 24 放到 32（脚本默认），产出率会上去，代价是多部件物体的渲染更慢、SAM3 提示词更多。

---

## 4. 路线 D：挖已有物体（零新物体，性价比高）

同样的 2 000 个物体里还有没榨出来的信息，而且这几项**不需要重新命名**——命名是路线 A 唯一的人力瓶颈。

### D1. 视角从 2 个扩到 6–12 个

概念库和 LoRA 现在都只训 `az0` / `az135`，但部署路径（`ext_bench.py`、`geosam2_masks.py`）最多用 12 视角。斜后方视角上部件被压缩、被遮挡，模型从没见过。

```bat
finetune\run_ft.bat render_views.py --dataset_root ...\pv --azimuths 0,60,135,180,240,300 --chunk 25
```

2 000 物体 × 4 个新方位 ≈ 3.4 s × 2 = 6.8 s/物体 → **约 4 h，磁盘 +12 GB**。这是本文档里**单位成本最低的一项**。

### D2. 用 PartVerse 的层级做多粒度

`anno_infos/<id>/<id>_info.json` 里有 **`ordered_part_level`**（还有 `bboxes`、`weights`）。也就是说 PartVerse 自带部件层级，我们现在只用了叶子层。把相邻层级的部件合并成粗粒度标签，等于**零新数据换一套新的监督**，而且直接对上「整块优先于语义」这个一直以来的核心诉求——粗粒度标签本身就是「整块」。

需要写代码（读 `ordered_part_level`、按层合并 `face2label`、生成第二套 `names.json` + `ids.npy`），估计 200 行。

### D3. 把剩下 682 个物体的 `variants/` 补上

`datasets/pv/` 的 2 000 个物体**全部**有 `names.json` / `parts/` / `input.glb` / 2 视角，但只有 **1 318** 个有 `variants/`（LoRA 训练样本）。剩下 682 个只差最后一步。按 `HANDOVER_cloud_segvigen.md` 的实测 1.5–2 min/物体：**约 20 h，磁盘 +90 GB**，LoRA 训练集 1 318 → 2 000（+52 %）。

---

## 5. SegviGen LoRA 的变体预算（这里磁盘才是约束）

抽样实测：一个完整物体的 `variants/` 有 **14 个变体**（`clean` × 2 视角、`corrupt` × 6、`partial` × 4、`sam3` × 2），每个 8.7 MB，合计 **约 132 MB/物体**。

| 方案 | 物体 × 变体 | 磁盘 | 生成耗时 |
|---|---|---|---|
| 现状 | 1 318 × 14 | 174 GB | 已完成 |
| 补齐本地 2 000 | 2 000 × 14 | 264 GB | +20 h |
| 路线 A 全量、每物体 14 变体 | 7 800 × 14 | **1.03 TB** | +170 h |
| 路线 A 全量、**每物体 6 变体** | 7 800 × 6 | **440 GB** | +73 h |

E 盘只剩 1.1 TB，14 变体的全量方案会把盘吃光。

**建议：固定预算下选「更多物体 × 更少变体」。** v1–v6 的教训是模型过拟合 PartVerse 的风格与部件划分口径（外部资产上 v6 相对 base 提升有限），这是**物体多样性**不足，不是每个物体的噪声增广不足。同一个物体的 6 个 `corrupt` 变体互相高度相关，边际信息远小于 6 个新物体。具体建议每物体保留 `clean` × 2 + `sam3` × 2 + `corrupt` × 2 = 6 个，`partial` 变体在 v5 已被证明无效（训练时能从 GT 知道涂灰 patch 属于哪个部件，推理时没有这个信息）可以直接砍掉。

**概念库不受这个约束**：它只要 `names.json` + 2 视角，1.46 MB/物体，全量 7 800 个也只有 11 GB。

---

## 6. 路线 B：PartVerse-XL（+16 000，零代码改动）

[`dscdyc/partversexl`](https://huggingface.co/datasets/dscdyc/partversexl)，FullPart（[arXiv 2510.26140](https://arxiv.org/abs/2510.26140)）的数据集，**40 K 物体 / 320 K 部件**，是 PartVerse 12 K/91 K 的升级版，来自 Objaverse-XL，200+ 类别。

**目录结构和我们本地的 PartVerse 完全一致**（`anno_infos/<id>/<id>_face2label.json` + `_info.json` + `_segmented.glb`、`normalized_glbs/<id>.glb`、`text_captions.json`），所以 **`import_partverse.py` 一行不用改**。

要下的只有两样，`textured_part_glbs`（67 个分卷）**不需要**——我们本来就是用 `face2label` 从整体网格上切几何，部件不需要纹理：

```bash
huggingface-cli download dscdyc/partversexl --repo-type dataset --local-dir /data/partverse_xl \
  --include "anno_infos.tar.gz" "normalized_glbs.tar.gz0*" "text_captions.json" "metadata.csv" "train.csv" "val.csv"
# anno_infos 8.97 GB + normalized_glbs 9 × 10.7 GB ≈ 106 GB
```

**先做一件便宜的事**：只下 `anno_infos.tar.gz` 或 `metadata.csv`，和本地 12 030 个 id 求交集，确认净新增量。论文说 XL 是 PartVerse 的 "expanded and **refined** extension"，"refined" 意味着部分旧物体被重新标注过——如果重标了，那 2 000 个人工复核过的名字与新标签的对应关系需要重新确认。这一步花 10 分钟，能避免下 106 GB 之后才发现问题。

净新增按 28 000 估、产出率 58 %，得约 16 000 个新物体：渲染 18 h、磁盘 107 GB、命名 320 批 Grok。

**注意**：B 只是把同一个分布（Objaverse 系、同一套标注流程与部件粒度口径）线性放大。它能改善长尾名字的覆盖率和词表规模，但**不解决**「外部资产上提升有限」这个问题——那是分布差异，要靠 C 或者真正的目标域资产。

---

## 7. 路线 C：PartNeXt（+23 500，自带人工部件名）

[`AuWang/PartNeXt`](https://huggingface.co/datasets/AuWang/PartNeXt) + [`AuWang/PartNeXt_mesh`](https://huggingface.co/datasets/AuWang/PartNeXt_mesh)（NeurIPS 2025 D&B，[arXiv 2510.20155](https://arxiv.org/abs/2510.20155)）。**23 519 个带纹理物体 / 350 187 个部件 / 50 类**，来源 Objaverse 14 811 + ABO 2 633 + 3D-FUTURE 6 075。

对我们有两个别处拿不到的东西：

1. **部件名是人工标的**，存在 `hierarchyList` 的每个节点的 `name` 字段里。**整条 Grok 重标 + 人工复核流程可以跳过**——而那是路线 A/B 唯一的人力瓶颈。
2. **层级是显式的**（深度 4–10 层，PartNet 式树结构），比 PartVerse 的 `ordered_part_level` 更完整。天然支持多粒度监督，直接服务「整块优先」。

代价是要写新 importer（约 300 行）：数据是 arrow 格式，字段为字符串；`masks` 是**叶子节点**的面索引，`mesh_face_num` 是逐 mesh 的面数（一个物体可能多个 mesh，要先拼成统一面序），`hierarchyList` 是树。目标是产出和现有一致的 `parts/*.glb` + `names.json`，之后 `render_views.py` 直接复用。

**分布互补性是它最大的价值**：ABO 和 3D-FUTURE 是家具 / 商品，PartVerse 是 Objaverse 的通用资产。我们的 10 个外部资产（角色、机械、玩具）两边都不完全覆盖，但两个分布混合训出来的概念库，泛化性应该好于单一分布放大。

**先验证再投入**：下 100 个物体，跑通 importer，看 `names` 的口径和我们的规则（1–3 词、小写、禁泛称词）差多少，再决定是否全量。PartNeXt 的名字来自 50 类的固定层级模板，可能比我们的自由命名更规范，也可能粒度太细（10 层深度的叶子可能是「螺丝」级别）。

---

## 8. 执行顺序与判据

| 批次 | 内容 | 成本 | 通过判据 |
|---|---|---|---|
| 1 | **D1** 视角 2 → 6（现有 2 000 个物体） | 4 h、12 GB | 概念库在新方位上的灰像素不显著高于 az0 |
| 2 | **A** 导入 + 重标 + 渲染剩余本地 PartVerse | 6.4 h 计算 + 116 批标注 | 自动校验 problems = 0；5 % 抽样人工过；`screen_dominant.py` 全量筛查完 |
| 3 | 用 A 的数据重训概念库 v4 | 70 min × N | **在旧 2 000 的留出集上**，阈值 0.5 mIoU ≥ 0.368 且难负例误检 ≤ 0.19；词表覆盖率 ≥ 80 % |
| 4 | **D3** 补齐 682 个 `variants/`（每物体 6 变体） | 20 h、40 GB | LoRA 训练集 2 000 个物体 |
| 5 | **B** 的 id 交集探查（只下 metadata） | 10 min | 确认净新增量与旧标签是否被 refine |
| 6 | **B** 或 **C** 全量 | 18–26 h + 下载 | 同批次 3 的判据 |

两条贯穿始终的纪律：

1. **留出集永远只从人工复核过的旧 2 000 个里选**，否则和 v3 的 0.368 没有可比性。
2. **每次扩容后要同时报 mIoU 和难负例误检**。数据变多最容易出现的假象是检出率整体上升、mIoU 变好，而实际是把误检也一起放大了。

数据扩容能改善的是**长尾名字覆盖**和**类别多样性**；它**不能**替代 `PLAN_concept_bank_v4.md` 里的 A（跨名字竞争损失）——那是「优化目标和部署目标不一致」的结构问题，加数据不会自动修好。两件事应该并行推进。
