# SAM3 概念库：训练数据格式

给准备新数据的人。`concept_bank.py` **只读下面这些文件**，不要把 SegviGen LoRA 那套 latent / variant 混进来。

现有 PartVerse 数据在本地 `datasets/pv/`，概念库权重在 `datasets/concept_bank_v3/bank.pt`（不进 git）。

---

## 1. 概念库在学什么

冻结 SAM3，只学两个加在文本 embedding 上的偏移：

- `E_0`：所有提示词共享的全局偏移（适应灰渲染、部件名词）
- `E[name]`：训练集里出现 ≥ `--min_count`（默认 8）次的名字各一个向量；长尾名字只吃 `E_0`

正样本：一张图里**可见**且非 `uncertain` 的部件名，目标掩码 = 该名字下所有部件像素的并集。  
负样本：其他物体的常见名字，目标 = 空。  
损失：正样本 BCE+Dice + 负样本 BCE + presence BCE。

推理时 `sam3_to_2dmap.py --concept_bank bank.pt` 把 `E_0 + E[name]` 加到 SAM3 文本特征上。

---

## 2. 每个物体最少要有什么

```
<root>/<object_id>/
  names.json                 # 必需。按部件 index 的名字列表
  names_meta.json            # 建议。整体物体名 + 逐部件 uncertain
  parts/0.glb  1.glb  ...    # 生成 ids.npy 时需要；训练本身不读
  input.glb                  # 生成 render.png 时需要；训练本身不读
  views/az0/
    render.png               # 必需。与 ids.npy 同机位、同分辨率
    ids.npy                  # 必需。int16，H×W，像素 = 部件 index，背景 = -1
    ids_preview.png          # 可选。给人看的伪彩图，训练不读
  views/az135/               # 第二个视角，格式同上；默认方位 0 和 135
    render.png
    ids.npy
```

`concept_bank.py` 发现物体的条件：存在 `views/az0/ids.npy`。  
一张图进训练集的条件：同目录同时有 `render.png` 和 `ids.npy`。

**不需要**：`shape_slat.pth`、`variants/`、`sam3_masks.npz`、`voxel_part.npy`、`input.vxz`。那些是 SegviGen LoRA 的。

`<object_id>` 可以是任意目录名（现有数据用 32 位 hex）。一个根目录下所有物体平铺，不要再套一层类别文件夹。

---

## 3. 各文件的精确格式

### 3.1 `names.json`（生产文件，SAM3 和概念库都读它）

JSON 数组，**第 i 个字符串 = `parts/` 自然排序后第 i 个部件**。

```json
["body", "barrel", "nozzle", "hatch", "barrel", "barrel", "arm", "cable", "barrel"]
```

规则：

| 项 | 要求 |
|---|---|
| 长度 | 必须等于 `parts/*.glb` 个数 |
| 空串 | 不允许。说不清就写保守功能名（`body` / `panel` / `base`）并把 `uncertain[i]=true` |
| 大小写 | 全小写，无标点 |
| 词数 | 1–3 个英文词，能短则短 |
| 同名 | 同一物体内同类部件必须同名（四条腿都叫 `leg`）。只有来源明确写了方位才加 `left` / `front` |
| 禁止 | 外观词和占位词：`component` `part` `piece` `section` `element` `object` `model` `detailed` `low-poly` `texture` `gradient` `geometric` `cylindrical` `rectangular` `spherical` `circular`，以及颜色词 |
| 语义 | 回答「这是什么部件」，不是「它长什么样」。`head` / `wheel` / `backrest` 对；`cylindrical component` 错 |
| 主导件 | 画面里占大半的那块，不要标成局部名（整椅被标成 `seat`、整狗被标成 `head` 是最常见的错） |

名字是 SAM3 的提示词，也是概念库词表。稀有名字（训练集出现 < 8 次）**不会**拿到独立 `E[name]`，只走 `E_0`。准备新数据时尽量复用已有高频词（`leg` `body` `head` `wheel` `arm` `torso` `base` `handle` `lid` …），不要发明近义变体（`hind leg` / `rear-leg` / `leg2`）。

### 3.2 `names_meta.json`

```json
{
  "source": "human",
  "object": "robot",
  "uncertain": [false, false, false, false, false, false, false, false, false]
}
```

| 字段 | 用途 |
|---|---|
| `object` | 整体物体名，1–3 词。模板不是裸 `{name}` 时会填进 prompt（默认 `--template name` 不用它） |
| `uncertain` | 与 `names.json` 等长。`true` 的部件默认**不作为正样本**（`--keep_uncertain` 才用） |
| `source` | 任意备注，训练不读 |

没有这个文件也能训：物体名视为空，全部 `uncertain=false`。

### 3.3 `views/<az>/render.png`

- RGB（或 RGBA，加载时当图读即可），**512×512**（`common.RESOLUTION`）
- 与 `data_toolkit/transforms.json` 同一套相机，方位角写在目录名里：`az0`、`az135`（`az{角度:g}`，135.0 写成 `az135`）
- 物体居中、铺满画面；背景尽量干净（现有数据是灰底纹理渲染）
- 必须和 `ids.npy` **同机位**：轮廓对不齐，监督就是错的。现有数据 render 前景与 ids 前景 IoU ≈ 0.96（差在抗锯齿）

生成：`finetune/render_views.py`（bpy Cycles 出 render，nvdiffrast 出 ids），**不走体素化**。

```bash
# Windows
finetune\run_ft.bat render_views.py --dataset_root <root> --azimuths 0,135 --chunk 25

# Linux（SegviGen venv，需要 bpy + nvdiffrast）
python finetune/render_views.py --dataset_root <root> --azimuths 0,135 --chunk 25
```

前提：该物体已有 `input.glb` 和 `parts/<i>.glb`。已存在的 `render.png` / `ids.npy` 会跳过。

### 3.4 `views/<az>/ids.npy`

- `dtype=int16`，形状 `(512, 512)`
- 像素值：`0 .. N-1` = `names.json` 的下标；**`-1` = 背景**
- 同名部件是**不同 index**（两条腿是 3 和 4，都叫 `leg`）。概念库会在训练时按名字求并集，你不要预先合并
- 不可见的部件：该视角里没有任何像素等于它的 index，这是正常的，该名字这一视角就不是正样本
- 不要写 `-2` 或其他哨兵；不确定的部件用 `names_meta.uncertain`，不要改 ids

`ids_preview.png` 由 `render_views.py` 顺手写出，伪彩，只给人校对，训练不读。

### 3.5 `parts/` 与 `input.glb`（只为了生成视图）

- `parts/0.glb`、`parts/1.glb`、… 自然排序后的顺序 **就是** `names.json` 的下标。不要 `part_00_head.glb` 这种会打乱排序的名字，或确保排序后仍与名字对齐
- 每个文件一块网格；世界坐标与 `input.glb` 一致
- `input.glb` 是带纹理的整物，用来渲 `render.png`

从已拆件的 GLB 导入：

```bat
finetune\run_ft.bat import_glb.py --glb your.glb --out <root>\<id> --names head body leg tail
```

---

## 4. 根目录旁的清单（可选）

| 文件 | 作用 |
|---|---|
| `pv_holdout_v3.txt`（或你自己的 holdout） | 一行一个 `object_id`（`#` 开头是注释）。训练时 `--holdout_file` 排除它们。现有清单 **55 个** = 35 mixed + 20 hard |
| 没有 holdout 文件 | `concept_bank.py` 按 `--holdout_frac 0.10` 随机留出 |

留出集实际构成：`--holdout_file` 里的 id 全部进留出，再随机补到 `--holdout_frac`。v3 是 55 固定 + 145 随机 = **200 个留出 / 1800 训练**，写在 `<out>/split.json`。补的那 145 个由 `--seed` 和**根目录下的物体列表**共同决定，所以物体集合一变，split 就变；要严格对比就直接沿用旧的 `split.json` 或保持物体集合不变。

---

## 5. 在云服务器上训练

### 5.1 只需要传这些（本地已打好包）

概念库不需要 `datasets/pv` 的全部 133 GB，只要第 2 节那份最小集：

| | 大小 |
|---|---|
| 2000 物体 × (`names.json` + `names_meta.json` + 2 视角 × (`render.png` + `ids.npy`)) | 2.87 GB（12 000 个文件） |
| 打包后 `cb_data.tar.gz` | **730 MB**（`ids.npy` 里几乎全是背景，gzip 压到 0.5 %） |

包里还含 `pv_holdout_v3.txt`、`pv_hard.txt`、`pv_holdout_mix.txt` 和 `concept_bank_v3/{bank.pt, split.json, text_cache.pt, log.jsonl}`（对照基线，共 3.2 MB）。

数据和权重已上传到 HF（**私有仓**，云端需要一个有读权限的 token）：

| 仓库 | 内容 |
|---|---|
| [`Zaun1996/segvigen-pv-2view`](https://huggingface.co/datasets/Zaun1996/segvigen-pv-2view)（dataset） | `cb_data.tar.gz` + 三份物体清单 |
| [`Zaun1996/sam3-concept-bank`](https://huggingface.co/Zaun1996/sam3-concept-bank)（model） | v3 的 `bank.pt` / `bank_epoch*.pt` / `split.json` / `log.jsonl` / `text_cache.pt` |

```bash
# 云端
export HF_TOKEN=<你的读 token>
huggingface-cli download Zaun1996/segvigen-pv-2view cb_data.tar.gz \
  --repo-type dataset --local-dir /data
cd /data && sha256sum cb_data.tar.gz && tar -xzf cb_data.tar.gz
# 得到 /data/pv/<id>/...、/data/pv_holdout_v3.txt、/data/concept_bank_v3/

# 只要 v3 权重做对照
huggingface-cli download Zaun1996/sam3-concept-bank --local-dir /data/concept_bank_v3
```

sha256 `b2b0ca006492ac9eb6e10a09c7b5b03a147418a6e413da655be3d955aa998ded`。
也可以直接 `scp E:/AI_New/ModelGen/datasets/cb_data.tar.gz user@cloud:/data/`。

重新生成这个包（本地）：`datasets/cb_filelist.txt` 是文件清单，`tar -czf cb_data.tar.gz -T cb_filelist.txt`（在 `datasets/` 下执行）。

环境只要 SAM3 那一个 venv（`transformers>=5`、torch、PIL、numpy），**不需要** TRELLIS.2 / bpy / nvdiffrast / o_voxel——那些只在生成 `render.png` / `ids.npy` 时用得到，而这两样已经在包里。SAM3 权重 `facebook/sam3` 是 gated，云端要先 `huggingface-cli login`。

### 5.2 训练命令（复现 v3 的配置）

```bash
.venv_holo/bin/python finetune/concept_bank.py \
  --dataset_root /data/pv \
  --out /data/concept_bank_v4 \
  --template name --epochs 3 --neg_mode mixed --negatives 3 --neg_weight 0.5 \
  --lr_e0 5e-4 --holdout_file /data/pv_holdout_v3.txt \
  --azimuths 0,135 --min_count 8 --max_prompts 12 \
  --eval_thresholds 0.4,0.5,0.6 --wandb --wandb_project segvigen-sam3
```

v3 的实际开销：3 epoch × 3600 step，**70 min**、峰值 **15.3 GB** 显存（SAM3 全程冻结，只有两个 256 维张量有梯度）。`--max_prompts 12` 控制单步的 `[N, Q, h, w]` 掩码显存，显存小就调低。

只评测已有 bank，不训练：

```bash
.venv_holo/bin/python finetune/concept_bank.py --dataset_root /data/pv --out /tmp/eval \
  --eval_only --resume /data/concept_bank_v3/bank.pt --holdout_file /data/pv_holdout_v3.txt \
  --eval_thresholds 0.4,0.5,0.6
```

---

## 6. 训练结果是什么

### 6.1 产物文件

| 文件 | 内容 | 用途 |
|---|---|---|
| `bank.pt` | 287 KB。`{E_0: [256], E: [N_names, 256], names: [...], template, base, meta}`。v3 是 **280 个名字**（训练集出现 ≥ 8 次的），`template = "{name}"`，`base = facebook/sam3`，`meta` 里存了全部训练参数 | **这是唯一的部署产物**。`sam3_to_2dmap.py` / `sam3_masks.py` / `geosam2_masks.py` 的 `--concept_bank` 加载它 |
| `bank_epoch1/2/3.pt` | 每个 epoch 的快照 | 挑最好的那个 epoch |
| `split.json` | `{train: [...], holdout: [...]}` | 复现同一划分 |
| `log.jsonl` | 每 50 step 一行 `{loss, e0_norm, e_norm, vram_gib}`；每 epoch 一行 `kind="eval"` 的完整指标 | 曲线与验收 |
| `text_cache.pt` | 2.8 MB。全部词表名字 + 物体名的 256 维文本向量（已加偏移） | v3–v5 的图例 token 用；v6/v7 路线**不用** |

模型本体没有任何改动——SAM3 权重全程冻结，学到的东西只有 `E_0`（一个共享 256 维偏移）和 `E[name]`（280 个名字各一个）。所以 287 KB 就是全部成果。

### 6.2 指标含义

每个 epoch 在留出集上跑一次评测（v3 用 240 张图 / 1068 个提示词），同一次前向在 4 个分数阈值上出数：

| 指标 | 定义 | 越大越好？ |
|---|---|---|
| `miou` | 每个提示词的「预测并集 vs 该名字的 GT 并集」IoU 的均值 | ↑ |
| `bind_rate` | IoU ≥ 0.5 的提示词占比 | ↑ |
| `false_positive_rate` | 随机负例名字（本物体没有的名字）里，吐出了任何像素的占比 | ↓ |
| `false_positive_rate_hard` | 同上，但负例是共现 / 文本相似的名字（更难） | ↓ |
| `part_bound_rate` | 可见部件实例中，被自己名字覆盖 ≥ 50 % 像素的占比 | ↑ |
| `grey_pixel_ratio` | 部件像素里最终没被任何名字绑定的比例 → 在 2D 图上变灰 | ↓ |

对下游最关键的是 **`grey_pixel_ratio`**（灰色 = SegviGen 拿不到颜色 = 漏件）和 **`false_positive_rate_hard`**（把 A 涂成 B）。`miou` 高但 hard FP 也高，说明只是把阈值效果整体放宽了，没有真正学会区分——所以要看整条阈值扫描，而不是单个数。

### 6.3 v3 实测（`concept_bank_v3/log.jsonl`，留出集 200 物体 / 1068 提示词）

epoch 0 = 裸 SAM3（不加 bank）的基线，epoch 3 = 最终产物：

| 阈值 | mIoU 基线 → v3 | bind 基线 → v3 | 灰像素 基线 → v3 | 随机 FP | 难负例 FP |
|---|---|---|---|---|---|
| 0.3 | 0.323 → **0.409** | 0.331 → 0.419 | 0.558 → **0.281** | 0.103 → 0.124 | 0.289 → 0.324 |
| 0.4 | 0.309 → **0.397** | 0.322 → 0.406 | 0.607 → **0.321** | 0.074 → 0.093 | 0.229 → 0.254 |
| **0.5**（部署常用） | 0.288 → **0.368** | 0.301 → 0.383 | 0.654 → **0.381** | 0.057 → 0.062 | 0.186 → 0.188 |
| 0.6 | 0.259 → **0.333** | 0.268 → 0.348 | 0.685 → 0.501 | 0.043 → 0.044 | 0.142 → 0.121 |

逐 epoch（阈值 0.5 的 mIoU）：0.288 → 0.351 → 0.365 → 0.368，**收益基本在第 1 个 epoch**，第 3 个只 +0.003；训练损失 1.105 → 0.851；`E_0` 范数 0.13 → 4.21，`E[name]` 平均范数 0.62 → 6.13。

怎么读这张表：

- **主收益是「找得到」**：灰像素在阈值 0.5 上 0.654 → 0.381（少了 42 %），部件绑定率 0.429 → 0.543。对 SegviGen 直接意味着 2D 图上被涂色的部件更多。
- **误检没有变坏**：阈值 0.5 上随机 FP 0.057 → 0.062、难负例 FP 0.186 → 0.188，基本持平；阈值 0.6 上难负例 FP 甚至更低（0.142 → 0.121）。这就是「不是靠整体放宽阈值换来的」的证据。
- **阈值 0.3 别用**：那里难负例 FP 从 0.289 涨到 0.324。部署统一用 **0.4**（`ext_bench.py` / `geosam2_masks.py` 的默认）或 0.5。
- **绝对值不高是正常的**：mIoU 0.37 是「名字级并集」的 2D IoU，GT 是 PartVerse 的部件划分，很多名字本身就有歧义（`body` 到哪算完）。它只作相对比较用。

### 6.4 新数据训完的验收标准

在**同一个留出集**上（要么沿用 `concept_bank_v3/split.json`，要么保证物体集合不变）：

1. 阈值 0.5 的 mIoU ≥ 0.368，且难负例 FP ≤ 0.19。只涨 mIoU 不看 FP 不算通过。
2. 阈值 0.5 的灰像素 ≤ 0.381。
3. `min_count 8` 的词表覆盖率（打印在启动日志里，"they cover X% of train part instances"）不低于 80 %。v3 是 280 个名字覆盖约 82 % 的部件实例；如果新数据全是一次性名字，覆盖率会掉，`E[name]` 学不到东西，只剩 `E_0`。
4. 下游抽查：拿新 bank 跑 `sam3_to_2dmap.py --threshold 0.4`，对 10 个外部资产的提示词命中率不低于 48/51（v3 的数）。

---

## 7. 交数据前自检

对每个物体：

```python
import json, os, numpy as np
from PIL import Image

root, oid = "/data/new_pv", "my_chair_001"
obj = os.path.join(root, oid)
names = json.load(open(os.path.join(obj, "names.json"), encoding="utf-8"))
meta = json.load(open(os.path.join(obj, "names_meta.json"), encoding="utf-8"))
parts = sorted(f for f in os.listdir(os.path.join(obj, "parts")) if f.lower().endswith(".glb"))
assert len(names) == len(parts) == len(meta["uncertain"])
assert all(n and n == n.lower() and n.strip() == n for n in names)
assert all(len(n.split()) <= 3 for n in names)

banned = {"component","part","piece","section","element","object","model",
          "detailed","cylindrical","rectangular","spherical","circular"}
assert not any(w in banned for n in names for w in n.split())

for az in ("az0", "az135"):
    v = os.path.join(obj, "views", az)
    im = np.array(Image.open(os.path.join(v, "render.png")))
    ids = np.load(os.path.join(v, "ids.npy"))
    assert ids.shape == im.shape[:2] == (512, 512)
    assert ids.dtype == np.int16
    vis = set(int(x) for x in np.unique(ids) if x >= 0)
    assert vis <= set(range(len(names)))
    assert (ids == -1).any()  # 应有背景
    print(oid, az, "visible", [(i, names[i]) for i in sorted(vis)])
```

还要**看图**：打开 `ids_preview.png` 和 `render.png`，抽几个大块确认名字指的是同一块东西。自动检查查不出「整椅叫 seat」。

批量统计：词表里出现 ≥ 8 次的名字应覆盖大部分实例（现有库约 82%）。若新数据几乎全是一次性名字，概念库学不到 `E[name]`，只剩 `E_0`。

---

## 8. 从现有仓库扩数据的最短路径

1. 每个新物体：`input.glb` + 按件拆开的 `parts/*.glb` + 按上面口径写好的 `names.json` / `names_meta.json`
2. `render_views.py --azimuths 0,135`（可再加更多方位；目录名跟着改，训练 `--azimuths` 对齐）
3. 跑第 7 节自检，抽查看图
4. 把新物体目录拷进同一个 `--dataset_root`（或再给 `concept_bank.py` 一个根；当前脚本只接受一个 root）
5. holdout 不要混进训练：新物体要么全部训练，要么自己列一份 holdout

不要为概念库去跑 `make_samples_a.py` / `prepare_object`。那是另一条管线，还会触发 `o_voxel`。

---

## 9. 和 SegviGen LoRA 数据的关系

| | 概念库 | SegviGen LoRA（v6/v7） |
|---|---|---|
| 监督 | 2D：名字 → 像素并集 | 3D：颜色图 → 纹理 latent |
| 必需文件 | `names.json` + `views/*/render.png` + `ids.npy` | 上表全部 + latent + `variants/` |
| 环境 | `.venv_holo`（transformers ≥ 5） | SegviGen `.venv`（transformers 4.57） |

两套数据可以共用同一个物体目录；概念库训练会忽略多出来的文件。
