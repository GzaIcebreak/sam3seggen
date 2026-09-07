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
| `pv_holdout_v3.txt`（或你自己的 holdout） | 一行一个 `object_id`。训练时 `--holdout_file` 排除它们。现有清单 69 个，**含全部 20 个 hard 物体** |
| 没有 holdout 文件 | `concept_bank.py` 按 `--holdout_frac 0.10` 随机留出 |

概念库训练命令（SAM3 环境，`transformers>=5`）：

```bash
.venv_holo/bin/python finetune/concept_bank.py \
  --dataset_root /data/pv \
  --out /data/concept_bank_v4 \
  --template name --epochs 3 --neg_mode mixed --negatives 3 --neg_weight 0.5 \
  --lr_e0 5e-4 --holdout_file /data/pv_holdout_v3.txt \
  --azimuths 0,135 --min_count 8
```

产物：`<out>/bank.pt`（推理加载）、`split.json`、`log.jsonl`、`text_cache.pt`。

验收对照（v3，阈值 0.5）：留出集 2D mIoU 0.288 → **0.368**，难负例误检率基本持平。新数据训完应不低于这个数。

---

## 5. 交数据前自检

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

## 6. 从现有仓库扩数据的最短路径

1. 每个新物体：`input.glb` + 按件拆开的 `parts/*.glb` + 按上面口径写好的 `names.json` / `names_meta.json`
2. `render_views.py --azimuths 0,135`（可再加更多方位；目录名跟着改，训练 `--azimuths` 对齐）
3. 跑第 5 节自检，抽查看图
4. 把新物体目录拷进同一个 `--dataset_root`（或再给 `concept_bank.py` 一个根；当前脚本只接受一个 root）
5. holdout 不要混进训练：新物体要么全部训练，要么自己列一份 holdout

不要为概念库去跑 `make_samples_a.py` / `prepare_object`。那是另一条管线，还会触发 `o_voxel`。

---

## 7. 和 SegviGen LoRA 数据的关系

| | 概念库 | SegviGen LoRA（v6/v7） |
|---|---|---|
| 监督 | 2D：名字 → 像素并集 | 3D：颜色图 → 纹理 latent |
| 必需文件 | `names.json` + `views/*/render.png` + `ids.npy` | 上表全部 + latent + `variants/` |
| 环境 | `.venv_holo`（transformers ≥ 5） | SegviGen `.venv`（transformers 4.57） |

两套数据可以共用同一个物体目录；概念库训练会忽略多出来的文件。
