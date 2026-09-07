# SegviGen 路线：云服务器接手方案

> 面向接手训练的人 / agent。目标读者不了解本地上下文，所以从"要搬什么、怎么装、先复现什么、再做什么、什么算通过"依次写。技术背景（v1–v6 的全部实验、模块原理、指标定义）见同目录 `REPORT_overview.md`；本文件只写**下一步**。
>
> 状态截至 2026-09-07。本地机器：Windows 11、RTX 5090D 32 GB。云端假定 Linux + 1 张 ≥ 40 GB 显卡（80 GB 更好：可开大 batch 与更长 rollout）。

## 0. 一页结论

- 现在最好的模型是 **v6**（`ckpt/full_seg_v6.ckpt`，LoRA r=16 合并进 `full_seg_w_2d_map`）。hard 20 物体上 mIoU 0.296 / 语义 mIoU 0.274 / 边界 F1 0.704，全部优于 base（0.278 / 0.265 / 0.666），逐物体 14 胜 5 负；10 个外部资产上整块性最好。
- v6 有效的原因只有一个：**轨迹监督**（训练时让模型在推理采样器自己走出的 \(x_t\) 上做 flow loss + 颜色 CE），修的是 teacher forcing 与 12 步 Euler 推理之间的曝光偏差。v3/v4/v5 试过的"往模型里塞文本"（图例 token、解耦注意力、per-token 名字）全部无效或有害，**不要再做**。
- 语义仍然只走一条路：**SAM3 + 概念库 → 2D 颜色图 → SegviGen 读颜色**。模型内部没有、也不需要读文字；名字只在 2D 图和图例 json 里。
- 云端要做的事按优先级：
  1. **E1 v7**：v6 加码（更多轨迹步、sam3 变体加权），最便宜、最确定。
  2. **E2 部署分布训练**：把训练用的 SAM3 变体从"GT 绑定"改成"legend 绑定"（推理时真实看到的那种图），并让目标是"按名字纠正后的"3D 着色。这是 v6 之后收益最大的一件事。
  3. **E3 数据扩容**：把剩余 681 个 PartVerse 物体准备完（1319 → 2000）再训一轮。
  4. **E4 第二视角条件**：把 SAM3 tracker 传到背面的图作为第二张条件图喂给 v6 型模型（v3 的双视角骨架可复用，但去掉文本部分）。
  5. 推理端（不训练、CPU 可做）：面邻接图上的 graph cut 标签正则化，与 GeoSAM2 路线共用，见 `HANDOVER_cloud_geosam2.md`。

## 1. 要搬到云端的东西

| 内容 | 本地路径 | 大小 | 说明 |
|---|---|---|---|
| 代码 | `SegviGen/`（git，远端 `GzaIcebreak/sam3seggen`，最新提交 e2db4a3 + 未提交的 `finetune/p3sam_run.py`、`run_p3sam.bat`、README/REPORT 修改） | 小 | 先 `git add finetune/p3sam_run.py finetune/run_p3sam.bat && git commit && git push sam3seggen`，云端 clone |
| TRELLIS.2 权重 | `SegviGen/microsoft/TRELLIS.2-4B/` | 15.3 GB | 或云端 `huggingface-cli download microsoft/TRELLIS.2-4B`（`pipeline.json` 与 ckpts）；国内用 `HF_ENDPOINT=https://hf-mirror.com` |
| SegviGen 官方 ckpt | `SegviGen/ckpt/full_seg.ckpt`、`full_seg_w_2d_map.ckpt`、`interactive_seg.ckpt` | 3 × 7.9 GB | 云端 `python download_ckpts.py` 重新下也行 |
| 我们的 ckpt | `SegviGen/ckpt/full_seg_v6.ckpt`（必需）、`full_seg_v3/v4/v5*.ckpt`（可不搬） | 2.6 GB 每个 | v6 是新训练的起点对照 |
| LoRA 权重 | `SegviGen/finetune/runs/pv_v6/`（`lora_last.pt`、`lora_final.pt`、`log.jsonl`、`done.json`） | 0.2 GB | 想从 v6 继续训就要它（`--resume_lora`） |
| DINOv3 | `weights/facebook/dinov3-vitl16-pretrain-lvd1689m/` | 1.1 GB | 或 HF 下载（需 gated 权限），环境变量 `SEGVIGEN_DINOV3` |
| RMBG-2.0 | `weights/briaai/RMBG-2.0/` | 5.0 GB | 只有部署推理的抠图用到，训练不需要 |
| SAM3 | HF cache `models--facebook--sam3` | ~3.4 GB | SAM3 环境下 `transformers` 自动下载（gated，需 HF token） |
| 概念库 | `datasets/concept_bank_v3/bank.pt` + `report.json` | < 10 MB | 必需 |
| 训练数据 | `datasets/pv/` | **132.8 GB** | 见下方"怎么瘦身" |
| 数据清单 | `datasets/pv_holdout_v3.txt`（69 个留出物体，**包含全部 20 个 hard 物体**，训练时用 `--holdout_file` 排除）、`pv_hard.txt`（20 个 hard 评测物体）、`pv_list_a.txt`、`pv_list_b.txt`、`pv_list_c_new.txt` | KB | 必需 |
| 颜色探针 | `SegviGen/finetune/color_probe.pt` | KB | 必需（v4–v6 颜色 CE 用的冻结线性探针；在 git 里） |
| 评测缓存 | `datasets/geosam2/`（5.9 GB）、`datasets/ext_bench/`（6.2 GB） | 可选 | 只是历史结果；重跑评测会重建 |

### 怎么瘦身 `datasets/pv`

每个已准备物体约 115 MB，其中 **`variants/*/cond.pth`（DINOv3 token，8 MB × 14 个变体 ≈ 110 MB）占了 95 %**。它可以在云端从 `map.png` 重算（一张图一次 DINOv3 前向，13.7k 张约 20–30 min）。所以：

```bash
# 本地 → 云端，排除 cond.pth 后大约 23 GB（Windows 侧用 WSL 的 rsync，或 tar 后 scp）
rsync -avP --exclude 'cond.pth' datasets/pv/ user@cloud:/data/pv/
rsync -avP datasets/pv_*.txt datasets/concept_bank_v3 user@cloud:/data/
```

云端重算 `cond.pth`（在 SegviGen 环境下，从 `SegviGen/` 目录运行）：

```python
# finetune/rebuild_cond.py（新建，20 行）
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cond_models, map_to_cond
root = sys.argv[1]
models = load_cond_models()
maps = sorted(glob.glob(os.path.join(root, "*", "variants", "*", "map.png")))
for i, m in enumerate(maps):
    out = os.path.join(os.path.dirname(m), "cond.pth")
    if not os.path.exists(out):
        map_to_cond(models, m, out)
    if i % 500 == 0:
        print(i, len(maps), flush=True)
```

`cond.pth` 是 `train.py` 读取的输入条件（`dataset.py` 里按变体目录加载），必须在训练前全部存在；用 `train.py --check_only` 校验。

### 每个物体目录里各文件的用途（决定哪些不能少）

| 文件 | 谁写的 | 谁读 |
|---|---|---|
| `input.glb`、`input.vxz`、`ids.vxz`、`ids.glb`、`ids_meta.json` | `common.prepare_object` | 评测（`eval_parts` / `eval_fidelity` 在 GT 上打分）、重新生成变体 |
| `shape_slat.pth`、`input_tex_slat.pth`、`common_coords.pth` | 同上 | **训练**（形状条件、纹理输入） |
| `voxel_part.npy` | 同上 | `cell_labels.py`、`pick_hard.py`、评测 |
| `cell_part.npz` | `cell_labels.py` | **训练**（颜色 CE 的 per-cell 类别） |
| `names.json`、`names_meta.json`、`captions.json` | 重标注（Grok 4.6 + 人工审阅 2000 条） | 变体生成、SAM3 提示词、概念库 |
| `views/az*/render.png`、`sam3_masks.npz`、`prompts.json` | `make_samples_a.py` | E2 需要（重新绑定颜色时复用 SAM3 掩码，不必重跑 SAM3） |
| `variants/<kind>_az<k>[_n]/{map.png, meta.json, cond.pth, output_tex_slat.pth, tokens.npz}` | `make_samples_a/b/partial.py` | **训练**。`kind` ∈ clean / corrupt / sam3 / partial；全库共 clean 1956、corrupt 5868、sam3 1975、partial 3884（v6 只用前三种，排除留出后 9465 个：clean 1900 / corrupt 5700 / sam3 1865） |
| `variants/sam3raw_az0/{map.png, map_legend.json, infer_*.glb}` | 评测脚本 | 只有 20 个 hard 物体有，是"部署路径"评测的产物，**没有 meta.json，不是训练变体**（`common.list_variants` 会跳过） |

## 2. 环境（三个 Python 环境 + Blender）

依赖冲突决定了必须分环境：SegviGen 锁 `transformers==4.57.6`，SAM3 要 `transformers>=5`，GeoSAM2 要 `numpy 2.x + opencv 4.14（带 OpenEXR）`。

### 2.1 SegviGen 环境（训练、评测、渲染）

按 `SegviGen/README.md` "部署"一节，即 TRELLIS.2 的 `setup.sh`：

```bash
python -m venv .venv && source .venv/bin/activate
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2 && ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm && cd ..
pip install mathutils transformers==4.57.6 bpy==4.1.0 --extra-index-url https://download.blender.org/pypi/
pip install --upgrade Pillow trimesh wandb open3d pymeshlab igraph scikit-image
sudo apt-get install -y libsm6 libxrender1 libxext6 libgl1     # bpy / open3d 需要
```

本地验证过的版本（Blackwell）：torch 2.11.0+cu128、flash_attn 2.8.3、spconv-cu126 2.3.8、flex-gemm 1.0.0、torch_scatter、torchsparse、o-voxel、cumesh、nvdiffrast、bpy 4.1。云端若是 Hopper/Ampere 用 TRELLIS.2 默认的 cu12x 组合即可。**bpy 与 Python 版本强绑定**（bpy 4.1 → Python 3.11）。

写一个 `finetune/run_ft.sh` 替代 Windows 的 `run_ft.bat`（所有 finetune 脚本都通过它跑）：

```bash
#!/usr/bin/env bash
# usage: finetune/run_ft.sh train.py --dataset_root ...
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm FLEX_GEMM_ALGO=explicit_gemm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEGVIGEN_DINOV3=${SEGVIGEN_DINOV3:-/data/weights/facebook/dinov3-vitl16-pretrain-lvd1689m}
export SEGVIGEN_RMBG=${SEGVIGEN_RMBG:-/data/weights/briaai/RMBG-2.0}
export SEGVIGEN_PY_SAM3=${SEGVIGEN_PY_SAM3:-$ROOT/../.venv_holo/bin/python}
cd "$ROOT"
exec "$ROOT/../.venv/bin/python" "finetune/$@"
```

`train_loop.bat`（崩溃自动续训）的 bash 版：

```bash
#!/usr/bin/env bash
# usage: finetune/train_loop.sh <attempts> <out_dir> <train.py args...>
N=$1; OUT=$2; shift 2
for i in $(seq 1 "$N"); do
  [ -f "$OUT/done.json" ] && { echo "done.json exists"; exit 0; }
  R=""; [ -f "$OUT/lora_last.pt" ] && R="--resume_lora $OUT/lora_last.pt"
  finetune/run_ft.sh train.py --out_dir "$OUT" $R "$@" || true
  [ -f "$OUT/done.json" ] && exit 0
  sleep 15
done
```

`train.py` 完成时写 `done.json`；重启时 `--resume_lora` 会把 LR schedule 接回保存的 step。

### 2.2 SAM3 环境 `.venv_holo`

```bash
python -m venv .venv_holo && .venv_holo/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv_holo/bin/pip install "transformers>=5.10" "numpy<2" pillow scipy huggingface_hub
huggingface-cli login      # facebook/sam3 是 gated 权重
```

用到它的脚本：`sam3_to_2dmap.py`、`finetune/sam3_masks.py`、`sam3_bank.py`、`concept_bank.py`、`bench_sam3.py`、`geosam2_masks.py`、`sam3_track.py`。`make_samples_a.py` 通过 `SEGVIGEN_PY_SAM3` 以子进程调用它。

### 2.3 GeoSAM2 环境 `.venv_geosam2`

只有评测对照需要，见 `HANDOVER_cloud_geosam2.md` §2。

### 2.4 需要改的硬编码路径

大部分脚本用相对路径或环境变量；只有下面几个写死了 `E:\AI_New\ModelGen`，到 Linux 上把常量改成云端路径（建议改成读环境变量 `MODELGEN_ROOT`）：

| 文件 | 常量 |
|---|---|
| `finetune/ext_bench.py` | `DS`、`PY`、`PY_SAM3`、`PY_GEO`、`BAT`、`P3SAM_BAT`（10 个外部资产评测才用） |
| `finetune/ablation_score.py` | `DS`（hard 20 / ext 消融打分） |
| `finetune/make_samples_a.py` | `DEFAULT_PY_SAM3` 的默认值（环境变量 `SEGVIGEN_PY_SAM3` 可覆盖，无需改） |
| `finetune/p3sam_run.py` | `P3SAM_ROOT` 默认值（环境变量可覆盖） |

其余 `.py` 里出现的 `E:\...` 都在 docstring 里。

### 2.5 wandb

`train.py --wandb --wandb_project segvigen-finetune --wandb_name <run>`；离线用 `--wandb_mode offline` 之后 `wandb sync`。run id 默认是 `out_dir` 的目录名并以 `resume="allow"` 初始化，所以同一 `out_dir` 续训自动接回同一条曲线；要另起曲线就换 `--wandb_id`。本地训练走了代理（`SEGVIGEN_PROXY`），云端一般不需要。历史曲线在 `wandb.ai/.../segvigen-finetune/runs/pv_v6` 等。

## 3. 上云后先做的验收（改代码之前）

目的：证明环境、数据、权重搬对了。三步，全部有本地参考数字。

**(1) 数据校验**

```bash
finetune/run_ft.sh train.py --dataset_root /data/pv --check_only --kinds clean corrupt sam3 --holdout_file /data/pv_holdout_v3.txt
```

期望打印 `9465 variants: clean=1900, corrupt=5700, sam3=1865`（与本地 `runs/pv_v6/console.log` 第一行一致），没有 missing 文件。

**(2) 训练 smoke test（20 步）**

```bash
finetune/run_ft.sh train.py --dataset_root /data/pv --out_dir finetune/runs/smoke \
  --kinds clean corrupt sam3 --holdout_file /data/pv_holdout_v3.txt --color_probe finetune/color_probe.pt \
  --color_weight 0.3 --color_tau 0.03 --color_t_min 0.8 --p_traj 0.5 --traj_steps 3 \
  --batch_size 4 --grad_accum 4 --max_steps 20 --check_every 0
```

期望：不 OOM（本地 batch 4 是 12.5 GB）、`log.jsonl` 每 `--log_every` 步有 `loss`、`ema`、`per_kind`、`color_acc`、`color_acc_traj`。v6 的完整参数在 `runs/pv_v6/args.json`（`lr 1e-4、warmup 100、p_uncond 0.1、grad_clip 1.0、save_every 250、check_every 250、check_limit 100、check_ts 0.5,0.95,1.0`）。

**(3) 评测复现：v6 在 hard 20 上的 mIoU**

```bash
# 对 pv_hard.txt 的 20 个物体（都在留出集里），用它们的 sam3_az0 变体作条件跑推理并打分
mkdir -p /data/eval_v6
while read id; do
  finetune/run_ft.sh eval_fidelity.py --object /data/pv/$id --variant sam3_az0 \
      --ckpt ckpt/full_seg_v6.ckpt --run_inference --output_glb /data/eval_v6/$id.glb --report /data/eval_v6/fid_$id.json
  finetune/run_ft.sh eval_parts.py --object /data/pv/$id --segvigen /data/eval_v6/$id.glb --variant sam3_az0 \
      --report /data/eval_v6/parts_$id.json
done < /data/pv_hard.txt
python - <<'EOF'
import json,glob,numpy as np
rs=[json.load(open(p)) for p in glob.glob('/data/eval_v6/parts_*.json')]
for k in ['miou','sem_miou','boundary_f1','small_part_recall','unlabelled_share']:
    print(k, round(float(np.mean([r[k] for r in rs])),3))
EOF
```

本地参考（v6）：mIoU 0.296、语义 mIoU 0.274、边界 F1 0.704、小件召回 0.037、留白 0.129。采样不固定种子，允许 ±0.01。base（`full_seg_w_2d_map.ckpt`）应为 0.278 / 0.265 / 0.666。**如果 v6 没有跑到 ≥ 0.285，先别训练，查环境**（常见：`ATTN_BACKEND` 没设导致 fallback、DINOv3 分辨率不同、`cond.pth` 重算时 `map.png` 读成 RGBA）。

## 4. 实验计划

按顺序做；每个实验给出改动、命令、要看的曲线、通过标准和止损点。所有训练都从 `full_seg_w_2d_map.ckpt` 上挂新 LoRA（v6 也是这样训的），不要在 v6 LoRA 上再叠 LoRA。

### E1 v7：v6 加码（1 天，先跑）

**改动**：三处，都是参数或 10 行代码。

1. `--p_traj 0.5 → 0.8`，`--traj_steps 3 → 6`。rollout 每步都是完整前向（no-grad），6 步 ≈ 训练时间 ×1.6。sched 前 7 个点 t = 1, .971, .937, .900, .857, .809, .756，覆盖到颜色锁定区（轨迹探针显示 t≈0.9–0.75 是 base 与 v6 拉开差距的区间）。
2. **sam3 变体加权**：`train.py` 新增 `--kind_weight sam3=2.0`。实现：`flow_loss` 目前 `loss = per_token.mean()`（第 224 行），改成按样本 kind 取权重、对 `coords_len_list` 切片加权平均；`color_loss` 同样传权重。理由：sam3 变体只占 20 %，但它是唯一与部署分布接近的那类；v6 的留出集轨迹探针 clean 0.85、sam3 只有 0.62，差距全在 sam3 这一类。
3. `--max_steps 2500`（v6 1500 步在 1200 步后 `color_acc_traj` 仍在爬）。80 GB 卡可以 `--batch_size 8 --grad_accum 2`，等效 batch 不变。

```bash
finetune/train_loop.sh 3 finetune/runs/pv_v7 --dataset_root /data/pv --kinds clean corrupt sam3 \
  --holdout_file /data/pv_holdout_v3.txt --color_probe finetune/color_probe.pt \
  --color_weight 0.3 --color_tau 0.03 --color_t_min 0.8 --p_traj 0.8 --traj_steps 6 --kind_weight sam3=2.0 \
  --batch_size 4 --grad_accum 4 --max_steps 2500 --save_every 250 --check_every 250 --check_limit 100 \
  --check_ts 0.5,0.95,1.0 --wandb --wandb_name pv_v7
finetune/run_ft.sh merge_lora.py --lora finetune/runs/pv_v7/lora_final.pt --out ckpt/full_seg_v7.ckpt
```

（`train.py` 里 wandb 的 run id 默认取 `out_dir` 的目录名并 `resume="allow"`，所以续训自动接回同一条曲线，不必手动传 `--wandb_id`。`merge_lora.py` 也接受 `lora_last.pt` / `lora_step*.pt`。）

**看什么**：wandb 里 `train/color_acc_traj`（v6 终点 0.77）、`train/color_acc_sam3`、`holdout/<kind>` 留出集 MSE（v6 终点 clean 0.276 / corrupt 0.229 / sam3 0.263，`check_ts 0.5,0.95,1.0`；明显高于这些说明 LoRA 把重建学坏了）。

**通过**：hard 20 mIoU ≥ 0.31 且边界 F1 ≥ 0.70，轨迹探针 sam3 列 ≥ 0.70（v6 0.62）。**止损**：1000 步时 `color_acc_traj` 不高于 v6 同步数，停。v6 的 `log.jsonl`（10 步窗口均值）：step 250 / 500 / 750 / 1000 / 1250 / 1500 → `color_acc_traj` 0.57 / 0.69 / 0.70 / 0.73 / 0.71 / 0.77，`ema` 0.91 / 0.60 / 0.53 / 0.65 / 0.65 / 0.63；日志里还有 `color_acc_clean` / `color_acc_corrupt` / `color_acc_sam3` 三个分项，`color_acc_sam3` 是要盯的那个。

### E2 部署分布训练：legend 绑定的 SAM3 变体（2–3 天，收益最大）

**问题**：现在 `sam3` 变体的颜色是 **GT 绑定**——`make_samples_a.py` 把 SAM3 掩码覆盖 ≥ `--cover` 的 GT 部件染成同色，没被任何提示词覆盖的部件在 2D 和 3D 里都涂灰。推理时（`sam3_to_2dmap.py`）却是 **legend 绑定**：每个提示词一个颜色，掩码按面积从小到大盖上去，多余像素归"未分配"。两种图看起来相似，但错误模式不同：部署图里 SAM3 把"手臂"也涂进"身体"的颜色时，训练集里从没出现过"该部件该颜色但被涂成了别的颜色"的样本。hard 20 上"SegviGen raw（部署路径）"的语义 mIoU 0.219 明显低于用 GT 绑定条件的 0.274，差的就是这一块。

**方案**：新增 `make_samples_legend.py`（从 `make_samples_a.py` 复制改），生成 `kind = sam3legend` 变体（不要叫 `sam3raw`：20 个 hard 物体下已有同名的评测目录）：

- **条件图**：直接调用 `sam3_to_2dmap.colorize` 的逻辑（同一套面积排序、前景裁剪、`pick_separated_colors` 调色板、未分配像素规则），输入是 `views/az*/sam3_masks.npz` 里已有的掩码（不用重跑 SAM3），输出 `map.png` + `legend.json`（名字 → 颜色）。
- **3D 目标**：对每个 GT 部件 p，取它的名字 `names.json[p]`；若该名字在 legend 里，整个部件着 legend 颜色；若不在（SAM3 没检出或阈值下没掩码），着灰。这就是"如果模型读懂了 2D 图里每种颜色代表什么部件，正确的 3D 着色应该是什么"。**目标不再复制 2D 图的错误**，而是名字级的正确答案——模型被迫学"颜色 → 部件"的对应，而不是"颜色 → 对应像素的几何"。
- 每个物体 2 个视角（az0、az135）各 1 个变体，1319 物体 ≈ 2.6k 个。`cell_labels.py` 的 `cell_part.npz` 不用变（按部件索引）；`meta.json` 需带 `groups`/`grey_parts`/`colors`，格式与 `sam3` 变体一致，`cells.variant_cell_targets` 就能直接吃。

**训练**：v7 配置 + `--kinds clean corrupt sam3 sam3legend --kind_weight sam3=1.5,sam3legend=2.0`（`dataset.py` 按 `meta.json` 的 `kind` 字段过滤，新 kind 只要 meta 写对就不用改 `dataset.py`；`--kind_weight` 是 E1 新加的参数）。

**评测**：hard 20 除了 `sam3_az0` 条件，还要用**部署路径**跑一遍（先 `sam3_to_2dmap.py` 出 map + legend，再 `inference_full.py`，再 `eval_parts.py --legend legend.json`，命令见 §6 第 2 步）——这才是真正的指标。v6 部署路径参考：mIoU 0.304、语义 0.219、命名 0.354。

**通过**：部署路径语义 mIoU ≥ 0.26（+0.04）、命名正确率 ≥ 0.45。**风险**：模型可能学会"忽略 2D 图、按几何先验着色"（因为目标与条件不完全一致）。检查手段：训练时加 `--check_shuffle`（留出集上把条件图与物体错配再算 MSE）——如果错配后 MSE 不涨，说明条件被忽略，需要降低 sam3legend 权重或提高 `--p_uncond`。

### E3 数据扩容：剩余 681 个物体 + 更多 SAM3 变体（GPU 后台 1–2 天，与 E1 并行）

`datasets/pv` 下 2000 个物体只有 1319 个准备完（`shape_slat.pth` 存在）。剩余 681 个各只有 `input.glb` + 名字。

```bash
# 未准备物体清单
python -c "
import os;r='/data/pv';print('\n'.join(d for d in sorted(os.listdir(r)) if len(d)>=30 and not os.path.exists(f'{r}/{d}/shape_slat.pth')))" > /data/pv_list_rest.txt
# path B（clean + 3 corrupt，两视角）与 path A（sam3，两视角）交替，o_voxel 崩了自动重试
finetune/run_ft.sh run_batch.py --dataset_root /data/pv --jobs b a \
  --objects_b /data/pv_list_rest.txt --objects_a /data/pv_list_rest.txt --chunk 8 \
  --args_b "--azimuths 0,135 --n_corrupt 3" --args_a "--azimuths 0,135 --threshold 0.4 --concept_bank /data/concept_bank_v3/bank.pt"
finetune/run_ft.sh cell_labels.py --dataset_root /data/pv          # 新物体的 cell_part.npz
```

本地速度约 1.5–2 min / 物体（含 SAM3 子进程），681 个 ≈ 20 h。`o_voxel` 的 access violation 在 Linux 上少见但仍要用 `run_batch.py` 包一层。**注意**：`pv_holdout_v3.txt` 与 `pv_hard.txt` 里的物体不能进训练；两个清单都从已准备的 1319 个里选的，新物体不会冲突，但如果后面重选 holdout 要排除 hard。

之后再跑一次 E1/E2 配置（`pv_v8`），看 2000 物体是否比 1319 提升。**预期不大**（v1→v2 数据翻倍无变化），但 E2 类"纠错"样本更多样会有帮助。

### E4 第二视角条件（3–4 天，E1/E2 之后）

10 个外部资产里，背面看不见的部件是 SegviGen 主要的漏件来源（前视图 SAM3 图只能着色可见部分，背面靠模型猜）。v3 曾接过第二视角（`LegendEncoder` 里 `[DINO; DINO₂; obj; legend×G]`），但那版被文本 token 拖累。

**方案**：v6 的 cond 只是 DINOv3 token；改成 `[DINO(view1); DINO(view2)]` 拼接（`build_cond` 里已有 `p_drop_view1` 的钩子，`dataset.py` 的双视角配对 `--pair` 逻辑可复用），不加任何文本。第二视角的训练图来自：

- 训练：同一物体 `az135` 的变体图（GT 绑定或 E2 的 legend 绑定），颜色与 `az0` 一致（同一 `groups`/`colors`），这在 `make_samples_*` 里天然满足（同一物体两视角共用调色板——**要确认** `make_samples_legend.py` 里两视角调色板一致，legend 按名字对齐）。
- 推理：`finetune/sam3_track.py` 用 SAM3 tracker 把正面掩码传播到背面（hard 20 上 tracker p2 的多视角一致性最好），再 `sam3_to_2dmap.colorize` 上色，颜色与正面 legend 对齐。

**评测**：外部 10 资产 `ext_bench.py` 的留白/整块性，hard 20 的小件召回（v6 0.037，目标 ≥ 0.06）。**止损**：训练 dropout 第二视角 0.3 时，若单视角推理指标掉到 v6 以下，说明模型过度依赖第二视角，回退。

### 不要再做的（已被 v3–v5 否定，详见 `REPORT_overview.md` §8.7 / §10）

- 往 DiT cond 里加文本 token（图例、物体名、per-token 名字）：shuffle 对照无差别或崩溃。
- 解耦图例交叉注意力：零差别。
- 单步 teacher-forced 颜色 CE 单独用（v4）：训练 CE 降 5 倍，端到端不变；它只有搭配轨迹监督才有用。
- 微调 DINOv3：它在 `image_feature_extractor.py` 里 `eval()` + `no_grad`，且 TRELLIS.2 的 cond 投影是按冻结特征训的；解冻的代价与收益都没有证据，先不碰。
- 用 corrupt 变体做更重的合成损坏：v2 加了也没用。

## 5. 训练监控与止损

- `runs/<dir>/log.jsonl` 每 `--log_every`（10）步一行：`loss`、`ema`、`lr`、`vram_gib`、`per_kind`（clean/corrupt/sam3 各自 MSE）、`color_acc`、`color_acc_traj`、`color_acc_clean/corrupt/sam3`。wandb 上对应 `train/loss`、`train/loss_<kind>`、`train/color_acc*`。`plot_run.py` / `report_html.py` 能出离线曲线。
- 留出集检查（`--check_every`）写 `check_step<N>.json`，wandb 上是 `holdout/<kind>`（teacher-forced MSE，在 `--check_ts` 的 t 上）与 `holdout/color_acc_<kind>`；加 `--check_shuffle` 后多一组 `holdout_shuffled/*`。v6 参考：step 250 → 1500，MSE clean 0.304 → 0.276、corrupt 0.254 → 0.229、sam3 0.266 → 0.263；color_acc clean 0.79 → 0.86、sam3 0.78 → 0.82。
- wandb 面板要盯的三条：`train/color_acc_traj`（最重要，与推理同分布）、`train/color_acc_sam3`、`holdout/sam3`（比 v6 同步数高 > 10 % 说明过拟合或 LR 太大）。
- LR：`--lr 1e-4`（LoRA r=16、alpha 32、目标 self+cross 注意力的 to_qkv/to_q/to_kv/to_out），warmup 100 步，cosine 到 0，`grad_clip 1.0`。batch 等效 16。v6 全程无 NaN。
- **止损规则**：任何一轮，1/3 步数时 `color_acc_traj` 不高于 v6 同步数，或 `holdout/<kind>` 比 v6 同步数高 > 15 %，停下改配置，不要跑完。

## 6. 完整评测流程（每个新 ckpt 都跑）

1. hard 20 × `sam3_az0` 条件（§3 第 3 步的循环，20 × 45 s ≈ 15 min）。
2. hard 20 × 部署路径（E2 起必做）：

```bash
mkdir -p /data/eval_raw
while read id; do
  # names.json 是按部件索引的名字列表，同名部件会重复，SAM3 提示词要去重
  P=$(python -c "import json;print(' '.join(sorted(set(json.load(open('/data/pv/$id/names.json'))))))")
  .venv_holo/bin/python sam3_to_2dmap.py --image /data/pv/$id/views/az0/render.png --prompts $P \
      --concept_bank /data/concept_bank_v3/bank.pt --threshold 0.4 --allow_missing \
      --out /data/eval_raw/$id.png --legend /data/eval_raw/$id.legend.json
  finetune/run_ft.sh ../inference_full.py --ckpt_path ckpt/full_seg_v7.ckpt --glb /data/pv/$id/input.glb \
      --input_vxz /data/pv/$id/input.vxz --img /data/eval_raw/$id.png --two_d_map --export_glb /data/eval_raw/$id.glb
  finetune/run_ft.sh eval_parts.py --object /data/pv/$id --segvigen /data/eval_raw/$id.glb \
      --legend /data/eval_raw/$id.legend.json --report /data/eval_raw/parts_$id.json
done < /data/pv_hard.txt
```

（阈值 0.4 与 `ext_bench.py` 一致，`sam3_to_2dmap.py` 默认是 0.3；`inference_full.py` 在仓库根目录，`run_ft.sh` 拼的是 `finetune/` 前缀，所以用 `../`。）

3. 轨迹探针（看曝光偏差是否进一步缩小）：

```bash
finetune/run_ft.sh trajectory_probe.py --dataset_root /data/pv --holdout_file /data/pv_holdout_v3.txt --limit 40 \
  --resume_lora finetune/runs/pv_v7/lora_final.pt --out /data/traj_v7.json
```

参考（20 留出物体，自由轨迹终点调色板准确率，clean / sam3）：base 0.733 / 0.520，v4 0.744 / 0.603，v6 0.850 / 0.621。E1/E2 的目标是把 sam3 这一列推到 ≥ 0.70。

4. 10 个外部资产（定性，`ext_bench.py` 需要三个环境 + Blender，路径常量要改；可选）。

## 7. 已知的坑

- `o_voxel` 体素化偶发崩溃：用 `run_batch.py`，已实现分块重试。
- `bpy` 渲染进程退出时可能 access violation，输出文件已写好；判断成功看文件不看退出码（`ext_bench.py` 已这样做）。
- Windows 下 `--check_ts 0.5,0.95,1.0` 要加引号，Linux 不用。
- 采样无固定种子：同一 ckpt 两次评测 mIoU 差 ±0.01，比较时看 20 个物体的均值和胜负数，不看单个。
- SAM3 的 `transformers` 版本：5.10.1 验证过；`Sam3TrackerVideoModel` 在多提示帧时需要先对每个提示帧跑一次前向填 `maskmem_features`（`sam3_track.py` 已处理）。
- 显存：v6 batch 4 = 12.5 GB，rollout 步数不增加显存（no-grad），只增时间。80 GB 卡 batch 8 应该在 25 GB 内。
- `merge_lora.py` 输出的 ckpt 是 2.6 GB（只含 texture flow 模型），`inference_full.py --ckpt_path` 直接可用。
