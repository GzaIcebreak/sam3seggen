# GeoSAM2 路线：云服务器接手方案

> 面向接手的人 / agent。GeoSAM2 路线到目前为止**没有训练过任何东西**，全部是零训练的消融与工程接入；本文件写清楚现状、怎么在云端复现基线，然后给三条可选的下一步（A 零训练工程化、B 逆向训练 GeoSAM2、C 换 SAM3 骨干），每条带工作量、验收标准和止损点。背景与全部数字见 `REPORT_overview.md` §9 与 `REPORT_geosam2_ablation.md`。SegviGen 路线的接手文件是 `HANDOVER_cloud_segvigen.md`，两条路线共用 SAM3 + 概念库这一上游。
>
> 状态截至 2026-09-07。

## 0. 一页结论

GeoSAM2（CVPR 2026，VAST）= SAM2 视频预测器（Hiera 骨干 + LoRA）+ 法线/位置图的残差几何融合，在 3D 部件数据上训练，把一个视角的 2D 掩码传播到 12 个转台视角，再深度测试反投影到 mesh、按面投票、连通域后处理。**没有公开训练代码**，只有推理与权重（`ckpt/geosam2.pt`，0.6 GB）。

我们在它上面做了零训练消融（同一批 SAM3 + 概念库掩码，同一套提升与后处理，只换 2D 掩码来源），hard 20 物体（有 GT）结论：

| 结论 | 证据 |
|---|---|
| 传播器不是瓶颈 | GeoSAM2 传播 p2 mIoU 0.254，SAM3 tracker 传播 p2（无几何、无 LoRA）0.276；SAM3 12 视角直接提升 0.230 |
| 几何融合没带来可测收益 | 上一行；GeoSAM2 自己的 p1/p2/p4 = 0.246 / 0.254 / 0.249 |
| 瓶颈在上游 2D 掩码质量与锚视角选择 | 单视角提示 0.209 → 2 视角 0.254–0.276；再多视角不涨（p4 ≤ p2），说明第 3、4 个视角的 SAM3 掩码引入的错误 ≥ 它补的信息 |
| GeoSAM2 系列的强项是小件召回，弱项是整块性 | 小件召回 0.070–0.080（SegviGen v6 0.037），但外部 10 资产的碎片数 2.5–2.7 / 件（v6 1.94），边界 F1 0.55–0.62（v6 0.704） |
| 与 SegviGen v6 是互补而非替代 | 逐物体 SAM3 tracker p2 vs v6：9 胜 11 负 |

所以：**如果目标是"更好的 GeoSAM2"，最便宜的路是修上游与后处理（方案 A），而不是训练**。训练（方案 B/C）只有在你明确要一个"语义部件专用"的传播器、且愿意先投入 2–3 天造数据时才值。

## 1. 要搬到云端的东西

| 内容 | 本地路径 | 大小 | 说明 |
|---|---|---|---|
| GeoSAM2 fork | `GeoSAM2/`（git，远端 `mine = GzaIcebreak/GeoSAM2-SegviGen`，上游 `VAST-AI-Research/GeoSAM2`，最新提交 4ba9670） | 小 | 含两个补丁：`inference.py` 的变换顺序修复、`utils/mode_ext.py` 的 OpenMP 扩展可选化。**云端 clone 我们的 fork**，不要 clone 上游 |
| GeoSAM2 权重 | `GeoSAM2/ckpt/geosam2.pt` | 0.6 GB | 或按上游 README 从 HF 下载 |
| 驱动脚本 | `SegviGen/finetune/geosam2_render.py`、`geosam2_masks.py`、`geosam2_run.py`、`geosam2_dual.py`、`geosam2_to_glb.py`、`geosam2_ablate.py`、`sam3_track.py`、`ablation_score.py`、`eval_parts.py` | 在 SegviGen git 里 | 全部在 SegviGen 仓库的 `finetune/`，随 SegviGen 一起 clone |
| 数据 | `datasets/pv/`（只需 20 个 hard 物体的 `input.glb`、`ids.*`、`voxel_part.npy`、`names.json`、`views/az0/render.png`）+ `datasets/pv_hard.txt` | ~0.3 GB | GeoSAM2 路线不需要 latent；方案 B 训练需要全部 1319 个物体的 `input.glb` + `parts/` + `names.json`（~10 GB） |
| 概念库 | `datasets/concept_bank_v3/bank.pt` | < 10 MB | 必需 |
| 渲染 / 结果缓存 | `datasets/geosam2/renders`（12 视角 × 20 物体的 rgb/normal/pos/depth EXR）、`results/`、`eval/` | 5.9 GB | 可选；搬了能省 1 h 渲染，也是校验用的参考结果 |
| 外部资产 | `datasets/ext_parts/*.glb`（10 个）、`datasets/ext_bench/`（结果） | 0.6 GB + 6.2 GB | 定性对照用，可选 |
| SAM3 | HF `facebook/sam3`（gated） | 3.4 GB | `.venv_holo` 里 transformers 自动下 |

## 2. 环境

### 2.1 `.venv_geosam2`（GeoSAM2 推理）

上游 README 的依赖 + 我们踩过的坑：

```bash
python -m venv .venv_geosam2 && source .venv_geosam2/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd GeoSAM2 && pip install -e . && cd ..          # sam2 包（fork 内）
pip install hydra-core omegaconf trimesh==5.1.* "numpy>=2" scipy scikit-learn tqdm pillow \
            "opencv-python==4.14.*"               # 4.14 带 OpenEXR；有些版本读 .exr 返回 None
export OPENCV_IO_ENABLE_OPENEXR=1                 # 驱动脚本已在子进程 env 里设了，直接跑 inference.py 时要自己设
```

`utils/mode_ext.py` 会 JIT 编译一个 C++ 众数扩展，Linux 有 gcc 就能编；编不过自动退回 Python 实现（我们的补丁），只是慢。GPU 显存：单物体 12 视角 1024² 约 8–10 GB。

### 2.2 渲染（bpy，在 SegviGen 的 `.venv` 里）

`geosam2_render.py` 用 GeoSAM2 自带的 Blender 脚本（`geosam2_render_upstream`）出 12 视角 rgb / normal / position / depth；需要 `bpy==4.1`，无显示器时 EEVEE 可能不可用，脚本 `--engine auto` 会退到 CYCLES（慢 3–4 倍，20 物体约 40 min）。环境变量 `GEOSAM2_ROOT` 指向 fork 目录（默认是 `SegviGen/../GeoSAM2`，目录结构不变就不用设）。

### 2.3 `.venv_holo`（SAM3 掩码与 SAM3 tracker）

同 `HANDOVER_cloud_segvigen.md` §2.2。`sam3_track.py` 需要 `transformers>=5.10`（`Sam3TrackerVideoModel`）。

### 2.4 硬编码路径

`finetune/ablation_score.py` 顶部的 `DS = r"E:\AI_New\ModelGen\datasets"` 要改；`ext_bench.py` 的 `DS/PY/PY_SAM3/PY_GEO/BAT/P3SAM_BAT` 同理（只有外部 10 资产对照用）。其余 GeoSAM2 脚本走 `--renders/--out/--dataset_root` 参数与 `GEOSAM2_ROOT` 环境变量。

## 3. 上云后先复现基线（半天）

按顺序跑，每步有本地参考数字。约定 `R=/data/geosam2/renders`，`O=/data/geosam2/results`。

```bash
# 1) 12 视角渲染（SegviGen venv，bpy）
finetune/run_ft.sh geosam2_render.py --dataset_root /data/pv --objects_file /data/pv_hard.txt --out $R

# 2) SAM3 + 概念库在 12 个视角上出标签图（SAM3 venv）；--view match 的锚视角会自动对齐 views/az0/render.png
.venv_holo/bin/python finetune/geosam2_masks.py --renders $R --dataset_root /data/pv --objects_file /data/pv_hard.txt \
    --threshold 0.4 --concept_bank /data/concept_bank_v3/bank.pt

# 3) 消融：GeoSAM2 传播 p1/p2/p4、SAM3 直接提升 p1/p2/p4/p12（GeoSAM2 venv）
.venv_geosam2/bin/python finetune/geosam2_ablate.py --renders $R --out $O --objects_file /data/pv_hard.txt --view match \
    --ref_render_pattern "/data/pv/{obj}/views/az0/render.png"

# 4) SAM3 tracker 传播 p1/p2/p4（SAM3 venv），再用同一套提升（GeoSAM2 venv）
.venv_holo/bin/python finetune/sam3_track.py --renders $R --out $O --objects_file /data/pv_hard.txt --view match \
    --ref_render_pattern "/data/pv/{obj}/views/az0/render.png" --n_prompts 1 2 4
.venv_geosam2/bin/python finetune/geosam2_ablate.py --renders $R --out $O --objects_file /data/pv_hard.txt --view match \
    --ref_render_pattern "/data/pv/{obj}/views/az0/render.png" \
    --modes lift:sam3track_p1_match lift:sam3track_p2_match lift:sam3track_p4_match

# 5) 打分（SegviGen venv；先改 ablation_score.py 的 DS）
finetune/run_ft.sh ablation_score.py hard && finetune/run_ft.sh ablation_score.py tables
```

目录约定：`sam3_track.py` 写 `<out>/<obj>/sam3track_p<n>_<tag>/masks/view_XXXX.npy`；`geosam2_ablate.py` 的 `lift:<name>` 读同一 `<out>/<obj>/<name>/masks`，结果写到 `<out>/<obj>/abl_lift_<name>_<tag>/`（`labels.json`、`faces.npy`、`mesh.glb`）；`ablation_score.py` 的 `ROWS` 按这些 stem 找结果。

**参考数字**（hard 20，`eval_parts`，mIoU / 语义 mIoU / 边界 F1 / 小件召回）：

| 行 | mIoU | 语义 | 边界 F1 | 小件 |
|---|---|---|---|---|
| sam3_p1 | 0.209 | 0.164 | 0.392 | 0.024 |
| sam3_p12 | 0.230 | 0.209 | 0.560 | 0.041 |
| geo_p1 / p2 / p4 | 0.246 / 0.254 / 0.249 | 0.182 / 0.216 / 0.215 | 0.549 / 0.549 / 0.508 | 0.050 / 0.037 / 0.037 |
| sam3track p1 / **p2** / p4 | 0.248 / **0.276** / 0.259 | 0.178 / 0.230 / 0.217 | 0.523 / 0.555 / 0.557 | 0.046 / 0.074 / 0.046 |

后处理是确定性的，2D 掩码也是确定性的，所以这些数应能复现到 ±0.005（GeoSAM2 传播里有随机采点，±0.01）。**geo_p2 若明显低于 0.25，先检查 `inference.py` 是否是 fork 版**（上游的 translate→scale 顺序会让非居中物体的深度测试失败，留白飙到 60 % 以上，这是我们修过的 bug）。

## 4. 下一步方案

### 方案 A：零训练工程化（推荐先做；3–5 天；CPU 为主）

目标：把"SAM3 + 概念库 → SAM3 tracker p2 传播 → GeoSAM2 提升"这条链做成一个可部署的管线，并把它的两个弱项（碎片、边界）用后处理补上，同时保住小件召回的优势。它与 SegviGen 路线不冲突：工程上可以做双管线，按物体类型（硬表面 / 软表面）或按 SegviGen 的置信度选用。

A1. **面邻接图上的标签正则化**（最确定的收益）。现在的后处理是 GeoSAM2 的 `complete_labels`：丢掉 < 2 % 最大连通域的小块 → 邻接填充 → 最近面填充。它不看几何边界，所以碎片多、边界锯齿。改成在面邻接图上做 α-expansion / graph cut：

- 数据项：每个面的 12 视角投票直方图（`geosam2_ablate.lift` 里 `mode_except_negative_one` 之前的原始票数），归一化后取 \(-\log p\)。
- 平滑项：相邻面二面角 θ 的函数，θ 小（平滑连接）罚大、θ 大（折边）罚小，例如 \(\lambda \cdot \exp(-(\theta/\theta_0)^2)\)，θ₀ ≈ 30°；再乘边长。
- 求解：`pygco` / `gco-wrapper`，或用 `igraph` 做 multi-label 的迭代 α-expansion；20k–200k 面在 CPU 上秒级。
- 这一步 SegviGen 路线也想用（GLB 输出的面标签同样可以正则化），做成独立模块 `finetune/label_regularize.py`：输入 mesh + 每面票数（或 one-hot 标签），输出面标签。
- **验收**：hard 20 上 sam3track p2 边界 F1 0.555 → ≥ 0.62，欠分割段不增，mIoU 不降；外部 10 资产碎片数 2.5 → ≤ 1.8 / 件。

A2. **锚视角与提示视角选择**。现在 `--view match` 用与 SegviGen 相同的正面，第二个提示视角是对面（+6）。数据说 p2 > p4，即多加视角会带进坏掩码。改成：对 12 个视角各跑一次 SAM3（已经有，`geosam2_masks.py` 输出全部 12 张），按"检出的提示词数 × 平均分数 × 前景占比"打分，取分最高的 2 个且夹角 ≥ 90° 作为提示视角；其余视角只作传播目标。**验收**：hard 20 mIoU ≥ 0.29（+0.015）。

A3. **传播回投一致性过滤**。SAM3 tracker 传播到的视角与该视角自己的 SAM3 掩码不一致时，现在全信传播。改成两者都算票（传播票权 1，SAM3 本视角票权 0.5，只在提示词一致时加），把明显的传播漂移（tracker 把"手臂"漂到"身体"上）压下去。这是 sam3_p12（0.230）与 sam3track_p2（0.276）的折中，预期 +0.01–0.02。

A4. **与 SegviGen v6 的名字级融合**（双管线的最终形态）。两条路线都输出"面 → 名字"。规则：以 SegviGen v6 为主（整块、边界好）；对 v6 输出里没有出现、但 GeoSAM2 链输出里连通域面积 ≥ 阈值、且 SAM3 分数 ≥ 0.6 的名字，把那块面从 v6 的对应大块里切出来。这直接补 v6 的小件召回（0.037 → 目标 ≥ 0.06）而不碎化大块。验收在 hard 20 与外部 10 资产的 `compare_*.png` 上一起看。

A 方案的产出：`finetune/geosam2_pipeline.py`（渲染 → SAM3 → tracker 传播 → 提升 → 正则化 → GLB），一条命令跑完一个 GLB；`label_regularize.py` 独立可用。

### 方案 B：逆向训练 GeoSAM2 为"语义部件传播器"（2–3 周；1 张 80 GB 卡）

只有在方案 A 的 A1–A3 做完、且传播错误（而非上游 SAM3 错误）仍是主要失败模式时才做。判断方法：hard 20 上把 SAM3 掩码换成 **GT 渲染的部件掩码**作提示（`render_part_ids` 已能出 GT id 图），如果 geo_p2 的 mIoU 从 0.25 跳到 > 0.6，说明传播器够好、上游才是瓶颈，**不要训练**；如果仍 < 0.45，传播器本身在语义部件上有系统误差，才值得训。

论文给出的训练配置（arXiv，§4）：约 4.7k 物体，12 视角 1024²，输入 rgb + 法线 + 位置图；掩码提示 → 视频式传播；损失是 SAM2 原版 focal + dice + IoU 头；8 × A800、batch 8、lr 5e-5、50 epoch。结构以 fork 里的 `sam2/configs/geosam2.yaml` 为准：`image_encoder` 与 `pos_map_encoder` 都是 `SAMLoraImgEncoder`（Hiera-B+，`embed_dim 112`，LoRA `rank 4`），几何编码器的特征残差加到 RGB 特征后进 memory attention / mask decoder。

**数据（最花时间的部分，2–3 天）**：

1. 物体：`datasets/pv` 的 1319 个已准备物体（有 `parts/` 与人工审阅过的 `names.json`），排除 `pv_holdout_v3.txt` 的 55 个。可再加 PartVerse-XL / PartNeXt 扩到 5k（GeoSAM2 的数据来自 PartVerse 系列 + 私有数据，重叠不可避免）。
2. 每物体跑 `geosam2_render.py` 出 12 视角 rgb / normal / pos / depth（1319 × 12，CYCLES 约 12 h，EEVEE 3 h）。
3. **新写 `geosam2_gt_masks.py`**：用 GeoSAM2 相同的 12 个相机（`geosam2_render_upstream` 里的相机参数）渲染每个部件的 id 图，写成每视角每部件的二值掩码。`common.render_part_ids` 已有部件 id 渲染（nvdiffrast），只需换相机。
4. 训练样本 = (提示视角 k 的某部件掩码, 其余 11 视角的该部件 GT 掩码)。每物体每部件 12 个锚 → 样本量约 1319 × 8 部件 × 12 ≈ 12 万条"视频"。

**训练代码**：GeoSAM2 fork 没有 `training/`。用 Meta 上游 `facebookresearch/sam2` 的 `training/` 目录（`train.py` + `sam2/configs/sam2.1_training/*.yaml`，MOSE 微调配置），改三处：

- 数据集类：把 12 视角当 12 帧视频，`img` = rgb，额外通道 `normal`、`pos` 走 GeoSAM2 的几何编码器（`sam2_video_predictor_geosam2.py` 里的 fusion 路径）；掩码提示而非点提示（`training/dataset` 里有 mask-prompt 采样开关）。
- 模型：加载 `ckpt/geosam2.pt`（含两路 LoRA 与融合权重），只训两路 LoRA（r=4）+ 几何融合 + mask decoder，冻结 Hiera-B+ 主干；lr 5e-5、batch 4 × grad_accum 2、bf16、5–10 epoch（从他们的权重出发，不是从头）。
- 损失：SAM2 原版（focal 20 : dice 1 : IoU 1），不加别的。

**评测**：`geosam2_ablate.py --modes geo_p1 geo_p2` 换权重路径重跑，`ablation_score.py hard`。**验收**：geo_p2 mIoU 0.254 → ≥ 0.29（超过 sam3track p2 的 0.276 至少 0.015），小件召回不降。**止损**：2 个 epoch 后 hard 20 的 geo_p2 没超过 0.265，停；大概率是上游掩码的问题，回到方案 A。

**风险**：训练目标（GT 掩码传播）与部署输入（SAM3 掩码，有噪声）不匹配。缓解：提示掩码做随机腐蚀/膨胀/漏块增强，或直接用训练集上 SAM3 的掩码作提示、GT 作目标（同 SegviGen 路线 E2 的思路）。

### 方案 C："GeoSAM3"——把 SAM2 骨干换成 SAM3（在 B 成立之后；1–2 周）

`sam3_track.py` 已证明 SAM3 tracker（Perception Encoder 骨干 + SAM2 式 memory attention）零训练就等于或好于 GeoSAM2 传播。所以"换骨干"本身在推理上已经完成；方案 C 说的是**给 SAM3 tracker 加几何分支并训练**：

- 结构：在 `Sam3TrackerVideoModel` 的图像编码器输出上加 GeoSAM2 式的残差几何融合（小编码器吃 normal + pos，1×1 conv 零初始化后相加），tracker 主干加 LoRA r=8。
- 训练：与方案 B 同一份数据与损失，起点是 SAM3 权重（无 GeoSAM2 权重可继承，几何分支从零）。
- 验收：sam3track p2 0.276 → ≥ 0.30。
- 只有方案 B 证明"几何分支 + 部件数据训练确实提升传播"之后才值得做，否则等于重新发现"几何没收益"（我们零训练时已经看到 GeoSAM2 几何 vs SAM3 tracker 无几何 = 0.254 vs 0.276）。

### 方案 D（共用上游）：概念库 v4 与 SAM3 掩码质量

两条路线的第一瓶颈都在 SAM3 的 2D 掩码。概念库 v3（`concept_bank.py`，BCE + Dice，E₀ 全局偏移 + 每名字偏移，3 epoch，混合负例）把留出集 2D mIoU 0.288 → 0.368。可做：

- 更多数据（1319 → 2000 物体的 `views/az*/sam3_masks.npz` + GT），更多视角（用 GeoSAM2 的 12 视角渲染作训练视角，而不只 az0/az135）。
- 难负例：把审阅时标出的"SAM3 常把 A 涂成 B"的名字对作为显式负例（`--neg_mode` 已支持 mixed）。
- 输出校准：按名字学一个阈值偏置（现在全局 0.4）。

命令模板见 `REPORT_overview.md` 附录 C；SAM3 venv，单卡 1–2 h。验收：留出集 2D mIoU ≥ 0.40，hard 20 的 sam3_p12 行 mIoU 0.230 → ≥ 0.25。

## 5. 推荐执行顺序（云端）

| 天 | 做什么 | 判断点 |
|---|---|---|
| 0.5 | 环境 + §3 复现基线 | 表里的数对得上 |
| 1–2 | A1 标签正则化（CPU）+ A2 锚视角选择 | 边界 F1 ≥ 0.62、mIoU ≥ 0.29 |
| 2–3 | D 概念库 v4（GPU，可与 A 并行） | 2D mIoU ≥ 0.40 |
| 3 | 方案 B 的前置判断：GT 掩码作提示跑 geo_p2 | > 0.6 → 不训练，回到 A/D；< 0.45 → 进入 B |
| 4–5 | A3、A4 双管线融合；出 `compare_*.png` 定性图 | 小件召回 ≥ 0.06 且外部 10 资产碎片 ≤ 1.8 |
| 6+ | （若判定要训）B：数据 3 天 + 训练 2 天 + 评测 | geo_p2 ≥ 0.29 |

## 6. 评测工具速查

- `finetune/eval_parts.py --object <pv/id> --faces mesh.glb --face_labels faces.npy --labels_json labels.json`：GeoSAM2 系列输出（面标签）打分；`--segvigen out.glb --legend legend.json`：SegviGen 输出打分。指标：mIoU（GT 部件 ↔ 预测段的匹配）、一对一 mIoU、语义 mIoU（按名字）、命名正确率、边界 F1、过分割件 / 欠分割段、小件召回、留白、段数。
- `finetune/ablation_score.py hard | ext | tables`：批量打分与 markdown 表。
- `finetune/ext_bench.py`：10 个外部资产全方法对照（含 SegviGen 原生 / base / v6、GeoSAM2 各行、P3-SAM），出 `compare_front.png` / `compare_back.png` 与 `detail/<key>.png`；需要三个 venv + Blender + P3-SAM，路径常量要改，云端可选。
- 逐物体胜负比比均值更可信：`ablation.json` 里有每物体的行。

## 7. 已知的坑

- **变换顺序**：上游 `inference.py` 先平移后缩放，渲染器是先缩放后居中；非居中物体全部面在深度测试里对不上。fork 已修（`inference.py` 与 `geosam2_ablate.prepare_mesh`），别用上游代码替换回去。
- **K < 12 提示视角时未标注视角不能投票**：`geosam2_ablate.lift` 用 `alpha` 列表让无标签视角不可见；否则它们投出的 999（"可见但无部件"）会淹没真实标签。
- **`complete_labels` 的实例拆分是空操作**：`faces_inst.npy` 与 `faces.npy` 相同（0 维 tensor 按身份哈希导致连通域没被拆开）。要实例拆分自己在面邻接图上做（方案 A1 顺带解决）。
- **大网格**：> 20 万面时 GeoSAM2 采点 + 反投影会申请上百 GB 内存（robot 资产 182 GB）。先 `decimate_glb.py` 减到 10 万面以内（bpy，退出时可能 access violation 但文件已写好，看文件不看退出码）。
- **SAM3 tracker 多提示帧**：`add_inputs_to_inference_session` 之后必须对每个提示帧先跑一次前向填 `maskmem_features`，否则 `propagate` 报 `maskmem_features ... cannot be empty`；`sam3_track.py` 已处理。
- **EXR 读取**：`cv2.imread` 返回 None → 装 `opencv-python==4.14.*` 并设 `OPENCV_IO_ENABLE_OPENEXR=1`。
- **无显示器 Blender**：EEVEE 可能起不来，`--engine CYCLES`。
- **GeoSAM2 结果的随机性**：采点随机，同一输入 mIoU 抖 ±0.01；比较看 20 物体均值与胜负数。
