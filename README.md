# SAM3-SegviGen（[English](README_EN.md)）

基于 [SegviGen](https://github.com/Nelipot-Lee/SegviGen) + [SAM3](https://huggingface.co/facebook/sam3)
的文本提示 3D 部件分割管线：输入一个 GLB 和一组语义提示词，输出一个 GLB——每个部件一个独立命名的
mesh，带真实贴图。

上游 SegviGen：[项目主页](https://fenghora.github.io/SegviGen-Page/) |
[论文](https://arxiv.org/abs/2603.16869) |
[在线 Demo](https://huggingface.co/spaces/fenghora/SegviGen) |
[权重](https://huggingface.co/fenghora/SegviGen)

## 🌟 相对原版的改进

**识别 —— 语义提示词替代手涂 2D 引导图。**
原版的 2D 引导模式需要手涂颜色图。本分支用 SAM3 按文本提示词分割条件渲染图
（`"helmet"`、`"body=head+face+hand"`），掩码上色后直接作为 SegviGen 的 2D 引导图。
提示词组可以把多个概念合并成一个部件，`--unassigned_to` 把没有提示词认领的区域归入指定部件，
保证输出部件数与提示词数严格一致。

**合并 —— 每个语义物体恰好一个 mesh。**
朴素 2D 引导会把复杂壳体拆成几十块碎片。`segment_vote.py` 换了一条路：先做无提示全量分割，
再多视角渲染交给 SAM3 打掩码，每个**部件**投票选覆盖率最高的提示词（覆盖率门槛防止掩码边缘渗漏
污染大部件），同名部件通过贴图图集合并成单个 mesh——UV 无损重映射，不重烘。

**正面视角自动选择。**
2D 引导模式对渲染朝向敏感。`--front_view` 自动挑选条件视角：

| 模式 | 决策方式 | 依赖 |
|---|---|---|
| `metric` | 轮廓对称性 + 覆盖率 + 居中度 | 无，纯离线 |
| `auto` | 指标 top-3，再用 SAM3 提示词置信度定夺 | SAM3 环境 |
| `vlm` | 指标 top-4 拼图，由 VLM（Kimi/Moonshot）选语义正面 | `MOONSHOT_API_KEY` |

**贴图烘焙作为可选项（默认开启）。**
每个输出部件在 Blender 中重新展开 UV 并把原模型贴图烘焙回去（`--no_texture` 完全跳过 Blender，
用占位纯色，速度快）。

**相对原版的健壮性修复。**
图例/清单严格校验与 `--sam3_only` 审计模式、离体碎片清理、图集合并的 glTF V 轴翻转与
metallic 系数修复、相机约定经渲染器实测标定（IoU 0.987）。

**微调 —— 让 SAM3 的颜色真正进入 SegviGen。**
原模型的 2D 引导图是从 3D 真值光栅出来的像素级完美图，而 SAM3 给的图边缘毛糙、有整件漏检、
有灰色未分配区域、有语义合并——这个域差是"SAM3 颜色进不了 SegviGen"的根因。`finetune/`
提供了完整的域适配 LoRA 微调管线（两条建样本路 + 训练 + 评估），详见下文
[微调](#-微调修-sam3-颜色进不了-segvigen)。

## 📷 效果

自动选出的正面 → SAM3 语义 2D 图 → 最终带贴图拆分结果（蘑菇 + 椅子，恰好两个 mesh）：

<p>
  <img src="docs/images/front_render.png" width="30%"/>
  <img src="docs/images/sam3_2d_map.png" width="30%"/>
  <img src="docs/images/vote_result_0.png" width="30%"/>
</p>
拆出的两个独立部件——提取的蘑菇 / 移除蘑菇后的椅子：

<p>
  <img src="docs/images/extracted_mushroom.png" width="30%"/>
  <img src="docs/images/chair_only.png" width="30%"/>
  <img src="docs/images/vote_result_90.png" width="30%"/>
</p>

## 🔨 部署

在 Windows 11 + RTX 5090D（32 GB）上开发验证；原版面向 Linux + ≥24 GB 显存——两者都可用。
需要**两个** Python 环境，因为 SAM3（transformers 5.x）与 SegviGen（transformers 4.57.6）依赖冲突：

1. SegviGen 环境（运行本仓库的环境）：先装 [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 依赖
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

    pip install mathutils
    pip install transformers==4.57.6   # 锁版本：TRELLIS.2 issue #101
    pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
    pip install --upgrade Pillow trimesh
    # 仅 Linux：sudo apt-get install -y libsm6 libxrender1 libxext6
    ```

2. SAM3 环境（另一个独立 venv）：transformers 5.x + `facebook/sam3` 权重
    ```sh
    pip install "transformers>=5"   # 外加与你 CUDA 匹配的 torch
    ```

3. 模型权重（见下节）

### 权重位置

仓库本身**不带**权重（`weights/` 已 gitignore）。部署用的东西分三类：上游公开仓、gated 仓、我们自己训的。
国内直连 huggingface.co 经常不通，下载脚本默认走 [`hf-mirror.com`](https://hf-mirror.com)
（`HF_ENDPOINT`，见 `env.sh` / `download_weights.sh`）。

| 用途 | 远程 | 落到本地 | 怎么下 |
|---|---|---|---|
| SegviGen 三个 ckpt（各 ~7.3 GB） | [`fenghora/SegviGen`](https://huggingface.co/fenghora/SegviGen) | `ckpt/full_seg.ckpt`、`full_seg_w_2d_map.ckpt`、`interactive_seg.ckpt` | `python download_ckpts.py` |
| TRELLIS.2-4B（体素 / 纹理编解码） | [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) | `microsoft/TRELLIS.2-4B/` | `./download_weights.sh` |
| 抠图 RMBG / BiRefNet | [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) 或 [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) | `weights/...`，由 `SEGVIGEN_RMBG` 指向 | `./download_weights.sh` |
| SAM3（gated，先在 HF 上同意许可） | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | `weights/facebook/sam3`（`SEGVIGEN_SAM3`） | `export HF_TOKEN=… && ./download_weights.sh --gated` |
| DINOv3（gated） | [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `weights/facebook/dinov3-vitl16-pretrain-lvd1689m`（`SEGVIGEN_DINOV3`） | 同上 |
| **SAM3 概念库 v3（部署默认）** | [`Zaun1996/sam3-concept-bank`](https://huggingface.co/Zaun1996/sam3-concept-bank) 根目录 `bank.pt` | 任意路径，推理时 `--concept_bank` | `huggingface-cli download Zaun1996/sam3-concept-bank bank.pt` |
| 概念库后续实验（**不上线**） | 同上仓的 [`v5/`](https://huggingface.co/Zaun1996/sam3-concept-bank/tree/main/v5)、[`mask_rank_v3/`](https://huggingface.co/Zaun1996/sam3-concept-bank/tree/main/mask_rank_v3) | 对照 / 复现 | 按子目录拉 |
| SegviGen LoRA（v6 等） | [`Zaun1996/segvigen-lora`](https://huggingface.co/Zaun1996/segvigen-lora) | 先下 `v6/lora_last.pt`，再 `finetune/merge_lora.py` 并进 `ckpt/full_seg_w_2d_map.ckpt` | 私有仓，需要 HF token |

```sh
# 上游 SegviGen 三个 ckpt → ckpt/
python download_ckpts.py

# 加上 TRELLIS.2-4B + 抠图；gated 的 SAM3 / DINOv3 再加 --gated（要先 accept license + HF_TOKEN）
./download_weights.sh
# ./download_weights.sh --gated

# 部署用的 SAM3 概念库（287 KB）
huggingface-cli download Zaun1996/sam3-concept-bank bank.pt --local-dir datasets/concept_bank_v3
```

推理接概念库：`python sam3_to_2dmap.py --concept_bank datasets/concept_bank_v3/bank.pt --threshold 0.4 …`。
`v5/`（逐像素 CE + 解码器 LoRA）和 `mask_rank_v3/`（候选级排序器）定量有提升，外部资产定性不够，**部署仍用根目录 `bank.pt` + 叠涂 @0.4**。各目录 README 写了指标和根因。

运行时配置：

- `SEGVIGEN_PY_SAM3` —— SAM3 环境的 Python 路径（默认 `../.venv_holo/Scripts/python.exe`）。
- GPU 后端（Blackwell 已验证）：`ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm FLEX_GEMM_ALGO=explicit_gemm`。
- VLM 正面模式：`MOONSHOT_API_KEY`（环境变量或仓库根目录下 gitignored 的 `.env` 文件）；
  `SEGVIGEN_VLM_BASE_URL` / `SEGVIGEN_VLM_MODEL` 可覆盖默认值
  （`https://api.moonshot.cn/v1`，`kimi-latest`）。

## 📒 接口

### `segment_api.py` —— 输入提示词，输出一个带命名部件的 GLB

```sh
python segment_api.py \
  --glb model.glb \
  --prompts "mushroom=small mushroom" chair \
  --unassigned_to chair \
  --front_view auto \
  --out out/parts.glb --work_dir out/work
```

- `--prompts`：每个输出部件一条；用 `+` 连接多个概念合并为一个部件
  （`body=head+face+hand`），可加 `name=` 显式命名。完全自定义，无写死内容。
- `--front_view metric|auto|vlm`：自动选择条件视角；`--azimuth`（角度）仍可手动固定视角。
- `--no_sam`：完全跳过 SAM3，用无提示 full_seg 权重在普通渲染图上分割（部件无命名，按颜色聚类）。
- `--no_texture`：跳过 Blender 重展开 UV + 烘焙（更快）；默认保留真实贴图。
- `--sam3_only`：在渲染 + SAM3 之后停止，保留 `render.png` / `sam3_2d_map.png` / 图例用于审核。

Python 调用：

```python
from segment_api import segment

manifest = segment(
    "model.glb", ["mushroom=small mushroom", "chair"], "out/parts.glb",
    with_texture=True,            # 贴图烘焙为可选项，默认开启
    front_view="auto",            # metric | auto | vlm | None
    unassigned_to="chair",
    work_dir="out/work",          # 保留中间产物便于检查
)
# manifest: [{"label": 0, "name": "mushroom", "node": "part_00_mushroom", "faces": ..., ...}]
```

### `segment_vote.py` —— 多视角投票，产出无碎片的语义物体

当单张 2D 引导图会把部件拆碎时，改用投票而非引导：

```sh
python segment_vote.py \
  --glb model.glb \
  --prompts "mushroom=small mushroom" chair \
  --unassigned_to chair --sam3_threshold 0.65 \
  --out out/merged.glb --work_dir out/vote_work
```

流程：全量分割 → 多视角渲染 → 每视角 SAM3 掩码 → 部件级投票（面级票数汇总到部件；
`min_cover` 防止掩码渗漏）→ 同名部件经贴图图集合并为一个 mesh。输出：每个提示词名恰好一个
mesh 节点。

### 原版推理脚本

原始入口保持不变：交互式分割（`inference_interactive.py`）、全量分割与 2D 引导分割
（`inference_full.py`，加 `--two_d_map`）。

### 测试

```sh
python -m unittest discover tests
```

## 🧪 微调（修 "SAM3 颜色进不了 SegviGen"）

完整文档见 [`finetune/README.md`](finetune/README.md)。所有脚本在仓库根目录下通过
`finetune\run_ft.bat <脚本> <参数>` 运行（与推理 .bat 相同的环境变量，使用 `.venv`；只有
`sam3_masks.py` 走 `.venv_holo`，由路 A 自动以子进程调用）。

**目标。** 对 `full_seg_w_2d_map` 模型做 LoRA 域适配，让它学会两件事：颜色严格跟着 2D 图走、
边界贴到几何上（锯齿 / 溢出 / 斑点不进 3D）；灰 = 未分配（2D 里整件是灰的，3D 也输出灰而不是
乱猜一个色；2D 只覆盖半件的，3D 整件补全同色）。

**两条建样本路。** 每个物体只体素化 / 编码一次，之后每个变体只是给体素重新上色再过一次纹理编码器，
所以一个物体出几十个变体很便宜：

| 路 | 2D 条件图来源 | 脚本 |
|---|---|---|
| A | Blender 渲染真实贴图 → SAM3 掩码 → 绑定到真值部件（覆盖率 / 精度规则，未绑定件变灰） | `make_samples_a.py` |
| B | 像素级完美图 + 合成腐蚀（边界抖动、斑点、整件变灰、半件擦除、灰洞、邻件合并） | `make_samples_b.py` + `corrupt.py` |

```bat
REM 数据：PartVerse（断点续传下载 → 按 anno_infos 面片标签拆件 → 从 caption 抽提示词）
finetune\run_ft.bat download_partverse.py --out E:\data\partverse
finetune\run_ft.bat import_partverse.py --partverse E:\data\partverse --out E:\data\pv --limit 2500

REM 建样本：分块子进程驱动（路 B 2000 个 + 路 A 500 个对象列表），隔离 o_voxel 原生崩溃，可恢复
finetune\run_ft.bat run_batch.py --dataset_root E:\data\pv --objects_b E:\data\pv_list_b.txt --objects_a E:\data\pv_list_a.txt

REM 训练 / 导出 / 评估
finetune\run_ft.bat train.py --dataset_root E:\data\pv --out_dir finetune\runs\v1 --max_steps 4000
finetune\run_ft.bat merge_lora.py --lora finetune\runs\v1\lora_last.pt --out ckpt\full_seg_w_2d_map_ft.ckpt
finetune\run_ft.bat eval_fidelity.py --object E:\data\pv\<obj> --variant <name> --run_inference --ckpt ckpt\full_seg_w_2d_map_ft.ckpt
```

**模块。** `common.py`（目录规范、ID 色板、相机复现、体素重着色、SLAT 编码、DINOv3 条件）、
`dataset.py` / `lora.py` / `model.py` / `train.py`（v-pred flow-matching LoRA 训练，含 step-0
分组损失检查、holdout）、`merge_lora.py`（合并为可直接被 `inference_full.py` 使用的 ckpt）、
`eval_fidelity.py`（输出 GLB 与 2D 图预期颜色的保真度 / 纯度，自动对齐 Y-up 帧）。

## ⚖️ License

本项目基于 [MIT License](LICENSE) 开源。
注意 **`trellis2`** 目录下的代码来自 [TRELLIS.2](https://github.com/Microsoft/TRELLIS.2)，
仍受其原始许可证约束；使用或再分发该部分代码时请遵守 TRELLIS.2 的许可要求。

## 引用

```
@article{li2026segvigen,
      title = {SegviGen: Repurposing 3D Generative Model for Part Segmentation}, 
      author = {Lin Li and Haoran Feng and Zehuan Huang and Haohua Chen and Wenbo Nie and Shaohua Hou and Keqing Fan and Pan Hu and Sheng Wang and Buyu Li and Lu Sheng},
      journal = {arXiv preprint arXiv:2603.16869},
      year = {2026}
}
``` 
