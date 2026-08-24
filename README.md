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

3. 模型权重（共约 24 GB，经 hf-mirror 断点续传下载）
    ```sh
    python download_ckpts.py   # -> ckpt/full_seg.ckpt, full_seg_w_2d_map.ckpt, interactive_seg.ckpt
    ```

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
