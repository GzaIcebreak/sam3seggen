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
朴素 2D 引导会把复杂壳体拆成几十块碎片。`segment_parts.py` 换了一条路：先做多次无提示全量
分割并对分区取交（故意过分割），再多视角渲染交给 SAM3 打掩码，每个**原子连通分量**投票选最
贴合它的提示词，同名部件通过贴图图集合并成单个 mesh——UV 无损重映射，不重烘。语言只决定名字，
边界全部来自几何，所以一个标错的像素再也撕不开一条边。

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

### `segment_parts.py` —— 当前主线：先过分割，再命名

```sh
python segment_parts.py \
  --glb model.glb \
  --prompts head torso arm hand leg foot \
  --unassigned_to torso \
  --out out/parts.glb --work_dir out/work
```

边界全部由几何决定，语言只负责取名。管线是固定的五步：

1. **平涂**：用固定视角网格渲染**原始模型**；如果渲染图本身没有颜色（无贴图的白模），拿
   一次 `full_seg` 的部件色给它涂一层临时平色（`flat_paint.py`）。有贴图的模型跳过这步。
2. **引导图**：SAM3 逐视角出掩码，并画出供人审阅的叠加图到 `work/guidance/`。这一步刻意
   排在采样之前——提示词选错在这里就看得出来，而拆分的钱还没花。
3. **拆分**：跑 `--samples` 次无提示 `full_seg`，条件相机在 `--azimuth` 两侧按
   `--azimuth_jitter` 抖动；对所有采样的分区取共同细化——只有每一次采样都同色的面才属于
   同一个原子，因此任何一次采样划出的边界都会保留（`data_toolkit/meet_samples.py`）。
4. **合并双壳**：原子切成连通分量，并把 remesh 的内壁并进它所衬的外壳。这必须发生在命名
   之前：两者没有共边，单独投票的内壁会继承离它最近的那个部件（`unit_vote.split_units`）。
5. **合并**（由 `--merge` 开关控制）：每个单元取第 2 步掩码覆盖它的名字，同时覆盖的取最
   贴合的那个；相机看不到的按最近可见面继承。按名字导出，并把原模型的 albedo 烘回每个部件。
6. **补全**（由 `--complete` 开关控制）：我们的部件在切口处是敞开的，X-Part 把每个部件重新
   生成为封闭实体（`xpart_complete.py`）。

第 3、6 步要花 GPU 时间，其余在渲染图缓存后都是秒级。

### 采样数买到的不是更细，是更少靠运气

同样是 5 次采样，一次抽样让 Mickey 只剩 13 个原子、最大的占 48%、耳朵从未与头分开，另一次抽样
就给出 34 个原子和全部六个部件。**不同抽样之间的差距比 5 次和 9 次之间的差距还大**，而多采样正
是用来收敛掉这个方差的：7 次和 9 次落在同样的六个部件上，所以默认值是 7——9 次只是多出原子
（40 对 34），不多出部件。

把每个样本读得更细不是替代品：把 `--color_tol` 从 20 降到 3 让每样本标签数翻了二十倍，原子数
仍然是 13（多出来的全是碎斑）。加大 `--azimuth_jitter` 也不是，而且可能倒退：60 度时有几次采样
只出了 2~6 个标签，这些近乎空白的分区会让 meet 沿着不存在的边界碎裂。

真正该调粗的是原子和单元的下限（`--min_atom_faces` 300、`--min_unit_faces` 600）。低于这个尺寸
的原子是采样之间的分歧碎屑，不是部件边界；从 150/300 提上来让机器人从 66 个原子降到 59、Mickey
从 40 降到 34，两个模型的命名部件一个没少，而**靠实际投票（而非最近邻兜底）命名的表面占比反而
上升**（机器人 75.0% → 76.1%）。再粗下去部件仍然保得住，但单元开始跨越两个部件，投票覆盖率又
掉回去（1000 时 73.3%）。

### 告诉 X-Part 部件长什么样（`--condition`）

包围盒是有损提示：盒子里还装着别的东西。X-Part 默认就是拿**落在盒子里的源表面**作为每个部件的
条件，于是机器人的躯干盒子连带装进了两条腿的上半截，生成回来的就是一个带腿的躯干。

但我们不止有盒子——拆分已经逐面决定了每个三角形属于谁。默认的 `--condition surface` 就从这些面
上采条件点，作为 `part_surface_inbbox` 直接交给 X-Part：同一个张量，只是由归属算出来的，而不是
裁出来的。盒子仍然一起传，它决定 token 预算。

用「生成实体上的点离该部件自身表面多远」来量（超过对角线 2% 记为跑偏）：

| | 盒子条件 | 表面条件 |
|---|---|---|
| 机器人 躯干 | 65.4%（体积 0.0530） | **5.0%**（体积 0.0183） |
| 机器人 两条腿 | 9.2% / 13.2% | **2.3% / 2.3%** |
| Mickey 均值（11 件） | 24.1% | **18.7%** |

`--condition box` 保留旧行为用于对照。

### X-Part 不是它提示词的函数

`partformer_dit` 每次前向都用 `torch.randperm` 抽一个部件身份嵌入，所以同一个部件、同样的提示，
结果会变。米奇那只小脚在六次运行里超出自己盒子的幅度依次是 675%、174%、7%、112%、1615%、882%。
**偶尔回来一个大出好几倍的部件是抽签抽坏了，不是提示词坏了。**

盒子正是让这件事可检验的东西：它说明了这个部件原本多大，而任何只为封住切口的生成都不该多出半
个盒宽。超过就用同样的条件重抽（`--redraws`，默认 2 次），并且**只在新的一抽确实更贴合盒子时才
采用**——上面那只脚的两次重抽是 1615% 和 882%，都被拒掉了。

### 太小的组件折回去，而不是单独分离（`--min_area_share`）

X-Part 在碎片上不可靠，而碎片通常也不是部件，只是切口留下的残渣。低于这个表面占比的连通分量会
按表面距离**并进紧邻的那个大分量**，名字随宿主。

这也修掉了旧行为的一个洞：这些组件以前是被**丢弃**的，于是那片表面进不了任何提示，X-Part 返回的
任何东西都覆盖不到它。折叠是「丢弃」本来就该是的样子。

注意这里的占比是在剔掉 remesh 内壁**之后**算的（内壁是丢弃而非折叠——它法线朝里，拿来做条件会
让一半的点描述一个里外翻转的形状），所以数值大约是双壳网格原始占比的两倍。

米奇的三只脚落在 1.24~1.30%，恰好在 X-Part 的能力边界上，两种条件都不稳。这就是这个旋钮要你自
己定的地方：默认 `0.005` 留着它们（会得到一只偶尔超框的脚），`0.015` 把它们并进底座，换来 8 个
零逃逸的干净实体，但脚不再是独立件。

### 拆分粒度（`--granularity`）

两个尺寸下限只有一起看才有意义（原子设了下限而单元又把它拆开，等于没设），所以合成一个名字：
`fine` 150/300、`medium` 300/600（默认）、`coarse` 800/1600。单独给 `--min_atom_faces` 或
`--min_unit_faces` 则覆盖对应的那一半。

要保住螺栓头、按钮这种小件就调 `fine`；部件本身很大、而拆分正在把平面碎成一块块面板时就调
`coarse`。

默认视角是 `45,225 × 10`——略偏的 3/4 对。正对 90 度会漏掉胸口的 `torso`，但抬高得不偿
失：35 度俯视会让躯干挡住腿脚和底座。`--flat_paint on|off` 可以强制或关掉第 1 步的平涂。

为什么要先过分割：`full_seg` 没有粒度旋钮，单次采样把相邻部件粘连的概率高到无法忽略——
机器人测试模型上，肩甲和两条手臂在**每一次**同视角采样里都是同一个 2.4 万面的原子，只有抖动
条件视角才能把它们分开。过分割对命名这一步毫无代价（它只会合并），欠分割则不可挽回。

`work/atoms.glb` 是投票前的原子（每块一色），`work/vote_report.json` 是逐单元的投票表。
部件不对时先看 `atoms.glb`：如果那条边界根本不存在，调提示词也变不出来。

### `merge_parts.py` —— 单独的合并（命名）接口

拆分很贵且与提示词无关，命名很便宜、而且是反复调的那一步。`--merge off` 停在第 4 步；给了
`--prompts` 的话引导图照出，所以可以先看图再决定怎么合：

```sh
python segment_parts.py --glb model.glb --merge off --out split/units.glb \
  --prompts head torso arm hand leg foot --unassigned_to torso
# 看过 split/work/guidance/*.png 之后
python merge_parts.py --glb model.glb --split split/work \
  --prompts head torso arm hand leg foot --unassigned_to torso \
  --out named/parts.glb
```

渲染图和掩码按生成它们的提示词缓存在拆分目录里，所以同样提示词重跑只要几秒，换提示词也
只重跑 SAM3。`--merge unit` 不做融合，每个投票单元一个节点——用来定位是哪个单元取错了名。

每个单元都是**逐视角**投票：每台看得够清楚的相机各投一票，选在该视角下最贴合它的掩码，
单元取得票最多的名字。把所有视角的像素混在一起统计，等于让恰好正对它的那台相机独自决定，
而那台相机往往正是它被别的部件挡住一半、因而被判成邻居的那台。机器人上换成逐视角投票后，
约 9700 面的肩甲从 `torso` 回到了 `arm`。

### `segment_api.py` —— 已弃用：输入提示词，输出一个带命名部件的 GLB

已由 `segment_parts.py` 取代。这条路是用单视角的 2D 图去引导生成模型，一个像素标错就会在
3D 上撕出一条边界。保留作为 2D 引导路线的参考实现，它的正面视角选择和 SAM3 上色选项仍在用。

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
- `--parts_output combined|separate`：默认 `combined`，只写 `--out` 那一个 GLB（每个部件一个
  节点）。`separate` 额外把每个部件单独导出到旁边的 `parts/` 目录，并给 `parts.json` 每行加
  `file` 字段；单件是从合并结果里切出来的，节点名和烘焙贴图与合并模式完全一致。
- `--split_mode stain|weld|refine`：染色如何变成部件边界。默认 `stain` 严格沿 SegviGen 的染色切，
  只把不到 100 面的碎块并给邻居（就是 SegviGen 自带 `split.py` 的规则），拆出来的件和染色图
  一致。`weld` 保留同样的切口，但以同色连通块为单位看 2D 引导图：一块有足够比例朝着相机、
  且可见面多数票明确指向另一个部件时，整块换名（块内不切、背面的块不动）。`refine` 是旧流程：
  逐像素回写可见面、投票接缝带、把游离小岛判给包围它的部件——正面和引导图更贴，但碎块数是
  `stain` 的 3–5 倍。`finetune/split_bench.py` 在 ext_bench 资产上对三者做无 GT 的量化对比。
- `--no_v6`：退回 base 2D-map 权重。默认用 `ckpt/full_seg_v6.ckpt`（轨迹监督 LoRA 已合并进去）；
  `--no_sam` 或显式 `--ckpt` 时它本来就不生效。
- `--no_sam`：完全跳过 SAM3，用无提示 full_seg 权重在普通渲染图上分割（部件无命名，按颜色聚类）。
- `--no_texture`：跳过 Blender 重展开 UV + 烘焙（更快）；默认保留真实贴图。
- `--sam3_only`：在渲染 + SAM3 之后停止，保留 `render.png` / `sam3_2d_map.png` / 图例用于审核。

2D 图默认由概念库 v3 的文本偏移 + 阈值 0.4 的小掩码优先叠涂生成。

- `--assign rank`：让 EASE Mask RankGNN 编辑叠涂集合（keep<0.1 丢弃、keep>=0.9 补入）。
  它在同分布上更好，但会把某个提示词的掩码整组删掉，所以默认不开。
- `--assign auto`：同一次前向里把两张图都画出来，只有排序器没丢掉任何提示词时才采用它的结果，
  决策写在 `<map>_auto.json`。代价是多画一遍，不是多跑一遍 SAM3。
- `--assign argmax`：v5 的逐像素竞争。
- `--no_concept_bank`：用原生 SAM3 词嵌入，不加载 v3 概念库。
- `--concept_bank` / `--rank_model`：换权重路径（也可用环境变量 `SEGVIGEN_CONCEPT_BANK` /
  `SEGVIGEN_RANK_MODEL`）。任一默认权重不在本机时会自动降级并打印提示，不会直接失败；
  显式指定的路径找不到则报错。
- `--sam3_threshold`：默认按画笔取标定值——带概念库 0.4，不带 0.3。

任一提示词没有掩码时，严格校验（默认）会报
`SAM3 produced no mask for requested component(s)`；放宽用 `--allow_partial`。

Python 调用：

```python
from segment_api import segment

manifest = segment(
    "model.glb", ["mushroom=small mushroom", "chair"], "out/parts.glb",
    with_texture=True,            # 贴图烘焙为可选项，默认开启
    front_view="auto",            # metric | auto | vlm | None
    use_v6=True,                  # 默认；False 用 base 2D-map 权重
    assign="paint",               # 默认；rank = EASE，auto = 两张图择优
    parts_output="combined",      # 默认；separate 额外每件一个 GLB
    split_mode="stain",           # 默认；weld = 按引导图整块换名，refine = 逐像素回写
    unassigned_to="chair",
    work_dir="out/work",          # 保留中间产物便于检查
)
# manifest: [{"label": 0, "name": "mushroom", "node": "part_00_mushroom", "faces": ..., ...}]
```

### `serve_api.py` —— HTTP 接口

```sh
./run_serve.sh --port 8020          # 交互式文档在 /docs

curl -X POST http://127.0.0.1:8020/segment \
  -F "glb=@model.glb" \
  -F "prompts=leaves" -F "prompts=fruit" \
  -F "unassigned_to=fruit" -F "samples=5"
```

`POST /segment` 走 `segment_parts.py`；`POST /segment_legacy` 是旧的 2D 引导路线，保留
`azimuth` / `front_view` / `assign` / `split_mode` 这些选项。

返回 `parts` 清单和下载链接：`GET /jobs/{id}/download` 取结果，
`GET /jobs/{id}/parts/{node}.glb` 取单件，`GET /jobs/{id}/atoms` 取投票前的原子、
`GET /jobs/{id}/report` 取投票表（旧路线的任务则是 `/map` 和 `/render`）。
`GET /health` 列出当前默认权重和 GPU 是否占用。

`prompts` 必须每个部件传一次，不要用空格拼在一起——概念本身可以带空格（`small mushroom`）。

各阶段仍是各自加载模型的子进程，一次请求约两分钟，而且只有一块卡：任务串行加锁，第二个请求
直接返回 409 而不是排队。它是给外部联调用的测试服务，不是吞吐服务。

### `segment_vote.py` —— 已弃用：单次采样 + 覆盖率投票

已由 `segment_parts.py` 取代：后者对多次采样取交而不是相信单次，并用 IoU 而不是覆盖率决胜
（覆盖率总是把单元判给最大的那张掩码，于是脚被判成腿、手臂被判成躯干）。

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
