# SegviGen `full_seg_w_2d_map` 域适配微调(修 "SAM3 颜色进不了 SegviGen")

原模型的 2D 条件图是从 3D 真值直接光栅出来的像素级完美图;SAM3 给的是边缘毛糙、
有整件漏检、有灰色未分配区域、有语义合并的图。这里用 LoRA 让模型学会两件事:

1. 颜色跟着 2D 图走,边界贴到几何上(锯齿 / 溢出 / 斑点不进 3D)
2. 灰 = 未分配:2D 里整件是灰的,3D 里也输出灰(150,150,150),而不是乱猜一个色;
   2D 只覆盖了半件的,3D 里整件补全同色

改动全貌、v1/v2 结果、数据修复与原论文的损失/条件结构对照见 [`REPORT_v3_changes.md`](REPORT_v3_changes.md)。

所有脚本在 SegviGen 根目录下通过 `finetune\run_ft.bat <脚本> <参数>` 运行
(它设置了与推理 .bat 相同的环境变量并使用 `.venv`);只有 `sam3_masks.py` 用 `.venv_holo`,
由 `make_samples_a.py` 自动以子进程调用。

## 目录约定

```
<root>/<object>/
  input.glb  parts/<k>.glb  names.json        # 输入:整体、按件拆分、部件名(SAM3 提示词)
  input.vxz  ids.vxz  voxel_part.npy          # 只算一次:体素化 + 每个体素属于哪一件
  shape_slat.pth  input_tex_slat.pth  common_coords.pth
  views/az<角度>/ids.npy  render.png  sam3_masks.npz  prompts.json
  variants/<name>/map.png  cond.pth  output_tex_slat.pth  meta.json
```

变体只需重新给 `ids.vxz` 的体素上色再过一次纹理编码器,不重复体素化,所以一个物体出几十个变体很便宜。
`meta.json.kind` ∈ {clean, corrupt, sam3},训练日志与 step-0 检查按它分组。

## 建样本

```bat
REM 本地 GLB(按节点拆件,名字用作提示词)
finetune\run_ft.bat import_glb.py --glb data_toolkit\assets\example.glb --out E:\data\ft\example --names wall chimney door ...

REM 路 B:每个视角 1 张干净图 + n_corrupt 张合成腐蚀图
finetune\run_ft.bat make_samples_b.py --dataset_root E:\data\ft --azimuths 0,135 --n_corrupt 3

REM 路 A:bpy 渲染 -> SAM3(.venv_holo) -> 掩码绑定真值部件 -> 图 + 目标
finetune\run_ft.bat make_samples_a.py --dataset_root E:\data\ft --azimuths 0,135
```

路 B 腐蚀算子(`corrupt.py`):`jitter` 边界溢出、`speckle` 边界斑点、`drop` 整件变灰(3D 也灰)、
`partial` 半件擦除(仅 2D)、`holes` 灰洞(仅 2D)、`merge` 相邻件合成一色(2D、3D 同步)。

路 A 绑定规则(`make_samples_a.py::bind_masks`):掩码覆盖某件可见像素 ≥ `--cover`(0.5)即绑定,
一个提示词绑到多件 = 一个颜色组(3D 同色);没有整件覆盖但精度 ≥ 0.5 且覆盖 ≥ 0.15 记为 partial
(3D 整件上色);没被任何提示词绑定的可见件 → 灰。掩码按面积从小到大画,溢出像素保留(这是 SAM3 的真实行为)。

`--hidden_policy gt|grey`:视角里完全看不见的件,3D 用自己的颜色(论文行为,默认)还是灰。

## 训练

```bat
REM step-0:不训练,固定 t={0.2,0.5,0.8} 和固定噪声,按 kind 输出平均损失(可比较不同 ckpt)
finetune\run_ft.bat train.py --dataset_root E:\data\ft --check_only --out_dir finetune\runs\check0

REM LoRA 训练(默认 r=16, alpha=32, 自注意力+交叉注意力的 q/k/v/out,lr 1e-4,p_uncond 0.1)
REM --holdout_file 里的对象不参与训练,每 --check_every 步在其上跑一次分组损失检查
finetune\run_ft.bat train.py --dataset_root E:\AI_New\ModelGen\datasets\pv --out_dir finetune\runs\v1 ^
    --batch_size 4 --grad_accum 1 --max_steps 4000 --save_every 500 ^
    --holdout_file E:\AI_New\ModelGen\datasets\pv_holdout.txt --check_root E:\AI_New\ModelGen\datasets\ft_smoke --check_every 500

REM 合并成 inference_full.py 直接可用的 ckpt
finetune\run_ft.bat merge_lora.py --lora finetune\runs\v1\lora_last.pt --out ckpt\full_seg_w_2d_map_sam3.ckpt
```

目标与 TRELLIS.2 训练器一致:v-prediction MSE、logit-normal t、sigma_min 1e-5、条件按 `p_uncond` 置零。
基座冻结、bf16;LoRA 参数 fp32;默认开梯度检查点。RTX 5090 上 batch 2 峰值约 6.7 GiB、约 4 s/步,
batch 4 峰值约 8.9 GiB、约 2.2 s/步(GPU 无争用时)。`--lora_targets cross` 只训交叉注意力(更保守)。

`--check_only` 在给了 `--holdout_file` 时只评留出对象(否则整个训练集都要过一遍);
`--check_limit N` 按 kind 等比抽样把校验集封顶,训练中的周期性校验同样受它约束。

### 长跑:别让训练跟着终端一起死

后台起的进程会随会话结束被回收。用计划任务(或任何脱离会话的方式)拉起 `train_loop.bat`:
它在 python 非正常退出后自动带 `--resume_lora <out_dir>\lora_last.pt` 重试,`train.py` 会从
checkpoint 里的 step 继续学习率 schedule,不重新 warmup。配合 `--save_every 250`,一次中断最多损失几分钟。

```bat
REM <attempts> <out_dir> <train.py 的其余参数>
finetune\train_loop.bat 30 finetune\runs\pv_v1 --dataset_root E:\data\pv --max_steps 4000 --save_every 250
```

看进度(读 `log.jsonl`,算 s/step 与 ETA,列出 checkpoint 与各次 holdout 校验):

```powershell
powershell -File finetune\watch_train.ps1 -Follow
```

注意计划任务默认以较低优先级运行;若发现磁盘 I/O 成为瓶颈,把任务 XML 里的 `<Priority>` 调到 4。

### wandb

```bat
.venv\Scripts\wandb login                      REM 或设置 WANDB_API_KEY
set SEGVIGEN_PROXY=http://127.0.0.1:7078       REM 需要代理才能出网时(run_ft.bat 会转成 HTTP(S)_PROXY)
finetune\run_ft.bat train.py --dataset_root E:\data\pv --out_dir finetune\runs\pv_v1 --wandb
```

记录 `train/loss`、`train/ema`、`train/lr`、`train/loss_<kind>`(clean / corrupt / sam3 分开)、
`train/s_per_step`、`train/vram_gib`,以及每次周期性校验的 `holdout/<kind>`;config 里带上变体总数与
各 kind 的构成。**`--wandb_id` 默认取 `out_dir` 的目录名并以 `resume="allow"` 初始化**,
所以 `train_loop.bat` 的多次重启会续在同一个 run 上,而不是每次断了就新开一条曲线。
出不了网就用 `--wandb_mode offline`,事后 `wandb sync <offline-run 目录>`。

要盯的核心指标是 `holdout/sam3`:它跌下来才说明真实 SAM3 条件下的表现在变好,
训练损失下降本身可能只是拟合了合成腐蚀。

## PartVerse

```bat
set HF_ENDPOINT=https://hf-mirror.com
.venv\Scripts\python finetune\download_partverse.py --out E:\datasets\partverse      REM 103 GB,可续传,自动拼接分卷
tar -xzf E:\datasets\partverse\normalized_glbs.tar.gz -C E:\datasets\partverse
tar -xzf E:\datasets\partverse\textured_part_glbs.tar.gz -C E:\datasets\partverse
.venv\Scripts\python finetune\import_partverse.py --partverse E:\datasets\partverse --out E:\data\pv --limit 1500 --min_parts 3 --max_parts 24
```

`import_partverse.py` 默认不用 80GB 的 `textured_part_glbs`:部件只需几何,直接用 `anno_infos` 的 `face2label`
从带贴图的整体上切出来(面片顺序不一致时按最近面心迁移标签,对不上的对象跳过)。
SAM3 提示词是短描述的头名词短语("A metal spike extracted from a baseball bat." → "metal spike"),
规则式抽取,约一到两成会得到 "gray cylindrical component" 这类弱提示——SAM3 找不到就按灰色规则进目标,
不算错误样本;要更好的名字可以用 VLM 再过一遍 `names.json`。

批量生产请用驱动器,不要直接长跑 `make_samples_*.py`:`o_voxel` 的原生扩展在个别网格上(以及长时间运行后)
会以 0xC0000005 崩掉整个进程。驱动器每 `--chunk` 个对象起一个子进程,失败的块逐个重试,仍失败的对象记进
`<root>/run_batch_state.json` 并跳过;A、B 交替执行避免共用 GPU;重复运行会自动跳过已完成的对象。

```bat
finetune\run_ft.bat run_batch.py --dataset_root E:\AI_New\ModelGen\datasets\pv --jobs b a ^
    --objects_b E:\AI_New\ModelGen\datasets\pv_list_b.txt --objects_a E:\AI_New\ModelGen\datasets\pv_list_a.txt --chunk 8
```

当前机器上的吞吐(串行、无争用):路 B 约 14 s/对象(2 视角 × (1 干净 + 3 腐蚀)),路 A 约 18 s/对象(含 bpy 渲染 + SAM3)。

## 评估

```bat
REM 用某个 ckpt 对一个变体的 map.png 跑 inference_full.py,再按 GT 部件采样输出表面颜色打分
finetune\run_ft.bat eval_fidelity.py --object E:\data\ft\example --variant sam3_az0 --ckpt ckpt\full_seg_w_2d_map.ckpt
finetune\run_ft.bat eval_fidelity.py --object E:\data\ft\example --variant sam3_az0 --ckpt ckpt\full_seg_w_2d_map_sam3.ckpt
```

打分是"最近邻调色板归属",不是颜色相等。SegviGen 的解码器复现调色板色时会有几十个 RGB 单位的
偏移,用绝对距离阈值会把颜色其实对了的件判成错:期望 (210,242,63) 输出成 (146,239,81) 是同一个
黄绿色、距离 66,而 60 的阈值给 0 分。分割真正需要的是归属对,所以每个采样点取最近的图例色、不设
阈值,偏移单独用距离报告,让两种失效模式可区分。

- `fidelity`:该件表面上最近邻图例色 = 2D 图分给它的色(灰件则为灰)的采样点比例;`parts_correct(>=0.8)` 计数
- `parts_assigned_correct`:主导归属正确的件数 —— 这是"颜色有没有进去"最直接的读数
- `dist_median` / `median_dist`:到期望色的距离,即偏色程度;`margin`:到期望色的距离减到最近竞争色的距离,
  负值 = 归属正确,接近 0 = 快要翻错(比 fidelity 更早预警)
- `purity`:该件表面上占主导归属的比例,低 = 被撕碎
- `fidelity_snap60` / `mean_fidelity_snap60`:旧的 60 单位阈值口径,只为和早期报告对比而保留
- `metric` 字段区分新旧报告;视角里完全不可见的件不计分(单独给 `hidden_mean_purity`)
- 输出 GLB 是 TRELLIS 的 Y-up 坐标(相对输入绕 X 转 90°),脚本自动在候选坐标系中选覆盖率最高的
- step-0 vs 训练后:`check_loss.json` 里 sam3/corrupt 与 clean 的差距应缩小
- 留出集建议 PartObjaverse-Tiny + 自己的资产(monk / dwarf / 1.glb)

基线(3 个 PartVerse 留出对象,基座 ckpt,sam3_az0):mean_fidelity 0.953,28 件里 27 件归属正确,
中位偏色 28~67 单位。唯一真错的是一把小钥匙被判给了邻件的蓝色。同一批数据在旧阈值口径下只有 0.656,
差距全部来自偏色被误判成归属错误。

v3 新增指标:`fragments`(每个颜色在表面均匀采样点上的连通块数,基于输入部件采样,输出网格拓扑不会虚增)、
`boundary_f1`(预测颜色边界 vs 期望边界的点级 F1,容差 1% 物体尺度);`--shuffle_legend` / `--swap_names`
分别做"打乱图例"对照和"互换两个名字看边界是否跟着动"的语义冲突测试。硬样本集由 `pick_hard.py` 按
`voxel_part.npy` 里"细长件贴在大件上"的接触面积选出。

## v3:SAM3 概念库 + 图例 token + 双视角

v1/v2 只做了 LoRA 域适配,颜色语义仍完全靠 latent MSE 隐式学习,留出 MSE 降了 8% 但颜色归属没有变好。
v3 分两层注入语义(细节与公式见 `REPORT_v3_changes.md` 第 4、6 节):

**SAM3 侧(`.venv_holo`)**:`render_views.py` 给每个对象补 `render.png` + GT `ids.npy`;`bench_sam3.py` 比较提示词模板
(结论:裸 `{name}` 最好);`concept_bank.py` 在冻结的 SAM3 上学两个 256 维文本偏移 `E_0`(共享)+ `E_name`(逐名字),
损失 = 正样本 BCE+Dice + 负样本 BCE + presence BCE,`--neg_mode random|hard|mixed`;导出 `bank.pt` 与 `text_cache.pt`。
评测同时报告 `--eval_thresholds` 下的多阈值结果,以便**在相同误检率下**比较不同 bank(概念库会整体抬高分数,
只看默认阈值 0.3 会把标定漂移误判成误检飙升)。

```bat
set HF_ENDPOINT=https://hf-mirror.com
.venv_holo\Scripts\python finetune\concept_bank.py --dataset_root E:\...\pv --out E:\...\concept_bank_v3 ^
    --holdout_file E:\...\pv_holdout_v3.txt --epochs 3 --neg_mode mixed --lr_e0 5e-4 --max_prompts 12 --wandb
.venv_holo\Scripts\python finetune\concept_bank.py ... --eval_only --resume E:\...\concept_bank_v3\bank.pt --eval_thresholds 0.4,0.5,0.6,0.7
```

`sam3_to_2dmap.py` / `sam3_masks.py` 用 `--concept_bank bank.pt` 加载偏移,并把每个部件实际用到的 256 维文本向量
一起写进图例,供 SegviGen 侧使用。

**SegviGen 侧(`.venv`)**:`model.py::LegendEncoder` 把条件从单路 DINO token 扩成
`[主视角 DINO + e_view0; 第二视角 DINO + e_view1; 物体名 token; 每个颜色组一个图例 token]`,
图例 token = LN(W_t·text_256 + MLP(rgb) + e_legend)。DiT 本体不改(交叉注意力原生接受变长上下文)。

```bat
finetune\run_ft.bat train.py --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt ^
    --text_cache E:\...\concept_bank_v3\text_cache.pt --pair --p_drop_legend 0.2 --p_drop_view1 0.3 ^
    --out_dir finetune\runs\pv_v3 --max_steps 1000 --wandb
REM 对照:同参数 + --legend_shuffle --out_dir finetune\runs\pv_v3_shuffle;若 fidelity 与 pv_v3 相同,说明模型没读文本
finetune\run_ft.bat merge_lora.py --lora finetune\runs\pv_v3\lora_final.pt --out ckpt\full_seg_v3.ckpt   REM 另存 ckpt\full_seg_v3_legend.pt
finetune\run_ft.bat eval_fidelity.py ... --ckpt ckpt\full_seg_v3.ckpt --legend_ckpt ckpt\full_seg_v3_legend.pt ^
    --text_cache ...\text_cache.pt --pair
```

`make_samples_a.py` / `make_samples_b.py` 默认给同一对象的两个视角一套共同调色板并共享 3D 目标(`--no_pair` 关闭),
`dataset.py` 据此返回 `cond_partner` 与图例。
