# SegviGen `full_seg_w_2d_map` 域适配微调(修 "SAM3 颜色进不了 SegviGen")

原模型的 2D 条件图是从 3D 真值直接光栅出来的像素级完美图;SAM3 给的是边缘毛糙、
有整件漏检、有灰色未分配区域、有语义合并的图。这里用 LoRA 让模型学会两件事:

1. 颜色跟着 2D 图走,边界贴到几何上(锯齿 / 溢出 / 斑点不进 3D)
2. 灰 = 未分配:2D 里整件是灰的,3D 里也输出灰(150,150,150),而不是乱猜一个色;
   2D 只覆盖了半件的,3D 里整件补全同色

当前状态的单一入口（路线、数据集、v1–v6 训练参数与结果、GeoSAM2 消融、外部资产定性图）见 [`REPORT_overview.md`](REPORT_overview.md)；
改动全貌、v1/v2 结果、数据修复与原论文的损失/条件结构对照见 [`REPORT_v3_changes.md`](REPORT_v3_changes.md)。
云服务器接手（要搬什么、Linux 环境、先复现哪些数字、下一步实验与验收标准）：SegviGen 路线见
[`HANDOVER_cloud_segvigen.md`](HANDOVER_cloud_segvigen.md)，GeoSAM2 路线见 [`HANDOVER_cloud_geosam2.md`](HANDOVER_cloud_geosam2.md)。

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

## v4:显式颜色监督 + partial 变体 + 解耦图例注意力

v3 的结论(9-05):pv_v3 与 pv_v3_shuffle 权重相差 5.5%,留出 MSE 却到小数点后 4 位相同,硬样本 fidelity 比基座还低——
图例 token 收到了梯度,但对输出没有可测量的影响。原因有三:MSE 不需要文本(2D 图已给出 97.7% cell 的颜色);
10 个图例 token 拼进约 1000 个 DINO token 里被 softmax 稀释;LoRA + MSE 三次微调都让颜色归属变差。v4 三件事一起改
(细节见 `REPORT_v3_changes.md` 第 7 节):

1. **显式损失** `cells.py` / `train.py::color_loss`:每步从 v 预测恢复 x0_hat,经冻结线性探针 `color_probe.pt`
   (32 维纹理 latent → RGB,留出对象 R²=0.86、最近调色板 91%)解码颜色,对每个部件纯度 ≥0.9 的 latent cell 做
   最近调色板分类 CE(类别 = 颜色组 + GREY,τ 作用在 RGB 平方距离上),**按类别均衡加权**,细小部件与主体同权;
   探针在目标 latent 上就读错的 cell 不计。这与 `eval_fidelity` 的"最近调色板归属"是同一个量。
2. **partial 变体** `make_samples_partial.py`:干净 2D 图上随机抹灰 1–3 个可见部件,3D 目标与图例不变
   (`write_variant(..., mask_2d=..., target_from=clean_*)`,只重算 DINO cond)。被抹部件的颜色只有图例知道,
   它们的 cell 上的 CE 只能靠读图例降下来。`masked_acc`(被抹 cell 的最近调色板准确率,`check_ts` 最大 t 处最诚实)
   就是"模型是否在用文本"的直接读数;基座在 t=0.95 处为 0.6%。
3. **解耦图例注意力** `model.py::LegendCrossAttention`(`--legend_attn`):每个 DiT block 的交叉注意力加一路
   图例专用 K/V(从图像 K/V 初始化,fp32,可训练)+ **零初始化读出矩阵** `to_out`(ControlNet 式 zero-linear,
   `--out_lr` 默认 3e-4),与图像注意力共用 q;第 0 步与基座完全一致。图例 token 不再拼进图像上下文,而是经
   `set_legend_context` 送给各 block。(第一版的零初始化标量 tanh 门控 500 步都打不开,见 REPORT 7.3(c)。)

```bat
finetune\run_ft.bat cell_labels.py --dataset_root E:\...\pv                       REM <obj>\cell_part.npz
finetune\run_ft.bat color_probe.py --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt
finetune\run_ft.bat make_samples_partial.py --dataset_root E:\...\pv --per_view 2   REM kind=partial
finetune\train_loop.bat 3 finetune\runs\pv_v4 --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt ^
    --text_cache E:\...\concept_bank_v3\text_cache.pt --pair --legend_attn --p_drop_legend 0.1 ^
    --color_probe finetune\color_probe.pt --color_weight 0.3 --color_tau 0.03 ^
    --max_steps 1500 --check_every 250 --check_ts 0.5,0.8,0.95 --wandb
REM 对照:同参数 + --legend_shuffle --out_dir finetune\runs\pv_v4_shuffle;看 holdout/masked_acc_hi_partial 是否分开
finetune\run_ft.bat merge_lora.py --lora finetune\runs\pv_v4\lora_last.pt --out ckpt\full_seg_v4.ckpt
REM 生成 ckpt\full_seg_v4_legend.pt(图例编码器 + 图例注意力 K/V 与读出矩阵,约 630 MB);推理/评估命令与 v3 相同
```

wandb 新增指标:`train/color`、`train/color_acc(_kind)`、`train/masked_acc`,`holdout/color_*`、`holdout/masked_acc_hi_partial`。

v4 结论(REPORT 7.6):颜色损失有效(留出 partial color CE 4.25→0.84),但真/乱图例读数三次检查完全一致,
且 `--check_only --check_no_legend` 消融显示拿掉图例反而更好——全局图例路径只学到对训练对象的记忆。

## v5:逐 token 语义注入

把部件名字加到它所在的 DINO patch token 上(而不是全局图例),颜色缺失处的 token 仍带名字,
模型只需做局部"按名查色";正常区域名字是"同一部件"的先验。REPORT 第 8 节。

- `token_labels.py`:每个变体一个 `tokens.npz`(32×32 patch 的部件名,复现 img_to_cond 的裁切;GT 变体用 `ids.npy`,
  sam3 变体用提示词掩码)。全量 13.7k 变体约 100 s。
- `dataset.py --token_text`:`token_text` [1029,256];`legend_shuffle` 同时置换图例与逐 token 名字。
- `model.py::LegendEncoder.tok_proj`:零初始化、无偏置的 256→1024 线性层,无名 token 不受影响;`tok_gain()` 看它是否在学。
- `inference_full.py`:v5 编码器下由色图颜色反推每 patch 名字(`token_text_from_map`),无需新输入。

```bat
finetune\run_ft.bat token_labels.py --dataset_root E:\...\pv
finetune\train_loop.bat 3 finetune\runs\pv_v5 --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt ^
    --text_cache E:\...\concept_bank_v3\text_cache.pt --pair --token_text --p_drop_legend 0.1 --check_shuffle ^
    --color_probe finetune\color_probe.pt --color_weight 0.3 --color_tau 0.03 ^
    --max_steps 1500 --check_every 250 --check_ts 0.5,0.8,0.95 --wandb
REM 消融:finetune\run_ft.bat train.py ... --check_only --check_no_legend --resume_lora finetune\runs\pv_v5\lora_step500.pt
```

wandb 新增 `train/tok_gain`。判据:holdout 与 holdout_shuffled 的 `masked_acc_hi_partial` 是否分开。

结果(REPORT 8.5、9.1):20 硬对象上 v5 带文本 fidelity 0.185(base 0.617),无文本 0.598;名字只被当成"这里该上色"的开关。

## 轨迹探针与 v6 轨迹训练

`trajectory_probe.py` 用推理采样器(12 步 Euler,rescale_t 3)在留出对象上逐步解码颜色,比较自由轨迹与
teacher-forced 两条曲线。结论(REPORT 9.2):颜色布局在 t=1 的第一步就定了,自由轨迹 acc 全程 ≈ 不变(base clean 0.72→0.73),
而 teacher-forced 在 t≤0.9 已 ≥0.92——训练时的单步颜色监督在"抄 x_t 里的答案"。

```bat
finetune\run_ft.bat trajectory_probe.py --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt --limit 20 --out E:\...\traj\base.json
finetune\run_ft.bat trajectory_probe.py ... --lora finetune\runs\pv_v4\lora_step1500.pt --out E:\...\traj\v4.json
```

v6(Path A):`--p_traj 0.5 --traj_steps 3` 让一半样本的 x_t 来自模型自己的采样轨迹(`rollout()`,同 cond、无 CFG),
颜色 CE 只在 `--color_t_min 0.8` 以上计。纯图像条件 + LoRA r16,不带图例/逐 token 文本:

```bat
finetune\train_loop.bat 3 finetune\runs\pv_v6 --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt ^
    --kinds clean corrupt sam3 --color_probe finetune\color_probe.pt --color_weight 0.3 --color_tau 0.03 --color_t_min 0.8 ^
    --p_traj 0.5 --traj_steps 3 --batch_size 4 --grad_accum 4 --max_steps 1500 --check_ts "0.5,0.95,1.0" --wandb
```

`--check_ts` 在 `cmd /c "..."` 里必须带引号,否则逗号被 cmd 拆成多个参数。看 `color_acc_traj`(rollout 样本上的颜色准确率)。

结果(REPORT 9.4、10.3):自由轨迹颜色准确率 clean 0.733→0.850、sam3 0.520→0.621;20 硬对象 fidelity 0.617→0.663(11 优/8 差),
`eval_parts` mIoU 0.278→0.296(14 优/5 差)、边界 F1 0.666→0.704——首个全面优于基座的版本。合并权重:`ckpt/full_seg_v6.ckpt`
(`merge_lora.py --lora finetune/runs/pv_v6/lora_step1500.pt`)。

## 统一 3D 评测:`eval_parts.py`

按独立 GT 部件计分(含隐藏部件、未覆盖部件计漏分),SegviGen 输出与外部面标签同一口径(REPORT 10.1):

```bat
finetune\run_ft.bat eval_parts.py --object E:\...\pv\<id> --segvigen <infer.glb> --variant sam3_az0 --report r.json
finetune\run_ft.bat eval_parts.py --object E:\...\pv\<id> --segvigen <infer.glb> --legend <map_legend.json>
finetune\run_ft.bat eval_parts.py --object E:\...\pv\<id> --faces <mesh.glb> --face_labels <labels.npy> --labels_json labels.json
```

`miou`(类无关,最佳匹配)/ `miou_matched`(一对一)/ `sem_miou`(按唯一名字,同名部件合并)/ `name_acc` /
`small_part_recall`(<1% 面积)/ `over_seg_parts` / `under_seg_segments` / `boundary_f1` / `unlabelled_share`。

## GeoSAM2 对照

GeoSAM2(CVPR 2026)装在 `E:\AI_New\ModelGen\GeoSAM2` + `.venv_geosam2`(Windows 修复见 REPORT 10.2)。
SAM3 概念库出的掩码作为它的 mask prompt,输出面标签,与 SegviGen 在 `eval_parts.py` 同表比较:

```bat
REM 1. 按 GeoSAM2 约定渲染 12 视角(bpy,主 venv)
finetune\run_ft.bat geosam2_render.py --dataset_root E:\...\pv --objects_file E:\...\pv_hard.txt --out E:\...\geosam2\renders
finetune\run_ft.bat geosam2_render.py --glb E:\...\ext_parts\小狗.glb --name dog --out E:\...\geosam2\renders
REM 2. SAM3 + 概念库 → 每视角标签图(.venv_holo)
.venv_holo\Scripts\python finetune\geosam2_masks.py --renders E:\...\geosam2\renders --dataset_root E:\...\pv ^
    --objects_file E:\...\pv_hard.txt --concept_bank E:\...\concept_bank_v3\bank.pt
.venv_holo\Scripts\python finetune\geosam2_masks.py --renders ... --objects dog --prompts head ear body leg tail --concept_bank ...
REM 3a. single:一个视角的标签图做 prompt + 对面视角自动分割(GeoSAM2 默认用法)
..\.venv_geosam2\Scripts\python finetune\geosam2_run.py --renders ... --out E:\...\geosam2\results --dataset_root E:\...\pv ^
    --objects_file E:\...\pv_hard.txt --view match
REM 3b. dual:v 与 v+6 两张标签图都做 prompt,不跑自动分割
..\.venv_geosam2\Scripts\python finetune\geosam2_dual.py --renders ... --out ... --dataset_root ... --objects_file ... --view match
REM 4. 面标签 → 每部件平色材质 GLB(原资产坐标系),可用 render_cond_view.py 渲染或直接拆件
finetune\run_ft.bat geosam2_to_glb.py --mesh <geosam2.glb> --labels <labels.npy> --labels_json labels.json --ref <input.glb> --out parts.glb
```

`--view match` 选与 SegviGen az0 轮廓最接近的视角(同一输入视角比较),`best` 选 SAM3 找到最多部件的视角。
`*_filled.npy` 把 GeoSAM2 留白的面(0/999)按最近已标面填充,与 SegviGen"从不留空"同口径。

结果(REPORT 10.3,20 硬对象):同一套 SAM3 掩码下,GeoSAM2 与 SegviGen 部署路径的语义 mIoU 打平(0.19–0.21 vs 0.22),
SegviGen 边界 F1 更高(0.67–0.70 vs 0.55–0.63)、碎片更少;GeoSAM2 小件召回更高、默认用法留白 15–17%,~200 万面的网格要先减面
(500–960 s/对象,机器人 CPU 内存溢出)。single 模式的 SAM2 自动分割约需 10 GB 显存,不要和训练同时跑。

## 外部资产对照:`ext_bench.py`

用户自己的 10 个模型(`datasets/ext_parts/`,即 `3D拆件.zip`)上,SegviGen 原生 / P3-SAM 原版 / base+SAM3 / v6+SAM3 / GeoSAM2 single / dual
同机位出图并统计。没有 GT,`score` 给的是结构统计(段数、每段连通块数、留白、边界密度)和方法间一致性,不是准确率:

```bat
finetune\run_ft.bat ext_bench.py all                    REM 或分阶段:
finetune\run_ft.bat ext_bench.py front mesh sam3        REM 8 机位选正面(FRONT_OVERRIDE 手工纠正前/背)、>200k 面减面、SAM3+概念库 2D 图
finetune\run_ft.bat ext_bench.py segvigen               REM full_seg / full_seg_w_2d_map / full_seg_v6,~1 min/次,峰值 21 GB
finetune\run_ft.bat ext_bench.py geo_prep               REM GeoSAM2 12 视角渲染 + SAM3 标签图(轻,可与上一步并行)
finetune\run_ft.bat ext_bench.py geosam2                REM single/dual × match/best,200k 面约 2 min/次
finetune\run_ft.bat ext_bench.py p3sam                  REM 原版 P3-SAM auto-mask(run_p3sam.bat -> p3sam_run.py),同一减面网格,20-60 s/次
finetune\run_ft.bat ext_bench.py score montage report   REM scores.json、compare_front/back.png、REPORT.md
```

输出在 `datasets/ext_bench/`:每资产一个目录(`render.png`/`map.png`/`legend.json`/`seg_*_upright.glb`/`geosam2_*_parts.glb`/`p3sam_parts.glb`/`vis/`),
`REPORT.md` 汇总。提示词与正面机位写在脚本顶部的 `ASSETS` / `FRONT_OVERRIDE`。`compare_*.png` 最右两列是消融里最好的两行
(`GeoSAM2 p2 (fixed)`、`SAM3 tracker p2 (best)`,GLB 由 `ablation_score.py ext` 生成),`GeoSAM2 dual (filled)` 列是变换 bug 修正前的旧结果。`decimate_glb.py`(bpy Decimate,保 UV)供 GeoSAM2 / P3-SAM
用;bpy 模块导出成功后常在退出时崩溃(0xC0000005),脚本按输出文件是否存在判断成功。

`P3-SAM (auto, no SAM3)` 列是 Hunyuan3D-Part 原版 `demo/auto_mask.py` 的结果(`E:\p3part\P3-SAM`,Sonata 骨干,400 个 FPS 点提示,
阈值 0.95,含默认后处理),不看文本也不看 SAM3,是纯几何部件先验的参照;它原来的 conda 环境已不在,`p3sam_run.py` 在 SegviGen 的 `.venv` 里跑
(补装了 addict / fpsample / numba / scikit-learn)。结果在 `datasets/ext_bench/p3sam/<key>/{mesh.glb, faces.npy, info.json}`,颜色按部件面积排序固定分配,与 SAM3 调色板无关。

结果(REPORT 11):SAM3+概念库对前视图 51 个提示词命中 48 个;v6 与 base 一致性 0.86,v6 多找回小件、留白更少但碎片略多(背面轮子/后脑易变色);
GeoSAM2 与 v6 类无关一致性 0.71,重复件颜色更一致、背面有异色补丁、raw 留白 7–8%;SegviGen 原生是另一套划分(一致性 0.51),块最整但无名字、不受控。

## GeoSAM2 消融 / SAM2→SAM3:`geosam2_ablate.py`、`sam3_track.py`

不训练 GeoSAM2,固定它的提升(`lift_2dmask_3d`)与填充(`complete_labels`),只换 12 视角 2D 标签图的来源,回答"SAM2+LoRA+几何传播值多少、
能不能用 SAM3 取代"。结论与三个路线问题(DINOv3 微调 / SAM2 换 SAM3 / 逆向训练)的评估在 `REPORT_geosam2_ablation.md`。

| 模式 | 2D 标签图来源 |
|---|---|
| `sam3_pK` (K=1/2/4/12) | K 个均匀视角各跑一次 SAM3(概念库+用户提示),不传播;`sam3_p12` = SAM3 完全取代 SAM2 |
| `geo_pK` (K=1/2/4) | K 个视角的 SAM3 掩码作提示,GeoSAM2 自己的传播;K 次传播结果逐像素多数票 |
| `lift:<dir>` | 外部标签图目录,这里接 `sam3_track.py` 的输出 = SAM3 tracker(`facebook/sam3`,PE 骨干,RGB,无几何,零训练)传播 |

```bat
REM 1) SAM3 tracker 传播(SegviGen venv):对每个物体、每个 K,把锚视角+均匀分布的 K-1 个视角当提示,正反两个半圈各跑一次,重叠帧 logits 取平均
finetune\run_ft.bat sam3_track.py --renders datasets\geosam2\renders --out datasets\geosam2\results --objects <ids...> --n_prompts 1 2 4 --view match --ref_render_pattern "<正面图路径,含 {obj}>"
REM 2) 消融(GeoSAM2 venv,.venv_geosam2):所有模式共用同一条提升+填充
set OPENCV_IO_ENABLE_OPENEXR=1
.venv_geosam2\Scripts\python SegviGen\finetune\geosam2_ablate.py --renders datasets\geosam2\renders --out datasets\geosam2\results --objects <ids...> ^
    --modes sam3_p1 sam3_p2 sam3_p4 sam3_p12 geo_p1 geo_p2 geo_p4 lift:sam3track_p1_match lift:sam3track_p2_match lift:sam3track_p4_match
REM 3) 打分:hard 集用 eval_parts 对 GT;外部资产出结构统计 + 一致性 + ablation_front/back.png;tables 出 markdown
finetune\run_ft.bat ablation_score.py hard
finetune\run_ft.bat ablation_score.py ext
finetune\run_ft.bat ablation_score.py tables       REM -> datasets\geosam2\eval\ablation_tables.md
```

每行输出在 `<results>/<obj>/abl_<mode>_match/`:`faces.npy`(面标签)、`labels.json`(id→名字、原始留白、秒数)、`mesh.glb`。
两个修掉的问题:(a) K<12 时未提示视角原来会投 999 票压过有标签像素,现在按 alpha 全 0 处理;(b) GeoSAM2 `prepare_mesh_and_point_cloud`
先平移后缩放,与渲染器(先缩放后平移)不一致,bbox 中心不在原点的资产(dog)深度测试只剩 9% 的面通过——`geosam2_ablate.prepare_mesh` 用
正确顺序,`GeoSAM2/inference.py` 已同步修。hard 集(偏移 0)不受影响;`ext_bench` 里 dog/mickey/pineapple/shelf 的旧 GeoSAM2 行受影响。

结果(hard 20,mIoU):SAM3 ×1 直接提升 0.209 < 任何传播 0.246–0.276;SAM3 tracker p2 0.276 ≥ GeoSAM2 传播最好 0.254(几何分支在
语义部件上无可测增益);SAM3 ×12 直接投票 0.230 且最碎;v6+SAM3 仍最好(0.296,边界 F1 0.704 vs 提升类 ≤0.56)。外部资产上各提升行几乎无差别。
