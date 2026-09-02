# SegviGen `full_seg_w_2d_map` 域适配微调(修 "SAM3 颜色进不了 SegviGen")

原模型的 2D 条件图是从 3D 真值直接光栅出来的像素级完美图;SAM3 给的是边缘毛糙、
有整件漏检、有灰色未分配区域、有语义合并的图。这里用 LoRA 让模型学会两件事:

1. 颜色跟着 2D 图走,边界贴到几何上(锯齿 / 溢出 / 斑点不进 3D)
2. 灰 = 未分配:2D 里整件是灰的,3D 里也输出灰(150,150,150),而不是乱猜一个色;
   2D 只覆盖了半件的,3D 里整件补全同色

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
基座冻结、bf16;LoRA 参数 fp32;默认开梯度检查点。RTX 5090 上 batch 2 峰值约 6.7 GiB、约 4 s/步。
`--lora_targets cross` 只训交叉注意力(更保守)。

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

- `fidelity`:该件表面上颜色等于 2D 图分给它的颜色(灰件则应为灰)的比例;`parts_correct(>=0.8)` 计数
- `purity`:该件表面上占主导颜色的比例,低 = 被撕碎;`other_share`:不属于任何图例色的比例
- 视角里完全不可见的件不计分(单独给 `hidden_mean_purity`)
- 输出 GLB 是 TRELLIS 的 Y-up 坐标(相对输入绕 X 转 90°),脚本自动在候选坐标系中选覆盖率最高的
- step-0 vs 训练后:`check_loss.json` 里 sam3/corrupt 与 clean 的差距应缩小
- 留出集建议 PartObjaverse-Tiny + 自己的资产(monk / dwarf / 1.glb)

基线(example.glb,基座 ckpt,sam3_az0):mean_fidelity 0.76,8/10 件正确;错的是 SAM3 给了红色却没进 3D 的小窗。
