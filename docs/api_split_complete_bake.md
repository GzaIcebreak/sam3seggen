# 一条龙 API：拆分 → 修复 → 烘焙

一条 HTTP / Python / CLI 调用走完整主线：几何过分割（开口的 `parts.glb`），提示词只负责取名、可以不写；再用混合修复把切口重生为封闭实体（先 X-Part，按评分决定换 HoloPart 还是退回开口面），最后把源模型 albedo 烘回封闭实体。开口件默认直接从原模型切出，带原贴图，不需要烘。

三个入口读同一份契约：`pipeline.PipelineOptions`。`POST /segment` 的字段名与它对齐。改完代码必须重启 `run_serve.sh`，否则 `/health` 仍是旧进程。

## 管线说明

```
paint → guidance → split → units → merge → complete → bake
```

| 阶段 | 默认是否跑 | 产物 |
|---|---|---|
| 平涂 | 渲染饱和度太低才跑 | `work/views_flat/`（仅无色模型） |
| 引导图 | 有提示词才跑 | `work/guidance/*.png`、SAM3 掩码 |
| 拆分 | 是 | `work/atoms.glb`、多次 `full_seg` 求交 |
| 单元 | 是 | 连通分量 + 双壳合并 |
| 命名 | 有提示词且 `merge=name` | 开口、已命名的 `parts.glb`；默认从原模型切（`export_from=source`），原 UV / 贴图 |
| 修复 | `complete=hybrid`（默认） | `complete/xpart_parts_raw.glb`、`xpart_instances.glb`、`decisions.json` |
| 回烘 | 默认开，且 `with_texture=true` | `complete/xpart_parts.glb` |

`complete` 默认就是 `hybrid`：先跑 X-Part，再按 `holopart_large` 决定每个实例用谁。默认 `score`：把 X-Part 实体和它的开口面比对打分（0–1），大件低于 `score_candidate=0.8`、小件低于 `score_candidate_small=0.6` 的再跑一次 HoloPart，取分高的；两者都低于 `score_floor=0.3` 就保留开口面，不硬塞一个错的实体。`escape` 是旧规则（大件且超框 > 50% 才换），`always` 大件一律换。不想修复时显式传 `complete=off`。`full` 仍是纯 X-Part，便于对照。


## HTTP 一条龙

```sh
HOST=https://u1045120-bc7b-28ae0187.westb.seetacloud.com:8443

curl -X POST "$HOST/segment" \
  --max-time 3600 \
  -F "glb=@model.glb" \
  -F "prompts=head, torso, arm, hand, leg, foot" \
  -F "unassigned_to=torso" \
  -F "merge=name" \
  -F "condition=surface" \
  -F "granularity=medium" \
  -F "flat_paint=auto" \
  -F "with_texture=true" \
  -F "texture_size=2048"
```

不传 `complete` 就是 `hybrid`。只要拆不要修：`-F "complete=off"`。只要 X-Part：`-F "complete=full"`。

`prompts` 是**一句逗号分隔**的部件名（中文逗号、顿号也行）。名字里可以有空格。`body=head+face` 仍把多个概念收成一个输出节点。不传则默认提示词是 **主体、底座**（`merge` / `granularity` 仍用请求值，默认 `name` / `medium`）。布尔字段按 multipart 传 `true` / `false`。`glb` 必须是文件字段（`curl -F "glb=@model.glb"` 或 `/docs` 里用 Choose File）；当成普通文本提交会 **422**。`/docs` 里没填的可选框不要留着灰色的 `string`；服务会把 `string` / 空数字当成没传。

### 不写提示词

可以。不传 `prompts` 就按 **主体 / 底座** 去命名，再走默认 `complete=hybrid`。要头/躯干这种更细的名字才需要自己写提示词。

```sh
curl -X POST "$HOST/segment" --max-time 3600 \
  -F "glb=@model.glb"
```

得到 `主体`、`底座` 两个名字（`merge=name` 时同名会焊成一块）。只要拆不修：再加 `-F "complete=off"`。显式 `-F "merge=off"` 才按几何单元出匿名件。

成功响应里看这些字段：

| 字段 | 含义 |
|---|---|
| `job_id` | 后续下载用（32 位 hex） |
| `seconds` | 墙钟 |
| `parts` | 开口件清单（名字、面数、节点） |
| `options` | 本次实际生效的开关 |
| `download` | 开口、已命名的 `parts.glb` |
| `complete` | 烘过贴图的封闭实体；没跑修复则为 `null` |
| `complete_raw` | 烘焙前的生成实体 |
| `complete_decisions` | 每个实例的评分（`q_xpart` / `q_holopart`）和最终选择：X-Part / HoloPart / 开口面 |
| `atoms` / `report` / `guidance` | 过分割原子、投票表、审阅叠加图 |

| 方法 | 路径 | 内容 |
|---|---|---|
| `GET` | `/health` | 六步、开关、默认、GPU 是否占用、`current_job` / `latest_job` |
| `POST` | `/segment` | 上传 GLB + 选项 → 清单和下载链接 |
| `GET` | `/jobs` | 最近任务列表（网关超时丢了 `job_id` 时用这个找回） |
| `GET` | `/jobs/latest` | 最新一单的状态、阶段、下载链接 |
| `GET` | `/jobs/{id}` | 指定任务的状态（不存在才是真 404） |
| `GET` | `/jobs/{id}/download` | 开口 `parts.glb` |
| `GET` | `/jobs/{id}/complete` | 封闭已烘（混合结果） |
| `GET` | `/jobs/{id}/complete_raw` | 烘焙前的生成实体 |
| `GET` | `/jobs/{id}/complete_decisions` | `decisions.json` |
| `GET` | `/jobs/{id}/atoms` | 投票前原子 |
| `GET` | `/jobs/{id}/report` | 逐单元投票表（没写提示词时没有） |
| `GET` | `/jobs/{id}/guidance/{name}` | 审阅叠加图 |

AutoDL 自定义服务的网关会掐掉浏览器对 `POST /segment` 的长连接，页面上常显示 **404**。这只是代理断了，**后台任务还在跑**。此时不要重提（会 **409**），用下面接口拿回 `job_id`：

```sh
curl -sS "$HOST/health"          # busy / current_job / latest_job
curl -sS "$HOST/jobs/latest"     # 最新一单：state、stage、links
curl -sS "$HOST/jobs"            # 最近任务列表
```

`state` 为 `running` / `done` / `error` / `incomplete`。`stage` 是盘上推出来的进度：`accepted` → `guidance` → `split` → `units` → `merge` → `complete` → `bake` → `done`。`done` 之后再下 `links.complete`。

下载：

```sh
HOST=https://u1045120-bc7b-28ae0187.westb.seetacloud.com:8443
JOB=...   # POST 响应或 GET /jobs/latest 里的 job_id

# 开口件（拆分 + 可选命名 + 开口烘焙）
curl -O "$HOST/jobs/${JOB}/download"

# 修复后、已回烘（一条龙的最终封闭模型）
curl -o xpart_parts.glb "$HOST/jobs/${JOB}/complete"

# 修复后、烘焙前（看生成几何、不看贴图）
curl -o xpart_parts_raw.glb "$HOST/jobs/${JOB}/complete_raw"

# 每个实例的后端选择
curl -O "$HOST/jobs/${JOB}/complete_decisions"

# 投票前原子（部件切错时先看这里：边界不存在，改提示词也变不出来）
curl -O "$HOST/jobs/${JOB}/atoms"
```

磁盘上对应：

```
$JOBS_DIR/{job_id}/
  input.glb
  parts.glb                          # GET /download
  parts.json
  work/atoms.glb
  work/vote_report.json
  work/guidance/*.png
  complete/xpart_parts.glb           # GET /complete（已烘，混合结果）
  complete/xpart_parts_raw.glb       # GET /complete_raw
  complete/xpart_instances.glb       # 纯 X-Part 逐件实体
  complete/hybrid_instances.glb      # 混合后的逐件实体
  complete/holopart_instances.glb    # 仅当评分挑出候选、跑过 HoloPart
  complete/open_instances.glb        # 交给生成器的开口实例
  complete/boxes.json
  complete/decisions.json            # GET /complete_decisions
```

有提示词时，`xpart_parts.glb` 的节点和 `parts.glb` 对齐：按名字收成一组的独立件（两只手）先拆成实例生成，再合并回组。不写提示词时按 **主体 / 底座** 命名，和写了这两个词一样。

响应里的 `options` 是请求字段的默认快照。不传 `prompts` 时看日志里的 `[split] no prompts; using 主体, 底座`。

## Python / CLI 等价调用

```python
from pipeline import PipelineOptions
from segment_parts import segment_parts

opts = PipelineOptions(
    merge="name",
    condition="surface",
    granularity="medium",
    flat_paint="auto",
    with_texture=True,
    texture_size=2048,
)
segment_parts(
    "model.glb",
    ["head", "torso", "arm", "hand", "leg", "foot"],
    "out/parts.glb",
    work_dir="out/work",
    unassigned_to="torso",
    **opts.segment_kwargs(),
)
# 开口件：out/parts.glb
# 封闭已烘：out/complete/xpart_parts.glb
```

`PipelineOptions()` 的 `complete` 已是 `hybrid`，不用再写。

```sh
python segment_parts.py \
  --glb model.glb \
  --prompts "head, torso, arm, hand, leg, foot" \
  --unassigned_to torso \
  --out out/parts.glb --work_dir out/work \
  --merge name --condition surface
```

CLI 里跳过烘焙写 `--no_texture`（没有 `--with_texture` 这种正向 flag）。HTTP / Python 则是 `with_texture=false`。只要拆不修：`--complete off`。不写提示词：省略 `--prompts`（自动填 **主体、底座**，`merge` / 粒度仍用默认）。

```sh
python segment_parts.py --glb model.glb --out out/parts.glb --work_dir out/work
```

拆分很贵、与提示词无关；命名很便宜。有提示词时也可以先 `--merge off` 只看 `work/guidance/`，再用 `merge_parts.py` 命名。一条龙不需要拆成两步。

## 开关说明

未传的字段用 `GET /health` 里的 `defaults`。下面按阶段分组。

### 总控：一条龙必须看的三个

| 开关 | HTTP / Python | CLI | 默认 | 作用 |
|---|---|---|---|---|
| `merge` | `name` / `unit` / `fragments` / `off` | 同左 | `name` | `name`：每个提示词一个节点，同名大件会焊在一起。`unit`：每个投票单元一个节点，用来定位是谁取错名。`fragments`：按几何切开留下，只把碎屑折回邻件（门槛是 `fragment_share`）。`off`：不命名，每个几何单元一个节点，**仍然修复**。不传 `prompts` 时填 **主体、底座**，`merge` 保持请求值。 |
| `fragment_share` | float | `--fragment_share` | `0.01` | 只在 `merge=fragments` 生效：面积低于表面这么多的单元才算碎屑。更小更碎、保留更多件；更大折得更狠。`0` 等于不折。独立名字的小件（按钮、耳朵）仍会留下。 |
| `complete` | `off` / `boxes` / `full` / `hybrid` | 同左 | `hybrid` | `off`：不修复。`boxes`：只写盒子提示和预览，不占 GPU。`full`：只跑 X-Part 再烘。`hybrid`：X-Part 之后，大件超框换成 HoloPart，再烘。 |
| `with_texture` | `true` / `false` | `--no_texture` 关掉 | `true` | 封闭实体走 Blender 重 UV + selected-to-active 烘焙；开口件在 `export_from=source` 时直接带原贴图，不烘，只有退回 `remesh` 时才烘。关掉则部件只给占位色，不需要 bpy。**这不表示源模型有没有贴图**，只表示要不要烘。 |
| `export_from` | `source` / `remesh` | `--export_from` | `source` | `source`：标签从重建网格转到原模型的面上（最近面 + 多数平滑），部件直接从原模型切，原分辨率、原 UV、原贴图；源模型不是单网格单材质时自动退回 `remesh`。`remesh`：从重建网格切再烘，旧行为。 |
| `texture_size` | int | `--texture_size` | `2048` | **小件**底图边长。面积 ≥ 8% 升到 2×（默认 4096），≥ 40% 升到 4×（默认 8192），封顶 8192。贴图按 PNG 打进 GLB。 |

一条龙有提示词：`merge=name` + `complete=hybrid` + `with_texture=true`。觉得同名焊得太狠：`-F "merge=fragments"`，再用 `-F "fragment_share=0.02"` 调折回门槛。不写提示词：自动 **主体、底座**，`merge=name`，粒度仍是 `medium`。

### 拆分（几何边界）

| 开关 | 默认 | 作用 |
|---|---|---|
| `granularity` | `medium` | 两个尺寸下限一起设：`fine` 150/300、`medium` 300/600、`coarse` 800/1600（原子面数 / 单元面数）。小件（螺栓、按钮）用 `fine`；大件被切成面板时用 `coarse`。 |
| `min_atom_faces` / `min_unit_faces` | 跟粒度走 | 显式覆盖对应那一半。原子下限挡住采样碎屑；单元下限挡住「单独投票的碎片」。 |
| `samples` | `7` | 无提示 `full_seg` 求交次数。买到的是更少靠运气，不是更细。 |
| `azimuth` / `azimuth_jitter` | `0` / `30` | 条件相机朝向，以及额外采样在两侧抖多远。抖太大（例如 60°）会出现近乎空白的分区，meet 会沿着不存在的边界碎裂。 |
| `color_tol` | `20` | 同一样本内两种颜色算不算两块。降到 3 只会多碎斑，不增加真正的部件。 |
| `mirror` | `auto` | 把每个采样沿该平面反射后再求交。`none` 关掉。 |

### 命名（语言只取名）

| 开关 | 默认 | 作用 |
|---|---|---|
| `prompts` | 可空 | 输出节点名。空 = **主体、底座**。`name=concept+concept` 合并概念。 |
| `unassigned_to` | `body` | 没有任何掩码认领的单元并进这个名字。必须是 `prompts` 里已有的名（默认 `body`）；不在名单里或传空则这些面从输出丢掉。Swagger 占位符 `string` 当成没填。 |
| `flat_paint` | `auto` | 无色渲染先平涂再给 SAM3。见下一节。 |
| `min_recall` | `0.5` | 一张掩码至少盖住单元这么多像素才认领它。 |
| `view_azimuths` × `view_elevations` | `45,135,225,315` × `10` | SAM3 投票视角。四角 3/4，避免单侧漏耳；正对 90° 容易漏胸口，抬太高躯干会挡住腿脚。 |
| `radius` / `resolution` | `2` / `512` | 投票用渲染的相机距离和分辨率。 |
| `sam3_threshold` | `0.4` | 概念库阈值。关掉概念库后画笔实际是 0.3。 |
| `concept_bank` / `no_concept_bank` | 环境变量里的 v3 `bank.pt` | 指定 bank 路径，或退回原生 SAM3 词嵌入。两个不要一起传。 |
| `allow_partial` | `true` | 某个提示词完全没有掩码时跳过该词，其余继续。 |
| `strict_parts` | `false` | 反过来：缺掩码或缺面就整单失败。与 `allow_partial` 不要打架；HTTP 里 `strict_parts` 优先。 |
| `reuse` | `true` | 复用 `work/` 里已有渲染和同提示词掩码。换源模型后应 `reuse=false`（CLI：`--no_reuse`）。 |
| `refine` / `refine_min_share` | `off` / `0.1` | `masks`：投票后把掩码按面投影，单元里如果有一块连贯区域（≥ 单元面积的 `refine_min_share`）被别的名字认领，就沿这条边界把它切出来。给"几何没分开、掩码分得开"的情况用；猴子的手背和护腕在掩码里也是一体，它帮不上，所以默认关。 |

### 修复（混合：X-Part + 按需 HoloPart）

| 开关 | 默认 | 作用 |
|---|---|---|
| `condition` | `surface` | `surface`：从拆分归属面上采条件点（盒子里装着别人的腿也不会被当成躯干）。`box`：盒内裁剪，旧行为，只作对照。盒子始终会传，用来算 token / 超框。 |
| `min_area_share` | `0.005` | 表面占比低于此值的连通分量折进最近大件，不单独生成。碎片上 X-Part 不可靠。占比在去掉 remesh 内壁之后算。 |
| `part_min_area_share` | 无 | 按部件覆盖上面的门槛，写法 `名=占比,名=占比`（`装饰品=0.001`）。树上几百个彩球没有一个到 0.5%，不这样写会全部折进树枝。 |
| `fold_within_part` | `false` | 过小件只折进**同名**部件里最近的那块；同名一块都没留下时才折进别的部件。 |
| `merge_gap` / `merge_max_share` | `0` / `0.05` | 同一部件的两块表面距离小于模型对角线的 `merge_gap`、且至少一块面积低于 `merge_max_share` 时接成一件（传递合并）。猴子的手被棍子切成两个半只手，X-Part 把半只手生成成 3–5 倍大的板；接成整只手后一次修好。`0` 关。挨在一起的装饰品会被接成一件，圣诞树这类模型别开。 |
| `redraws` | `2` | 生成实体超出自己的盒子时，用同样条件重抽这么多次；只有更贴盒子的那一抽才会被采用。身份嵌入是随机的，偶发巨型件是抽签，不是提示词坏了。 |
| `octree_resolution` | `512` | X-Part 重建分辨率。 |
| `seed` | `42` | 生成种子。不能消掉身份嵌入的随机，只能让可复现的那部分固定。 |
| `py_xpart` / `xpart_root` / `xpart_weights` | 环境变量 | HTTP 不暴露路径；只在本机 CLI / Python 里改。 |
| `holopart_large` | `score` | `score`：按评分选后端（见下）。`escape`：大件且超框 > 50% 才换 HoloPart（旧规则）。`always`：大件一律换。 |
| `score_candidate` / `score_candidate_small` | `0.8` / `0.6` | `score`：大件（面积 ≥ 8% 或某轴 ≥ 源模 55%）/ 小件的 X-Part 实体低于此分，再跑一次 HoloPart 对比。小件 X-Part 本来就在行，问得少一些。 |
| `score_floor` | `0.3` | `score`：X-Part 和 HoloPart 都低于此分时保留开口面。 |
| `py_holopart` / `holopart_root` / `holopart_weights` | 环境变量 | 同上。没有候选实例时不会启动 HoloPart。HoloPart 实体保留到 20 万面（`holopart_complete --max_faces`），它自带脚本的 1 万面上限会把护甲抹成光壳。 |

评分（`hybrid_complete.py`）：把生成实体和它的开口面双向比对，距离按盒子对角线归一化。分 = 覆盖率（开口面采样点落在实体 2% 距离内的比例）×（1 − 多余几何 p90 / 0.2）×（1 − 出框率）× 最大连通壳的面积占比 ×（1 − 侵入率 / 0.1）。壳完整项是必需的：HoloPart 曾把一只手生成成 3012 个碎片，碎片全贴在原表面上，只看距离反而比完整的拳头分高。**侵入率**（部件互斥）：实体表面上“贴着别的部件的开口面、却离自己的开口面很远”的比例。X-Part 重生身体时会把邻居也长回来——小狗的身体修复件带着第二条尾巴和四只爪子（4.2% 的表面贴在尾巴部件上），多余几何 p90 对不到 10% 的小块是盲的、尾巴又在身体的盒子里，只有这一项抓得到。打分前先把这类区域切掉：裁掉的边缘够短（尾巴根、脚踝，≤ 1 倍对角线）就切并封口，边缘很长（护甲下的躯干皮肤）就留着；剩下的侵入再扣分，`decisions.json` 里 `xpart_cut` / `holopart_cut` 是切掉的面积占比。评分看不见"细节丢失"——一个光滑的护甲只要贴着原表面就能拿高分，所以 HoloPart 的面数上限一定要放开。每个实例的分数写在 `decisions.json`。

旧的 `escape` 规则只看超框：大件（面积 `0.08` / 轴向 `0.55`）超框 `0.5` 才换。圣诞树的树身 0% 出框却是个圆块（X-Part 0.70，HoloPart 0.89），它抓不到。

封闭实体的笼子上限仍是 `0.05` / `0.15`，但会按该件到源表面的中位距离收紧：贴得近的用开口件那档（`0.02` / `0.05`），只有飘得远的才用满上限。`texture_size` 同时作用于开口烘焙和封闭回烘，大件自动加像素。

## 如何分辨输入模型有无贴图

管线里有两件不同的事，不要混：

1. **源模型渲染出来有没有颜色** —— 决定要不要 `flat_paint`，让 SAM3 能看见部件。
2. **输出要不要把源 albedo 烘回去** —— 这是 `with_texture`，和源模型是否带贴图文件无关。白模也可以 `with_texture=true`，烘回去的只是灰白。

### 管线自己怎么判（`flat_paint=auto`）

不读 glTF 的 `baseColorTexture` 字段。它渲染固定视角网格，再算**轮廓内像素**的平均饱和度：

```
saturation = mean( max(R,G,B) - min(R,G,B) )   # 只统计 alpha > 16 的前景
```

实现：`flat_paint.is_colorless()`。门槛 `COLORLESS_SATURATION = 8.0`。

| 实测 | 平均饱和度 | 判定 |
|---|---|---|
| 无贴图机器人 | 0.25–0.42 | 无色，会平涂 |
| 有 albedo 的米奇 | 43–53 | 有色，跳过平涂 |

中间空了两个数量级，一般不用按模型调。日志里会打：

```
mean render saturation 0.31 (colourless, threshold 8)
mean render saturation 48.20 (textured, threshold 8)
```

`auto` 判成无色（或你强制 `flat_paint=on`）时，会拿**一次** `full_seg` 的部件色，重映射到互相远离的调色板，栅格化到同一组相机，写出 `work/views_flat/`。这层颜色只给 SAM3 取名用，不改几何。

| `flat_paint` | 行为 |
|---|---|
| `auto` | 饱和度 &lt; 8 才平涂（推荐） |
| `on` | 有贴图也平涂。贴图很灰、SAM3 抓不住时用 |
| `off` | 永不平涂。白模上 SAM3 常报 `no instance`，整片背面会掉进 `unassigned_to` |

渲染饱和度才是管线用的标准。提交前也可以先看文件：

**A. 有没有 albedo 贴图（glTF）**

```python
import trimesh

scene = trimesh.load("model.glb", process=False)
meshes = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
for mesh in meshes:
    visual = getattr(mesh, "visual", None)
    image = getattr(getattr(visual, "material", None), "baseColorTexture", None)
    print(mesh, "baseColorTexture", image is not None)
```

有 `baseColorTexture` 通常就是带贴图。爆炸图脚本 `data_toolkit/render_compare_sheet.py` 的 `has_texture()` 也是这个判断：有贴图才显示 albedo，否则用部件染色。

**B. 只有顶点色、没有贴图图**

`visual.kind == "vertex"` 时有逐顶点 RGB，渲染往往仍有饱和度，`flat_paint=auto` 会当成有色。烘焙走的是源表面的 albedo / Emission，顶点色不一定回得来。

**C. 白模 / 金属灰**

没有贴图、也没有可用的顶点色时，渲染饱和度接近 0。`auto` 会平涂。PBR 金属金等在 Diffuse 上几乎是黑的，回烘时 Blender 走 Emission 通道，避免烘成一片黑。

**D. 和 `with_texture` 怎么配**

| 源模型 | 建议 |
|---|---|
| 有真实 albedo | `flat_paint=auto`（会跳过平涂），`with_texture=true`，封闭件带回原贴图 |
| 白模 / 无色 | `flat_paint=auto`（会平涂，SAM3 才找得到头和躯干），`with_texture=true` 仍可开（烘回的是灰），或 `false` 省掉 Blender |
| 有贴图但看起来像灰模、命名失败 | 改 `flat_paint=on`，不要关 `with_texture` |

判断「有没有贴图」看渲染饱和度或 `baseColorTexture`；判断「输出要不要贴图」看 `with_texture`。

## 推荐的一条龙取值

默认值就是推荐组合：下面这张表是 `GET /health` 里 `defaults` 的意思，以及每一项为什么是这个值。不传就是这样跑。

### 默认组合（什么都不传时生效）

| 阶段 | 开关 | 默认 | 为什么 |
|---|---|---|---|
| 命名 | `prompts` | 空 → 自动提名（`auto_prompts=true`） | 不写提示词时 SAM3 过一遍概念库 280 个词，挑主体名 + 2–6 个部件名（小狗 `head, leg, body`，椅子 `backrest, chair leg, seat cushion`）。`auto_prompts=false` 回到 `主体, 底座` |
| 命名 | `merge` | `name` | 每个提示词一个节点，同名件焊在一起，输出部件数和提示词数一致 |
| 命名 | `unassigned_to` | `body` | 没有掩码认领的单元并进这个名字。**必须是提示词里的名字**，否则被忽略、无票面从输出丢掉（日志 `unassigned_to=... ignored`）；用中文或别的主体名时要显式传，如 `unassigned_to=主体` |
| 命名 | `mode` | `auto` | `smart`（智能分割模式）让视觉大模型（Qwen `qwen3.8-max`）先看图命名、SAM3 只测它给的词：比规则提名快 10 倍以上（不扫全库），形状词误认（跑车的 `wing`）也不会出现。它给的部件词里，单个最大连通掩码盖住剪影 60% 以上的（长剑的 `crossbar`）当整体词丢弃——按单块而不是按总面积判，六块隔板加起来过半的 `shelf` 仍是部件；主体名 + 1 个部件词（菠萝的 `fruit` + `leaves`）就够用，不再回退规则命名。只多几秒、需要 key；有 key 就建议开 |
| 命名 | `refine` | `off` | 按面切分单元只在"几何没分开、掩码分得开"时有用；猴子的手背在掩码里也是护腕，它帮不上，还会挪错小块 |
| 拆分 | `granularity` | `medium`（300 / 600 面） | 小件（螺栓、按钮）才降到 `fine`，大件被切成面板才升 `coarse` |
| 拆分 | `samples` / `mirror` | `7` / `auto` | 多采样买到的是少靠运气；对称求交免费 |
| 投票 | `view_azimuths` × `view_elevations` | `45,135,225,315` × `10` | 四个 3/4 视角两侧都看得到；正对或抬高会漏胸口、挡腿脚 |
| 投票 | `sam3_threshold` | `0.4` | 概念库下的门槛（无库 0.3） |
| 导出 | `export_from` | `source` | 部件直接从原模型切，原分辨率、原 UV、原贴图；重建网格上手部比原模型粗 4.5 倍，手指在那一步就没了 |
| 导出 | `with_texture` / `texture_size` | `true` / `2048` | 只对封闭实体和 `remesh` 退路生效；大件自动升到 4096 / 8192 |
| 修复 | `complete` | `hybrid` | 先 X-Part，再按评分决定 |
| 修复 | `condition` | `surface` | 条件点从拆分归属面采，盒子里装着别人的腿也不会被当成躯干 |
| 修复 | `min_area_share` / `redraws` | `0.005` / `2` | 碎片折进邻件；超框实体重抽两次、只留更贴盒的 |
| 修复 | `merge_gap` / `fold_within_part` / `part_min_area_share` | `0` / `false` / 无 | 都关。它们是给两类特殊模型开的，见下 |
| 修复 | `holopart_large` | `score` | 每个 X-Part 实体和开口面比对打分；只看出框率抓不到 0% 出框却修成圆块的树身 |
| 修复 | `score_candidate` / `score_candidate_small` | `0.8` / `0.6` | 大件多问一次 HoloPart（数量少、修坏代价大）；小件 X-Part 本来就在行，问得少（圣诞树上 0.8 全用会把一半装饰品白送去 HoloPart） |
| 修复 | `score_floor` | `0.3` | 两个后端都差就保留开口面，不硬塞错的实体 |

一句话：**只上传 glb**，或者只加一句 `prompts`，其余不动。

### 什么时候要改

| 模型长什么样 | 症状 | 改这些 |
|---|---|---|
| 主体上挂满几十上百个小物件（圣诞树、挂饰、按钮阵列） | 小物件全部被折进主体，只剩一个大框 | `part_min_area_share=小物件名=0.001`、`fold_within_part=true`；`merge_gap` 必须保持 `0`，否则挨着的小物件会被接成一件 |
| 小件被别的部件穿过、切成几段（握棍的手、穿过袖口的手臂） | 半只手被 X-Part 生成成 3–5 倍大的板 | `merge_gap=0.01`（同一部件、表面相距 < 对角线 1% 的小块接回一件） |
| 没有贴图的白模 / 灰模 | SAM3 什么都认不出、投票全 0 | `flat_paint=auto` 默认就会平涂；仍然不行时 `flat_paint=on` |
| 某个提示词整只丢了 | 输出少一个部件 | 先看 `work/guidance/*.png`：掩码没有就是提示词的事，换概念词；掩码有、面没有就是 `unassigned_to` 或 `min_recall` |
| 同名两件被焊在一起、想分开看 | 两只手一个节点 | `merge=unit` 或 `merge=fragments` 看单元级结果 |
| 修复太慢 | 一次几十分钟 | `score_candidate_small` 降到 `0.5`，或 `holopart_large=escape` 回到只在超框时换 |
| 修复后大件是光滑圆块 | 树身、躯干没细节 | 这是 X-Part 的局限；确认 `holopart_large=score`（会换 HoloPart），仍不满意就 `complete=off` 保留开口件 |
| 源模型超过 ~50 万面 | 内存暴涨、任务被杀 | 上传前先减面到 30 万左右（Blender Decimate），管线自身不减面；90 GB 的机器曾被 187 万面的树撑爆 |
| 源模型多网格 / 多材质 | 日志里 `source is not one textured mesh` | `export_from` 自动退回 `remesh` + 烘焙，不用改；想要原 UV 就先在 DCC 里合并成单网格单材质 |

### 三组写法

只上传 glb（自动提名 + 评分修复）：

```
# 实际生效：prompts=概念库自动提出（work/auto_prompts.json 里能看到），merge=name，
#           granularity=medium，complete=hybrid，holopart_large=score，export_from=source
# 不想自动提名：options={"auto_prompts": false} -> 主体, 底座
```

有名字的人形 / 道具（猴子这类）：

```
prompts=armor, staff, base, body=head+face+hand+boot+leg
unassigned_to=body
merge_gap=0.01          # 手被棍子切开时才需要；没有这种情况保持 0
# 其余全部默认
```

主体上挂满小物件（圣诞树）：

```
prompts=装饰品=bauble+christmas ball+star+bow+pinecone, 树=christmas tree+tree stand
unassigned_to=树
part_min_area_share=装饰品=0.001
fold_within_part=true
# merge_gap 保持 0；提示词避开 ornament / garland / tinsel 这类会盖住整棵树的词
```
