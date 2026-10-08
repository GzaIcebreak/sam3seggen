# 接口调用说明（`serve_api.py`）

上传一个 GLB，得到按语义拆开、可选修复成封闭实体、带贴图的部件 GLB。本文是全部端点的参考。只接一条龙 `POST /pipeline` 的话看更短的 [pipeline_api.md](pipeline_api.md)。这里讲**怎么调**：地址、流程、每个端点的输入输出、错误码、示例代码。每个开关为什么是这个值、什么时候该改，见 [api_split_complete_bake.md](api_split_complete_bake.md) 的「开关说明」和「推荐的一条龙取值」。

三个入口读同一份契约 `pipeline.PipelineOptions`：HTTP 表单字段、Python 关键字参数、CLI 参数同名同义。改完代码要重启 `run_serve.sh`，否则 `/health` 还是旧进程。

## 1. 地址与限制

| 项 | 值 |
|---|---|
| 服务进程 | `./run_serve.sh`（默认端口 **6006**） |
| 本机访问 | `http://127.0.0.1:6006` |
| AutoDL 外网 | `https://<实例>.westb.seetacloud.com:8443`（自定义服务映射到 6006） |
| 交互式文档 | `/docs`（Swagger） |
| 并发 | GPU 一次只跑一单。`POST /pipeline` 排队（返回 `position`）；同步的 `POST /segment` 在忙时返回 **409** |
| 单次耗时 | 拆分 3–5 分钟；加修复通常 5–20 分钟，取决于部件数和有多少实例需要 HoloPart 对比 |
| 上传 | `glb` 必须是 multipart **文件**字段；建议 ≤ 30 万面。超过约 50 万面会把内存撑爆，请先减面 |
| 输出 | GLB（部件各一个节点，贴图内嵌 PNG）+ JSON 清单 |

**AutoDL 网关会掐掉长 POST**，所以对外请用 `POST /pipeline`：它 0.1 秒就返回票据，之后只有短请求。同步的 `POST /segment` 经外网常在几分钟后收到 404 / 连接断开——那不是任务失败，后台还在跑，用 `GET /jobs/latest` 找回。

## 2. 一条龙：三条命令跑通

输入模型和部件名，拿回拆分并修复好的模型。`POST /pipeline` **立刻**返回票据（202），任务排队在后台跑；轮询 `status_url`，`done` 后下载 `result_url`——那就是最终模型本身，不是链接清单。

```sh
HOST=http://127.0.0.1:6006

# 1) 提交（0.1 s 返回；不写 prompts 就按 主体 / 底座 拆）
curl -sS -X POST "$HOST/pipeline" \
  -F "glb=@monk.glb" \
  -F "prompts=armor, staff, base, body=head+face+hand+boot+leg" \
  -F "unassigned_to=body" | tee ticket.json
# {"job_id":"23a1…","state":"queued","position":1,
#  "status_url":"/jobs/23a1…","result_url":"/jobs/23a1…/result"}

# 2) 轮询，直到 state 为 done（拆分 + 修复通常 10–20 分钟）
JOB=$(python -c "import json;print(json.load(open('ticket.json'))['job_id'])")
watch -n 20 "curl -sS $HOST/jobs/$JOB | python -c \"import json,sys;d=json.load(sys.stdin);print(d['state'],d['stage'])\""

# 3) 下载最终模型：每个部件一个节点，封闭实体，贴图内嵌
curl -o repaired_parts.glb "$HOST/jobs/$JOB/result"
```

### 结果存在哪

- `curl -o repaired_parts.glb …` 存到**运行命令时的当前目录**（`-o` 是相对路径；在服务器终端里就是那个终端的 `pwd`，在本机跑就是本机当前目录）。要放到固定位置写绝对路径：`curl -o /root/autodl-tmp/out/repaired_parts.glb …`。
- 浏览器直接打开 `…/jobs/{id}/result`：响应头是 `Content-Disposition: attachment; filename="repaired_parts.glb"`，浏览器存进默认下载文件夹。`complete=off` 的任务文件名是 `parts.glb`。
- 不下载也能用：文件一直在服务器的任务目录里（第 9 节），`$SEGVIGEN_JOBS_DIR/{job_id}/complete/xpart_parts.glb` 就是 `/result` 返回的那个文件；开口部件是同目录下的 `parts.glb`。`job_id` 在票据里，丢了就 `GET /jobs`。
- 外网调用时地址换成 AutoDL 映射的 `https://<实例>.westb.seetacloud.com:8443`，路径不变。

要改开关就加一个 `options` 字段，内容是 JSON，键名和 `/segment` 的表单字段一样（第 5 节）：

```sh
-F 'options={"merge_gap": 0.01}'                     # 手被棍子切开的模型
-F 'options={"complete": "off"}'                     # 只拆不修，result 就是开口部件
-F 'options={"part_min_area_share": "装饰品=0.001", "fold_within_part": true}'   # 挂满小物件
```

## 3. 调用流程

```
POST /pipeline ──202──▶ {job_id, state: queued, position, status_url, result_url}
                              │
        GET status_url ◀──────┘  每 15–30 s 一次
              │ state=queued   排队中，position 是前面还有几单（含自己）
              │ state=running  stage 给进度：guidance → split → units → merge → complete → bake
              │ state=done  ──▶ GET result_url  ──▶ 最终 GLB
              └ state=error ──▶ error 字段是失败原因；中间产物仍在任务目录
```

- 任务**排队不拒绝**：GPU 忙时第二单返回 `queued` 和 `position`，按提交顺序执行。（旧的同步接口 `POST /segment` 在忙时仍然 409。）
- `GET result_url` 在 `queued` / `running` 时返回 **409**，`detail` 里带 `state` / `stage` / `position`，可以直接把它当轮询接口用；`error` 时返回 500。
- 提交后连接就结束了，网关超时的问题不再存在。丢了 `job_id` 就 `GET /jobs` 按 `filename` / `queued` 时间找。
- `stage` 是按盘上产物推出来的进度。`result` 在有修复时是 `complete/xpart_parts.glb`（封闭、已烘），`complete=off` 时是 `parts.glb`（开口）。
- 任务目录在 `SEGVIGEN_JOBS_DIR`，产物不会自动清理。

### Python 示例

```python
import json, time, requests

HOST = "http://127.0.0.1:6006"

def split_and_repair(glb_path, prompts="", unassigned_to=None, poll=20, **options):
    """上传 -> 排队 -> 轮询 -> 返回最终 GLB 的字节。"""
    data = {"prompts": prompts, "options": json.dumps(options)}
    if unassigned_to:
        data["unassigned_to"] = unassigned_to
    with open(glb_path, "rb") as f:
        ticket = requests.post(f"{HOST}/pipeline", files={"glb": f}, data=data, timeout=120)
    ticket.raise_for_status()                      # 400 = 选项写错，看 detail
    ticket = ticket.json()
    while True:
        status = requests.get(HOST + ticket["status_url"], timeout=30).json()
        if status["state"] == "done":
            break
        if status["state"] == "error":
            raise RuntimeError(f"job {ticket['job_id']} failed: {status['error']}")
        print(status["state"], status["stage"], status.get("position"))
        time.sleep(poll)
    model = requests.get(HOST + ticket["result_url"], timeout=300)
    model.raise_for_status()
    return model.content, status

glb, status = split_and_repair("monk.glb",
                               prompts="armor, staff, base, body=head+face+hand+boot+leg",
                               unassigned_to="body", merge_gap=0.01)
open("repaired_parts.glb", "wb").write(glb)
# 想看每个实例用了 X-Part 还是 HoloPart：
decisions = requests.get(HOST + status["links"]["complete_decisions"]).json()
```

### 同步接口 `POST /segment`（保留）

同一条管线，但请求会一直挂到跑完才返回 JSON（含 `parts` 清单和各下载链接），GPU 忙时 409。经 AutoDL 网关调用时长连接常被掐断，页面上显示 404 但任务仍在跑——此时用 `GET /jobs/latest` 找回。新接入请用 `/pipeline`。

```sh
curl -sS -X POST "$HOST/segment" --max-time 3600 -F "glb=@model.glb" | tee result.json
```

## 4. 端点参考

### `GET /health`

服务是否在、GPU 是否被占、所有开关的取值范围和**当前默认值**。客户端不该写死默认值，从这里读。

```json
{
  "status": "ok",
  "busy": false,
  "current_job": null,
  "latest_job": "a94e44a5…",
  "jobs_dir": "/root/autodl-tmp/christmas_tree_jobs",
  "pipeline": "segment_parts",
  "stages": [{"name": "paint", "always": true, "what": "…"}, …],
  "switches": {"complete": ["off", "boxes", "full", "hybrid"], "holopart_large": ["escape", "always", "score"], …},
  "defaults": {"merge": "name", "complete": "hybrid", "holopart_large": "score", "export_from": "source", "prompts": ["主体", "底座"], …}
}
```

### `POST /pipeline`（一条龙，推荐）

`multipart/form-data`：`glb`（必填）、`prompts`、`unassigned_to`、`options`（JSON 对象，键为第 5 节任意字段）。立刻返回 **202**：

```json
{"job_id": "23a1…", "state": "queued", "position": 1,
 "status_url": "/jobs/23a1…", "result_url": "/jobs/23a1…/result"}
```

选项写错返回 400，此时不会入队。

### `GET /jobs/{job_id}/result`

最终模型文件（`model/gltf-binary`）。有修复时是封闭、已烘贴图的部件（文件名 `repaired_parts.glb`），`complete=off` 时是开口部件（`parts.glb`）。任务未完成返回 **409**，`detail` = `{state, stage, position, status_url}`；任务失败返回 **500**。

### `POST /segment`（同步）

`multipart/form-data`。字段见第 5 节；只有 `glb` 必填。请求挂到跑完，成功返回 **200** 和结果对象（第 6.1 节）；GPU 忙时 409。

### `GET /jobs?limit=20`

最近任务，新的在前。

```json
{"busy": false, "current_job": null, "latest_job": "…",
 "jobs": [{"job_id": "…", "state": "done", "stage": "done", "filename": "monk.glb",
           "started": 1789476251.8, "finished": 1789476567.2, "error": null,
           "mtime": 1789476567.2, "links": {"download": "/jobs/…/download", "complete": "/jobs/…/complete", …}}]}
```

### `GET /jobs/latest`、`GET /jobs/{job_id}`

单个任务的摘要（同上一行的结构）加 `result`（任务完成后等于 `POST /segment` 的响应；未完成为 `null`）。`/jobs/latest` 在有任务运行时返回运行中的那单，否则返回最新一单。`job_id` 是 32 位十六进制；不存在返回 404。

### 下载类

| 方法 | 路径 | 内容 | 什么时候有 |
|---|---|---|---|
| `GET` | `/jobs/{id}/result` | **最终模型**：有修复时同 `/complete`，否则同 `/download`；未完成 409 | `done` |
| `GET` | `/jobs/{id}/download` | `parts.glb`，开口部件，每个部件一个节点，节点名 `part_00_<名字>` | 命名阶段之后 |
| `GET` | `/jobs/{id}/complete` | `complete/xpart_parts.glb`，修复后的封闭实体，已烘贴图，节点名与 `parts.glb` 对齐 | `complete≠off`，烘焙完成后 |
| `GET` | `/jobs/{id}/complete_raw` | 烘焙前的生成几何（看形状不看贴图） | 同上，早一步 |
| `GET` | `/jobs/{id}/complete_decisions` | `decisions.json`：每个实例的评分和最终选择 | `complete=hybrid` |
| `GET` | `/jobs/{id}/atoms` | `work/atoms.glb`，投票前的几何过分割原子 | 拆分之后 |
| `GET` | `/jobs/{id}/report` | `work/vote_report.json`，逐单元投票表 | 有提示词且 `merge≠off` |
| `GET` | `/jobs/{id}/guidance/{name}` | `work/guidance/*.png`，SAM3 掩码叠在渲染上的审阅图 | 有提示词 |
| `GET` | `/jobs/{id}/parts/{node}.glb` | 单个部件文件 | 仅旧路线 `parts_output=separate` |
| `GET` | `/jobs/{id}/map`、`/render` | 旧路线的 2D 引导图和条件渲染 | 仅 `POST /segment_legacy` |

产物不存在时返回 404，`detail` 说明缺哪个文件（例如 `job … has no complete/xpart_parts.glb`：修复没跑或还没跑完）。

### `POST /segment_legacy`（已弃用）

旧的 2D 引导路线（`front_view` / `assign` / `split_mode` / `use_v6`），只为复现旧结果保留，新接入不要用。

## 5. `POST /segment` 字段

未传的字段用 `/health` 里的 `defaults`。布尔按文本 `true` / `false` 传。Swagger 里留着灰色占位符 `string` 的字段会被当成没传。下表默认值是当前部署的值，以 `/health` 为准。

### 5.1 输入与命名

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `glb` | 文件 | **必填** | 源模型。单网格单材质时部件保留原 UV / 贴图；多网格或多材质会自动退回「重建网格 + 烘焙」 |
| `prompts` | 文本 | 空 → 自动提名 | 一句逗号分隔的部件名（中文逗号、顿号也行）。不传时由 `auto_prompts` 从概念库自动提出主体名和 2–6 个部件名（见 `work/auto_prompts.json`）；`auto_prompts=false` 时回到 `主体, 底座`。`名=概念+概念` 把多个概念收成一个部件，如 `body=head+face+hand+boot+leg`。输出部件数 = 名字数 |
| `unassigned_to` | 文本 | `body` | 没有掩码认领的单元并进这个部件。**必须是 `prompts` 里的名字**，否则被忽略、这些面从输出丢掉。用中文提示词时要显式传，如 `主体` |
| `merge` | `name` / `unit` / `fragments` / `off` | `name` | `name` 每个名字一个节点；`unit` 每个投票单元一个节点（排查取错名）；`fragments` 按几何留碎块只折回碎屑；`off` 不命名，按几何单元出匿名件 |
| `fragment_share` | float | `0.01` | 仅 `merge=fragments`：低于表面这么多的单元算碎屑 |
| `strict_parts` | bool | `false` | 某个名字完全没有掩码或没有面时整单失败；默认跳过该词继续 |
| `allow_partial` | bool | `true` | 与上面相反的写法；两者同传时 `strict_parts` 优先 |
| `reuse` | bool | `true` | 复用任务目录里已有的渲染和同提示词掩码（同一单内） |
| `auto_prompts` | bool | `true` | 没传 `prompts` 时：SAM3 过一遍概念库 280 个词（8 视角，约 1 分钟），挑主体名 + 2–6 个互不重叠的部件名，再走正常投票。`merge=off` 时不提名 |
| `mode` | `auto` / `smart` | `auto` | **`smart` = 智能分割模式**：提名阶段把候选词和 4 张渲染图交给视觉大模型（任何 OpenAI 兼容接口；当前部署 Qwen `qwen3.8-max`，也可换 Kimi），由它判断物体类别、剔除不属于该物体的词、给出主体名和部件名；只保留概念库里有的词，形状词（plank、panel…）也不要；单块掩码盖住剪影 60% 以上的部件词当整体词丢弃（按最大连通块判，多实例的 `shelf` 不受影响）；主体名 + 1 个部件词即可采用。需要 `SEGVIGEN_VLM_API_KEY`，没有则 400。Qwen 关闭思考时每次 2–4 秒 |

### 5.2 拆分（几何边界）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `granularity` | `fine` / `medium` / `coarse` | `medium` | 原子 / 单元的面数下限：150/300、300/600、800/1600。小件用 `fine`，大件被切成面板用 `coarse` |
| `min_atom_faces` / `min_unit_faces` | int | 跟粒度 | 显式覆盖 |
| `samples` | int | `7` | 无提示词过分割的采样次数 |
| `azimuth` / `azimuth_jitter` | float | `0` / `30` | 条件相机朝向与抖动 |
| `color_tol` | float | `20` | 同一采样内两种颜色算不算两块 |
| `mirror` | `auto` / `none` / `x` / `y` / `z` | `auto` | 沿对称面反射后一并求交 |

### 5.3 投票（语言只取名）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `view_azimuths` / `view_elevations` | 角度列表 | `45,135,225,315` / `10` | SAM3 投票视角 |
| `radius` / `resolution` | float / int | `2` / `512` | 投票用渲染 |
| `sam3_threshold` | float | `0.4` | 概念库下的掩码阈值（无库时 0.3） |
| `concept_bank` / `no_concept_bank` | 路径 / bool | 环境变量 | 换概念库，或退回原生 SAM3；不要同传 |
| `flat_paint` | `auto` / `on` / `off` | `auto` | 无贴图模型先平涂再给 SAM3 |
| `min_recall` | float | `0.5` | 掩码至少盖住单元这么多像素才认领 |
| `refine` / `refine_min_share` | `masks` / `off`，float | `off` / `0.1` | 投票后按面读掩码，把单元里一整块被别的名字认领的区域切出来。给「几何没分开、掩码分得开」的模型用 |

### 5.4 导出

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `export_from` | `source` / `remesh` | `source` | `source`：部件直接从原模型切，原分辨率、原 UV、原贴图，不烘。`remesh`：从重建网格切再烘（旧行为） |
| `with_texture` | bool | `true` | 封闭实体是否烘贴图；`export_from=remesh` 时也控制开口件是否烘。关掉只有占位色 |
| `texture_size` | int | `2048` | 小件贴图边长；面积 ≥ 8% 自动 ×2，≥ 40% 自动 ×4，封顶 8192 |

### 5.5 修复（封闭实体）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `complete` | `off` / `boxes` / `full` / `hybrid` | `hybrid` | `off` 不修复；`boxes` 只写盒子提示不占 GPU；`full` 只用 X-Part；`hybrid` X-Part 之后按 `holopart_large` 决定 |
| `holopart_large` | `escape` / `always` / `score` | `score` | `score`：每个 X-Part 实体和它的开口面比对打分，低分再跑 HoloPart 取高分，都差就保留开口面。`escape`：大件且超框 > 50% 才换。`always`：大件一律换 |
| `score_candidate` / `score_candidate_small` | float | `0.8` / `0.6` | 大件 / 小件低于此分才跑 HoloPart 对比。降低可以省时间 |
| `score_floor` | float | `0.3` | 两个后端都低于此分时保留开口面 |
| `condition` | `surface` / `box` | `surface` | X-Part 的条件点来自拆分归属面（推荐）还是盒内裁剪 |
| `min_area_share` | float | `0.005` | 低于表面占比的连通块折进最近大件，不单独生成 |
| `part_min_area_share` | `名=占比,…` | 无 | 按部件覆盖上一项，如 `装饰品=0.001` |
| `fold_within_part` | bool | `false` | 小块只折进同名部件 |
| `merge_gap` / `merge_max_share` | float | `0` / `0.05` | 把被别的部件切开的同名小块接回一件（`0.01` 适合握棍的手）；`0` 关。挨在一起的小物件会被接成一件，圣诞树类模型保持 `0` |
| `redraws` | int | `2` | 生成实体超出盒子时重抽次数，只留更贴盒的 |
| `octree_resolution` / `seed` | int | `512` / `42` | X-Part 重建分辨率与种子 |

## 6. 响应对象

### 6.1 `POST /segment` 成功响应

```json
{
  "job_id": "3f2a…",
  "seconds": 316.4,
  "pipeline": "segment_parts",
  "samples": 7,
  "atoms": 102,
  "parts": [ …见 6.2… ],
  "download": "/jobs/3f2a…/download",
  "complete": "/jobs/3f2a…/complete",
  "complete_raw": "/jobs/3f2a…/complete_raw",
  "complete_decisions": "/jobs/3f2a…/complete_decisions",
  "atoms_glb": "/jobs/3f2a…/atoms",
  "report": "/jobs/3f2a…/report",
  "guidance": ["/jobs/3f2a…/guidance/az45_el10.png", …],
  "files": [],
  "options": { …本次实际生效的全部开关… }
}
```

`complete*` 和 `report` 在没跑对应阶段时为 `null`。

### 6.2 `parts` 每一行（也是 `parts.json`）

| 字段 | 含义 |
|---|---|
| `label` | 标签序号，与 `prompts` 顺序一致 |
| `name` | 部件名 |
| `node` | GLB 里的节点名，`part_00_armor` |
| `part_color` | 该部件在 `atoms` / 审阅图里的颜色 |
| `faces` / `area` / `area_share` | 面数、面积、面积占比 |
| `texture_size` | 贴图边长；`export_from=source` 时是源贴图的边长 |
| `source` | `original` 表示从原模型切出（源贴图、原 UV）；没有该字段表示来自重建网格 + 烘焙 |

### 6.3 `decisions.json` 每一行（`complete=hybrid`）

| 字段 | 含义 |
|---|---|
| `node` / `name` / `instance` | 实例（同名部件的独立连通块各是一个实例，如两只手）；`node` 形如 `08_part_03_body`，与 `complete_raw` 里的节点对应 |
| `area_share` / `large` / `threshold` | 面积占比；是否算大件；用的候选阈值 |
| `xpart` | X-Part 实体的指标：`cover`（开口面被覆盖比例）、`fit_p90`、`extra_p90`（多余几何）、`escape`（出框率）、`largest_shell`（最大连通壳面积占比）、`intrusion`（切除后仍贴在别的部件开口面上的比例，部件互斥） |
| `xpart_cut` / `holopart_cut` | 打分前从实体上切掉的“长回来的邻居”面积占比（小狗身体上的第二条尾巴和爪子） |
| `q_xpart` | X-Part 得分 0–1 |
| `candidate` | 是否低于阈值、跑了 HoloPart 对比 |
| `holopart` / `q_holopart` | 候选实例才有 |
| `backend` | 最终采用：`xpart` / `holopart` / `open`（退回开口面） |
| `score` | 采用者的得分 |

`holopart_large=escape` / `always` 时行结构是旧的：`escape`、`large`、`backend`、`xpart_node`。

## 7. 错误码

| 码 | 原因 | 怎么办 |
|---|---|---|
| **400** | 枚举值不在范围内（`merge` / `complete` / `condition` / `holopart_large` / `refine` / `export_from` / `flat_paint` / `mirror` / `granularity`）、数字字段填了非数字、`part_min_area_share` 写法不对、`concept_bank` 与 `no_concept_bank` 同传 | 看 `detail`，对照 `/health` 的 `switches` |
| **404** | `job_id` 不存在；或该任务没有请求的产物（`detail` 会说缺哪个文件） | 用 `/jobs` 核对；修复类产物要等 `stage=done` |
| **409** | `POST /segment`：GPU 上已有任务在跑；`GET /jobs/{id}/result`：任务还没完成（`detail` 里有 `state` / `stage` / `position`） | 用 `/pipeline` 就不会因为忙而被拒；`/result` 的 409 当轮询继续等 |
| **422** | `glb` 不是文件字段（当成文本提交了） | `curl -F "glb=@file.glb"`；Swagger 用 Choose File |
| **500** | 某个阶段的子进程失败 | `detail` 是子进程的最后几行；任务目录里的中间产物会保留；`/jobs/{id}` 的 `error` 同样能看到 |

管线内部在提示词没有任何掩码时默认**跳过该词继续**（`allow_partial=true`），不报错；输出里就少那个部件，`/report` 和 `guidance` 能看出来。

## 8. 常见场景

下面的开关在 `/pipeline` 里都放进 `options` JSON；在 `/segment` 里是同名表单字段。

| 场景 | 传什么 |
|---|---|
| 不知道怎么填 | 只传 `glb`。部件名自动从概念库提出（如 `head, leg, body`），默认评分修复；加 `mode=smart` 让视觉大模型复核命名（更准，多几秒） |
| 人形 / 道具，想按语义拆 | `prompts=armor, staff, base, body=head+face+hand+boot+leg`，`unassigned_to=body` |
| 手握着棍子、被切成两半 | 上一行再加 `merge_gap=0.01` |
| 主体上挂满小物件（圣诞树） | `prompts=装饰品=bauble+star+bow+pinecone, 树=christmas tree`，`unassigned_to=树`，`part_min_area_share=装饰品=0.001`，`fold_within_part=true`；`merge_gap` 保持 `0` |
| 只要拆分，不要修复 | `complete=off` |
| 只看几何单元，不命名 | `merge=off`（可以不传 `prompts`；此时不自动提名） |
| 白模 / 灰模 | 默认 `flat_paint=auto` 会平涂；还不行传 `flat_paint=on` |
| 修复太慢 | `score_candidate_small=0.5` 或 `holopart_large=escape` |
| 模型 > 50 万面 | 先减面到 30 万左右再上传 |

## 9. 运维

```sh
# 启动 / 重启（改了代码必须重启）
pkill -f "sam3seggen/serve_api.py"; SEGVIGEN_JOBS_DIR=/root/autodl-tmp/jobs nohup ./run_serve.sh > serve.log 2>&1 &

# 确认跑的是新代码：看 defaults 里有没有新字段
curl -sS http://127.0.0.1:6006/health | python -c "import json,sys;print(json.load(sys.stdin)['defaults']['holopart_large'])"
```

| 环境变量 | 作用 |
|---|---|
| `SEGVIGEN_JOBS_DIR` | 任务目录；不设则用系统临时目录下 `segvigen_jobs/` |
| `SEGVIGEN_PY_SAM3` / `SEGVIGEN_PY_XPART` / `SEGVIGEN_PY_HOLOPART` | 三个子环境的 Python |
| `SEGVIGEN_XPART_ROOT` / `SEGVIGEN_HOLOPART_ROOT` 及 `_WEIGHTS` | 生成模型代码与权重位置 |
| `SEGVIGEN_SAM3` / `SEGVIGEN_DINOV3` / `SEGVIGEN_CONCEPT_BANK` | SAM3 权重与概念库 |
| `SEGVIGEN_VLM_API_KEY`、`SEGVIGEN_VLM_BASE_URL`、`SEGVIGEN_VLM_MODEL`、`SEGVIGEN_VLM_THINKING`（都可放仓库根目录 `.env`，已 gitignore） | 智能分割模式的视觉大模型。当前部署：DashScope `https://dashscope.aliyuncs.com/compatible-mode/v1` + `qwen3.8-max`，思考默认关（`THINKING=on` 打开，慢 30 倍、结果一样）。换 Kimi：base 改 `https://api.moonshot.cn/v1`，模型可不填（自动选支持图片的）；旧变量 `MOONSHOT_API_KEY` 仍可作 key 的兜底 |

任务目录布局（`/health` 的 `jobs_dir` 就是当前部署的 `$SEGVIGEN_JOBS_DIR`，本机部署是 `/root/autodl-tmp/christmas_tree_jobs`）：

```
$SEGVIGEN_JOBS_DIR/{job_id}/
  job.json                       # state / started / finished / error
  result.json                    # 完成后 = POST 响应
  input.glb
  parts.glb  parts.json          # GET /download
  work/
    views/  guidance/            # 渲染、审阅图
    atoms.glb  atoms.npy         # GET /atoms
    labels.npy  label_names.json # 每面标签（重建网格上）
    vote_report.json             # GET /report
  complete/
    boxes.json  open_instances.glb        # 交给生成器的实例
    xpart_instances.glb  holopart_instances.glb
    hybrid_instances.glb  decisions.json  # GET /complete_decisions
    xpart_parts_raw.glb                    # GET /complete_raw
    xpart_parts.glb                        # GET /complete = GET /result（有修复时）
```

产物不会自动清理；一单连中间产物约 100–300 MB，定期删旧的 `{job_id}/` 目录即可，服务不依赖它们。
