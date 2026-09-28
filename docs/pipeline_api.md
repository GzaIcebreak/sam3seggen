# Pipeline API 调用说明

一次调用完成「拆分 → 命名 → 修复成封闭实体 → 保留贴图」：上传一个 GLB 和部件名，拿回一个按部件分好节点、每个部件都是封闭实体、贴图内嵌的 GLB。

本文只讲 `POST /pipeline` 这条一条龙接口。全部端点、每个字段的取值和错误码见 [api_reference.md](api_reference.md)；开关为什么是这个值见 [api_split_complete_bake.md](api_split_complete_bake.md)。

## 1. 三步

```
① POST /pipeline          上传模型 + 部件名        → 立刻返回票据（202）
② GET  /jobs/{job_id}     每 15–30 秒问一次        → state 变成 done
③ GET  /jobs/{job_id}/result                        → 最终模型（GLB 文件）
```

```sh
HOST=http://127.0.0.1:6006        # 外网：https://<实例>.westb.seetacloud.com:8443

# ① 提交
curl -sS -X POST "$HOST/pipeline" \
  -F "glb=@monk.glb" \
  -F "prompts=armor, staff, base, body=head+face+hand+boot+leg" \
  -F "unassigned_to=body"
# → {"job_id":"23a10e71…","state":"queued","position":1,
#    "status_url":"/jobs/23a10e71…","result_url":"/jobs/23a10e71…/result"}

# ② 轮询
curl -sS "$HOST/jobs/23a10e71…"
# → {"state":"running","stage":"split", …}      还在跑
# → {"state":"done","stage":"done", …}          好了

# ③ 下载
curl -o repaired_parts.glb "$HOST/jobs/23a10e71…/result"
```

提交只用 0.1 秒，之后全是短请求，经 AutoDL 网关调用不会被掐断。一单通常 10–20 分钟（拆分约 5 分钟，修复按部件数和需要 HoloPart 对比的实例数浮动）。

## 2. 请求：`POST /pipeline`

`multipart/form-data`，只有 `glb` 必填。

| 字段 | 必填 | 说明 |
|---|---|---|
| `glb` | 是 | 模型文件。单网格、单材质的 GLB 效果最好（部件直接保留原 UV / 贴图）；多网格或多材质会自动退回「重建网格 + 烘焙」。建议 ≤ 30 万面，超过约 50 万面请先减面 |
| `prompts` | 否 | 部件名，**一句逗号分隔**：`armor, staff, base, body`。**不传则自动提名**：SAM3 过一遍概念库的 280 个词，挑出主体名和 2–6 个部件名（小狗 → `head, leg, body`；椅子 → `backrest, chair leg, seat cushion, body`），结果写在任务目录 `work/auto_prompts.json`。想回到 `主体, 底座` 传 `{"auto_prompts": false}`。名字里可以有空格；`名=概念+概念` 把多个概念收成一个部件：`body=head+face+hand+boot+leg` 输出一个叫 `body` 的部件。输出部件数 = 名字数 |
| `unassigned_to` | 否 | 没有任何提示词认领的面并进这个部件。**必须是 `prompts` 里的名字**，否则被忽略、这些面从输出丢掉。默认 `body`；用中文名时要显式传，如 `unassigned_to=主体`。一般给最大的那个部件 |
| `options` | 否 | JSON 对象，任何其他开关都放这里，键名见第 5 节：`{"merge_gap": 0.01}` |

提示词用 SAM3 认得的**常见英文名词**：`head`、`torso`、`arm`、`leg`、`wheel`、`window`、`leaves`、`fruit`、`base`。它认不出方位和序数（`left arm`、`upper blade`、`middle fruit`）——这类需求目前做不到，见第 7 节。

### 响应（202）

```json
{
  "job_id": "23a10e7122c844d095eea6606b94564b",
  "state": "queued",
  "position": 1,
  "status_url": "/jobs/23a10e7122c844d095eea6606b94564b",
  "result_url": "/jobs/23a10e7122c844d095eea6606b94564b/result"
}
```

`position` 是队列位置（含自己）；GPU 一次只跑一单，后来的排队，不拒绝。选项写错返回 **400**，此时不会入队。

## 3. 轮询：`GET /jobs/{job_id}`

```json
{
  "job_id": "23a1…",
  "state": "running",
  "stage": "complete",
  "position": null,
  "filename": "monk.glb",
  "started": 1789524903.5,
  "finished": null,
  "error": null,
  "links": {"download": "/jobs/23a1…/download", "atoms": "/jobs/23a1…/atoms", "report": "/jobs/23a1…/report"},
  "status_url": "/jobs/23a1…",
  "result_url": "/jobs/23a1…/result",
  "result": null
}
```

| `state` | 含义 | 客户端做什么 |
|---|---|---|
| `queued` | 排队中，`position` 是前面还有几单 | 继续等 |
| `running` | 在跑，`stage` 给进度：`guidance → split → units → merge → complete → bake` | 继续等 |
| `done` | 完成，`result` 是完整清单（同旧同步接口的响应） | 下载 `result_url` |
| `error` | 失败，`error` 是原因 | 中间产物仍在任务目录；常见原因见第 6 节 |

也可以直接对 `result_url` 轮询：没完成时它返回 409，`detail` 里带同样的 `state` / `stage` / `position`。

## 4. 下载：`GET /jobs/{job_id}/result`

返回 `model/gltf-binary`，`Content-Disposition: attachment; filename="repaired_parts.glb"`。

- 每个部件一个节点，节点名 `part_00_<名字>`，与 `prompts` 顺序一致；
- 部件是**封闭实体**（生成器修复了被切开的地方），贴图按 PNG 内嵌；
- 如果传了 `{"complete": "off"}`（只拆不修），返回的是开口部件，文件名 `parts.glb`。

其他可下载的东西（同一 `job_id`）：

| 路径 | 内容 |
|---|---|
| `/jobs/{id}/download` | 开口部件 `parts.glb`（修复前的拆分结果，原贴图） |
| `/jobs/{id}/complete_raw` | 修复后、烘贴图前的几何 |
| `/jobs/{id}/complete_decisions` | 每个实例的评分和采用的后端（X-Part / HoloPart / 保留开口面） |
| `/jobs/{id}/report` | 逐单元投票表——某个部件取错名时先看这里 |
| `/jobs/{id}/guidance/{name}.png` | SAM3 掩码叠在渲染上的审阅图 |

`curl -o` 存到运行命令的当前目录；浏览器直接打开存到默认下载文件夹。文件本身一直在服务器 `$SEGVIGEN_JOBS_DIR/{job_id}/complete/xpart_parts.glb`，不下载也能用。

## 5. `options` 常用键

放进 `options` 的 JSON 里，键名与 `GET /health` 的 `defaults` 一致；没写的用默认值。完整表见 api_reference.md 第 5 节。

| 键 | 默认 | 什么时候改 |
|---|---|---|
| `auto_prompts` | `true` | `false`：不传 `prompts` 时不自动提名，直接叫 `主体, 底座` |
| `complete` | `"hybrid"` | `"off"` 只拆不修；`"full"` 只用 X-Part 不比对 |
| `merge_gap` | `0` | 小件被别的部件穿过、切成几段（握着棍子的手）：`0.01` |
| `part_min_area_share` | 无 | 主体上挂满小物件（圣诞树上的球）：`"装饰品=0.001"`，否则小物件全被并进主体 |
| `fold_within_part` | `false` | 与上一条一起用：`true` |
| `merge` | `"name"` | `"off"` 不命名、按几何单元出匿名件（无需求拆件用这个，`prompts` 可以不传）；`"unit"` 每个投票单元一个节点，用来排查 |
| `granularity` | `"medium"` | 小件多用 `"fine"`；大件被切成面板用 `"coarse"` |
| `flat_paint` | `"auto"` | 白模 / 灰模识别不出时 `"on"` |
| `score_candidate_small` | `0.6` | 修复太慢：降到 `0.5` |
| `holopart_large` | `"score"` | `"escape"` 回到只在大件超框时才换 HoloPart，更快 |
| `export_from` | `"source"` | `"remesh"` 从重建网格切再烘贴图（旧行为） |
| `texture_size` | `2048` | 修复实体的烘焙贴图边长（小件；大件自动加倍） |

```sh
-F 'options={"merge_gap": 0.01}'
-F 'options={"complete": "off", "merge": "off"}'
-F 'options={"part_min_area_share": "装饰品=0.001", "fold_within_part": true}'
```

## 6. 错误

| 码 | 在哪一步 | 原因 | 处理 |
|---|---|---|---|
| 400 | 提交 | `options` 不是 JSON 对象；某个开关值不在范围内；`part_min_area_share` 写法不对 | 看 `detail`；对照 `GET /health` 的 `switches` |
| 422 | 提交 | `glb` 当成文本字段传了 | `curl -F "glb=@file.glb"`（带 `@`）；Swagger 用 Choose File |
| 404 | 轮询 / 下载 | `job_id` 不存在 | `GET /jobs` 列出最近任务找回 |
| 409 | 下载 | 任务还没完成 | `detail.state` 就是当前状态，继续等 |
| 500 | 下载 | 任务失败 | 同 `GET /jobs/{id}` 的 `error` |

`state=error` 的常见 `error`：

- `xpart_complete.py failed (exit -9)` 或 `(exit -15)`：进程被系统或内存看门狗杀掉，通常是模型太大内存撑爆——先减面到 30 万面左右再传；
- `… failed (exit 1); intermediates kept in job …`：某个阶段的子进程报错，服务器日志里有完整堆栈；中间产物在任务目录。

某个提示词 SAM3 完全没认出来**不算错误**：默认跳过它继续跑，输出里少那个部件；`/report` 和 `/guidance` 能看出是哪个词没有掩码。要它失败就传 `{"strict_parts": true}`。

## 7. 现在做不到的

- **方位 / 序数**：`left arm` / `right arm`、`upper blade`、`front left wheel`、`fruit 上中下三段`。SAM3 只认概念，左右手在它看来是同一个东西。会在几何层面另做（对称面、主轴等分），提示词里写了也没用。
- **同名多实例分开编号**：四个车轮都叫 `wheel`，现在出一个 `wheel` 节点。
- **细长薄件**：剑刃、横撑、螺旋桨这类渲染只有几像素宽的东西，掩码经常拿不到。
- **无提示词时的命名不总是对**：自动提名在 10 个测试模型上 7 个合理；形状像别的东西时会错（跑车被提出 `wing`，机甲被提出 `engine`），细长薄件（长剑）几乎提不出词。提名结果在 `work/auto_prompts.json`，不满意就把里面的词改好后作为 `prompts` 再提一单。

## 8. 客户端示例

### Python

```python
import json, time, requests

HOST = "http://127.0.0.1:6006"

def split_and_repair(glb_path, prompts="", unassigned_to=None, poll=20, **options):
    """上传 -> 排队 -> 轮询 -> 返回 (最终 GLB 字节, 任务摘要)。"""
    data = {"prompts": prompts}
    if unassigned_to:
        data["unassigned_to"] = unassigned_to
    if options:
        data["options"] = json.dumps(options, ensure_ascii=False)
    with open(glb_path, "rb") as f:
        ticket = requests.post(f"{HOST}/pipeline", files={"glb": f}, data=data, timeout=120)
    ticket.raise_for_status()                       # 400 = 选项写错
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

glb, status = split_and_repair(
    "monk.glb",
    prompts="armor, staff, base, body=head+face+hand+boot+leg",
    unassigned_to="body",
    merge_gap=0.01,
)
open("repaired_parts.glb", "wb").write(glb)
print([p["node"] for p in status["result"]["parts"]])
```

### JavaScript（浏览器 / Node 18+）

```js
const HOST = "http://127.0.0.1:6006";

async function splitAndRepair(file, prompts, unassignedTo, options = {}, poll = 20000) {
  const form = new FormData();
  form.append("glb", file);                          // File 或 Blob
  form.append("prompts", prompts);
  if (unassignedTo) form.append("unassigned_to", unassignedTo);
  if (Object.keys(options).length) form.append("options", JSON.stringify(options));

  const ticket = await fetch(`${HOST}/pipeline`, { method: "POST", body: form }).then(r => {
    if (!r.ok) throw new Error(`submit ${r.status}`);
    return r.json();
  });

  for (;;) {
    const status = await fetch(HOST + ticket.status_url).then(r => r.json());
    if (status.state === "done") break;
    if (status.state === "error") throw new Error(status.error);
    console.log(status.state, status.stage, status.position);
    await new Promise(res => setTimeout(res, poll));
  }
  return fetch(HOST + ticket.result_url).then(r => r.blob());   // GLB
}
```

### 只用 curl 的等待循环

```sh
HOST=http://127.0.0.1:6006
JOB=$(curl -sS -X POST "$HOST/pipeline" -F "glb=@model.glb" -F "prompts=head, torso, arm, leg" -F "unassigned_to=torso" \
      | python -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
until [ "$(curl -sS $HOST/jobs/$JOB | python -c "import json,sys; print(json.load(sys.stdin)['state'])")" = done ]; do sleep 20; done
curl -o repaired_parts.glb "$HOST/jobs/$JOB/result"
```

## 9. 与旧同步接口的关系

`POST /segment` 还在：同一条管线，但请求挂到跑完才返回 JSON 清单和链接，GPU 忙时 409，经网关常被掐断。新接入一律用 `/pipeline`；两者的任务都在 `GET /jobs` 里，下载端点通用。
