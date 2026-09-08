# 数据清洗接手文档：部件名重标注

面向接手这项工作的人或 agent，在**另一台服务器**上独立完成。
2026-09-08。工具在 `finetune/relabel/`（本文件所在仓库内，已进 git）。

前置阅读：数据格式 `DATA_concept_bank.md`、扩容背景与成本 `PLAN_data_expansion.md`。
不需要读 SegviGen 训练相关的文档——这项工作和模型训练完全解耦。

---

## 1. 你要解决的问题

SegviGen 的条件只有一张 2D 色图，模型不知道每块颜色代表什么部件。部件名（`names.json`）有两个用途：
**生成 SAM3 的提示词**，以及作为文本条件。

PartVerse 自带的名字是从 caption 里启发式抽取的，质量不能用。真实样本：

```
detailed three-dimensional model    cylindrical component    texture    part
representing the ground             highlighted component    various components
```

第一批 2000 个物体已经用 Grok 4.6 读 caption 重标 + 三轮人工复核解决（改动了 92 % 的名字，14 303 个部件全覆盖，`uncertain` 全部清零）。**你要做的是对新一批物体重复这个流程。**

**输入**：`batch_NNN.json`，每批约 50 个物体的 caption + 占位名字，纯文本约 200 KB。
**输出**：`out_NNN.json`，同结构的干净名字。
**验收**：自动校验 `problems: 0` + 抽样看图确认名字和掩码指的是同一块东西。

标注这一步**不需要任何 3D 数据或图片**，只读写文本。看图只在审阅阶段需要。

---

## 2. 数据在哪

三个 HF 私有仓，**已重标注的和未重标的分开存放**——把未复核的名字混进人工复核过的那批，会让
mIoU 0.368 这个基线失去可比性：

| 仓库 | 物体 | 名字状态 | 用途 |
|---|---|---|---|
| `Zaun1996/segvigen-pv-2view` | 2 000 | 已重标 + 三轮人工复核 | 训练 / 评测 / 留出集。**你不要动它** |
| `Zaun1996/segvigen-pv-raw` | 5 826 | 原始启发式，未清洗 | 你要清洗的数据本体（两视角渲染） |
| `Zaun1996/segvigen-relabel-work` | 5 826（同上） | — | 你的作业包（批次 / 审阅图 / 参考输出） |

这三个仓是固定的，后续来源以新归档形式加入（见 §8），不会新建仓。
注意**工具不在 HF 上**，在 git 仓 `finetune/relabel/`（§3）：规则改一次要让所有批次的校验口径同步变化，所以工具必须版本管理。

日常只需要第三个仓；只有要跑完整训练验证时才需要第二个。工作包（需要一个有读权限的 token）：

```bash
export HF_TOKEN=<读 token>
huggingface-cli download Zaun1996/segvigen-relabel-work --repo-type dataset --local-dir /data/relabel_work
cd /data/relabel_work && tar -xzf batches.tar.gz          # batch_NNN.json，标注只需要这个
cd /data && tar -xzf /data/relabel_work/review_views.tar.gz  # 渲染图 + ids.npy，审阅阶段需要
```

| 内容 | 体积 | 什么时候需要 |
|---|---|---|
| `batches.tar.gz` → `batch_NNN.json` | 4.4 MB | **标注阶段**（第 4 节） |
| `example_out_000.json` | 30 KB | 已通过校验的参考输出，先复现它（§4.3） |
| `review_views.tar.gz` → `pv_new/<id>/{names.json, captions.json, views/az0/{render.png, ids.npy}}` | 1.1 GB | **审阅阶段**（第 5 节） |
| `finetune/relabel/` | 仓库内 | 全程 |

目录结构（`--root` 指向 `pv_new`）：

```
pv_new/<object_id>/
  names.json        # ["head", "body", ...]  索引 = 部件 id。这是你要改的生产文件
  names_v1.json     # 旧名字备份，apply_names.py 首次写入时自动创建，可安全回滚
  names_meta.json   # {source, object, uncertain[]}
  captions.json     # PartVerse 原始 caption（只读，标注的依据）
  views/az0/render.png   # 512×512 RGBA 纹理渲染
  views/az0/ids.npy      # int16 512×512，值 = 部件 id，-1 = 背景，与 render.png 严格对齐
```

工作包里只有 `az0` 一个视角（审阅工具默认也只看 az0），`az135` 和 `parts/*.glb`、`input.glb` 留在本机，
因为审阅不需要它们。所以 4.2 GB 未压缩的内容打包后只有 1.1 GB。

---

## 3. 环境

只要 Python + 三个包，**不需要** GPU、bpy、torch、SAM3：

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy pillow
# check_names.py / prep_batches.py / apply_names.py 只用标准库
# review_sheet.py / screen_large.py 需要 numpy + pillow
```

`finetune/relabel/rules.py` 会自动找系统字体（Windows / DejaVu / Liberation / macOS）；一个都没有时退回 PIL 位图字体，图还是能出，只是字丑。Linux 上装 `fonts-dejavu-core` 更好看。

---

## 4. 标注阶段

### 4.1 命名口径（判断"标得对不对"的唯一标准）

1. **名字要回答"这是什么部件"，不是"它长什么样"。** `head`、`wheel`、`backrest`、`chimney` 对；`cylindrical component`、`detailed section` 错。
2. **1–3 个词，全小写，无标点。** 能用 1–2 个词就不用 3 个。
3. **禁止外观词和占位词**：颜色词，以及 `component / part / piece / section / element / object / model / representation / detailed / low-poly / texture / gradient / geometric / cylindrical / rectangular / spherical / circular`。完整清单在 `finetune/relabel/rules.py` 的 `JUNK`，那是唯一的来源，不要在别处再抄一份。
   看到 `cylindrical component` 要回 caption 里找它到底是什么（管子？腿？枪管？）。
4. **优先用长 caption 的最后一句。** 短 caption 常常只是视觉描述，长 caption 结尾才会说"这个部件对应整体的哪个部分"。
5. **同一物体内同类部件用同名。** 三条腿都叫 `leg`。只有 caption 明确写了方位才加 `left` / `front`，**不许自己编方位**。
6. **caption 说不清或为空时**，给一个保守的功能名词（`panel`、`base plate`、`support beam`、`body`），并把该部件的 `uncertain` 置 `true`。不允许留空。
7. 整体物体名写进 `object` 字段，也是 1–3 个词。

### 4.2 给标注模型的 prompt（照抄，把 `NNN` 换成批号）

一批 50 个物体，输入约 200 KB，正常 1–3 分钟完成。实际执行时并行发 5–8 个 subagent，每个负责一批。

```
You are relabeling 3D part names for a part-segmentation dataset (PartVerse). Read caption text and write ONE clean semantic part name per part. Do not modify any file other than the output file specified below.

## Input
Read this JSON file: /data/relabel_work/batch_NNN.json

Structure:
{
  "<object_id>": {
    "part_order": ["0", "1", ...],
    "parts": {
      "<part_id>": {
        "current_name": "...",
        "short_caption": "...",
        "long_caption": "..."
      }
    }
  }
}

## What a good part name is
A name that answers "what IS this part of the object". Good: head, torso, left arm, tail, wheel, seat cushion, chair leg, blade, handle, lid, window, door, upper shell, lower shell, roof, headlight.

Rules:
1. 1 to 3 words, lowercase, no trailing punctuation. Prefer 1-2 words when unambiguous.
2. Name the semantic identity, NOT appearance or rendering. NEVER use these or similar as the name: color words (red, teal, black...), "component", "part", "piece", "section", "element", "object", "model", "3d model", "representation", "view", "close-up", "image", "render", "low-poly", "detailed", "highlighted", "shape", "texture", "gradient", "design", "geometric", "cylindrical", "rectangular", "spherical", "circular". "cylindrical component" is WRONG - find what the thing actually is.
3. Use the long caption's final sentence(s) when they say what the part corresponds to.
4. If two parts of one object are the same kind of thing, give them the same name, disambiguated by position only if the caption states it. Do not invent positions.
5. Decide what the whole object is from all its part captions and write it as "object" (1-3 words).
6. If the captions do not say what a part is (including empty captions), choose the best geometric-function noun you can and set "uncertain": true. Never leave a name empty.
7. Preserve every object_id and part_id. Do not add, drop or reorder anything.

## Output
Write EXACTLY this JSON to: /data/relabel_work/out_NNN.json

{
  "<object_id>": {
    "object": "<1-3 word name for the whole object>",
    "parts": {
      "<part_id>": {"name": "<clean part name>", "uncertain": false}
    }
  }
}

Every object_id and every part_id from the input must be present; `uncertain` must be present (true/false) for every part. Valid JSON, UTF-8.

When done, reply with only: number of objects and parts written, and how many parts you marked uncertain.
```

### 4.3 参考批次（已跑通，可直接对标）

`batch_000` 已经用上面这段 prompt 实跑并通过全部校验，输出作为 `example_out_000.json` 放在 HF 仓里。
**先复现它的数字**，再开始批量作业——对不上说明 prompt 或模型配置有问题。

```
batch 000: 50 objects, 326 parts, 296 names changed, 52 uncertain, 0 junk-word
problems: 0
```

改动率 296/326 = 91 %，与第一批 2000 个物体的 92 % 一致。典型改动：

| 占位名字 | 重标后 |
|---|---|
| `cylindrical component` | `legging` |
| `textured fabric component` | `sleeve` |
| `resembling the sole` | `sole` |
| `spherical object` | `ball` |
| `human leg model` | `leg` |
| `character's hair` | `hair` |

注意最后两行：**去掉 `model`、`character's` 这类修饰，只留核心名词**，并且同一物体的三条腿都归到 `leg`
（该批 `leg` 出现 23 次、`leaf` 覆盖全部叶片）。这个「同类归并」是概念库能学到东西的前提，见 §7 关于唯一名字数的说明。

### 4.4 逐批校验

每批标完立刻校验，不要攒到最后：

```bash
python finetune/relabel/check_names.py --dir /data/relabel_work --batch 007
python finetune/relabel/check_names.py --dir /data/relabel_work --batch 007 --dump   # 逐条对照名字/旧名字/caption
```

检查项与通过标准：

| 检查项 | 通过标准 |
|---|---|
| 物体覆盖 | 输入输出物体数一致，无 missing / extra |
| 部件覆盖 | 每个 `part_id` 都在，无空名字 |
| `uncertain` 字段 | 每个部件都有（true/false） |
| 词数 | ≤ 3 |
| 大小写 / 空白 | 全小写、无首尾空格、无结尾句号 |
| 泛称/外观词 | 命中数应为 0（例外见 §6.1） |
| **`problems`** | **0** |

退出码非零表示有问题，可以直接用在驱动脚本里。

### 4.5 全量合并

全部批次通过后：

```bash
python finetune/relabel/check_names.py --dir /data/relabel_work --all
# -> /data/relabel_work/names_v2.json，并列出坏批次
```

**`bad batches: 0` 才能往下走。** 有坏批次时只重跑那一批（重新让标注模型读 `batch_NNN.json` 写 `out_NNN.json`），不要全量重来。

---

## 5. 审阅阶段（这一步不能省）

自动校验能证明名字**符合规则**，只有看图能证明名字**指的是那块几何**。第一批 2000 个物体的 257 处错误全部是自动校验查不出来的。

### 5.1 强制：主导部件筛查

这是投入产出比最高的一步，第一批 139 处错误来自这里，且是**最伤 SAM3 的一类错**。

```bash
python finetune/relabel/screen_large.py --root /data/pv_new \
  --names /data/relabel_work/names_v2.json --out /data/relabel_work/review/large
```

它筛出「az0 可见占比 ≥ 45 % 却没有 body 类名字」的部件，每个出一张高亮缩略图，拼成联络表 `sheet_NNN.png` + `flagged.json`。

**逐格看，判断标准**：高亮区域是否覆盖了整个物体 / 主体？

- 是 → 改成 `body`（或更准确的 `hull` / `tabletop` / `upper body` / `door` / `wall` / `base`）
- 否 → 保留。前景视角天然主导的正确局部名是存在的：太阳镜 `lens`、剑 `blade`、灯罩 `lampshade`

`BODY_LIKE` 白名单在 `rules.py`，觉得漏了合理的整体名可以加，但**加之前先看图确认**。

### 5.2 抽样看图

```bash
python finetune/relabel/review_sheet.py --root /data/pv_new \
  --out /data/relabel_work/review --n 30 --seed 42
# -> review/review.md，带渲染图、GT 掩码图、逐部件高亮缩略图
```

大批量建议抽 **5 %**（5800 个物体 ≈ 290 个，分 10 轮换种子做）。不给 `--seed` 是均匀间隔抽样、结果可复现；给种子是随机抽，用于换一批复审。

对着掩码图核对四件事：

1. **名字和掩码指的是同一块东西吗。** 最重要的一条。
2. **名字和整体物体自洽吗。** 物体是 `chair`，部件出现 `blade` 就要怀疑。
3. **同类部件是否同名。** 三条腿应该都叫 `leg`。
4. **`uncertain` 标得合理吗。** caption 为空却标 false，或 caption 写得很清楚却标 true，都要查。

### 5.3 改名字的正确做法

**改 `out_NNN.json`，不要直接改 `names.json`。** 原因：`names.json` 会被下一次 `apply_names.py` 覆盖，改在那里会丢；而 `out_NNN.json` 是可追溯的源头。

第一批的做法是把每轮修复写成一个一次性脚本（`fix_review_0903.py`、`fix_review_0904.py`），里面是 `(object_id, part_id, new_name)` 清单，直接改 `out_NNN.json`。这样修复历史可复查。建议照做，文件名带日期。

改完必须重跑：

```bash
python finetune/relabel/check_names.py --dir /data/relabel_work --all   # 重新合并
python finetune/relabel/apply_names.py --root /data/pv_new --names /data/relabel_work/names_v2.json --dry
python finetune/relabel/apply_names.py --root /data/pv_new --names /data/relabel_work/names_v2.json
```

`--dry` 先看改动数量。`names_v1.json` 只在首次写入时创建，重复跑 `apply_names.py` 不会破坏回滚副本。

---

## 6. 已知问题与处理办法

### 6.1 `orange` 等被判为颜色词（可能是误报）

`JUNK` 里有颜色词 `orange`，但第一批有 3 个部件真的是**橙子**（caption 原文 `A stylized, pixelated orange fruit with green leaves.`）。

**处理**：黑名单命中不等于错。回 caption / 掩码图核对再决定，确认无误就忽略。同类还有 `gold`（金子 vs 金色）、`silver`（银器 vs 银色）。

### 6.2 `uncertain = false` 不等于正确（雪人事件）

第一批物体 `0e61ed24`（雪人）的 p0 叫 `head`、p1 叫 `top hat`，都**没有**被标 uncertain，但显色复核发现 p0 其实是最大的底球（应为 `body`）、p1 是嘴部煤球（应为 `mouth`），真正的头是 p7。9 个部件全部重定名。

**教训**：清零 uncertain 的那一轮只扫 uncertain 部件是不够的。对「同一物体多个部件同名」或「语义别扭」的物体，要顺手把**全部**部件过一遍显色掩码。

### 6.3 对 SAM3 有歧义的词

`wing`（房子侧翼 vs 鸟翅膀）、`boot`（汽车后备箱 vs 靴子）、`temple`（眼镜腿 vs 寺庙）在建筑/汽车语境下没错，但 SAM3 很可能理解成另一个意思。

**处理**：这类词只在跑 SAM3 后才暴露。先照常标注，等下游检查绑定率时若某个词异常低，改用「物体名 + 部件名」的组合提示词（例如 `house wing`）。**不要**为此在标注阶段就把名字改错。

### 6.4 `apply_names.py` 报 "old has N names, new has M names"

某个物体的部件数对不上，脚本会跳过并打印。原因通常是该物体的 `captions.json` 和 `names.json` 长度本来就不一致——**数据生成阶段就有问题**。

**处理**：应该把该物体**排除出数据集**，而不是硬改名字。检查 `captions.json` 的 `source_part_ids` 长度、`names.json` 长度、`parts/` 里的 glb 数量，三者不一致就删掉这个物体。

### 6.5 某批 `out_NNN.json` 缺失或校验不过

`check_names.py --all` 会列出坏批次，形如 `('013', 'missing')` 或 `('021', '3 problems, e.g. [...]')`。只重跑那一批。

### 6.6 回滚

```bash
# 单个物体
cp /data/pv_new/<id>/names_v1.json /data/pv_new/<id>/names.json
# 全部
for d in /data/pv_new/*/; do [ -f "$d/names_v1.json" ] && cp "$d/names_v1.json" "$d/names.json"; done
```

---

## 7. 交回来什么

只要两个文件 + 修复脚本，不用回传任何图片或 3D 数据：

| 交付物 | 说明 |
|---|---|
| `names_v2.json` | 合并后的最终名字（这是核心产物） |
| `out_000..NNN.json` | 逐批原始输出，含修复后的内容，可追溯 |
| `fix_review_<日期>.py` | 每轮人工修复清单 |
| 一份简短报告 | 见下面的质量基线表 |

报告里要有这几个数（对照第一批的基线）：

| 指标 | 第一批（2000 物体）基线 | 你这批 |
|---|---|---|
| 物体 / 部件 | 2000 / 14 303，全覆盖 | |
| 名字被改动 | 13 156 个（92 %） | |
| 唯一名字数 | 1554（其中 678 个只用过一次，占 44 %；出现 ≥ 8 次的 299 个） | |
| `uncertain` | 0（全库清零） | |
| 自动校验 problems | 0 | |
| 泛称词命中 | 3，均为误报 | |
| 人工复核覆盖 | 抽样 12 物体 + 全库主导部件筛查 316 个候选 + 全部 1054 个 uncertain 部件 | |

**「唯一名字数」和「出现 ≥ 8 次的名字数」要报**：概念库的 `--min_count 8` 只学高频名字，第一批是 299 个名字覆盖约 82 % 的部件实例。如果你这批几乎全是一次性名字，说明命名过于具体（把 `leg` 写成 `left rear chair leg`），概念库学不到东西。这比 problems = 0 更能反映质量。

---

## 8. 下一批数据也走同一条流程

这批 5826 个物体做完之后还有两个来源要清洗，**流程和本文档完全一致**，只有两处不同。
完整的七步模板、仓库命名约定与每个来源的差异见 `PLAN_data_expansion.md` §8。

| 来源 | 物体量 | 与本批的差异 |
|---|---|---|
| **PartVerse-XL** | 25 406 净新增 → 约 14 700 可用（294 批） | 无差异。caption 来源、目录结构、工具全部相同。**导入时必须只喂 `xl_new_ids.txt`**，因为 XL 会把我们已人工标好的 1 245 个物体重新标注，整包导入会静默冲掉那批人工成果 |
| **PartNeXt** | 23 519 | **不走重标，走校验。** 它的部件名是人工标的，直接用 `check_names.py` 全量校验，只把不通过的部件送去改写。预计主要问题是词数超 3（层级路径拼接）和粒度过细 |

**仓库不会变，还是这三个**（见 §2）。新来源以新归档的形式加进去，文件名带来源前缀：
`segvigen-pv-raw` 里多一个 `xl_raw_2view.tar.gz`，`segvigen-relabel-work` 里多一个 `xl_batches.tar.gz`，
并各带一份 `ids_xl.txt`。你清洗完的成果作为新归档进 `segvigen-pv-2view`，**不要重新生成已有的 `cb_data.tar.gz`**。

## 9. 三条纪律

1. **每一步的通过标准都是 `problems: 0` 和坏批次为空。** 不要在有坏批次的情况下往下走。
2. **不要碰已复核的那 2000 个物体**（`segvigen-pv-2view` / 本机 `datasets/pv/`）。它们是所有指标的对照基线和留出集来源。你只在未清洗的那个 root 上工作。
3. **改名字改在 `out_NNN.json`**，不是 `names.json`。后者会被覆盖。
