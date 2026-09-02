# 组合提示词与四组件 GLB 设计

## 目标

扩展 `segment_api.py`，使 Python API 和 CLI 都能使用单个提示词、多个独立提示词，以及将多个 SAM3 概念合并为一个输出组件。组件名称、数量和组合关系全部由每次调用动态决定，不在接口中写死。`monk.glb` 的四组件输出仅作为验收样例。

## 接口

Python API 的 `prompts` 接受：

- 单字符串：`"armor"`
- 字符串序列：`["armor", "staff", "base"]`
- 命名组合：`"body=head+face+hand+boot+leg"`

CLI 保持 `--prompts` 多参数形式，并使用相同的 `name=a+b+c` 组合语法。

可选的 `unassigned_to` 参数指定由哪个动态组件吸收所有未被 SAM3 提示词覆盖的前景。例如 `unassigned_to="body"` 会归入本次请求中的 `body`，避免产生额外的 `<unassigned>` 节点。

接口先把输入统一解析为动态组件规格：

```python
[
    {"name": "armor", "concepts": ["armor"]},
    {"name": "body", "concepts": ["head", "face", "hand", "boot", "leg"]},
]
```

上述结构由调用参数生成；接口不包含 `armor`、`body` 等业务名称判断。调用者以后可以任意指定例如：

```python
["roof", "opening=door+window", "wall=facade+brick"]
```

## 数据流

1. 将单字符串归一化为单元素列表，校验提示词和输出名称。
2. 用正面方位渲染输入模型。
3. SAM3 对组合中的每个唯一概念分别推理。
4. 将同一命名组合内的多个 mask 做并集，得到一个组件 mask。
5. 按面积由小到大解决 mask 重叠，并把未覆盖前景归入 `unassigned_to` 动态指定的组件。
6. SegviGen 根据四颜色 2D map 生成四类表面标签。
7. 对四类面片进行平滑、孤岛修正、拆分和贴图重烘焙。
8. 导出一个 GLB，包含四个 Mesh 节点。

## 输出约束

严格模式下，期望的组件名称和数量从本次 `prompts` 动态推导。最终 manifest 必须与动态规格完全一致。

任一目标提示词未产生有效 mask、重烘焙后缺少组件、出现额外组件或节点名重复时，接口应明确报错，不静默返回不完整模型。校验逻辑不得引用 monk 样例中的任何固定名称。

## monk 测试

调用参数：

```text
prompts = [
  "armor",
  "staff",
  "base",
  "body=head+face+hand+boot+leg"
]
unassigned_to = "body"
azimuth = 135
with_texture = True
```

验证内容：

- 单字符串与字符串列表都能被正确归一化。
- 组合提示词只产生一个 `body` legend 项。
- 未分配前景归入 `body`，不产生第五个灰色组件。
- 输出 GLB 只有四个 Mesh 节点，名称与请求一致。
- 每个节点包含面片和贴图。
- 输出正面、侧面和背面三张检查图。
- 额外使用与 monk 无关的任意名称验证解析和严格校验均为动态逻辑。

## 错误处理

- 空提示词、空组合或重复输出名称：在启动模型推理前报错。
- `unassigned_to` 不属于请求组件：在启动模型推理前报错。
- SAM3 未识别某个组件：终止并列出缺失名称。
- 最终 manifest 与请求不一致：保留工作目录和中间文件并报错，便于调参。

## SAM3-only 审核模式

接口增加 `sam3_only=True`，CLI 对应 `--sam3_only`。该模式仅执行条件视图渲染与 SAM3 二维分割，输出：

- `render.png`
- `sam3_2d_map.png`
- `sam3_2d_map_legend.json`

它仍执行动态提示词解析、`unassigned_to` 校验、缺失组件检查和严格 legend 名称校验，但不会加载 SegviGen、运行 3D 推理、拆件或调用 Blender 烘焙。

Python API 返回结构化审核结果：

```python
{
    "render": ".../render.png",
    "map": ".../sam3_2d_map.png",
    "legend": [...],
}
```

调用方审核二维 map 后，可用相同提示词和方位角启动完整流水线。审核模式不包含任何固定组件名称。
