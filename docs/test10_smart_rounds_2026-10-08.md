# 10 模型一条龙测试：智能分割模式（Qwen qwen3.8-max）两轮对比

日期 2026-10-08。全部走 `POST /pipeline`，只传 `glb` + `mode=smart`，其余默认（`complete=hybrid`、
`holopart_large=score`、`export_from=source`，概念库 v6 / 884 词）。每单结束后用
`xmas_split2/explode_render.py` 渲染爆炸图（`xmas_split2/smart10/<模型>/sheet.png` 第一轮，
`smart10_v2/` 第二轮），统计见 `xmas_split2/rounds_summary.json`。

## 两轮之间改了什么

第一轮跑完暴露了 4 个问题，修完（`f400fa5`…`58fca08`）重启服务跑第二轮：

| 问题 | 第一轮现象 | 修法 |
|---|---|---|
| 无贴图模型修复直接失败 | 长剑：源模型无贴图 → 退回重建网格导出，薄剑身被加厚，X-Part 形状一致性检查 11.7% > 5% 拒绝 | 单网格源模型不管有无贴图都从原网格切（`source_export.py`） |
| 身体修复件长出邻居 | 小狗：身体带第二条尾巴和四只爪子（4.2% 表面贴在尾巴部件上），多余几何 p90 看不到 <10% 的小块 | **部件互斥**：实体表面"贴别的部件开口面、离自己开口面 > τ"的区域，边缘短就切掉封口；残余侵入率进评分（`hybrid_complete.py`） |
| VLM 部件词被整体词规则误杀 | 置物架 shelf / rung、老鼠 head：掩码总面积过半被当整体词丢弃，回退规则命名 | 按单个最大连通块判（≥ 60% 才算整体词），六块隔板加起来过半不再误杀 |
| 主体 + 1 部件词被判不够 | 菠萝：Qwen 给 fruit + leaves 正确，却回退成 plant / plant pot | 有主体名时 1 个部件词即接受 |

## 逐模型结果

| 模型 | 第一轮 | 第二轮 | 变化 |
|---|---|---|---|
| chair | backrest / slat / seat / leg / 横撑（5） | slat（含靠背横梁）/ seat / leg / 横撑（4） | 提名相同，backrest 这次没投到面（采样随机） |
| human | head / arm / body / leg / foot | head / arm / torso / leg（+0.8% 碎片） | Qwen 把 foot 换成 torso；仍干净 |
| airplane | wing / propeller / cockpit / engine / airframe | 同 | 稳定，第二轮多 4 个实例走 HoloPart（2188 s vs 1079 s） |
| dog | ear / leg(爪) / tail / body（身体带第二条尾巴） | ear / leg / paw / tail / body，**身体不再长尾巴和爪子** | 修复目标达成；head 两轮都投不到面 |
| mouse | base / character / ear（ear 带走头顶） | base / body / ear / face | head 不再被误杀，脸分出来了 |
| pineapple | plant / plant pot（命名错） | leaves / body | 命名改对 |
| rack | cabinet（整架）/ wooden base（两块板） | frame / shelf（顶板）/ wooden base（5 板 + 底箱） | 架子正确分出；隔板命名仍不统一 |
| robot | body / boot（整腿）/ legs（膝甲） | head / arm / body / leg | 明显变好（Qwen 这次没提 armor） |
| car | car body / car wheel | body / wheel | 未变：hood / roof / door / bumper 掩码都在、分数 0.57–0.89，但整块车壳是一个几何单元，按多数票归 body |
| sword | bracket / main body / body（三块） | bracket / body（前后两半） | 未变甚至更差：薄件 SAM3 掩码盖满整剑，Qwen 的 handle / crossbar 被判整体词，回退规则命名 |

耗时：第一轮 701–1282 s/单，第二轮 777–2188 s/单。第二轮更慢的原因是侵入惩罚压低了 X-Part 分数，
触发了更多 HoloPart 对比（人形 3 个、飞机 4 个实例换成了 HoloPart）。

## 仍未解决

1. **几何单元粒度**：车壳、椅背横梁这类与主体连成一片的部件，SegviGen 的 7 次采样切不开，
   SAM3 掩码再准也投不出来。管线里的 `refine=masks`（按掩码切单元，默认关）正是对这个的，
   待在跑车上验证。
2. **薄件**（长剑）：SAM3 对薄长物体的每个词都给整体掩码，命名层无能为力，需要几何兜底。
3. **同名多实例不区分左右**，`head` 在狗上两轮都拿不到面（被 body 吃掉）。
4. **VLM 运行间差异**：同一模型两轮提名不同（human foot→torso，robot legs/armor/boot→head/arm/leg/boot），
   想要可复现就固定 `prompts`。
5. 第二轮更慢（见上），`score_candidate_small` 降到 0.5 可换回速度。

## 复现

```sh
# 提交（服务已带两轮之间的修复）
curl -sS -X POST localhost:6006/pipeline -F "glb=@/root/autodl-tmp/test10/dog.glb" -F "mode=smart"
# 爆炸图
/root/autodl-tmp/envs/trellis2/bin/python /root/autodl-tmp/xmas_split2/explode_render.py \
    --job /root/autodl-tmp/christmas_tree_jobs/<job_id> --out /tmp/dog --title dog --device cpu
```

## 第三轮：Qwen 先给候选词，SAM3 只测这些词（`6eb6fc6`）

第二轮的阶段计时显示每单一半时间花在"SAM3 过概念库 884 词 × 8 视角"（6–12 分钟）。第三轮把顺序反过来：
Qwen 先看 4 张渲染图和整张词表给出主体名 + 1–8 个部件词（2–4 秒），SAM3 只测这几个词（约 20 秒），
找不到的词丢掉；VLM 失败或一个词都找不到才回退全库扫描。结果在 `xmas_split2/smart10_v3/`。

| 模型 | 第二轮耗时 | 第三轮耗时 | 第三轮结果 | 对比第二轮 |
|---|---|---|---|---|
| sword | 909 s | 264 s | blade / hilt | **首次正确**：hilt 面积 14%，不再被整体词规则丢弃 |
| dog | 777 s | 424 s | body / ear / leg | 退：这次只切出 16 个几何单元，tail 零票、一条腿留在身体里 |
| mouse | 959 s | 504 s | base / body / ear / head / leg | 持平；ear 仍带走头顶（33%） |
| pineapple | 818 s | 347 s | body / leaves | 持平 |
| rack | 919 s | 293 s | frame / shelf | 进：隔板统一叫 shelf（上三块仍留在 frame 里） |
| human | 1506 s | 432 s | head / arm / hand / leg / foot / body | 进：六件全对 |
| robot | 1439 s | 424 s | head / torso / legs / body(双臂) | 持平 |
| chair | 1147 s | 455 s | backrest / slat / seat / leg / crossbar | 进：五件全对 |
| car | 1272 s | 732 s | body / wheel / windshield | 略进：多出挡风玻璃；hood / roof 仍投不出（车壳一体） |
| airplane | 2188 s | 731 s | fuselage / wing / propeller / engine / tail / seat / strut | 进：七件 |

- 耗时 264–732 s/单（第二轮 777–2188 s），提名阶段 373–732 s → 20–35 s。
- 命名质量总体持平或更好（长剑、人形、椅子、飞机、置物架变好；小狗因分割采样随机退了一次）。
- 仍未解决：几何单元切不开的车壳/靠背（`refine=masks` 在跑车上移动 0 面片，无效）；同一模型多次运行的随机性（SegviGen 采样 + VLM 提名）。

## 引导图一条龙（`POST /pipeline_guided`，开发分支 `part-exclusivity` 的 `634e86c`）

上传模型 + 一张"每个部件涂一种平色"的参考图，Qwen 对照它命名，SAM3 只测这些词。乐高人仔配 9 色 GT 图实测：
`head, arm, hand, leg, waist + torso`（waist SAM3 未找到被丢弃），正是 GT 的粒度。
颜色边界参与切几何（能分左右）是下一步：轮廓匹配找相机角度在对称模型上有镜像/前后歧义（人仔 IoU 0.86–0.88 平坦），
需要 VLM 辅助判定视角方向，或让用户在接口里直接给 `guide_azimuth / guide_elevation`。
