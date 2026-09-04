"""Self-contained HTML status page for a finetune/train.py run.

Charts are hand-drawn inline SVG, so the page needs no network, no CDN and no browser
extensions: open the file and it renders. With --watch the page is regenerated on an
interval and carries a meta-refresh, which turns it into a live dashboard while training.

    finetune\run_ft.bat report_html.py --run finetune\runs\pv_v1 ^
        --check0 finetune\runs\pv_check0\check_loss.json ^
        --fidelity <obj>\variants\sam3_az0\fid_base.json ^
        --out E:\data\pv_v1.html --watch 60
"""
from __future__ import annotations

import argparse
import datetime
import html
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_stats import KINDS, checkpoints, read_checks, read_fidelity, read_log, read_run_args, step_times

KIND_COLOR = {"clean": "#6c8ebf", "corrupt": "#d79b5e", "sam3": "#63a86e"}
INK = "#e6e6e6"
MUTED = "#8b8b8b"
GRID = "#2c2c2c"

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 32px 48px; background: #151515; color: #e6e6e6;
       font: 14px/1.55 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
h1 { font-size: 21px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 15px; margin: 34px 0 10px; font-weight: 600; }
.sub { color: #8b8b8b; font-size: 12px; margin: 0; }
.stats { display: flex; gap: 34px; flex-wrap: wrap; margin: 22px 0 4px; }
.stat b { display: block; font-size: 23px; font-weight: 600; letter-spacing: -0.02em; }
.stat span { color: #8b8b8b; font-size: 12px; }
.good b { color: #63a86e; }
.warn b { color: #d79b5e; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
.panel { border: 1px solid #2c2c2c; border-radius: 8px; padding: 14px 16px 10px; }
.panel h3 { font-size: 13px; margin: 0 0 2px; font-weight: 600; }
.panel p { color: #8b8b8b; font-size: 11.5px; margin: 2px 0 8px; }
.note { color: #8b8b8b; font-size: 11.5px; margin: 8px 0 0; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid #262626; font-size: 13px; }
th:first-child, td:first-child { text-align: left; }
th { color: #8b8b8b; font-weight: 500; font-size: 11.5px; text-transform: uppercase;
     letter-spacing: 0.04em; }
tr:last-child td { border-bottom: none; }
.callout { border: 1px solid #3d3323; background: #1e1a13; border-radius: 8px;
           padding: 12px 16px; margin-top: 30px; }
.callout b { color: #d79b5e; }
code { background: #202020; padding: 1px 5px; border-radius: 4px; font-size: 12.5px;
       font-family: "Cascadia Mono", Consolas, monospace; }
svg text { font-family: "Segoe UI", system-ui, sans-serif; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def axes_frame(w: int, h: int, pad: dict, y_lo: float, y_hi: float, x_label: str, y_label: str,
               fmt: str = "{:.3f}") -> list[str]:
    """Y grid lines with labels plus axis titles; returns SVG fragments."""
    out = []
    for i in range(5):
        frac = i / 4
        y = pad["t"] + (h - pad["t"] - pad["b"]) * (1 - frac)
        val = y_lo + (y_hi - y_lo) * frac
        out.append(f'<line x1="{pad["l"]}" y1="{y:.1f}" x2="{w - pad["r"]}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{pad["l"] - 8}" y="{y + 3.5:.1f}" fill="{MUTED}" font-size="10" '
                   f'text-anchor="end">{fmt.format(val)}</text>')
    out.append(f'<text x="{(w + pad["l"] - pad["r"]) / 2:.0f}" y="{h - 4}" fill="{MUTED}" '
               f'font-size="10.5" text-anchor="middle">{esc(x_label)}</text>')
    out.append(f'<text x="12" y="{(h - pad["b"] + pad["t"]) / 2:.0f}" fill="{MUTED}" font-size="10.5" '
               f'text-anchor="middle" transform="rotate(-90 12 {(h - pad["b"] + pad["t"]) / 2:.0f})">'
               f'{esc(y_label)}</text>')
    return out


def line_chart(xs: list[float], ys: list[float], x_label: str, y_label: str, color: str,
               w: int = 520, h: int = 240, ref: tuple[float, str] | None = None,
               value_fmt: str = "{:.3f}") -> str:
    if not xs:
        return ""
    pad = {"l": 62, "r": 16, "t": 12, "b": 34}
    y_lo, y_hi = min(ys), max(ys)
    if ref:
        y_lo, y_hi = min(y_lo, ref[0]), max(y_hi, ref[0])
    span = (y_hi - y_lo) or max(abs(y_hi), 1e-6)
    y_lo, y_hi = y_lo - span * 0.08, y_hi + span * 0.08
    x_lo, x_hi = min(xs), max(xs)
    x_span = (x_hi - x_lo) or 1

    def px(x): return pad["l"] + (w - pad["l"] - pad["r"]) * (x - x_lo) / x_span

    def py(y): return pad["t"] + (h - pad["t"] - pad["b"]) * (1 - (y - y_lo) / (y_hi - y_lo))

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img">']
    parts += axes_frame(w, h, pad, y_lo, y_hi, x_label, y_label, value_fmt)
    if ref:
        parts.append(f'<line x1="{pad["l"]}" y1="{py(ref[0]):.1f}" x2="{w - pad["r"]}" '
                     f'y2="{py(ref[0]):.1f}" stroke="#6c8ebf" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{w - pad["r"] - 4}" y="{py(ref[0]) - 5:.1f}" fill="#6c8ebf" '
                     f'font-size="10" text-anchor="end">{esc(ref[1])}</text>')
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    for x in (x_lo, (x_lo + x_hi) / 2, x_hi):
        parts.append(f'<text x="{px(x):.1f}" y="{h - pad["b"] + 14}" fill="{MUTED}" font-size="10" '
                     f'text-anchor="middle">{x:.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bars(categories: list[str], series: list[tuple[str, list[float], str]],
                 x_label: str, y_label: str, w: int = 520, h: int = 260,
                 y_max: float | None = None, ref: tuple[float, str] | None = None) -> str:
    if not categories:
        return ""
    pad = {"l": 62, "r": 16, "t": 14, "b": 50}
    values = [v for _, data, _ in series for v in data if v is not None]
    y_hi = y_max if y_max is not None else max(values + ([ref[0]] if ref else [])) * 1.18
    plot_w = w - pad["l"] - pad["r"]
    plot_h = h - pad["t"] - pad["b"]
    slot = plot_w / len(categories)
    bar_w = min(34.0, slot * 0.72 / len(series))

    def py(y): return pad["t"] + plot_h * (1 - y / y_hi)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img">']
    parts += axes_frame(w, h, pad, 0, y_hi, x_label, y_label)
    if ref:
        parts.append(f'<line x1="{pad["l"]}" y1="{py(ref[0]):.1f}" x2="{w - pad["r"]}" '
                     f'y2="{py(ref[0]):.1f}" stroke="#8172b3" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{w - pad["r"] - 4}" y="{py(ref[0]) - 5:.1f}" fill="#8172b3" '
                     f'font-size="10" text-anchor="end">{esc(ref[1])}</text>')
    for ci, cat in enumerate(categories):
        centre = pad["l"] + slot * (ci + 0.5)
        for si, (_, data, color) in enumerate(series):
            val = data[ci]
            if val is None:
                continue
            x = centre + (si - (len(series) - 1) / 2) * bar_w - bar_w / 2
            y = py(val)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                         f'height="{pad["t"] + plot_h - y:.1f}" fill="{color}" rx="1.5"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" fill="{INK}" font-size="9.5" '
                         f'text-anchor="middle">{val:.3f}</text>')
        parts.append(f'<text x="{centre:.1f}" y="{pad["t"] + plot_h + 15:.1f}" fill="{MUTED}" '
                     f'font-size="10.5" text-anchor="middle">{esc(cat)}</text>')
    legend_y = h - 8
    x = pad["l"]
    for name, _, color in series:
        parts.append(f'<rect x="{x}" y="{legend_y - 8}" width="9" height="9" fill="{color}" rx="1.5"/>')
        parts.append(f'<text x="{x + 14}" y="{legend_y}" fill="{MUTED}" font-size="10.5">{esc(name)}</text>')
        x += 22 + 6.4 * len(name)
    parts.append("</svg>")
    return "".join(parts)


def build(run: str, check0: str | None, fidelity: list[str] | None,
          fidelity_after: list[str] | None, watch: int) -> str:
    rows = read_log(run)
    checks = read_checks(run, check0)
    run_args = read_run_args(run)
    max_steps = run_args.get("max_steps", 0)
    step = rows[-1]["step"] if rows else 0
    t_steps, t_secs = step_times(rows)
    reports = read_fidelity(fidelity)
    after = dict(read_fidelity(fidelity_after))
    ckpts = checkpoints(run)

    rate = sum(t_secs[-3:]) / len(t_secs[-3:]) if t_secs else 0
    eta = datetime.timedelta(seconds=int((max_steps - step) * rate)) if rate and max_steps else None
    sam3_delta = None
    if len(checks) >= 2 and checks[0][1].get("sam3") and checks[-1][1].get("sam3"):
        first, last = checks[0][1]["sam3"]["mean"], checks[-1][1]["sam3"]["mean"]
        sam3_delta = (last - first) / first * 100

    out = ["<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">",
           f"<title>{esc(os.path.basename(os.path.normpath(run)))} 训练状态</title>"]
    if watch:
        out.append(f'<meta http-equiv="refresh" content="{max(10, watch)}">')
    out.append(f"<style>{CSS}</style></head><body>")
    out.append(f"<h1>{esc(os.path.basename(os.path.normpath(run)))} · SegviGen LoRA 域适配</h1>")
    out.append(f'<p class="sub">生成于 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}'
               + (f" · 每 {max(10, watch)} 秒自动刷新" if watch else "")
               + f' · 数据源 <code>{esc(run)}</code></p>')

    out.append('<div class="stats">')
    pct = f"{100 * step / max_steps:.1f}%" if max_steps else ""
    out.append(f'<div class="stat"><b>{step} / {max_steps}</b><span>训练步数 {pct}</span></div>')
    if sam3_delta is not None:
        cls = "good" if sam3_delta < 0 else "warn"
        out.append(f'<div class="stat {cls}"><b>{sam3_delta:+.1f}%</b>'
                   f'<span>holdout sam3 损失（相对基线）</span></div>')
    if reports:
        mean_fid = sum(s["mean_fidelity"] for _, s in reports) / len(reports)
        out.append(f'<div class="stat warn"><b>{mean_fid:.2f}</b><span>基线平均保真度</span></div>')
    if rate:
        out.append(f'<div class="stat"><b>{rate:.1f} s</b><span>当前每步耗时</span></div>')
    if eta is not None:
        out.append(f'<div class="stat"><b>{eta}</b><span>剩余时间估计</span></div>')
    out.append("</div>")

    out.append("<h2>留出集损失：按条件类型分组</h2>")
    if checks:
        cats = ["基线" if s == 0 else f"step {s}" for s, _ in checks]
        series = [(k, [summary.get(k, {}).get("mean") for _, summary in checks], KIND_COLOR[k])
                  for k in KINDS if any(summary.get(k) for _, summary in checks)]
        out.append('<div class="panel">')
        out.append(grouped_bars(cats, series, "checkpoint", "v-pred MSE（固定 t 与噪声）", w=1080, h=300))
        out.append('<p class="note">在不参与训练的对象上、用固定的 t={0.2, 0.5, 0.8} 与固定噪声计算，'
                   '因此可跨 checkpoint 直接比较。越低越好（"全预测零"约为 2.0）。</p>')
        out.append("</div>")
        out.append("<table><tr><th>条件类型</th>"
                   + "".join(f"<th>{esc(c)}</th>" for c in cats)
                   + "<th>相对基线</th><th>变体数</th></tr>")
        for kind, data, _ in series:
            first = data[0]
            delta = f"{(data[-1] - first) / first * 100:+.1f}%" if first and data[-1] else "—"
            n = next((summary[kind]["n"] for _, summary in checks if summary.get(kind)), "")
            out.append(f"<tr><td>{esc(kind)}</td>"
                       + "".join(f"<td>{v:.4f}</td>" if v else "<td>—</td>" for v in data)
                       + f"<td>{delta}</td><td>{n}</td></tr>")
        out.append("</table>")

    out.append('<h2>训练曲线与吞吐</h2><div class="grid">')
    out.append('<div class="panel"><h3>训练损失（EMA 0.98）</h3>'
               '<p>单步 batch 很小，震荡属正常；这个损失被几何与纹理重建主导，对颜色对应关系不敏感。</p>')
    out.append(line_chart([r["step"] for r in rows], [r["ema"] for r in rows],
                          "训练步数", "v-pred MSE", "#6c8ebf"))
    out.append("</div>")
    out.append('<div class="panel"><h3>每步耗时</h3>'
               '<p>取相邻日志点之间的实测值。尖峰通常意味着 GPU 被其他进程占用。</p>')
    ref = (min(t_secs), f"最快 {min(t_secs):.1f} s/步") if t_secs else None
    out.append(line_chart(t_steps, t_secs, "训练步数", "秒 / 步", "#d79b5e", ref=ref, value_fmt="{:.1f}"))
    out.append("</div></div>")

    if reports:
        out.append("<h2>颜色保真度 vs 纯度</h2>")
        out.append('<div class="panel">')
        series = [("fidelity 基座", [s["mean_fidelity"] for _, s in reports], "#b0555a")]
        if after:
            series.append(("fidelity 微调后", [after[n]["mean_fidelity"] if n in after else 0
                                            for n, _ in reports], "#8172b3"))
        series.append(("purity 基座", [s["mean_purity"] for _, s in reports], "#63a86e"))
        if after:
            series.append(("purity 微调后", [after[n]["mean_purity"] if n in after else 0
                                          for n, _ in reports], "#4c72b0"))
        out.append(grouped_bars([n for n, _ in reports], series, "留出对象", "采样表面点占比",
                                w=1080, h=300, y_max=1.15, ref=(0.8, "0.8 达标线")))
        out.append('<p class="note">fidelity 只看最近邻图例色归属对不对，偏色多少看下表的中位偏色：'
                   '归属高 + 偏色大 = 颜色进去了但整体偏移；归属低 = 真的染错了件。'
                   'margin 接近 0 表示该件快要翻到别的图例色上。</p>')
        out.append("</div>")
        out.append("<table><tr><th>对象</th><th>ckpt</th><th>部件数</th><th>归属正确</th>"
                   "<th>达标部件 (≥0.8)</th><th>fidelity</th><th>中位偏色</th><th>中位 margin</th>"
                   "<th>purity</th><th>旧阈值口径</th></tr>")
        for name, s in reports:
            for tag, row in (("基座", s), ("微调后", after.get(name))):
                if row is None:
                    continue
                # median_dist and friends only exist on reports written after the metric rewrite
                dist = f"{row['median_dist']:.1f}" if row.get("median_dist") is not None else "—"
                margin = f"{row['median_margin']:+.1f}" if row.get("median_margin") is not None else "—"
                old = f"{row['mean_fidelity_snap60']:.3f}" if row.get("mean_fidelity_snap60") is not None else "—"
                assigned = (f"{row['parts_assigned_correct']} / {row['n_visible_scored']}"
                            if row.get("parts_assigned_correct") is not None else "—")
                out.append(f"<tr><td>{esc(name)}</td><td>{tag}</td><td>{row['n_parts']}</td>"
                           f"<td>{assigned}</td>"
                           f"<td>{row['parts_correct(>=0.8)']} / {row['n_visible_scored']}</td>"
                           f"<td>{row['mean_fidelity']:.3f}</td><td>{dist}</td><td>{margin}</td>"
                           f"<td>{row['mean_purity']:.3f}</td><td>{old}</td></tr>")
        out.append("</table>")

    if ckpts:
        out.append(f'<p class="note">已保存 {len(ckpts)} 个 checkpoint，最新 '
                   f'<code>{esc(ckpts[-1][1])}</code>；'
                   f'batch {run_args.get("batch_size")} × 累积 {run_args.get("grad_accum")}，'
                   f'lr {run_args.get("lr")}，LoRA r={run_args.get("lora_r")} '
                   f'alpha={run_args.get("lora_alpha")} 目标 {esc(run_args.get("lora_targets"))}。</p>')

    if reports and len(checks) >= 2:
        out.append('<div class="callout"><b>判定标准提醒</b><br>'
                   '留出损失下降幅度有限，且该指标基线本就很低，对颜色对应关系不敏感。'
                   '真正的判据是训练后在同样这些对象上复测 fidelity 能否明显抬升——'
                   '需要 <code>merge_lora.py</code> 导出 ckpt 后再跑一次 <code>eval_fidelity.py</code>。</div>')
    out.append("</body></html>")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--check0", default=None, help="step-0 baseline check_loss.json")
    parser.add_argument("--fidelity", nargs="*", default=None, help="eval_fidelity.py reports, base ckpt")
    parser.add_argument("--fidelity_after", nargs="*", default=None,
                        help="the same objects evaluated with the finetuned ckpt, shown alongside")
    parser.add_argument("--out", default=None)
    parser.add_argument("--watch", type=int, default=0,
                        help="Regenerate every N seconds and add a meta-refresh (0 = write once)")
    args = parser.parse_args()

    out = args.out or os.path.join(args.run, "report.html")
    while True:
        with open(out, "w", encoding="utf-8") as f:
            f.write(build(args.run, args.check0, args.fidelity, args.fidelity_after, args.watch))
        print(f"wrote {out} at {datetime.datetime.now():%H:%M:%S}")
        if not args.watch:
            return
        time.sleep(max(10, args.watch))


if __name__ == "__main__":
    main()
