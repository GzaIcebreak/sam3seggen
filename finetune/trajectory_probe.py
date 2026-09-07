"""Path A diagnostic: does the colour assignment survive the real sampler?

Training scores x0_hat from a TEACHER-FORCED x_t = (1-t) x0 + sigma_t eps. Inference runs the
12-step Euler sampler (rescale_t 3, no CFG at strength 1) from pure noise, so every x_t after the
first step is the model's own. This script runs that sampler on held-out variants and, at every
step, probes the colour of x0_hat both on the sampler's trajectory and teacher-forced at the same
t (same noise), plus the final sample. The gap between the two curves is the exposure bias the
v4 colour loss cannot see; where the trajectory curve locks in tells which t matter.

    finetune\run_ft.bat trajectory_probe.py --dataset_root E:\...\pv --holdout_file E:\...\pv_holdout_v3.txt ^
        --limit 40 --out E:\...\traj_base.json [--resume_lora finetune\runs\pv_v4\lora_last.pt --text_cache ... --pair]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import cells
import common
import trellis2.modules.sparse as sp
from dataset import VariantDataset, collate
from lora import inject_lora, load_lora_state_dict
from model import LegendEncoder, load_gen3dseg, inject_legend_attention, load_legend_attn_state, set_legend_context
from train import build_cond, x0_from_v, to_sparse, t_schedule, SIGMA_MIN


def color_metrics(rgb: torch.Tensor, batch: dict, probe, device: str, tau: float) -> dict:
    """Balanced CE / accuracy / masked accuracy of ONE sample's decoded colours (probe already applied)."""
    cls = batch["cell_cls"].to(device)
    palette = batch["palette"][0].to(device)
    with torch.no_grad():
        rgb_true = probe(batch["tex_out"].to(device))
        nearest_true = torch.cdist(rgb_true, palette).argmin(1)
        cls = torch.where(nearest_true == cls, cls, torch.full_like(cls, -1))
    out = cells.palette_ce(rgb, cls, palette, tau)
    if out is None:
        return {}
    ce, acc, _ = out
    row = {"ce": ce.item(), "acc": acc.item()}
    masked = batch.get("cell_masked")
    if masked is not None and bool(masked.any()):
        m = cells.nearest_accuracy(rgb, cls, palette, masked.to(device))
        if m is not None:
            row["masked_acc"] = m.item()
    return row


@torch.no_grad()
def run_variant(model, batch, cond, device, probe, tau, ts, decoupled, legend_ctx):
    coords = batch["coords"].to(device)
    x_0 = batch["tex_out"].to(device)
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(x_0.shape, generator=gen).to(device)
    tex_in, shape = to_sparse(batch, "tex_in", device), to_sparse(batch, "shape", device)
    lens = batch["coords_len_list"]

    def predict(x, t):
        if decoupled:
            set_legend_context(model.flow_model, legend_ctx)
        tt = torch.full((1,), t, device=device)
        return model(sp.SparseTensor(x, coords), tex_in, shape, tt * 1000, cond, lens).feats.float()

    x = noise.clone()
    traj, forced = [], []
    for i in range(len(ts) - 1):
        t, t_prev = ts[i], ts[i + 1]
        t_tok = torch.full((x.shape[0], 1), t, device=device)
        v = predict(x, t)
        traj.append(color_metrics(probe(x0_from_v(x, v, t_tok)), batch, probe, device, tau))
        x_tf = (1 - t_tok) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t_tok) * noise
        v_tf = predict(x_tf, t)
        forced.append(color_metrics(probe(x0_from_v(x_tf, v_tf, t_tok)), batch, probe, device, tau))
        x = x - (t - t_prev) * v
    final = color_metrics(probe(x), batch, probe, device, tau)
    final["mse_x0"] = ((x - x_0) ** 2).mean().item()
    return traj, forced, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", nargs="+", required=True)
    ap.add_argument("--holdout_file", required=True)
    ap.add_argument("--kinds", nargs="*", default=["sam3", "partial", "clean"])
    ap.add_argument("--limit", type=int, default=40, help="variants per kind")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume_lora", default=None)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    ap.add_argument("--lora_targets", default="self,cross")
    ap.add_argument("--text_cache", default=None)
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--token_text", action="store_true")
    ap.add_argument("--no_legend", action="store_true")
    ap.add_argument("--color_probe", default=os.path.join(common.ROOT, "finetune", "color_probe.pt"))
    ap.add_argument("--color_tau", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(common.ROOT)
    device = "cuda"

    holdout = [l.strip() for l in open(args.holdout_file, encoding="utf-8") if l.strip()]
    ds = VariantDataset(args.dataset_root, kinds=args.kinds, objects=holdout, text_cache=args.text_cache,
                        pair=args.pair, token_text=args.token_text, cell_targets=True, seed=args.seed)
    # same variants per kind for every checkpoint
    rng = np.random.default_rng(args.seed)
    by_kind = defaultdict(list)
    for i, (_, _, meta) in enumerate(ds.items):
        by_kind[meta["kind"]].append(i)
    keep = sorted(j for k, idx in by_kind.items() for j in rng.choice(idx, min(args.limit, len(idx)), replace=False))
    ds.items = [ds.items[j] for j in keep]
    print(f"{len(ds)} variants: " + ", ".join(f"{k}={min(args.limit, len(v))}" for k, v in sorted(by_kind.items())))

    model = load_gen3dseg(args.ckpt or __import__("model").DEFAULT_CKPT, device)
    for p in model.parameters():
        p.requires_grad_(False)
    legend, decoupled = None, False
    if args.resume_lora:
        payload = torch.load(args.resume_lora, map_location="cpu")
        inject_lora(model.flow_model, r=args.lora_r, alpha=args.lora_alpha, targets=args.lora_targets.split(","))
        if payload.get("legend_attn"):
            # same order as train.py: the wrappers sit around the LoRA-injected cross_attn ("...inner...")
            inject_legend_attention(model.flow_model)
            load_legend_attn_state(model, payload["legend_attn"])
            decoupled = True
        load_lora_state_dict(model, payload["lora"])
        if payload.get("legend") and args.text_cache:
            legend = LegendEncoder(text_dim=ds.text.dim)
            legend.load_state_dict(payload["legend"])
            legend = legend.to(device).eval()
        print(f"LoRA from {args.resume_lora} (step {payload.get('step')}), legend={'yes' if legend else 'no'}, decoupled={decoupled}")
    model.to(device).eval()
    if args.no_legend:
        legend = None
    probe = cells.ColorProbe(args.color_probe, device)
    ts = t_schedule(args.steps, args.rescale_t)
    print("t schedule:", " ".join(f"{t:.3f}" for t in ts))

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    rows = []
    for i, batch in enumerate(loader):
        cond = build_cond(batch, device, 0.0, np.random.default_rng(0), legend, 0.0, 0.0, decoupled)
        legend_ctx = None
        if decoupled:
            cond, legend_ctx = cond
        traj, forced, final = run_variant(model, batch, cond, device, probe, args.color_tau, ts, decoupled, legend_ctx)
        rows.append({"name": batch["names"][0], "kind": batch["kinds"][0], "traj": traj, "forced": forced, "final": final})
        print(f"  {batch['names'][0]:<48s} {batch['kinds'][0]:<8s} final acc={final.get('acc', float('nan')):.3f} "
              f"ce={final.get('ce', float('nan')):.3f}" + (f" masked={final['masked_acc']:.3f}" if "masked_acc" in final else "")
              + f" | traj acc@t1={traj[0].get('acc', float('nan')):.3f} forced acc@t1={forced[0].get('acc', float('nan')):.3f}",
              flush=True)

    def mean_of(rs, key, field):
        vals = [r[key][field] for r in rs if field in r[key]] if key == "final" else None
        return float(np.mean(vals)) if vals else None

    summary = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        s = {"n": len(rs), "final": {f: mean_of(rs, "final", f) for f in ("acc", "ce", "masked_acc", "mse_x0")}}
        for key in ("traj", "forced"):
            s[key] = {}
            for f in ("acc", "ce", "masked_acc"):
                per_step = []
                for k in range(len(ts) - 1):
                    vals = [r[key][k][f] for r in rs if f in r[key][k]]
                    per_step.append(float(np.mean(vals)) if vals else None)
                s[key][f] = per_step
        summary[kind] = s
        print(f"\n[{kind}] n={len(rs)}  final sample: acc={s['final']['acc']:.3f} ce={s['final']['ce']:.3f}"
              + (f" masked_acc={s['final']['masked_acc']:.3f}" if s['final']['masked_acc'] is not None else ""))
        print("   t      traj acc  forced acc | traj ce  forced ce" + ("  | traj masked  forced masked" if kind == "partial" else ""))
        for k in range(len(ts) - 1):
            line = f"  {ts[k]:.3f}   {s['traj']['acc'][k]:.3f}     {s['forced']['acc'][k]:.3f}   |  {s['traj']['ce'][k]:.3f}    {s['forced']['ce'][k]:.3f}"
            if kind == "partial" and s["traj"]["masked_acc"][k] is not None:
                line += f"   |   {s['traj']['masked_acc'][k]:.3f}        {s['forced']['masked_acc'][k]:.3f}"
            print(line)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "ts": ts, "summary": summary, "rows": rows}, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
