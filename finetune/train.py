"""LoRA fine-tuning of SegviGen full_seg_w_2d_map on SAM3-style conditions.

Objective is the TRELLIS.2 trainer's flow matching (v-prediction MSE, logit-normal t,
sigma_min 1e-5, condition dropout for CFG); only the LoRA adapters train.

Step-0 check (no training; the loss gap between clean and sam3/corrupt maps IS the domain shift):
    finetune\run_ft.bat train.py --dataset_root <root> --check_only
Train:
    finetune\run_ft.bat train.py --dataset_root <root> --out_dir <dir> --max_steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cells
import common
import trellis2.modules.sparse as sp
from dataset import VariantDataset, collate
from lora import inject_lora, lora_state_dict, load_lora_state_dict
from model import (LegendEncoder, load_gen3dseg, set_gradient_checkpointing, DEFAULT_CKPT,
                   inject_legend_attention, set_legend_context, legend_attn_state_dict, load_legend_attn_state)

SIGMA_MIN = 1e-5


def sample_t(batch_size: int, generator=None, stratified: bool = True) -> torch.Tensor:
    """logitNormal(0, 1) timesteps, stratified across the batch by default.

    The loss varies several-fold with t, so batch_size independent draws make each step's loss
    largely a readout of which timesteps happened to be sampled. Taking one draw per
    equal-probability stratum keeps the marginal distribution identical while cutting that
    variance, which shows up as both a steadier curve and a less noisy gradient.
    """
    u = torch.rand(batch_size, generator=generator)
    if stratified and batch_size > 1:
        u = (torch.randperm(batch_size, generator=generator) + u) / batch_size
    return torch.sigmoid(torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6)))


def to_sparse(batch: dict, key: str, device: str):
    return sp.SparseTensor(batch[key].to(device), batch["coords"].to(device))


def t_schedule(steps: int = 12, rescale_t: float = 3.0) -> list[float]:
    """The inference sampler's timesteps (pipeline.json tex_slat_sampler: 12 Euler steps, rescale_t 3):
    1.000 0.971 0.937 0.900 0.857 0.808 0.750 0.682 0.600 0.500 0.375 0.214 0."""
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    return t_seq.tolist()


@torch.no_grad()
def rollout(model, batch: dict, cond, ks, sched: list[float], noise: torch.Tensor, device: str, decoupled: bool):
    """Path A: x_t from the model's OWN sampler trajectory instead of teacher forcing.

    Sample i is carried k_i Euler steps of the real sampler from pure noise (same cond as the graded
    pass, no CFG as at inference), ending at t = sched[k_i]. trajectory_probe.py shows the colour
    layout is fixed by the first 1-3 steps (t >= 0.9) and never revised, while a teacher-forced x_t
    at those t already leaks the answer (forced acc 0.92 vs 0.74 at t=0.9); so the colour loss is only
    honest on the trajectory the sampler actually visits."""
    coords = batch["coords"].to(device)
    x = noise.to(device).clone()
    tex_in, shape = to_sparse(batch, "tex_in", device), to_sparse(batch, "shape", device)
    lens = batch["coords_len_list"]
    sample_of_tok = coords[:, 0].long()
    ks_t = torch.as_tensor(np.asarray(ks), device=device)
    if decoupled:
        cond, legend_ctx = cond
    for k in range(1, int(max(ks)) + 1):
        t, t_prev = sched[k - 1], sched[k]
        if decoupled:
            set_legend_context(model.flow_model, legend_ctx)
        tt = torch.full((len(lens),), t, device=device)
        v = model(sp.SparseTensor(x, coords), tex_in, shape, tt * 1000, cond, lens).feats.float()
        active = (ks_t >= k)[sample_of_tok].unsqueeze(-1)
        x = torch.where(active, x - (t - t_prev) * v, x)
    return x


def build_cond(batch: dict, device: str, p_uncond: float, rng: np.random.Generator, legend=None,
               p_drop_legend: float = 0.0, p_drop_view1: float = 0.0, decoupled: bool = False):
    """v1/v2: the [B, T, 1024] DINO tensor. v3 (legend encoder given): a list of per-sample
    contexts [T (+T) (+1) (+G), 1024]. Unconditional samples are all-zero T tokens, as the base
    model's neg_cond is.
    v4 (decoupled): returns (image contexts, legend contexts); the legend list goes to the
    LegendCrossAttention wrappers instead of being concatenated."""
    cond = batch["cond"].to(device)
    B = cond.shape[0]
    drop_all = rng.random(B) < p_uncond
    if legend is None:
        drop = torch.from_numpy(drop_all).to(device)
        out = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
        batch["legend_dropped"] = [True] * B
        return (out, None) if decoupled else out
    drop_leg = rng.random(B) < p_drop_legend
    drop_v1 = rng.random(B) < p_drop_view1
    # the colour loss must not ask for greyed parts' colours when the legend was not shown
    batch["legend_dropped"] = [bool(drop_all[i] or drop_leg[i]) for i in range(B)]
    out, legend_ctx = [], []
    for i in range(B):
        if drop_all[i]:
            out.append(torch.zeros_like(cond[i]))
            legend_ctx.append(None)
            continue
        partner = batch["cond_partner"][i]
        partner = None if (partner is None or drop_v1[i]) else partner.to(device)
        # the per-token names (v5) are dropped together with the legend: one "no text" mode
        if drop_leg[i] or "legend_text" not in batch:
            lt, lc, ot, ct, pt = None, None, None, None, None
        else:
            lt = batch["legend_text"][i].to(device)
            lc = batch["legend_rgb"][i].to(device)
            ot = batch["obj_text"][i]
            ot = ot.to(device) if ot is not None else None
            ct = batch.get("token_text", [None] * B)[i]
            ct = ct.to(device) if ct is not None else None
            pt = batch.get("token_text_partner", [None] * B)[i]
            pt = pt.to(device) if (pt is not None and partner is not None) else None
        if decoupled:
            image, leg = legend.tokens(cond[i], partner, lt, lc, ot, cond_text=ct, partner_text=pt)
            out.append(image)
            legend_ctx.append(leg)
        else:
            out.append(legend(cond[i], partner, lt, lc, ot, cond_text=ct, partner_text=pt))
    return (out, legend_ctx) if decoupled else out


def x0_from_v(x_t: torch.Tensor, v: torch.Tensor, t_tok: torch.Tensor) -> torch.Tensor:
    """Invert x_t = (1-t) x0 + (s_min + (1-s_min) t) eps, v = (1-s_min) eps - x0 for x0."""
    a = (SIGMA_MIN + (1 - SIGMA_MIN) * t_tok) / (1 - SIGMA_MIN)
    return (x_t - a * v) / (1 - t_tok + a)


def color_loss(pred_v: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor, t_tok: torch.Tensor,
               batch: dict, probe, tau: float, device: str, t_min: float = 0.0):
    """v4 explicit supervision: decode x0_hat to RGB with the frozen probe and classify every
    label-pure latent cell against the variant's palette. Cells whose TARGET latent the probe
    already misreads (~9%) are skipped, so the loss never asks for something the VAE latent does not
    express. Uniform weight over t on purpose: the high-noise regime is where sampling starts and
    where a thin part is first assigned the body's colour, so it must not be down-weighted.
    Returns (loss, per-sample rows)."""
    rgb = probe(x0_from_v(x_t, pred_v.float(), t_tok))
    with torch.no_grad():
        rgb_true = probe(x_0)
    cls = batch["cell_cls"].to(device)
    masked_all = batch.get("cell_masked")
    masked_all = masked_all.to(device) if masked_all is not None else None
    total, n_terms, rows = rgb.new_zeros(()), 0, []
    begin = 0
    for i, n in enumerate(batch["coords_len_list"]):
        if float(t[i]) < t_min:
            # below t_min the x0 estimate is mostly read off the ground truth in x_t (trajectory_probe:
            # teacher-forced acc 0.92 at t=0.9 vs 0.74 on the sampler's own path), so the CE teaches nothing
            begin += n
            rows.append(None)
            continue
        palette = batch["palette"][i].to(device)
        c = cls[begin:begin + n]
        with torch.no_grad():
            nearest_true = torch.cdist(rgb_true[begin:begin + n], palette).argmin(1)
            c = torch.where(nearest_true == c, c, torch.full_like(c, -1))
        masked = masked_all[begin:begin + n] if masked_all is not None else None
        # partial variants: accuracy on the greyed parts is THE read-out of legend use (reported
        # always); their loss is dropped when the legend was not shown, as nothing could tell
        masked_acc = cells.nearest_accuracy(rgb[begin:begin + n], c, palette, masked) if masked is not None else None
        if masked is not None and batch.get("legend_dropped", [False] * len(batch["coords_len_list"]))[i]:
            c = torch.where(masked, torch.full_like(c, -1), c)
        out = cells.palette_ce(rgb[begin:begin + n], c, palette, tau)
        begin += n
        if out is None:
            rows.append(None)
            continue
        ce, acc, n_cells = out
        total = total + ce
        n_terms += 1
        row = {"color": ce.item(), "color_acc": acc.item(), "n_cells": n_cells}
        if masked_acc is not None:
            row["masked_acc"] = masked_acc.item()
        rows.append(row)
    return total / max(1, n_terms), rows


def flow_loss(model, batch: dict, t: torch.Tensor, noise: torch.Tensor, device: str, p_uncond: float,
              rng: np.random.Generator, legend=None, p_drop_legend: float = 0.0, p_drop_view1: float = 0.0,
              probe=None, color_weight: float = 0.0, color_tau: float = 0.01, decoupled: bool = False,
              cond=None, x_t: torch.Tensor | None = None, color_t_min: float = 0.0):
    """Returns (total loss, per-sample v-pred MSE, per-sample colour rows or None).

    `cond` / `x_t` may be given (Path A trajectory batches): x_t then comes from the sampler's own
    path and the regression target is the velocity that would carry THAT x_t to x_0, i.e. the
    implied noise eps' = (x_t - (1-t) x_0) / sigma_t replaces the drawn one."""
    coords = batch["coords"].to(device)
    x_0 = batch["tex_out"].to(device)
    noise = noise.to(device)
    t_tok = t.to(device)[coords[:, 0].long()].unsqueeze(-1)
    sigma = SIGMA_MIN + (1 - SIGMA_MIN) * t_tok
    if x_t is None:
        x_t = (1 - t_tok) * x_0 + sigma * noise
    else:
        noise = (x_t - (1 - t_tok) * x_0) / sigma
    if cond is None:
        cond = build_cond(batch, device, p_uncond, rng, legend, p_drop_legend, p_drop_view1, decoupled)
    if decoupled:
        cond, legend_ctx = cond
        set_legend_context(model.flow_model, legend_ctx)
    pred = model(sp.SparseTensor(x_t, coords), to_sparse(batch, "tex_in", device), to_sparse(batch, "shape", device),
                 t.to(device) * 1000, cond, batch["coords_len_list"])
    target = (1 - SIGMA_MIN) * noise - x_0
    per_token = ((pred.feats.float() - target) ** 2).mean(-1)
    loss = per_token.mean()
    per_sample = []
    begin = 0
    for n in batch["coords_len_list"]:
        per_sample.append(per_token[begin:begin + n].mean().item())
        begin += n
    color_rows = None
    if probe is not None and "cell_cls" in batch:
        c_loss, color_rows = color_loss(pred.feats, x_t, x_0, t, t_tok, batch, probe, color_tau, device, color_t_min)
        loss = loss + color_weight * c_loss
    return loss, per_sample, color_rows


@torch.no_grad()
def check_loss(model, dataset: VariantDataset, ts: list[float], device: str, seed: int, legend=None,
               probe=None, color_tau: float = 0.01, decoupled: bool = False) -> dict:
    """Fixed t grid + fixed noise per sample so numbers are comparable across runs/checkpoints.
    With a probe, also the colour CE / nearest-palette accuracy of x0_hat (mean over the t grid)."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    by_kind = defaultdict(list)
    color_by_kind = defaultdict(lambda: defaultdict(list))
    rows = []
    for i, batch in enumerate(loader):
        gen = torch.Generator().manual_seed(seed + i)
        noise = torch.randn(batch["tex_out"].shape, generator=gen)
        losses, colors = {}, []
        for t_val in ts:
            t = torch.full((1,), t_val)
            _, per_sample, color_rows = flow_loss(model, batch, t, noise, device, 0.0, np.random.default_rng(0), legend,
                                                  probe=probe, color_tau=color_tau, decoupled=decoupled)
            losses[t_val] = per_sample[0]
            if color_rows and color_rows[0] is not None:
                colors.append(color_rows[0])
        mean = float(np.mean(list(losses.values())))
        by_kind[batch["kinds"][0]].append(mean)
        row = {"name": batch["names"][0], "kind": batch["kinds"][0], "mean": mean,
               **{f"t{t_val:g}": v for t_val, v in losses.items()}}
        extra = ""
        if colors:
            row["color"] = float(np.mean([c["color"] for c in colors]))
            row["color_acc"] = float(np.mean([c["color_acc"] for c in colors]))
            # at ts[-1] (=1.0 when asked) x_t carries no ground truth: this is what the sampler's first
            # step decides, and trajectory_probe shows later steps never revise it
            row["color_acc_hi"] = float(colors[-1]["color_acc"])
            row["color_hi"] = float(colors[-1]["color"])
            for key in ("color", "color_acc", "color_acc_hi", "color_hi"):
                color_by_kind[batch["kinds"][0]][key].append(row[key])
            extra = f"  color={row['color']:.3f} acc={row['color_acc']:.3f} (t={ts[-1]:g}: {row['color_acc_hi']:.3f})"
            masked = [c["masked_acc"] for c in colors if "masked_acc" in c]
            if masked:
                row["masked_acc"] = float(np.mean(masked))
                # at the largest t the x0 estimate owes least to the ground truth mixed into x_t,
                # so this is the honest read-out of "did the legend tell the model the colour"
                row["masked_acc_hi"] = float(masked[-1])
                color_by_kind[batch["kinds"][0]]["masked_acc"].append(row["masked_acc"])
                color_by_kind[batch["kinds"][0]]["masked_acc_hi"].append(row["masked_acc_hi"])
                extra += f" masked_acc={row['masked_acc']:.3f} (t={ts[-1]:g}: {row['masked_acc_hi']:.3f})"
        rows.append(row)
        print(f"  {batch['names'][0]:<48s} {batch['kinds'][0]:<8s} "
              + " ".join(f"t={t_val:g}:{v:.4f}" for t_val, v in losses.items()) + extra)
    summary = {k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in by_kind.items()}
    for k, d in color_by_kind.items():
        for key in ("color", "color_acc", "color_acc_hi", "color_hi"):
            summary[k][key] = float(np.mean(d[key]))
        if d.get("masked_acc"):
            summary[k]["masked_acc"] = float(np.mean(d["masked_acc"]))
            summary[k]["masked_acc_hi"] = float(np.mean(d["masked_acc_hi"]))
    print("\nper-kind mean loss (v-pred MSE; predicting zero would score about 2.0):")
    for k, s in sorted(summary.items()):
        extra = f" color={s['color']:.3f} color_acc={s['color_acc']:.3f} (t={ts[-1]:g}: {s['color_acc_hi']:.3f})" if "color" in s else ""
        extra += f" masked_acc={s['masked_acc']:.3f} (t={ts[-1]:g}: {s['masked_acc_hi']:.3f})" if "masked_acc" in s else ""
        print(f"  {k:<8s} n={s['n']:<4d} mean={s['mean']:.4f} std={s['std']:.4f}{extra}")
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", nargs="+", required=True)
    parser.add_argument("--kinds", nargs="*", default=None, help="Subset of clean/corrupt/sam3")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", default=os.path.join(common.ROOT, "finetune", "runs", "default"))
    parser.add_argument("--resume_lora", default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_targets", default="self,cross")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--check_no_legend", action="store_true",
                        help="check_only: withhold the legend entirely (ablation: does the legend path change outputs?)")
    parser.add_argument("--check_ts", default="0.2,0.5,0.8")
    parser.add_argument("--check_every", type=int, default=0, help="Run the loss check on the holdout every N steps")
    parser.add_argument("--check_limit", type=int, default=0,
                        help="Cap the check set to this many variants, sampled evenly per kind (0 = all)")
    parser.add_argument("--check_root", nargs="*", default=None, help="Extra roots evaluated (not trained on)")
    parser.add_argument("--holdout_file", default=None,
                        help="Object names (one per line) excluded from training and added to the holdout check")
    parser.add_argument("--no_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", default="segvigen-finetune")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None, help="Display name (default: out_dir basename)")
    parser.add_argument("--wandb_id", default=None,
                        help="Run id to resume; defaults to out_dir basename so train_loop restarts "
                             "continue one run instead of starting a new one each time")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline"])
    # ---- v3: legend tokens + paired views
    parser.add_argument("--text_cache", default=None,
                        help="Stage-B text_cache.pt; enables legend tokens (object + one per colour group)")
    parser.add_argument("--pair", action="store_true", help="Condition on the partner view too when the variant has one")
    parser.add_argument("--p_drop_legend", type=float, default=0.2, help="Drop legend (keep images) with this prob")
    parser.add_argument("--p_drop_view1", type=float, default=0.3, help="Drop the partner view with this prob")
    parser.add_argument("--legend_shuffle", action="store_true",
                        help="CONTROL: permute group names inside each object; fidelity equal to the "
                             "real run means the model ignores the text")
    parser.add_argument("--new_lr", type=float, default=1e-3, help="LR of the (from-scratch) legend encoder")
    # ---- v4: explicit colour supervision
    parser.add_argument("--color_probe", default=None,
                        help="color_probe.pt (frozen latent->RGB); enables the nearest-palette CE on x0_hat")
    parser.add_argument("--color_weight", type=float, default=0.5, help="Weight of the colour CE next to the v-pred MSE")
    parser.add_argument("--color_tau", type=float, default=0.01, help="Softmax temperature on squared RGB distance (0..1 scale)")
    parser.add_argument("--min_purity", type=float, default=0.9, help="Cells below this part purity are ignored by the colour loss")
    parser.add_argument("--legend_attn", action="store_true",
                        help="v4: legend tokens through their own gated cross-attention (LegendCrossAttention) "
                             "instead of being concatenated to the image tokens")
    parser.add_argument("--attn_lr", type=float, default=1e-4, help="LR of the legend-attention K/V projections")
    parser.add_argument("--out_lr", type=float, default=3e-4, help="LR of the zero-initialised legend read-out linears")
    parser.add_argument("--check_shuffle", action="store_true",
                        help="At every check also score the holdout with permuted legend names "
                             "(holdout_shuffled/*): same weights, wrong legend -> the gap is legend use")
    # ---- v5: per-token part names
    parser.add_argument("--token_text", action="store_true",
                        help="v5: add the text vector of the part under each DINO patch to that token "
                             "(needs <variant>/tokens.npz from token_labels.py and --text_cache)")
    # ---- Path A: supervise on the sampler's own trajectory
    parser.add_argument("--p_traj", type=float, default=0.0,
                        help="Fraction of micro-batches whose x_t comes from k in [0, traj_steps] no-grad Euler steps "
                             "of the real sampler from pure noise (t = 1, 0.971, 0.937, 0.9, ...) instead of teacher forcing")
    parser.add_argument("--traj_steps", type=int, default=3, help="Max sampler steps rolled out before the graded pass")
    parser.add_argument("--color_t_min", type=float, default=0.0,
                        help="Colour CE only for samples with t >= this (below, x_t already holds the answer)")
    args = parser.parse_args()
    if args.legend_attn and not args.text_cache:
        parser.error("--legend_attn needs --text_cache")
    if args.token_text and not args.text_cache:
        parser.error("--token_text needs --text_cache")

    os.chdir(common.ROOT)
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    holdout = []
    if args.holdout_file:
        with open(args.holdout_file, "r", encoding="utf-8") as f:
            holdout = [line.strip() for line in f if line.strip()]
    ds_kw = dict(text_cache=args.text_cache, pair=args.pair, seed=args.seed,
                 cell_targets=bool(args.color_probe), min_purity=args.min_purity, token_text=args.token_text)
    probe = cells.ColorProbe(args.color_probe, device) if args.color_probe else None
    if probe is not None:
        print(f"colour probe {args.color_probe}: {probe.info}")
    dataset = VariantDataset(args.dataset_root, kinds=args.kinds, exclude=holdout,
                             legend_shuffle=args.legend_shuffle, **ds_kw)
    kinds = dataset.kinds()
    print(f"{len(dataset)} variants: " + ", ".join(f"{k}={kinds.count(k)}" for k in sorted(set(kinds))))
    if args.pair:
        print(f"paired variants (partner view available): {dataset.n_paired()}")

    def subsample(ds: VariantDataset, limit: int) -> VariantDataset:
        """Even stride per kind, so the check set keeps the clean/corrupt/sam3 mix."""
        if not limit or len(ds) <= limit:
            return ds
        by_kind = defaultdict(list)
        for item in ds.items:
            by_kind[item[2]["kind"]].append(item)
        per_kind = max(1, limit // len(by_kind))
        picked = []
        for kind in sorted(by_kind):
            items = by_kind[kind]
            stride = max(1, len(items) // per_kind)
            picked += items[::stride][:per_kind]
        ds.items = picked
        return ds

    model = load_gen3dseg(args.ckpt, device)
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = inject_lora(model.flow_model, r=args.lora_r, alpha=args.lora_alpha,
                              targets=args.lora_targets.split(","), dropout=args.lora_dropout)
    attn_params, out_params = [], []
    if args.legend_attn:
        attn_params, out_params = inject_legend_attention(model.flow_model)
    model.to(device)
    legend = None
    if args.text_cache:
        legend = LegendEncoder(text_dim=dataset.text.dim).to(device)
        legend.train()
    start_step = 0
    if args.resume_lora:
        payload = torch.load(args.resume_lora, map_location="cpu")
        load_lora_state_dict(model, payload["lora"])
        if legend is not None and payload.get("legend"):
            legend.load_state_dict(payload["legend"])
        if args.legend_attn and payload.get("legend_attn"):
            load_legend_attn_state(model, payload["legend_attn"])
        # Continue the LR schedule instead of re-warming up, so an interrupted run resumes cleanly.
        start_step = int(payload.get("step", 0))
        print(f"resumed LoRA from {args.resume_lora} at step {start_step}")
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params {n_lora / 1e6:.2f}M of {n_total / 1e9:.2f}B ({100 * n_lora / n_total:.3f}%)")
    if legend is not None:
        print(f"legend encoder params {sum(p.numel() for p in legend.parameters()) / 1e6:.2f}M "
              f"(text vectors missing for {sum(dataset.text.missing.values())} lookups so far)")
    if args.token_text:
        n_tok = sum(1 for obj, v, _ in dataset.items if os.path.exists(os.path.join(obj.variant_dir(v), "tokens.npz")))
        print(f"per-token names: {n_tok}/{len(dataset)} training variants have tokens.npz")
    if args.legend_attn:
        print(f"legend attention: {len(out_params) // 2} blocks, "
              f"{sum(p.numel() for p in attn_params) / 1e6:.1f}M K/V params + "
              f"{sum(p.numel() for p in out_params) / 1e6:.1f}M zero-init read-out params")
    set_gradient_checkpointing(model.flow_model, not args.no_checkpointing)
    ts = [float(v) for v in args.check_ts.split(",")]

    if args.check_only:
        model.eval()
        # With a holdout the baseline must come from the held-out objects, so it stays comparable
        # to the periodic checks during training.
        target = VariantDataset(args.dataset_root, kinds=args.kinds, objects=holdout, **ds_kw) if holdout else dataset
        target = subsample(target, args.check_limit)
        print(f"checking {len(target)} variants ({'holdout' if holdout else 'train set'})")
        if legend is not None:
            legend.eval()
        result = check_loss(model, target, ts, device, args.seed, None if args.check_no_legend else legend,
                            probe, args.color_tau, args.legend_attn)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "check_loss.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    check_dataset = None
    if args.check_root or holdout:
        check_items = []
        if holdout:
            check_items.append(VariantDataset(args.dataset_root, kinds=args.kinds, objects=holdout, **ds_kw))
        if args.check_root:
            check_items.append(VariantDataset(args.check_root, **ds_kw))
        check_dataset = check_items[0]
        for extra in check_items[1:]:
            check_dataset.items += extra.items
        check_dataset = subsample(check_dataset, args.check_limit)
        print(f"holdout: {len(check_dataset)} variants")
    check_shuffled = None
    if check_dataset is not None and legend is not None and args.check_shuffle:
        # Same holdout items with permuted legend names: the colour-accuracy gap to the true
        # legend (on partial variants above all) is the direct readout of semantic injection.
        check_shuffled = VariantDataset(args.dataset_root, kinds=args.kinds, objects=holdout or None,
                                        legend_shuffle=True, **ds_kw)
        check_shuffled.items = list(check_dataset.items)

    run = None
    if args.wandb:
        import wandb
        # Resuming one id keeps train_loop's restarts as a single continuous run.
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
            id=args.wandb_id or os.path.basename(os.path.normpath(args.out_dir)),
            name=args.wandb_name or os.path.basename(os.path.normpath(args.out_dir)),
            resume="allow",
            config={**vars(args), "n_variants": len(dataset), "n_lora_params": n_lora,
                    "variants_per_kind": {k: kinds.count(k) for k in sorted(set(kinds))},
                    "n_paired": dataset.n_paired() if args.pair else 0,
                    "n_holdout_variants": len(check_dataset) if check_dataset is not None else 0},
        )
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("train/color_acc", summary="max")
        wandb.define_metric("holdout/*", summary="min")
        print(f"wandb: {run.url or args.wandb_mode}")

    groups = [{"params": lora_params, "base_lr": args.lr}]
    if legend is not None:
        groups.append({"params": list(legend.parameters()), "base_lr": args.new_lr})
    if attn_params:
        groups.append({"params": attn_params, "base_lr": args.attn_lr})
        groups.append({"params": out_params, "base_lr": args.out_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    trainable = [p for g in groups for p in g["params"]]

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    def lr_at(step):
        return args.lr * lr_scale(step)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, drop_last=True)
    log_path = os.path.join(args.out_dir, "log.jsonl")
    model.train()
    step, micro, ema = start_step, 0, None
    t0 = time.time()
    t_last, step_last = t0, step
    kind_acc = defaultdict(list)
    color_acc = defaultdict(list)
    sched = t_schedule()
    while step < args.max_steps:
        for batch in loader:
            B = len(batch["coords_len_list"])
            noise = torch.randn(batch["tex_out"].shape)
            traj = args.p_traj > 0 and rng.random() < args.p_traj
            if traj:
                ks = rng.integers(0, args.traj_steps + 1, size=B)
                t = torch.tensor([sched[k] for k in ks], dtype=torch.float32)
                cond = build_cond(batch, device, args.p_uncond, rng, legend, args.p_drop_legend, args.p_drop_view1,
                                  args.legend_attn)
                x_t = rollout(model, batch, cond, ks, sched, noise, device, args.legend_attn)
            else:
                t, cond, x_t = sample_t(B), None, None
            loss, per_sample, color_rows = flow_loss(model, batch, t, noise, device, args.p_uncond, rng, legend,
                                                     args.p_drop_legend, args.p_drop_view1,
                                                     probe, args.color_weight, args.color_tau, args.legend_attn,
                                                     cond=cond, x_t=x_t, color_t_min=args.color_t_min)
            (loss / args.grad_accum).backward()
            for k, v in zip(batch["kinds"], per_sample):
                kind_acc[k].append(v)
                if traj:
                    kind_acc["traj"].append(v)
            if color_rows:
                for k, row in zip(batch["kinds"], color_rows):
                    if row is not None:
                        color_acc["color"].append(row["color"])
                        color_acc["color_acc"].append(row["color_acc"])
                        color_acc[f"color_acc_{k}"].append(row["color_acc"])
                        if traj:
                            # the honest colour read-out: x_t is the sampler's own, no ground truth leaks in
                            color_acc["color_acc_traj"].append(row["color_acc"])
                            color_acc["color_traj"].append(row["color"])
                        if "masked_acc" in row:
                            color_acc["masked_acc"].append(row["masked_acc"])
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % args.log_every == 0 or step == 1:
                per_kind = {k: float(np.mean(v)) for k, v in kind_acc.items()}
                # "loss" is the mean over every sample since the last log point, not this step's
                # value: a single step's loss is dominated by which samples and timesteps it drew
                # (measured at ~30x the size of the improvement we are looking for), so logging it
                # raw produces a curve that jitters just as much on a frozen model.
                window = [v for vals in kind_acc.values() for v in vals]
                kind_acc.clear()
                color_stats = {k: float(np.mean(v)) for k, v in color_acc.items() if v}
                color_acc.clear()
                rec = {"step": step, "loss": float(np.mean(window)), "loss_step": loss.item(),
                       "n_samples": len(window), "ema": ema, "lr": lr_at(step), "per_kind": per_kind,
                       "elapsed_s": round(time.time() - t0, 1), "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                if color_stats:
                    rec.update(color_stats)
                if args.token_text:
                    rec["tok_gain"] = round(legend.tok_gain(), 5)
                print(json.dumps(rec))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if run is not None:
                    # Rate since the previous log point, so GPU contention shows up instead of
                    # being averaged away over the whole run.
                    now = time.time()
                    run.log({"train/loss": rec["loss"], "train/loss_step": rec["loss_step"],
                             "train/ema": ema, "train/lr": rec["lr"],
                             "train/vram_gib": rec["vram_gib"],
                             "train/s_per_step": (now - t_last) / max(1, step - step_last),
                             **{f"train/loss_{k}": v for k, v in per_kind.items()},
                             **{f"train/{k}": v for k, v in color_stats.items()},
                             **({"train/tok_gain": rec["tok_gain"]} if args.token_text else {})}, step=step)
                    t_last, step_last = now, step
            if step % args.save_every == 0 or step == args.max_steps:
                payload = {"lora": lora_state_dict(model), "step": step, "args": vars(args),
                           "legend": {k: v.detach().cpu() for k, v in legend.state_dict().items()} if legend is not None else None,
                           "legend_attn": legend_attn_state_dict(model) if args.legend_attn else None}
                torch.save(payload, os.path.join(args.out_dir, f"lora_step{step}.pt"))
                torch.save(payload, os.path.join(args.out_dir, "lora_last.pt"))
            if check_dataset is not None and args.check_every and step % args.check_every == 0:
                model.eval()
                if legend is not None:
                    legend.eval()
                checks = [("holdout", check_dataset, "")]
                if check_shuffled is not None:
                    checks.append(("holdout_shuffled", check_shuffled, "_shuffled"))
                log = {}
                for prefix, ds, suffix in checks:
                    if suffix:
                        print(f"\n[{prefix}] same variants, legend names permuted")
                    result = check_loss(model, ds, ts, device, args.seed, legend, probe, args.color_tau, args.legend_attn)
                    with open(os.path.join(args.out_dir, f"check_step{step}{suffix}.json"), "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)
                    summ = result["summary"]
                    log.update({f"{prefix}/{k}": s["mean"] for k, s in summ.items()})
                    log.update({f"{prefix}/color_{k}": s["color"] for k, s in summ.items() if "color" in s})
                    log.update({f"{prefix}/color_acc_{k}": s["color_acc"] for k, s in summ.items() if "color" in s})
                    log.update({f"{prefix}/color_acc_hi_{k}": s["color_acc_hi"] for k, s in summ.items() if "color" in s})
                    log.update({f"{prefix}/color_hi_{k}": s["color_hi"] for k, s in summ.items() if "color" in s})
                    log.update({f"{prefix}/masked_acc_{k}": s["masked_acc"] for k, s in summ.items() if "masked_acc" in s})
                    log.update({f"{prefix}/masked_acc_hi_{k}": s["masked_acc_hi"] for k, s in summ.items() if "masked_acc" in s})
                if run is not None:
                    run.log(log, step=step)
                model.train()
                if legend is not None:
                    legend.train()
            if step >= args.max_steps:
                break
    print(f"done: {step} steps, {(time.time() - t0) / 60:.1f} min, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    # Sentinel for train_loop.bat: a restart driver must not depend on exit codes surviving
    # nested .bat calls, or it will happily start training again from step 0.
    with open(os.path.join(args.out_dir, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "max_steps": args.max_steps,
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
