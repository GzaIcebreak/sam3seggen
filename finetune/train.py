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

import common
import trellis2.modules.sparse as sp
from dataset import VariantDataset, collate
from lora import inject_lora, lora_state_dict, load_lora_state_dict
from model import load_gen3dseg, set_gradient_checkpointing, DEFAULT_CKPT

SIGMA_MIN = 1e-5


def sample_t(batch_size: int, generator=None) -> torch.Tensor:
    return torch.sigmoid(torch.randn(batch_size, generator=generator))  # logitNormal(0, 1)


def to_sparse(batch: dict, key: str, device: str):
    return sp.SparseTensor(batch[key].to(device), batch["coords"].to(device))


def flow_loss(model, batch: dict, t: torch.Tensor, noise: torch.Tensor, device: str, p_uncond: float, rng: np.random.Generator):
    coords = batch["coords"].to(device)
    x_0 = batch["tex_out"].to(device)
    noise = noise.to(device)
    t_tok = t.to(device)[coords[:, 0].long()].unsqueeze(-1)
    x_t = (1 - t_tok) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t_tok) * noise
    cond = batch["cond"].to(device)
    if p_uncond > 0:
        drop = torch.from_numpy(rng.random(cond.shape[0]) < p_uncond).to(device)
        cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
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
    return loss, per_sample


@torch.no_grad()
def check_loss(model, dataset: VariantDataset, ts: list[float], device: str, seed: int) -> dict:
    """Fixed t grid + fixed noise per sample so numbers are comparable across runs/checkpoints."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    by_kind = defaultdict(list)
    rows = []
    for i, batch in enumerate(loader):
        gen = torch.Generator().manual_seed(seed + i)
        noise = torch.randn(batch["tex_out"].shape, generator=gen)
        losses = {}
        for t_val in ts:
            t = torch.full((1,), t_val)
            _, per_sample = flow_loss(model, batch, t, noise, device, 0.0, np.random.default_rng(0))
            losses[t_val] = per_sample[0]
        mean = float(np.mean(list(losses.values())))
        by_kind[batch["kinds"][0]].append(mean)
        rows.append({"name": batch["names"][0], "kind": batch["kinds"][0], "mean": mean,
                     **{f"t{t_val:g}": v for t_val, v in losses.items()}})
        print(f"  {batch['names'][0]:<48s} {batch['kinds'][0]:<8s} " + " ".join(f"t={t_val:g}:{v:.4f}" for t_val, v in losses.items()))
    summary = {k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in by_kind.items()}
    print("\nper-kind mean loss (v-pred MSE; predicting zero would score about 2.0):")
    for k, s in sorted(summary.items()):
        print(f"  {k:<8s} n={s['n']:<4d} mean={s['mean']:.4f} std={s['std']:.4f}")
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
    args = parser.parse_args()

    os.chdir(common.ROOT)
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    holdout = []
    if args.holdout_file:
        with open(args.holdout_file, "r", encoding="utf-8") as f:
            holdout = [line.strip() for line in f if line.strip()]
    dataset = VariantDataset(args.dataset_root, kinds=args.kinds, exclude=holdout)
    kinds = dataset.kinds()
    print(f"{len(dataset)} variants: " + ", ".join(f"{k}={kinds.count(k)}" for k in sorted(set(kinds))))

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
    model.to(device)
    start_step = 0
    if args.resume_lora:
        payload = torch.load(args.resume_lora, map_location="cpu")
        load_lora_state_dict(model, payload["lora"])
        # Continue the LR schedule instead of re-warming up, so an interrupted run resumes cleanly.
        start_step = int(payload.get("step", 0))
        print(f"resumed LoRA from {args.resume_lora} at step {start_step}")
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params {n_lora / 1e6:.2f}M of {n_total / 1e9:.2f}B ({100 * n_lora / n_total:.3f}%)")
    set_gradient_checkpointing(model.flow_model, not args.no_checkpointing)
    ts = [float(v) for v in args.check_ts.split(",")]

    if args.check_only:
        model.eval()
        # With a holdout the baseline must come from the held-out objects, so it stays comparable
        # to the periodic checks during training.
        target = VariantDataset(args.dataset_root, kinds=args.kinds, objects=holdout) if holdout else dataset
        target = subsample(target, args.check_limit)
        print(f"checking {len(target)} variants ({'holdout' if holdout else 'train set'})")
        result = check_loss(model, target, ts, device, args.seed)
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
            check_items.append(VariantDataset(args.dataset_root, kinds=args.kinds, objects=holdout))
        if args.check_root:
            check_items.append(VariantDataset(args.check_root))
        check_dataset = check_items[0]
        for extra in check_items[1:]:
            check_dataset.items += extra.items
        check_dataset = subsample(check_dataset, args.check_limit)
        print(f"holdout: {len(check_dataset)} variants")

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
                    "n_holdout_variants": len(check_dataset) if check_dataset is not None else 0},
        )
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("holdout/*", summary="min")
        print(f"wandb: {run.url or args.wandb_mode}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, drop_last=True)
    log_path = os.path.join(args.out_dir, "log.jsonl")
    model.train()
    step, micro, ema = start_step, 0, None
    t0 = time.time()
    t_last, step_last = t0, step
    kind_acc = defaultdict(list)
    while step < args.max_steps:
        for batch in loader:
            t = sample_t(len(batch["coords_len_list"]))
            noise = torch.randn(batch["tex_out"].shape)
            loss, per_sample = flow_loss(model, batch, t, noise, device, args.p_uncond, rng)
            (loss / args.grad_accum).backward()
            for k, v in zip(batch["kinds"], per_sample):
                kind_acc[k].append(v)
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % args.log_every == 0 or step == 1:
                per_kind = {k: float(np.mean(v)) for k, v in kind_acc.items()}
                kind_acc.clear()
                rec = {"step": step, "loss": loss.item(), "ema": ema, "lr": lr_at(step), "per_kind": per_kind,
                       "elapsed_s": round(time.time() - t0, 1), "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                print(json.dumps(rec))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if run is not None:
                    # Rate since the previous log point, so GPU contention shows up instead of
                    # being averaged away over the whole run.
                    now = time.time()
                    run.log({"train/loss": rec["loss"], "train/ema": ema, "train/lr": rec["lr"],
                             "train/vram_gib": rec["vram_gib"],
                             "train/s_per_step": (now - t_last) / max(1, step - step_last),
                             **{f"train/loss_{k}": v for k, v in per_kind.items()}}, step=step)
                    t_last, step_last = now, step
            if step % args.save_every == 0 or step == args.max_steps:
                payload = {"lora": lora_state_dict(model), "step": step, "args": vars(args)}
                torch.save(payload, os.path.join(args.out_dir, f"lora_step{step}.pt"))
                torch.save(payload, os.path.join(args.out_dir, "lora_last.pt"))
            if check_dataset is not None and args.check_every and step % args.check_every == 0:
                model.eval()
                result = check_loss(model, check_dataset, ts, device, args.seed)
                with open(os.path.join(args.out_dir, f"check_step{step}.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                if run is not None:
                    run.log({f"holdout/{k}": s["mean"] for k, s in result["summary"].items()}, step=step)
                model.train()
            if step >= args.max_steps:
                break
    print(f"done: {step} steps, {(time.time() - t0) / 60:.1f} min, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
