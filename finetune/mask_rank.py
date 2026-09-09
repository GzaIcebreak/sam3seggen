"""Stage 1 of the Mask RankGNN plan: features, model and training for candidate re-ranking.

SAM3 weights every (name, query) candidate by a score it assigned without looking at the other
candidates in the image, and `REPORT_oracle_candidates.md` shows what that costs: the best candidate
is the top-scoring one only 3.7 % of the time, and 88 % of the mis-painted pixels are covered by some
candidate of their own name that simply lost the logsumexp. This module learns a per-candidate weight
from the relations between candidates and feeds it back into the same painter:

    L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} + lambda * log keep_{n,q} ]

lambda = 0 is the current behaviour, so the whole thing is a pure post-process with a free fallback.

Only the top-K candidates per name are re-ranked: the oracle showed the best of the top-5 by score
already reaches IoU 0.553 against 0.428 for the top-1 and 0.702 for the whole 200-query pool, so the
head of the list carries most of the headroom at a fraction of the graph size.

`candidate_features` is shared by the offline dump (`mask_rank_feats.py`) and the online repaint
(`mask_rank_paint.py`) so training and inference cannot drift apart.

    python finetune/mask_rank.py --feats runs/mask_rank_v3/feats_train \
        --val_feats runs/mask_rank_v3/feats_holdout --out runs/mask_rank_v3

The checkpoint holds only tensors and primitives, so it loads under torch's safe (weights_only) path.
The network has no dropout or batch norm (LayerNorm only), so it behaves identically whether or not
torch would consider it "training"; there is no mode switching anywhere.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SCALAR = 15
D_TEXT = 256
D_VIS = 256
N_EDGE = 5


# --------------------------------------------------------------------------- features

def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """[N, Q] scores -> [N, k] query indices, highest score first."""
    return scores.topk(min(k, scores.shape[1]), dim=1).indices


@torch.no_grad()
def candidate_features(logits: torch.Tensor, scores: torch.Tensor, fpn: torch.Tensor,
                       fg: torch.Tensor, text_vecs: torch.Tensor, idx: torch.Tensor) -> dict:
    """Node and edge features of one image's top-K candidates.

    logits [N, Q, h, w] mask logits, scores [N, Q], fpn [256, h, w] the mask-resolution FPN level,
    fg [h, w] silhouette, text_vecs [N, 256] the bank-shifted pooled text vector of each name,
    idx [N, k] selected queries. Returns node [C, N_SCALAR + 512], edge [C, C, N_EDGE] and the
    (name, query) each row came from, with C = N * k in row-major (name, rank) order."""
    n, k = idx.shape
    h, w = logits.shape[-2:]
    dev = logits.device
    sel = logits.gather(1, idx[:, :, None, None].expand(-1, -1, h, w)).float()   # [N, k, h, w]
    sc = scores.gather(1, idx).clamp(min=1e-6)                                   # [N, k]
    m = (sel > 0) & fg
    mf = m.reshape(n * k, h * w).float()                                         # [C, hw]
    area = mf.sum(1)
    fga = float(fg.sum().clamp(min=1))
    safe = area.clamp(min=1)

    ys, xs = torch.meshgrid(torch.arange(h, device=dev).float() / h,
                            torch.arange(w, device=dev).float() / w, indexing="ij")
    cy = (mf @ ys.reshape(-1)) / safe
    cx = (mf @ xs.reshape(-1)) / safe
    rows, cols = m.any(-1).reshape(n * k, h), m.any(-2).reshape(n * k, w)
    y0 = rows.float().argmax(1) / h
    y1 = (h - 1 - rows.flip(1).float().argmax(1)) / h
    x0 = cols.float().argmax(1) / w
    x1 = (w - 1 - cols.flip(1).float().argmax(1)) / w

    flat = sel.reshape(n * k, h * w)
    lmax = flat.max(1).values
    lmean = (flat * mf).sum(1) / safe
    # share of its own name's logsumexp this candidate carries, among the selected ones
    mass = torch.softmax(sel.amax((-1, -2)) + sc.log(), dim=1).reshape(-1)

    scalar = torch.stack([
        sc.reshape(-1), sc.reshape(-1).log(),
        torch.arange(k, device=dev).float().div(k).repeat(n),
        area / fga, (area / fga + 1e-6).log(),
        x0, y0, x1, y1, cx, cy,
        lmax, lmean, mass,
        torch.full((n * k,), n / 10.0, device=dev),
    ], dim=1)                                                                     # [C, N_SCALAR]

    vis = (mf @ fpn.reshape(fpn.shape[0], -1).T.float()) / safe[:, None]          # [C, 256]
    text = text_vecs[:, None, :].expand(n, k, D_TEXT).reshape(n * k, D_TEXT).float()
    node = torch.cat([scalar, text, vis], dim=1)

    inter = mf @ mf.T
    union = area[:, None] + area[None, :] - inter
    same = torch.arange(n * k, device=dev) // k
    edge = torch.stack([
        inter / union.clamp(min=1),
        inter / safe[:, None],
        inter / safe[None, :],
        ((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2).sqrt(),
        (same[:, None] == same[None, :]).float(),
    ], dim=2)
    return {"node": node, "edge": edge, "area": area, "name": same, "query": idx.reshape(-1)}


# --------------------------------------------------------------------------- model

class RankMessagePassing(nn.Module):
    """SAMV-DUSt3R's message passing unit: one batched multi-head attention over all node pairs,
    then a residual update that concatenates the original features with the message and fuses them.
    The edge features enter as a per-head additive bias on the attention logits, which is what lets
    "this candidate is contained in that one" reach the ranking; the paper's version has no edges."""

    def __init__(self, d: int, heads: int, edge_dim: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.edge = nn.Linear(edge_dim, heads)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
        c, d = x.shape
        q, k, v = self.qkv(x).view(c, 3, self.h, self.dh).permute(1, 2, 0, 3)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                        # [H, C, C]
        if e is not None:
            att = att + self.edge(e).permute(2, 0, 1)
        msg = (att.softmax(-1) @ v).transpose(0, 1).reshape(c, d)
        return self.norm(x + self.fuse(torch.cat([x, self.o(msg)], dim=1)))


class MaskRankGNN(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4,
                 use_edge: bool = True, use_text: bool = True, use_vis: bool = True):
        super().__init__()
        self.use_edge = use_edge
        self.scalar = nn.Sequential(nn.LayerNorm(N_SCALAR), nn.Linear(N_SCALAR, d))
        self.text = nn.Sequential(nn.LayerNorm(D_TEXT), nn.Linear(D_TEXT, d)) if use_text else None
        self.vis = nn.Sequential(nn.LayerNorm(D_VIS), nn.Linear(D_VIS, d)) if use_vis else None
        self.layers = nn.ModuleList([RankMessagePassing(d, heads, N_EDGE) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x = self.scalar(node[:, :N_SCALAR])
        if self.text is not None:
            x = x + self.text(node[:, N_SCALAR:N_SCALAR + D_TEXT])
        if self.vis is not None:
            x = x + self.vis(node[:, N_SCALAR + D_TEXT:])
        for layer in self.layers:
            x = layer(x, edge if self.use_edge else None)
        return self.head(x).squeeze(-1).sigmoid()


def save_model(model: nn.Module, args: argparse.Namespace, path: str) -> None:
    cfg = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}
    torch.save({"state": model.state_dict(), "cfg": cfg}, path)


def load_model(path: str, device: str) -> tuple[MaskRankGNN, dict]:
    d = torch.load(path, map_location=device, weights_only=True)
    c = d["cfg"]
    m = MaskRankGNN(c["dim"], c["layers"], c["heads"], not c["no_edge"], not c["no_text"],
                    not c["no_vis"]).to(device)
    m.load_state_dict(d["state"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m, c


# --------------------------------------------------------------------------- loss

def gt_score(prec: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
    """Precision-leaning target: a bleeding mask is copied into 3D, a missing one only leaves grey."""
    return prec * rec.clamp(min=0).sqrt()


def score_rank_loss(keep: torch.Tensor, g: torch.Tensor, prec: torch.Tensor, alpha: float = 0.2,
                    margin: float = 0.1, gap: float = 0.1, bce_w: float = 0.5,
                    good: float = 0.8) -> tuple[torch.Tensor, dict]:
    """Score Rank Loss (SAMV-DUSt3R eq. 1) plus a calibration BCE.

    The ranking term is what the painter needs (which candidate should win this pixel); the BCE is what
    the lambda weighting needs (a keep whose absolute value means something). g is used raw rather than
    min-max normalised per image: unlike the paper's covisible-point sums, precision and recall already
    have an absolute meaning, and stretching an all-bad image to [0, 1] would train on noise."""
    mse = F.mse_loss(keep, g)
    m = (g[:, None] - g[None, :]) > gap
    rank = (F.relu(margin - (keep[:, None] - keep[None, :])[m]).mean() if bool(m.any())
            else keep.new_zeros(()))
    bce = F.binary_cross_entropy(keep.clamp(1e-6, 1 - 1e-6), (prec >= good).float())
    return alpha * mse + (1 - alpha) * rank + bce_w * bce, {
        "mse": mse.detach().item(), "rank": rank.detach().item(), "bce": bce.detach().item()}


# --------------------------------------------------------------------------- data

class FeatSet:
    """The npz files written by mask_rank_feats.py, loaded lazily and cached in RAM."""

    def __init__(self, root: str, limit: int | None = None):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))[:limit]
        self.cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.files)

    def get(self, i: int, device: str) -> dict:
        if i not in self.cache:
            d = np.load(self.files[i])
            self.cache[i] = {k: torch.from_numpy(d[k]) for k in ("node", "edge", "prec", "rec")}
        return {k: v.to(device).float() for k, v in self.cache[i].items()}


def shuffle_text(node: torch.Tensor, k: int) -> torch.Tensor:
    """Control: permute which name's concept vector each candidate carries, inside the image."""
    n = node.shape[0] // k
    t = node[:, N_SCALAR:N_SCALAR + D_TEXT].view(n, k, D_TEXT)[torch.randperm(n, device=node.device)]
    return torch.cat([node[:, :N_SCALAR], t.reshape(-1, D_TEXT), node[:, N_SCALAR + D_TEXT:]], dim=1)


# --------------------------------------------------------------------------- rank quality

@torch.no_grad()
def rank_quality(model: nn.Module, ds: FeatSet, device: str, k: int, limit: int | None = None) -> dict:
    """Offline proxy for the repaint: per (image, name), the IoU of the candidate the ranker puts
    first, against the score-top-1 and against the best of the K. No SAM3 needed."""
    by_score, by_keep, best = [], [], []
    for i in range(min(len(ds), limit or len(ds))):
        d = ds.get(i, device)
        keep = model(d["node"], d["edge"])
        s = d["prec"] + d["rec"]
        inter = d["prec"] * d["rec"]
        iou = torch.where(s > 0, inter / (s - inter).clamp(min=1e-6), torch.zeros_like(inter))
        for n in range(keep.shape[0] // k):
            sl = slice(n * k, (n + 1) * k)
            by_score.append(float(iou[sl][0]))                    # rows are score-sorted
            by_keep.append(float(iou[sl][keep[sl].argmax()]))
            best.append(float(iou[sl].max()))
    return {"names": len(best), "iou_score_top1": float(np.mean(by_score)),
            "iou_rank_top1": float(np.mean(by_keep)), "iou_best_of_k": float(np.mean(best))}


# --------------------------------------------------------------------------- train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="training feature dir (mask_rank_feats.py)")
    ap.add_argument("--val_feats", default=None, help="holdout feature dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=32, help="must match the dump")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32, help="images per optimiser step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--bce_w", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_edge", action="store_true", help="ablation: drop the edge bias")
    ap.add_argument("--no_text", action="store_true", help="ablation: drop the concept vectors")
    ap.add_argument("--no_vis", action="store_true", help="ablation: drop the pooled FPN features")
    ap.add_argument("--shuffle_text", action="store_true",
                    help="control: permute the names' text vectors inside each image")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = FeatSet(args.feats, args.limit)
    val = FeatSet(args.val_feats) if args.val_feats else None
    print(f"train {len(train)} images, val {len(val) if val else 0}", flush=True)

    model = MaskRankGNN(args.dim, args.layers, args.heads, not args.no_edge,
                        not args.no_text, not args.no_vis).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, len(train) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    t0, step = time.time(), 0

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        run: dict[str, float] = {}
        n_seen = 0
        for b0 in range(0, len(order) - args.batch + 1, args.batch):
            opt.zero_grad()
            tot = 0.0
            for i in order[b0:b0 + args.batch]:
                d = train.get(int(i), device)
                node = shuffle_text(d["node"], args.topk) if args.shuffle_text else d["node"]
                keep = model(node, d["edge"])
                loss, parts = score_rank_loss(keep, gt_score(d["prec"], d["rec"]), d["prec"],
                                              args.alpha, args.margin, bce_w=args.bce_w)
                (loss / args.batch).backward()
                tot += float(loss)
                for k_, v_ in parts.items():
                    run[k_] = run.get(k_, 0.0) + v_
                n_seen += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                print(f"[{(time.time() - t0) / 60:5.1f} min] epoch {epoch} step {step}/{steps} "
                      f"loss {tot / args.batch:.4f} ("
                      + " ".join(f"{k_}={v_ / n_seen:.3f}" for k_, v_ in run.items()) + ")", flush=True)
        row = {"epoch": epoch, "step": step, **{k_: v_ / max(1, n_seen) for k_, v_ in run.items()}}
        for tag, ds in (("train", train), ("val", val)):
            if ds is not None and len(ds):
                q = rank_quality(model, ds, device, args.topk, limit=200)
                row[tag] = q
                print(f"  {tag} rank quality: score-top1 {q['iou_score_top1']:.3f} -> "
                      f"rank-top1 {q['iou_rank_top1']:.3f} (best of K {q['iou_best_of_k']:.3f})", flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
        save_model(model, args, os.path.join(args.out, f"rank_epoch{epoch}.pt"))
        save_model(model, args, os.path.join(args.out, "rank.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
