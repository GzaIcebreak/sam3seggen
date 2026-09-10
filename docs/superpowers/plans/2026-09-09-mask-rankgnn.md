# Mask RankGNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether SAM3's candidate masks contain the right answer (oracle gate), then train a small attention-based ranker over those candidates that replaces `colorize`'s area ordering.

**Architecture:** A pure-function library (`finetune/mask_cands.py`) extracts per-(name, query) candidates from one SAM3 forward pass and scores them against GT; an oracle CLI (`finetune/oracle_candidates.py`) reports upper bounds and an error decomposition, and dumps per-image features; a ranker (`finetune/mask_rank.py`) trains `RankMessagePassing` layers with Score Rank Loss on the dump and evaluates with the same F metrics. `sam3_to_2dmap.py` gets an optional `--rank_model` switch.

**Tech Stack:** Python 3.x in `/root/autodl-tmp/envs/sam3/bin/python` (torch 2.8.0+cu128, transformers 5.16.1, numpy 2.2.6). `unittest` tests in `tests/`, run from the repo root `/root/autodl-tmp/sam3seggen`.

Spec: `docs/superpowers/specs/2026-09-09-mask-rankgnn-design.md`.

## Global Constraints

- Every GPU job starts only after `/root/autodl-tmp/runs/v5_ce_bank/train.log` contains a line starting with `EXIT:`. Use the wait loop in Task 3 Step 2.
- Data: `/root/autodl-tmp/datasets/pv` (objects with `names.json`, `names_meta.json`, `views/az{0,135}/{render.png,ids.npy}`); split `/root/autodl-tmp/datasets/concept_bank_v3/split.json` (`{"train": [1800 ids], "holdout": [200 ids]}`); hard ids `/root/autodl-tmp/datasets/pv_hard.txt`; MakerWorld `/root/autodl-tmp/datasets/makerworld/concept_bank_clean_v2/objects`; bank `/root/autodl-tmp/datasets/concept_bank_v3/bank.pt`; SAM3 weights `/root/autodl-tmp/sam3seggen/weights/facebook/sam3`.
- Holdout eval images = `concept_bank.list_images(root, holdout_objs, ["az0", "az135"])[:240]` (same order as `concept_bank.py` line 678). Baseline must reproduce `pixel_acc 0.541 / pixel_wrong 0.205 / pixel_unassigned 0.254` at score threshold 0.5 within ±0.005, else stop and debug.
- Candidates: `score > 0.05`, mask = `sigmoid(pred_masks) > 0.5` upsampled bilinear to the image, clipped to `ids >= 0`, dropped if area < 16 px, at most 12 per name (highest score first).
- Do not modify `finetune/concept_bank.py` or `finetune/sam3_bank.py`. Import only.
- Never `git config`. Commit with `GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com git commit ...`. Only `git add` the files named in the task; the repo has unrelated uncommitted changes.
- Run all Python as `cd /root/autodl-tmp/sam3seggen && HF_HUB_OFFLINE=1 /root/autodl-tmp/envs/sam3/bin/python ...`. Abbreviated below as `$PY`.
- Tests: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_mask_cands -v` (no pytest in the env).
- Serialisation: `.npz` dumps hold only numeric arrays plus one JSON string array (`names`), so `np.load` runs with its default (no object arrays). `rank.pt` holds only tensors / ints / bools so `torch.load` runs with its default safe mode. Put torch modules in inference mode with `.train(False)`.

---

## File Structure

| File | Responsibility |
|---|---|
| `finetune/mask_cands.py` (create) | Pure functions: candidate scoring vs GT, painting operators (area order / keep order), oracle paints, error decomposition, pairwise edge features. No SAM3 import at module top. Plus one GPU function `extract_candidates()` that calls `sam3_bank` and returns a `Cands` dict. |
| `finetune/oracle_candidates.py` (create) | CLI. Loads split, samples, SAM3 + bank; runs `extract_candidates` per image; aggregates baseline / oracle / decomposition; writes JSON; `--dump` writes per-image `.npz`. `--gate` prints the spec's verdict. |
| `finetune/mask_rank.py` (create) | `RankMessagePassing`, `MaskRankGNN`, `score_rank_loss`, dump loading, `train()`, `evaluate_dump()`, `rank_paint()`, CLI `train` / `eval`. |
| `sam3_to_2dmap.py` (modify) | `segment_candidates()` (instances, not unions) and `--rank_model` in `main()`; default behaviour unchanged. |
| `tests/test_mask_cands.py` (create) | Unit tests for the pure functions. |
| `tests/test_mask_rank.py` (create) | Unit tests for loss, model shapes, `rank_paint`. |
| `tests/test_rank_colorize.py` (create) | Unit test for `rank_colorize`. |

The `Cands` dict (produced by `extract_candidates`, consumed everywhere):

```python
Cands = {
    "masks":     np.ndarray[bool, (K, H, W)],   # clipped to silhouette
    "name_idx":  np.ndarray[int64, (K,)],       # index into `names`
    "names":     list[str],                     # GT names present in this image (== list(sample.gts.keys()))
    "score":     np.ndarray[float32, (K,)],
    "area":      np.ndarray[int64, (K,)],
    "bbox":      np.ndarray[float32, (K, 4)],   # x0,y0,x1,y1 normalised to [0,1]
    "centroid":  np.ndarray[float32, (K, 2)],
    "text_vec":  np.ndarray[float32, (K, 256)],
    "vis_feat":  np.ndarray[float32, (K, C)],   # masked mean of the finest FPN map
    "precision": np.ndarray[float32, (K,)],
    "recall":    np.ndarray[float32, (K,)],
    "iou":       np.ndarray[float32, (K,)],
    "gt_lab":    np.ndarray[int64, (H, W)],     # name index per pixel, -1 outside GT-named parts
    "fg":        np.ndarray[bool, (H, W)],      # ids >= 0
}
```

---

### Task 1: `finetune/mask_cands.py` pure functions (TDD)

**Files:**
- Create: `finetune/mask_cands.py`
- Test: `tests/test_mask_cands.py`

**Interfaces:**
- Produces:
  - `gt_label_map(ids: np.ndarray, part_names: list[str], names: list[str]) -> np.ndarray[int64, (H,W)]` — `names.index(part_names[ids[p]])` per pixel, -1 where `ids < 0` or the part's name is not in `names`.
  - `score_candidates(masks: np.ndarray[bool,(K,H,W)], name_idx: np.ndarray[(K,)], gt_lab: np.ndarray[(H,W)]) -> tuple[precision, recall, iou]` each `np.ndarray[float32,(K,)]`.
  - `paint_area(masks, name_idx, keep: np.ndarray[bool,(K,)], fg, n_names: int) -> np.ndarray[int64,(H,W)]` — union kept masks per name, then smallest-union-first painting exactly like `sam3_bank.paint`; -1 unpainted.
  - `paint_keep(masks, name_idx, keep_score: np.ndarray[float32,(K,)], fg, tau: float) -> np.ndarray[int64,(H,W)]` — each pixel gets the name of the covering candidate with the highest `keep_score >= tau`; -1 otherwise.
  - `oracle_pixel(masks, name_idx, gt_lab) -> np.ndarray[int64,(H,W)]` — `gt_lab` where some candidate of that name covers the pixel, else -1.
  - `f_metrics(painted, gt_lab) -> dict` with keys `pixel_acc, pixel_wrong, pixel_unassigned, n_valid` (fractions over `gt_lab >= 0`).
  - `decompose(painted, gt_lab, masks, name_idx) -> dict` with keys `right, wrong_recoverable, wrong_missed, grey_recoverable, grey_missed` (fractions over valid pixels; sum to 1).
  - `pair_features(masks, name_idx, centroid) -> np.ndarray[float32,(K,K,5)]` = `[iou, inter/|A|, inter/|B|, centroid_dist, same_name]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mask_cands.py
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "finetune"))
import mask_cands as mc


def _toy():
    # 4x4 image; GT: name0 = left half, name1 = right-top quadrant, background bottom-right
    ids = np.array([[0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, -1, -1],
                    [0, 0, -1, -1]])
    part_names = ["body", "arm"]
    names = ["body", "arm"]
    gt = mc.gt_label_map(ids, part_names, names)
    fg = ids >= 0
    m_body_exact = gt == 0
    m_body_bleed = np.ones((4, 4), bool)            # body candidate that swallows everything
    m_arm_exact = gt == 1
    masks = np.stack([m_body_exact, m_body_bleed, m_arm_exact])
    name_idx = np.array([0, 0, 1])
    return ids, names, gt, fg, masks, name_idx


class GtLabelMapTest(unittest.TestCase):
    def test_maps_parts_to_name_indices_and_minus_one_elsewhere(self):
        ids = np.array([[0, 1], [-1, 2]])
        gt = mc.gt_label_map(ids, ["a", "b", "zzz"], ["a", "b"])
        np.testing.assert_array_equal(gt, np.array([[0, 1], [-1, -1]]))


class ScoreCandidatesTest(unittest.TestCase):
    def test_precision_recall_iou(self):
        _, _, gt, _, masks, name_idx = _toy()
        p, r, iou = mc.score_candidates(masks, name_idx, gt)
        np.testing.assert_allclose(p, [1.0, 8 / 16, 1.0])
        np.testing.assert_allclose(r, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(iou, [1.0, 8 / 16, 1.0])


class PaintTest(unittest.TestCase):
    def test_paint_area_smallest_first_no_overwrite(self):
        _, _, gt, fg, masks, name_idx = _toy()
        keep = np.array([False, True, True])       # bleeding body + exact arm
        out = mc.paint_area(masks, name_idx, keep, fg, 2)
        # arm (4 px) painted before body (12 px in silhouette): arm survives, body fills the rest of fg
        self.assertEqual(out[0, 2], 1)
        self.assertEqual(out[0, 0], 0)
        self.assertEqual(out[2, 2], -1)            # outside silhouette stays -1

    def test_paint_keep_highest_score_wins_and_tau_cuts(self):
        _, _, gt, fg, masks, name_idx = _toy()
        out = mc.paint_keep(masks, name_idx, np.array([0.9, 0.2, 0.8], np.float32), fg, tau=0.5)
        np.testing.assert_array_equal(out, gt)     # exact body (0.9) and exact arm (0.8) win, bleed cut by tau
        out2 = mc.paint_keep(masks, name_idx, np.array([0.1, 0.9, 0.8], np.float32), fg, tau=0.5)
        self.assertEqual(out2[0, 2], 0)            # bleeding body (0.9) beats arm (0.8) at an arm pixel

    def test_oracle_pixel_marks_recoverable_pixels(self):
        _, _, gt, _, masks, name_idx = _toy()
        out = mc.oracle_pixel(masks[[1, 2]], name_idx[[1, 2]], gt)   # only the bleeding body + arm
        np.testing.assert_array_equal(out, gt)


class MetricsTest(unittest.TestCase):
    def test_f_metrics_and_decompose(self):
        _, _, gt, fg, masks, name_idx = _toy()
        painted = gt.copy()
        painted[0, 0] = 1        # body pixel painted arm; a body candidate covers it -> wrong_recoverable
        painted[3, 0] = -1       # body pixel grey; a body candidate covers it -> grey_recoverable
        m = mc.f_metrics(painted, gt)
        self.assertEqual(m["n_valid"], 12)
        self.assertAlmostEqual(m["pixel_acc"], 10 / 12)
        self.assertAlmostEqual(m["pixel_wrong"], 1 / 12)
        self.assertAlmostEqual(m["pixel_unassigned"], 1 / 12)
        d = mc.decompose(painted, gt, masks, name_idx)
        self.assertAlmostEqual(d["right"], 10 / 12)
        self.assertAlmostEqual(d["wrong_recoverable"], 1 / 12)
        self.assertAlmostEqual(d["grey_recoverable"], 1 / 12)
        self.assertAlmostEqual(d["wrong_missed"] + d["grey_missed"], 0.0)
        d2 = mc.decompose(painted, gt, masks[[2]], name_idx[[2]])    # no body candidate at all
        self.assertAlmostEqual(d2["wrong_missed"], 1 / 12)
        self.assertAlmostEqual(d2["grey_missed"], 1 / 12)


class PairFeaturesTest(unittest.TestCase):
    def test_shapes_and_values(self):
        _, _, _, _, masks, name_idx = _toy()
        cen = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.25]], np.float32)
        e = mc.pair_features(masks, name_idx, cen)
        self.assertEqual(e.shape, (3, 3, 5))
        self.assertAlmostEqual(e[0, 1, 0], 8 / 16)     # iou(body_exact, bleed)
        self.assertAlmostEqual(e[0, 1, 1], 1.0)        # inter/|A|: body_exact inside bleed
        self.assertAlmostEqual(e[0, 1, 2], 8 / 16)     # inter/|B|
        self.assertEqual(e[0, 1, 4], 1.0)              # same name
        self.assertEqual(e[0, 2, 4], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_mask_cands -v`
Expected: `ModuleNotFoundError: No module named 'mask_cands'`

- [ ] **Step 3: Implement the pure functions**

```python
# finetune/mask_cands.py
"""Candidate masks from one SAM3 pass and the operators the oracle / ranker share.

A candidate is one (name, DETR query) pair whose score is above a low floor. Everything here is
numpy and independent of SAM3 except `extract_candidates`, which needs the loaded model.
"""
from __future__ import annotations

import numpy as np


def gt_label_map(ids: np.ndarray, part_names: list[str], names: list[str]) -> np.ndarray:
    lut = np.full(max(1, len(part_names)), -1, dtype=np.int64)
    pos = {n: i for i, n in enumerate(names)}
    for p, nm in enumerate(part_names):
        nm = (nm or "").strip()
        if nm in pos:
            lut[p] = pos[nm]
    out = np.full(ids.shape, -1, dtype=np.int64)
    fg = ids >= 0
    out[fg] = lut[np.clip(ids[fg], 0, len(lut) - 1)]
    return out


def score_candidates(masks: np.ndarray, name_idx: np.ndarray, gt_lab: np.ndarray):
    k = masks.shape[0]
    prec = np.zeros(k, np.float32); rec = np.zeros(k, np.float32); iou = np.zeros(k, np.float32)
    for i in range(k):
        m = masks[i]
        g = gt_lab == name_idx[i]
        inter = np.logical_and(m, g).sum()
        a, b = m.sum(), g.sum()
        prec[i] = inter / a if a else 0.0
        rec[i] = inter / b if b else 0.0
        union = a + b - inter
        iou[i] = inter / union if union else 0.0
    return prec, rec, iou


def _unions(masks: np.ndarray, name_idx: np.ndarray, keep: np.ndarray, n_names: int) -> np.ndarray:
    u = np.zeros((n_names, *masks.shape[1:]), bool)
    for i in np.nonzero(keep)[0]:
        u[name_idx[i]] |= masks[i]
    return u


def paint_area(masks, name_idx, keep, fg, n_names: int) -> np.ndarray:
    """sam3_bank.paint semantics: one union per name, smallest first, never overwrite."""
    u = _unions(masks, name_idx, keep, n_names)
    out = np.full(fg.shape, -1, dtype=np.int64)
    for n in np.argsort(u.reshape(n_names, -1).sum(1)):
        free = u[n] & fg & (out < 0)
        out[free] = n
    return out


def paint_keep(masks, name_idx, keep_score, fg, tau: float) -> np.ndarray:
    """Every silhouette pixel goes to the name of the covering candidate with the highest keep score."""
    out = np.full(fg.shape, -1, dtype=np.int64)
    best = np.full(fg.shape, -np.inf, dtype=np.float32)
    for i in np.argsort(keep_score):                      # ascending: later (higher) overwrite earlier
        if keep_score[i] < tau:
            continue
        m = masks[i] & fg
        win = m & (keep_score[i] > best)
        out[win] = name_idx[i]
        best[win] = keep_score[i]
    return out


def oracle_pixel(masks, name_idx, gt_lab) -> np.ndarray:
    out = np.full(gt_lab.shape, -1, dtype=np.int64)
    for i in range(masks.shape[0]):
        hit = masks[i] & (gt_lab == name_idx[i])
        out[hit] = name_idx[i]
    return out


def f_metrics(painted, gt_lab) -> dict:
    valid = gt_lab >= 0
    n = int(valid.sum())
    right = int((valid & (painted == gt_lab)).sum())
    wrong = int((valid & (painted >= 0) & (painted != gt_lab)).sum())
    d = max(1, n)
    return {"pixel_acc": right / d, "pixel_wrong": wrong / d,
            "pixel_unassigned": (n - right - wrong) / d, "n_valid": n}


def decompose(painted, gt_lab, masks, name_idx) -> dict:
    valid = gt_lab >= 0
    cover = oracle_pixel(masks, name_idx, gt_lab) >= 0        # some candidate of the true name covers the pixel
    right = valid & (painted == gt_lab)
    wrong = valid & (painted >= 0) & ~right
    grey = valid & (painted < 0)
    d = max(1, int(valid.sum()))
    return {"right": int(right.sum()) / d,
            "wrong_recoverable": int((wrong & cover).sum()) / d,
            "wrong_missed": int((wrong & ~cover).sum()) / d,
            "grey_recoverable": int((grey & cover).sum()) / d,
            "grey_missed": int((grey & ~cover).sum()) / d}


def pair_features(masks, name_idx, centroid) -> np.ndarray:
    k = masks.shape[0]
    flat = masks.reshape(k, -1).astype(np.float32)
    area = flat.sum(1)
    inter = flat @ flat.T
    union = area[:, None] + area[None, :] - inter
    e = np.zeros((k, k, 5), np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        e[..., 0] = np.where(union > 0, inter / union, 0.0)
        e[..., 1] = np.where(area[:, None] > 0, inter / area[:, None], 0.0)
        e[..., 2] = np.where(area[None, :] > 0, inter / area[None, :], 0.0)
    e[..., 3] = np.linalg.norm(centroid[:, None, :] - centroid[None, :, :], axis=-1)
    e[..., 4] = (name_idx[:, None] == name_idx[None, :]).astype(np.float32)
    return e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_mask_cands -v`
Expected: `OK` with 7 tests.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/sam3seggen && git add finetune/mask_cands.py tests/test_mask_cands.py && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "mask_cands: candidate scoring, paint operators, oracle and error decomposition (pure numpy)"
```

---

### Task 2: `extract_candidates()` (GPU) in `finetune/mask_cands.py`

**Files:**
- Modify: `finetune/mask_cands.py` (append)

**Interfaces:**
- Consumes: `sam3_bank.encode_image`, `text_features_batch`, `text_query_vecs`, `bank_forward`, `batch_scores`, `fill_template`; `ConceptBank.offset(name, use_e0=..., query_vec=...)` (called the same way at `sam3_to_2dmap.py` line 143); `concept_bank.ImageSample` (fields `image`, `ids`, `names`, `gts`, `obj_name`).
- Produces: `extract_candidates(processor, model, bank, sample, device, score_min=0.05, max_per_name=12, min_area=16, chunk=12) -> Cands`.

- [ ] **Step 1: Append the function**

```python
# append to finetune/mask_cands.py
import torch
import torch.nn.functional as F


def _finest_fpn(vis) -> torch.Tensor:
    """[1, C, h, w] feature map with the largest spatial size among the vision outputs."""
    cands = []
    fpn = getattr(vis, "fpn_hidden_states", None)
    if fpn is not None:
        cands += [t for t in fpn if torch.is_tensor(t) and t.dim() == 4]
    lh = getattr(vis, "last_hidden_state", None)
    if torch.is_tensor(lh) and lh.dim() == 4:
        cands.append(lh)
    if not cands:
        raise RuntimeError("no 4-D vision feature map found on the SAM3 vision output")
    return max(cands, key=lambda t: t.shape[-1] * t.shape[-2])


@torch.no_grad()
def extract_candidates(processor, model, bank, sample, device: str, score_min: float = 0.05,
                       max_per_name: int = 12, min_area: int = 16, chunk: int = 12) -> dict:
    import sam3_bank as sb
    names = list(sample.gts.keys())
    h, w = sample.ids.shape
    fg = sample.ids >= 0
    gt_lab = gt_label_map(sample.ids, sample.names, names)
    vis = sb.encode_image(processor, model, sample.image, device)
    feat = _finest_fpn(vis).float()                                   # [1, C, fh, fw]
    template = bank.template if bank is not None else "{name}"
    use_qv = bank is not None and bank.nn_cos > 0 and bank.tvec is not None
    fg_t = torch.from_numpy(fg).to(device)

    masks, nidx, score, tvecs, vfeat = [], [], [], [], []
    for k0 in range(0, len(names), chunk):
        part = names[k0:k0 + chunk]
        prompts = [sb.fill_template(template, n, sample.obj_name) for n in part]
        tf, am = sb.text_features_batch(processor, model, prompts, device)
        qv = sb.text_query_vecs(tf, am)                               # [n, 256] un-shifted
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out, bias = sb.bank_forward(model, vis, tf, am, bank, part, qv if use_qv else None)
        scores = sb.batch_scores(out, bias).float()                   # [n, Q]
        for j, n in enumerate(part):
            keep = torch.nonzero(scores[j] > score_min).flatten()
            if keep.numel() == 0:
                continue
            order = torch.argsort(scores[j][keep], descending=True)[:max_per_name]
            keep = keep[order]
            m = F.interpolate(out.pred_masks[j][keep].float().sigmoid()[None], size=(h, w),
                              mode="bilinear", align_corners=False)[0] > 0.5           # [k, h, w]
            m &= fg_t
            area = m.flatten(1).sum(1)
            ok = area >= min_area
            if not bool(ok.any()):
                continue
            m, keep = m[ok], keep[ok]
            ms = F.interpolate(m[None].float(), size=feat.shape[-2:], mode="area")[0]  # [k, fh, fw]
            pooled = (feat[0][None] * ms[:, None]).sum((-1, -2)) / ms.sum((-1, -2)).clamp(min=1e-6)[:, None]
            tv = qv[j]
            if bank is not None:
                tv = tv + bank.offset(n, use_e0=True, query_vec=qv[j]).float()
            kk = int(ok.sum())
            masks.append(m.cpu().numpy()); nidx += [k0 + j] * kk
            score.append(scores[j][keep].cpu().numpy())
            tvecs.append(tv[None].expand(kk, -1).cpu().numpy())
            vfeat.append(pooled.cpu().numpy())
        del out
    if masks:
        M = np.concatenate(masks, 0); S = np.concatenate(score).astype(np.float32)
        T = np.concatenate(tvecs).astype(np.float32); V = np.concatenate(vfeat).astype(np.float32)
    else:
        M = np.zeros((0, h, w), bool); S = np.zeros(0, np.float32)
        T = np.zeros((0, 256), np.float32); V = np.zeros((0, feat.shape[1]), np.float32)
    N = np.asarray(nidx, dtype=np.int64)
    K = M.shape[0]
    bbox = np.zeros((K, 4), np.float32); cen = np.zeros((K, 2), np.float32); area = np.zeros(K, np.int64)
    for i in range(K):
        ys, xs = np.nonzero(M[i])
        area[i] = len(ys)
        bbox[i] = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
        cen[i] = [xs.mean() / w, ys.mean() / h]
    prec, rec, iou = score_candidates(M, N, gt_lab)
    return {"masks": M, "name_idx": N, "names": names, "score": S, "area": area, "bbox": bbox,
            "centroid": cen, "text_vec": T, "vis_feat": V, "precision": prec, "recall": rec, "iou": iou,
            "gt_lab": gt_lab, "fg": fg}
```

If `ConceptBank.offset` has different keyword names, match them (read `finetune/sam3_bank.py` around lines 150–216).

- [ ] **Step 2: Smoke test on one image (GPU; wait for v5_ce_bank first)**

```bash
cd /root/autodl-tmp/sam3seggen && while ! grep -q '^EXIT:' /root/autodl-tmp/runs/v5_ce_bank/train.log; do sleep 60; done && \
HF_HUB_OFFLINE=1 $PY - <<'EOF'
import sys, json, torch
sys.path.insert(0, "finetune")
import sam3_bank as sb, mask_cands as mc
from concept_bank import ImageSample
from sam3_to_2dmap import load_sam3
dev = "cuda"
proc, model = load_sam3("/root/autodl-tmp/sam3seggen/weights/facebook/sam3", dev)
bank = sb.ConceptBank.load("/root/autodl-tmp/datasets/concept_bank_v3/bank.pt", dev)
with open("/root/autodl-tmp/datasets/concept_bank_v3/split.json") as f:
    hold = json.load(f)["holdout"]
s = ImageSample("/root/autodl-tmp/datasets/pv", hold[0], "az0", True)
c = mc.extract_candidates(proc, model, bank, s, dev)
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in c.items() if k != "names"}, c["names"])
print("score", c["score"].round(2), "prec", c["precision"].round(2))
EOF
```

Expected: shapes print; `masks` is `(K, H, W)` with `K` between 1 and `12 * len(names)`; `vis_feat` second dim is the FPN channel count (record it); no exception.

- [ ] **Step 3: Commit**

```bash
cd /root/autodl-tmp/sam3seggen && git add finetune/mask_cands.py && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "mask_cands: extract per-(name, query) SAM3 candidates with GT scores and pooled features"
```

---

### Task 3: `finetune/oracle_candidates.py` — oracle report, decomposition, dump, gate

**Files:**
- Create: `finetune/oracle_candidates.py`

**Interfaces:**
- Consumes: `mask_cands.extract_candidates`, `paint_area`, `oracle_pixel`, `f_metrics`, `decompose`, `pair_features`; `concept_bank.ImageSample`, `list_images`, `list_objects`, `read_id_file`.
- Produces:
  - `<out>/{holdout,hard,mw}.json` — per set: `baseline` (thresholds 0.4/0.5/0.6 → f_metrics), `oracle_select` (p → f_metrics), `oracle_pixel`, `decompose`, `best_score_below_0.5` (fraction of names whose best-IoU candidate has score < 0.5), `top_confusions` (list of `[painted_name, gt_name, pixels]`, top 20).
  - `--dump DIR`: per image `DIR/<tag>/<obj>_<az>.npz` with every `Cands` key except `masks`/`gt_lab`/`fg` when mode is `feats`, and all keys when mode is `full` (masks packed with `np.packbits`, `mask_w` stored); `names` stored as one JSON string array; plus `pair` `[K,K,5]`, `gt_score = precision * sqrt(recall)`, `fg_area`.
  - `--gate`: prints `GATE PASS` / `GATE FAIL` / `GATE MID` per the spec §3.5.

- [ ] **Step 1: Write the CLI**

```python
# finetune/oracle_candidates.py
"""Oracle ceilings and error decomposition for SAM3 candidate masks (spec 2026-09-09-mask-rankgnn §3).

    $PY finetune/oracle_candidates.py --out /root/autodl-tmp/runs/oracle_v3/v3 \
        --bank /root/autodl-tmp/datasets/concept_bank_v3/bank.pt [--dump DIR --dump_mode feats|full --dump_train]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

import sam3_bank as sb
import mask_cands as mc
from concept_bank import ImageSample, list_images, list_objects, read_id_file
from sam3_to_2dmap import load_sam3

P_GRID = (0.6, 0.7, 0.8, 0.9)
T_GRID = (0.4, 0.5, 0.6)


def _acc():
    return {"baseline": {t: Counter() for t in T_GRID}, "oracle_select": {p: Counter() for p in P_GRID},
            "oracle_pixel": Counter(), "decompose": Counter(), "best_lowscore": [0, 0],
            "confusions": Counter(), "images": 0}


def _add(cnt: Counter, m: dict, n: int):
    for k in ("pixel_acc", "pixel_wrong", "pixel_unassigned"):
        cnt[k] += m[k] * n
    cnt["n"] += n


def _fin(cnt: Counter) -> dict:
    n = max(1, cnt["n"])
    return {k: cnt[k] / n for k in ("pixel_acc", "pixel_wrong", "pixel_unassigned")} | {"n_valid": cnt["n"]}


def process(c: dict, acc: dict):
    masks, nidx, names, gt, fg = c["masks"], c["name_idx"], c["names"], c["gt_lab"], c["fg"]
    n_valid = int((gt >= 0).sum())
    if n_valid == 0 or masks.shape[0] == 0:
        return
    for t in T_GRID:
        painted = mc.paint_area(masks, nidx, c["score"] > t, fg, len(names))
        _add(acc["baseline"][t], mc.f_metrics(painted, gt), n_valid)
        if t == 0.5:
            for k, v in mc.decompose(painted, gt, masks, nidx).items():
                acc["decompose"][k] += v * n_valid
            acc["decompose"]["n"] += n_valid
            wrong = (gt >= 0) & (painted >= 0) & (painted != gt)
            for (a, b), cnt in Counter(zip(painted[wrong].tolist(), gt[wrong].tolist())).items():
                acc["confusions"][(names[a], names[b])] += cnt
    for p in P_GRID:
        painted = mc.paint_area(masks, nidx, c["precision"] >= p, fg, len(names))
        _add(acc["oracle_select"][p], mc.f_metrics(painted, gt), n_valid)
    _add(acc["oracle_pixel"], mc.f_metrics(mc.oracle_pixel(masks, nidx, gt), gt), n_valid)
    for ni in range(len(names)):
        sel = np.nonzero(nidx == ni)[0]
        if len(sel) == 0:
            continue
        best = sel[np.argmax(c["iou"][sel])]
        acc["best_lowscore"][0] += int(c["score"][best] < 0.5)
        acc["best_lowscore"][1] += 1
    acc["images"] += 1


def finish(acc: dict) -> dict:
    n = max(1, acc["decompose"]["n"])
    return {"images": acc["images"],
            "baseline": {f"{t:g}": _fin(acc["baseline"][t]) for t in T_GRID},
            "oracle_select": {f"{p:g}": _fin(acc["oracle_select"][p]) for p in P_GRID},
            "oracle_pixel": _fin(acc["oracle_pixel"]),
            "decompose": {k: v / n for k, v in acc["decompose"].items() if k != "n"},
            "best_score_below_0.5": acc["best_lowscore"][0] / max(1, acc["best_lowscore"][1]),
            "top_confusions": [[a, b, c] for (a, b), c in acc["confusions"].most_common(20)]}


def dump(c: dict, path: str, mode: str):
    keep = {k: v for k, v in c.items() if k not in ("masks", "gt_lab", "fg", "names")}
    keep["names"] = np.array(json.dumps(c["names"]))          # plain str array, no object dtype
    keep["pair"] = mc.pair_features(c["masks"], c["name_idx"], c["centroid"])
    keep["gt_score"] = (c["precision"] * np.sqrt(c["recall"])).astype(np.float32)
    keep["fg_area"] = np.int64(c["fg"].sum())
    if mode == "full":
        keep["masks"] = np.packbits(c["masks"], axis=-1); keep["mask_w"] = np.int64(c["masks"].shape[-1])
        keep["gt_lab"] = c["gt_lab"].astype(np.int16); keep["fg"] = c["fg"]
    np.savez_compressed(path, **keep)


def gate(r: dict) -> str:
    sel = r["oracle_select"]["0.8"]["pixel_acc"]; pix = r["oracle_pixel"]["pixel_acc"]
    rec = r["decompose"]["wrong_recoverable"] + r["decompose"]["grey_recoverable"]
    if sel >= 0.65 and rec >= 0.15:
        return f"GATE PASS  (select@0.8 {sel:.3f} >= 0.65, recoverable {rec:.3f} >= 0.15)"
    if pix <= 0.60 or rec < 0.08:
        return f"GATE FAIL  (oracle_pixel {pix:.3f}, recoverable {rec:.3f})"
    sel7 = r["oracle_select"]["0.7"]["pixel_acc"]
    return (f"GATE MID   (select@0.8 {sel:.3f}, select@0.7 {sel7:.3f}, recoverable {rec:.3f})"
            f" -> treat as FAIL unless select@0.7 >= 0.65")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/root/autodl-tmp/datasets/pv")
    ap.add_argument("--split", default="/root/autodl-tmp/datasets/concept_bank_v3/split.json")
    ap.add_argument("--hard_file", default="/root/autodl-tmp/datasets/pv_hard.txt")
    ap.add_argument("--mw_root", default="/root/autodl-tmp/datasets/makerworld/concept_bank_clean_v2/objects")
    ap.add_argument("--model", default="/root/autodl-tmp/sam3seggen/weights/facebook/sam3")
    ap.add_argument("--bank", default=None, help="bank.pt; omit for raw SAM3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_images", type=int, default=240)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--dump_mode", choices=["feats", "full"], default="full")
    ap.add_argument("--dump_train", action="store_true", help="also dump the 1800 train objects (feats mode)")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_sam3(args.model, dev)
    bank = sb.ConceptBank.load(args.bank, dev) if args.bank else None
    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)
    az = ["az0", "az135"]
    hold_imgs = list_images(args.dataset_root, split["holdout"], az)[:args.eval_images]
    hard = read_id_file(args.hard_file) & set(split["holdout"])
    mw_objs = list_objects(args.mw_root)
    mw_imgs = list_images(args.mw_root, mw_objs, az)
    sets = {"holdout": _acc(), "hard": _acc(), "mw": _acc()}
    t0 = time.time()

    def run(root, imgs, keys, tag, mode):
        for i, (o, a) in enumerate(imgs):
            s = ImageSample(root, o, a, True)
            if not s.gts:
                continue
            c = mc.extract_candidates(proc, model, bank, s, dev)
            for k in keys(o):
                process(c, sets[k])
            if args.dump:
                os.makedirs(os.path.join(args.dump, tag), exist_ok=True)
                dump(c, os.path.join(args.dump, tag, f"{o}_{a}.npz"), mode)
            if i % 50 == 0:
                print(f"[{(time.time() - t0) / 60:5.1f} min] {tag} {i}/{len(imgs)}", flush=True)

    run(args.dataset_root, hold_imgs, lambda o: ["holdout", "hard"] if o in hard else ["holdout"], "holdout", args.dump_mode)
    run(args.mw_root, mw_imgs, lambda o: ["mw"], "mw", args.dump_mode)
    res = {k: finish(v) for k, v in sets.items()}
    for k, r in res.items():
        with open(os.path.join(args.out, f"{k}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        b, s8, op, d = r["baseline"]["0.5"], r["oracle_select"]["0.8"], r["oracle_pixel"], r["decompose"]
        print(f"{k:8s} imgs {r['images']:4d} | v3@0.5 acc {b['pixel_acc']:.3f} wrong {b['pixel_wrong']:.3f} grey {b['pixel_unassigned']:.3f}"
              f" | select@0.8 acc {s8['pixel_acc']:.3f} wrong {s8['pixel_wrong']:.3f} | pixel-oracle {op['pixel_acc']:.3f}"
              f" | recoverable wrong {d['wrong_recoverable']:.3f} grey {d['grey_recoverable']:.3f}"
              f" | missed wrong {d['wrong_missed']:.3f} grey {d['grey_missed']:.3f} | best<0.5 {r['best_score_below_0.5']:.2f}", flush=True)
        print("   confusions:", r["top_confusions"][:8], flush=True)
    if args.gate:
        print(gate(res["holdout"]), flush=True)
    if args.dump and args.dump_train:
        train_imgs = list_images(args.dataset_root, split["train"], az)
        run(args.dataset_root, train_imgs, lambda o: [], "train", "feats")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the oracle with the v3 bank (GPU; after v5_ce_bank)**

```bash
cd /root/autodl-tmp/sam3seggen && mkdir -p /root/autodl-tmp/runs/oracle_v3 && \
while ! grep -q '^EXIT:' /root/autodl-tmp/runs/v5_ce_bank/train.log; do sleep 60; done && \
HF_HUB_OFFLINE=1 $PY finetune/oracle_candidates.py --out /root/autodl-tmp/runs/oracle_v3/v3 \
  --bank /root/autodl-tmp/datasets/concept_bank_v3/bank.pt --gate \
  --dump /root/autodl-tmp/runs/mask_rank_v1/feats --dump_mode full 2>&1 | tee /root/autodl-tmp/runs/oracle_v3/v3.log
```

Expected: `holdout` line shows `v3@0.5 acc 0.54x wrong 0.20x grey 0.25x` (within ±0.005 of 0.541 / 0.205 / 0.254). If not, the candidate union differs from `sam3_bank.batch_union_masks` — for one image compare `paint_area(masks, nidx, score > 0.5, ...)` against `sb.paint(sb.batch_union_masks(out, (h, w), 0.5, bias=bias)[:len(names)], fg)` and fix before continuing (likely culprits: `min_area` dropping, `max_per_name` truncation, or the `fg` clip — `sb.paint` also clips to `fg`). Ends with a `GATE ...` line. Runtime ≈ 10–20 min.

- [ ] **Step 3: Run the raw-SAM3 control**

```bash
cd /root/autodl-tmp/sam3seggen && HF_HUB_OFFLINE=1 $PY finetune/oracle_candidates.py --out /root/autodl-tmp/runs/oracle_v3/none 2>&1 | tee /root/autodl-tmp/runs/oracle_v3/none.log
```

Expected: `holdout` baseline ≈ `0.318 / 0.096 / 0.587` (raw SAM3 @0.5 from `REPORT_concept_bank_v4_eval.md` §2).

- [ ] **Step 4: Commit and report**

```bash
cd /root/autodl-tmp/sam3seggen && git add finetune/oracle_candidates.py && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "oracle_candidates: ceilings, error decomposition and feature dump for SAM3 candidate masks"
```

Report the three summary lines (holdout / hard / mw), the GATE verdict, and the top confusions to the user. **STOP here if GATE FAIL (or MID without select@0.7 ≥ 0.65).** Tasks 4–7 run only on PASS.

- [ ] **Step 5 (PASS only): dump train features**

```bash
cd /root/autodl-tmp/sam3seggen && HF_HUB_OFFLINE=1 $PY finetune/oracle_candidates.py --out /root/autodl-tmp/runs/oracle_v3/v3_train \
  --bank /root/autodl-tmp/datasets/concept_bank_v3/bank.pt --dump /root/autodl-tmp/runs/mask_rank_v1/feats --dump_mode full --dump_train \
  2>&1 | tee /root/autodl-tmp/runs/oracle_v3/v3_train.log
```

Expected: `feats/train/*.npz` ≈ 3600 files (feats mode, a few hundred KB each); ≈ 1–2 h. (The holdout/mw dumps are rewritten identically; harmless.)

---

### Task 4: `finetune/mask_rank.py` — model, loss, features, rank_paint (TDD)

**Files:**
- Create: `finetune/mask_rank.py`
- Test: `tests/test_mask_rank.py`

**Interfaces:**
- Consumes: dump `.npz` keys `score, area, bbox, centroid, text_vec, vis_feat, name_idx, pair, gt_score, precision, fg_area` (+ `masks, mask_w, gt_lab, fg` in full mode); `mask_cands.paint_keep`, `paint_area`, `f_metrics`.
- Produces:
  - `node_features(d: dict) -> np.ndarray[float32, (K, F)]` = `[score, log(area / fg_area), bbox(4), centroid(2), multi_instance_flag, text_vec(256), vis_feat(C)]`; `F = 9 + 256 + C`.
  - `class RankMessagePassing(nn.Module)`: `__init__(d=128, heads=4, edge_dim=5, use_edge=True)`, `forward(x: [B,K,d], edge: [B,K,K,edge_dim], mask: [B,K] bool) -> [B,K,d]`.
  - `class MaskRankGNN(nn.Module)`: `__init__(in_dim, d=128, layers=3, heads=4, use_edge=True)`, `forward(x, edge, mask) -> keep: [B,K]` in (0,1), zero on padded slots.
  - `score_rank_loss(keep, gt, mask, alpha=0.2, margin=0.1, bce_w=0.5, prec=None) -> Tensor`.
  - `rank_paint(masks, name_idx, keep: np.ndarray, fg, tau, mode="keep"|"area") -> painted`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mask_rank.py
import os, sys, unittest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "finetune"))
import mask_rank as mr


class ModelShapeTest(unittest.TestCase):
    def test_forward_shapes_and_range(self):
        B, K, F = 2, 7, 9 + 256 + 32
        x = torch.randn(B, K, F); e = torch.rand(B, K, K, 5)
        m = torch.ones(B, K, dtype=torch.bool); m[1, 5:] = False
        net = mr.MaskRankGNN(in_dim=F, d=32, layers=2, heads=4)
        keep = net(x, e, m)
        self.assertEqual(keep.shape, (B, K))
        self.assertTrue(bool(((keep >= 0) & (keep <= 1)).all()))
        self.assertTrue(bool((keep[1, 5:] == 0).all()))

    def test_padding_does_not_change_real_nodes(self):
        F = 9 + 256 + 8
        net = mr.MaskRankGNN(in_dim=F, d=16, layers=1, heads=2).train(False)
        x = torch.randn(1, 4, F); e = torch.rand(1, 4, 4, 5); m = torch.ones(1, 4, dtype=torch.bool)
        m2 = m.clone(); m2[0, 3] = False
        x2 = x.clone(); x2[0, 3] = 99.0                    # garbage in the padded slot must not leak
        a = net(x, e, m2)[0, :3]
        b = net(x2, e, m2)[0, :3]
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


class LossTest(unittest.TestCase):
    def test_perfect_order_gives_lower_loss_than_inverted(self):
        gt = torch.tensor([[1.0, 0.5, 0.0]]); m = torch.ones(1, 3, dtype=torch.bool)
        prec = torch.tensor([[0.9, 0.7, 0.1]])
        good = mr.score_rank_loss(torch.tensor([[0.9, 0.5, 0.1]]), gt, m, prec=prec)
        bad = mr.score_rank_loss(torch.tensor([[0.1, 0.5, 0.9]]), gt, m, prec=prec)
        self.assertLess(good.item(), bad.item())

    def test_padded_nodes_ignored(self):
        gt = torch.tensor([[1.0, 0.0, 0.3]]); m = torch.tensor([[True, True, False]])
        prec = torch.tensor([[0.9, 0.1, 0.5]])
        a = mr.score_rank_loss(torch.tensor([[0.8, 0.2, 0.0]]), gt, m, prec=prec)
        b = mr.score_rank_loss(torch.tensor([[0.8, 0.2, 1.0]]), gt, m, prec=prec)
        self.assertAlmostEqual(a.item(), b.item(), places=6)


class RankPaintTest(unittest.TestCase):
    def test_keep_mode_uses_highest_keep_and_area_mode_uses_union_order(self):
        fg = np.ones((2, 2), bool)
        masks = np.array([[[1, 1], [1, 1]], [[1, 0], [0, 0]]], bool)     # big body, small arm at (0,0)
        name_idx = np.array([0, 1])
        keep = np.array([0.9, 0.6], np.float32)
        out_keep = mr.rank_paint(masks, name_idx, keep, fg, tau=0.5, mode="keep")
        self.assertEqual(out_keep[0, 0], 0)                                 # body 0.9 beats arm 0.6
        out_area = mr.rank_paint(masks, name_idx, keep, fg, tau=0.5, mode="area")
        self.assertEqual(out_area[0, 0], 1)                                 # smallest-first keeps arm


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_mask_rank -v`
Expected: `ModuleNotFoundError: No module named 'mask_rank'`

- [ ] **Step 3: Implement model, loss, features, rank_paint**

```python
# finetune/mask_rank.py
"""Mask RankGNN: rank SAM3 candidate masks within one image (spec 2026-09-09-mask-rankgnn §4).

    $PY finetune/mask_rank.py train --feats /root/autodl-tmp/runs/mask_rank_v1/feats --out /root/autodl-tmp/runs/mask_rank_v1
    $PY finetune/mask_rank.py eval  --feats ... --ckpt .../rank.pt [--shuffle_text] [--tau 0.3,0.4,0.5,0.6]
"""
from __future__ import annotations

import argparse, glob, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mask_cands as mc

N_BASE = 9   # score, log_area, bbox(4), centroid(2), multi_instance


def node_features(d: dict) -> np.ndarray:
    K = int(d["score"].shape[0])
    if K == 0:
        return np.zeros((0, N_BASE + d["text_vec"].shape[1] + d["vis_feat"].shape[1]), np.float32)
    fg_area = float(d["fg_area"]) if "fg_area" in d else float(d["area"].max())
    uniq, counts = np.unique(d["name_idx"], return_counts=True)
    multi = np.isin(d["name_idx"], uniq[counts > 1]).astype(np.float32)
    base = np.concatenate([d["score"][:, None], np.log(d["area"] / max(1.0, fg_area))[:, None],
                           d["bbox"], d["centroid"], multi[:, None]], 1).astype(np.float32)
    return np.concatenate([base, d["text_vec"], d["vis_feat"]], 1).astype(np.float32)


class RankMessagePassing(nn.Module):
    """Global multi-head self-attention over candidates with an additive per-head edge bias
    (SAMV-DUSt3R RankMessagePassing + edge bias), residual + LayerNorm + FFN."""

    def __init__(self, d: int = 128, heads: int = 4, edge_dim: int = 5, use_edge: bool = True):
        super().__init__()
        self.h, self.dk, self.use_edge = heads, d // heads, use_edge
        self.qkv = nn.Linear(d, 3 * d)
        self.edge = nn.Linear(edge_dim, heads) if use_edge else None
        self.out = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x, edge, mask):
        B, K, d = x.shape
        q, k, v = self.qkv(x).view(B, K, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)     # [B,h,K,dk]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)                              # [B,h,K,K]
        if self.use_edge:
            att = att + self.edge(edge).permute(0, 3, 1, 2)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        att = torch.softmax(att, -1).nan_to_num(0.0)
        msg = (att @ v).transpose(1, 2).reshape(B, K, d)
        x = self.n1(x + self.out(msg))
        x = self.n2(x + self.ffn(x))
        return x * mask[..., None]


class MaskRankGNN(nn.Module):
    def __init__(self, in_dim: int, d: int = 128, layers: int = 3, heads: int = 4, use_edge: bool = True):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.LayerNorm(d))
        self.layers = nn.ModuleList([RankMessagePassing(d, heads, 5, use_edge) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x, edge, mask):
        x = self.inp(x) * mask[..., None]
        for layer in self.layers:
            x = layer(x, edge, mask)
        return torch.sigmoid(self.head(x).squeeze(-1)) * mask


def score_rank_loss(keep, gt, mask, alpha: float = 0.2, margin: float = 0.1, bce_w: float = 0.5, prec=None):
    """alpha * MSE + (1 - alpha) * margin ranking over pairs with |gt_i - gt_j| > 0.1, + bce_w * BCE(keep, prec >= 0.8)."""
    m = mask.float()
    mse = ((keep - gt) ** 2 * m).sum() / m.sum().clamp(min=1)
    diff_gt = gt[:, :, None] - gt[:, None, :]                                      # [B,K,K]
    pair = (diff_gt.abs() > 0.1) & mask[:, :, None] & mask[:, None, :]
    sign = torch.sign(diff_gt)
    rank = F.relu(margin - sign * (keep[:, :, None] - keep[:, None, :]))
    rank = (rank * pair).sum() / pair.float().sum().clamp(min=1)
    loss = alpha * mse + (1 - alpha) * rank
    if prec is not None and bce_w > 0:
        tgt = (prec >= 0.8).float()
        bce = F.binary_cross_entropy(keep.clamp(1e-6, 1 - 1e-6), tgt, reduction="none")
        loss = loss + bce_w * (bce * m).sum() / m.sum().clamp(min=1)
    return loss


def rank_paint(masks, name_idx, keep, fg, tau: float, mode: str = "keep"):
    if mode == "keep":
        return mc.paint_keep(masks, name_idx, keep, fg, tau)
    n_names = int(name_idx.max()) + 1 if len(name_idx) else 0
    return mc.paint_area(masks, name_idx, keep >= tau, fg, n_names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_mask_rank -v`
Expected: `OK` with 5 tests.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/sam3seggen && git add finetune/mask_rank.py tests/test_mask_rank.py && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "mask_rank: RankMessagePassing ranker, Score Rank Loss and rank_paint"
```

---

### Task 5: `mask_rank.py` dataset, training and evaluation CLI

**Files:**
- Modify: `finetune/mask_rank.py` (append)

**Interfaces:**
- Consumes: Task 4 classes; dump directory layout `feats/{train,holdout,mw}/<obj>_<az>.npz`; hard ids from `/root/autodl-tmp/datasets/pv_hard.txt`.
- Produces: `<out>/rank.pt` (dict of tensors/ints/bools: `state, in_dim, d, layers, heads, use_edge, use_vis`), `<out>/log.jsonl`, `<out>/eval.json` with per set (`holdout`, `hard`, `mw`) × tau × mode (`keep`, `area`) → f_metrics. `load_model(ckpt, device) -> (net, ckpt_dict)`; `batch_items(items, device) -> (X, E, M, G, P)`.

- [ ] **Step 1: Append dataset, train and eval**

```python
# append to finetune/mask_rank.py

def load_item(path: str, use_vis: bool, shuffle_text: bool, rng: random.Random | None) -> dict:
    d = dict(np.load(path))
    if not use_vis:
        d["vis_feat"] = np.zeros((d["vis_feat"].shape[0], 0), np.float32)
    if shuffle_text and d["text_vec"].shape[0] > 1:
        perm = list(range(d["text_vec"].shape[0]))
        (rng or random).shuffle(perm)
        d["text_vec"] = d["text_vec"][np.asarray(perm)]
    return d


def batch_items(items: list[dict], device):
    K = max(1, max(int(x["score"].shape[0]) for x in items))
    Fdim = node_features(items[0]).shape[1]
    X = torch.zeros(len(items), K, Fdim); E = torch.zeros(len(items), K, K, 5)
    M = torch.zeros(len(items), K, dtype=torch.bool); G = torch.zeros(len(items), K); P = torch.zeros(len(items), K)
    for b, d in enumerate(items):
        k = int(d["score"].shape[0])
        if k == 0:
            continue
        X[b, :k] = torch.from_numpy(node_features(d)); E[b, :k, :k] = torch.from_numpy(d["pair"])
        M[b, :k] = True; G[b, :k] = torch.from_numpy(d["gt_score"]); P[b, :k] = torch.from_numpy(d["precision"])
    return X.to(device), E.to(device), M.to(device), G.to(device), P.to(device)


def load_model(ckpt: str, device):
    c = torch.load(ckpt, map_location=device)
    net = MaskRankGNN(int(c["in_dim"]), int(c["d"]), int(c["layers"]), int(c["heads"]), bool(c["use_edge"])).to(device).train(False)
    net.load_state_dict(c["state"])
    return net, c


@torch.no_grad()
def evaluate_dump(feats: str, ckpt: str, hard_file: str, taus=(0.3, 0.4, 0.5, 0.6), shuffle_text=False, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, c = load_model(ckpt, dev)
    hard = set()
    if os.path.exists(hard_file):
        with open(hard_file, encoding="utf-8") as f:
            hard = {l.strip() for l in f if l.strip() and not l.startswith("#")}
    rng = random.Random(seed)
    acc = {s: {f"{t:g}": {m: {"acc": 0.0, "wrong": 0.0, "grey": 0.0, "n": 0} for m in ("keep", "area")} for t in taus}
           for s in ("holdout", "hard", "mw")}
    for tag in ("holdout", "mw"):
        for f in sorted(glob.glob(os.path.join(feats, tag, "*.npz"))):
            d = load_item(f, bool(c["use_vis"]), shuffle_text, rng)
            if d["score"].shape[0] == 0 or "masks" not in d:
                continue
            X, E, M, _, _ = batch_items([d], dev)
            keep = net(X, E, M)[0, :d["score"].shape[0]].cpu().numpy()
            masks = np.unpackbits(d["masks"], axis=-1, count=int(d["mask_w"])).astype(bool)
            gt, fg = d["gt_lab"].astype(np.int64), d["fg"]
            n_valid = int((gt >= 0).sum())
            sets = [tag] + (["hard"] if tag == "holdout" and os.path.basename(f).split("_")[0] in hard else [])
            for t in taus:
                for mode in ("keep", "area"):
                    m = mc.f_metrics(rank_paint(masks, d["name_idx"], keep, fg, t, mode), gt)
                    for s in sets:
                        a = acc[s][f"{t:g}"][mode]
                        a["acc"] += m["pixel_acc"] * n_valid; a["wrong"] += m["pixel_wrong"] * n_valid
                        a["grey"] += m["pixel_unassigned"] * n_valid; a["n"] += n_valid
    out = {}
    for s, per_t in acc.items():
        out[s] = {}
        for t, per_m in per_t.items():
            out[s][t] = {m: {"pixel_acc": a["acc"] / max(1, a["n"]), "pixel_wrong": a["wrong"] / max(1, a["n"]),
                             "pixel_unassigned": a["grey"] / max(1, a["n"]), "n_valid": a["n"]} for m, a in per_m.items()}
    return out


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed); torch.manual_seed(args.seed)
    files = sorted(glob.glob(os.path.join(args.feats, "train", "*.npz")))
    data = [load_item(f, args.use_vis, False, None) for f in files]
    data = [d for d in data if d["score"].shape[0] > 0]
    in_dim = node_features(data[0]).shape[1]
    net = MaskRankGNN(in_dim, args.d, args.layers, args.heads, args.use_edge).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * math.ceil(len(data) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "rank.pt")
    step, t0 = 0, time.time()
    with open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8") as log:
        for ep in range(args.epochs):
            net.train(True)
            rng.shuffle(data)
            for i in range(0, len(data), args.batch):
                X, E, M, G, P = batch_items(data[i:i + args.batch], dev)
                loss = score_rank_loss(net(X, E, M), G, M, prec=P)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
                step += 1
                if step % 20 == 0:
                    log.write(json.dumps({"step": step, "epoch": ep + 1, "loss": float(loss)}) + "\n"); log.flush()
                    print(f"[{(time.time() - t0) / 60:5.1f} min] ep {ep + 1} step {step}/{steps} loss {float(loss):.4f}", flush=True)
            torch.save({"state": net.state_dict(), "in_dim": in_dim, "d": args.d, "layers": args.layers, "heads": args.heads,
                        "use_edge": bool(args.use_edge), "use_vis": bool(args.use_vis)}, ckpt)
            r = evaluate_dump(args.feats, ckpt, args.hard_file, taus=(0.5,), shuffle_text=False)
            h = r["holdout"]["0.5"]["keep"]
            print(f"  epoch {ep + 1} holdout tau0.5 keep: acc {h['pixel_acc']:.3f} wrong {h['pixel_wrong']:.3f} grey {h['pixel_unassigned']:.3f}", flush=True)
            log.write(json.dumps({"kind": "eval", "epoch": ep + 1, **r}) + "\n"); log.flush()


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--feats", required=True); t.add_argument("--out", required=True)
    t.add_argument("--hard_file", default="/root/autodl-tmp/datasets/pv_hard.txt")
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--d", type=int, default=128)
    t.add_argument("--layers", type=int, default=3); t.add_argument("--heads", type=int, default=4)
    t.add_argument("--no_edge", dest="use_edge", action="store_false")
    t.add_argument("--no_vis", dest="use_vis", action="store_false")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--feats", required=True); e.add_argument("--ckpt", required=True)
    e.add_argument("--hard_file", default="/root/autodl-tmp/datasets/pv_hard.txt")
    e.add_argument("--tau", default="0.3,0.4,0.5,0.6"); e.add_argument("--shuffle_text", action="store_true")
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
        return
    taus = tuple(float(x) for x in args.tau.split(","))
    r = evaluate_dump(args.feats, args.ckpt, args.hard_file, taus, args.shuffle_text)
    for s in ("holdout", "hard", "mw"):
        for t in r[s]:
            k, a = r[s][t]["keep"], r[s][t]["area"]
            print(f"{s:8s} tau {t}: keep acc {k['pixel_acc']:.3f} wrong {k['pixel_wrong']:.3f} grey {k['pixel_unassigned']:.3f}"
                  f" | area acc {a['pixel_acc']:.3f} wrong {a['pixel_wrong']:.3f} grey {a['pixel_unassigned']:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train**

```bash
cd /root/autodl-tmp/sam3seggen && HF_HUB_OFFLINE=1 $PY finetune/mask_rank.py train --feats /root/autodl-tmp/runs/mask_rank_v1/feats \
  --out /root/autodl-tmp/runs/mask_rank_v1 2>&1 | tee /root/autodl-tmp/runs/mask_rank_v1/train.log
```

Expected: loss decreases from ≈ 0.6 in the first 20 steps; per-epoch holdout line printed; `rank.pt` written. ≈ 1–3 min per epoch (the eval unpacks ≈ 400 images of masks).

- [ ] **Step 3: Evaluate, shuffle control, ablations**

```bash
cd /root/autodl-tmp/sam3seggen && R=/root/autodl-tmp/runs/mask_rank_v1 && \
HF_HUB_OFFLINE=1 $PY finetune/mask_rank.py eval --feats $R/feats --ckpt $R/rank.pt --out $R/eval.json | tee $R/eval.log && \
HF_HUB_OFFLINE=1 $PY finetune/mask_rank.py eval --feats $R/feats --ckpt $R/rank.pt --shuffle_text --out $R/eval_shuffle.json | tee $R/eval_shuffle.log && \
HF_HUB_OFFLINE=1 $PY finetune/mask_rank.py train --feats $R/feats --out ${R}_noedge --no_edge > ${R}_noedge.log 2>&1 && \
HF_HUB_OFFLINE=1 $PY finetune/mask_rank.py train --feats $R/feats --out ${R}_novis --no_vis > ${R}_novis.log 2>&1 && \
grep 'epoch 20 holdout' ${R}_noedge.log ${R}_novis.log
```

Pass criteria (spec §4.7): some tau on `holdout` with `keep` mode has `pixel_acc >= 0.583` (v3@0.4) and `pixel_wrong <= 0.205` (v3@0.5); `hard` `pixel_wrong <= 0.25`; `mw` `pixel_acc >= 0.58`. Stop rule: best holdout `pixel_acc` < 0.6 × Oracle-select(0.8) from Task 3 → stop, do not sweep. Report the shuffle delta and the two ablations verbatim.

- [ ] **Step 4: Commit**

```bash
cd /root/autodl-tmp/sam3seggen && git add finetune/mask_rank.py && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "mask_rank: dump dataset, training loop, evaluation with shuffle control and ablation flags"
```

---

### Task 6: `sam3_to_2dmap.py --rank_model` (deployment switch)

**Files:**
- Modify: `sam3_to_2dmap.py` — add `segment_candidates()` after `segment_prompts()` (ends line 174), add `rank_colorize()` after `colorize()` (ends line 293), add `--rank_model` / `--rank_tau` in `main()` (argparse block starts line 296) and branch at the `colorize(...)` call (line 333).
- Test: `tests/test_rank_colorize.py`

**Interfaces:**
- Consumes: `mask_rank.load_model`, `batch_items`; `mask_cands.extract_candidates`, `pair_features`, `paint_keep`.
- Produces: `segment_candidates(processor, model, image, prompts, device, bank=None, score_min=0.05, max_per_name=12) -> dict` (Cands minus GT fields); `rank_colorize(image, cands, keep, tau) -> (Image, legend)` with the same legend row schema as `colorize` (`prompt, part, color, pixels, score, text_vec`); CLI `--rank_model PATH --rank_tau 0.5`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rank_colorize.py
import unittest
import numpy as np
from PIL import Image

import sam3_to_2dmap as module


class RankColorizeTest(unittest.TestCase):
    def test_paints_highest_keep_and_writes_legend(self):
        img = Image.new("RGBA", (2, 2), (255, 255, 255, 255))
        cands = {"masks": np.array([[[1, 1], [1, 1]], [[1, 0], [0, 0]]], bool), "name_idx": np.array([0, 1]),
                 "names": ["body", "arm"], "score": np.array([0.7, 0.6], np.float32),
                 "text_vec": np.zeros((2, 4), np.float32), "fg": np.ones((2, 2), bool)}
        out, legend = module.rank_colorize(img, cands, keep=np.array([0.4, 0.9], np.float32), tau=0.5)
        px = np.array(out)
        parts = {row["part"]: row for row in legend}
        self.assertIn("arm", parts); self.assertNotIn("body", parts)     # body cut by tau
        self.assertEqual(parts["arm"]["pixels"], 1)
        self.assertEqual(tuple(px[0, 0]), tuple(parts["arm"]["color"]))
        self.assertEqual(parts["<unassigned>"]["pixels"], 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_rank_colorize -v`
Expected: `AttributeError: module 'sam3_to_2dmap' has no attribute 'rank_colorize'`

- [ ] **Step 3: Implement**

Add near the top of `sam3_to_2dmap.py` (after the existing imports):

```python
FINETUNE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune")


def _finetune_module(name: str):
    import importlib
    if FINETUNE_DIR not in sys.path:
        sys.path.insert(0, FINETUNE_DIR)
    return importlib.import_module(name)
```

Insert after `colorize()`:

```python
def rank_colorize(image: Image.Image, cands: dict, keep: np.ndarray, tau: float) -> tuple[Image.Image, list[dict]]:
    """colorize() with the ranker's per-candidate keep score deciding ownership instead of mask area."""
    mc = _finetune_module("mask_cands")
    fg = cands["fg"] & foreground_mask(image)
    painted = mc.paint_keep(cands["masks"], cands["name_idx"], keep, fg, tau)
    canvas = np.full((*fg.shape, 3), 255, dtype=np.uint8)
    canvas[fg] = UNASSIGNED
    present = [n for n in range(len(cands["names"])) if (painted == n).any()]
    colors = pick_separated_colors(len(present))
    legend = []
    for n, color in zip(present, colors):
        sel = painted == n
        canvas[sel] = color
        rows = np.nonzero(cands["name_idx"] == n)[0]
        best = rows[np.argmax(keep[rows])]
        legend.append({"prompt": cands["names"][n], "part": cands["names"][n], "color": list(color),
                       "pixels": int(sel.sum()), "score": float(cands["score"][best]),
                       "text_vec": [round(float(x), 5) for x in cands["text_vec"][best]]})
    left = int((fg & (painted < 0)).sum())
    if left:
        legend.append({"prompt": "<unassigned>", "part": "<unassigned>", "color": list(UNASSIGNED),
                       "pixels": left, "score": None, "text_vec": None})
    return Image.fromarray(canvas, mode="RGB"), legend
```

Insert after `segment_prompts()`:

```python
@torch.no_grad()
def segment_candidates(processor, model, image: Image.Image, prompts: list[str], device: str, bank=None,
                       score_min: float = 0.05, max_per_name: int = 12) -> dict:
    """Per-(prompt, query) candidates for the ranker; finetune/mask_cands.extract_candidates without GT."""
    mc = _finetune_module("mask_cands")

    class _Sample:      # duck-typed stand-in for concept_bank.ImageSample
        pass
    s = _Sample()
    s.image = image
    fg = foreground_mask(image)
    s.ids = np.where(fg, 0, -1)          # one pseudo part covering the silhouette
    s.names = ["<fg>"]                   # not a prompt, so gt_lab is -1 everywhere and GT scores come out 0
    s.gts = {p: fg for p in prompts}     # keys drive the prompt list
    s.obj_name = None
    c = mc.extract_candidates(processor, model, bank, s, device, score_min=score_min, max_per_name=max_per_name)
    for k in ("precision", "recall", "iou", "gt_lab"):
        c.pop(k, None)
    return c
```

In `main()` add the arguments:

```python
    parser.add_argument("--rank_model", default=None,
                        help="finetune/mask_rank.py rank.pt: rank SAM3 candidates instead of area ordering")
    parser.add_argument("--rank_tau", type=float, default=0.5)
```

Replace the single line `colored, legend = colorize(image, parts, instance=False, unassigned_to=args.unassigned_to)` with:

```python
    if args.rank_model:
        if args.unassigned_to:
            print("warning: --unassigned_to is ignored with --rank_model")
        mr = _finetune_module("mask_rank"); mc = _finetune_module("mask_cands")
        unique = list(dict.fromkeys(p for _, ps in specs for p in ps))
        cands = segment_candidates(processor, model, image, unique, device, bank=bank)
        net, ck = mr.load_model(args.rank_model, device)
        if not bool(ck["use_vis"]):
            cands["vis_feat"] = np.zeros((cands["score"].shape[0], 0), np.float32)
        cands["pair"] = mc.pair_features(cands["masks"], cands["name_idx"], cands["centroid"])
        cands["gt_score"] = np.zeros(cands["score"].shape[0], np.float32)
        cands["precision"] = np.zeros(cands["score"].shape[0], np.float32)
        cands["fg_area"] = np.int64(cands["fg"].sum())
        X, E, M, _, _ = mr.batch_items([cands], device)
        keep = net(X, E, M)[0, :cands["score"].shape[0]].detach().cpu().numpy()
        colored, legend = rank_colorize(image, cands, keep, args.rank_tau)
    else:
        colored, legend = colorize(image, parts, instance=False, unassigned_to=args.unassigned_to)
```

Read `main()` lines 296–333 first and use the exact local names it has for the prompt specs, bank, device, processor and model (they are created there before `segment_parts` is called).

- [ ] **Step 4: Run tests**

Run: `cd /root/autodl-tmp/sam3seggen && $PY -m unittest tests.test_rank_colorize tests.test_sam3_grouping -v`
Expected: `OK` (the existing grouping tests still pass — default path unchanged).

- [ ] **Step 5: End-to-end smoke on one holdout render**

```bash
cd /root/autodl-tmp/sam3seggen && O=$(python3 -c "import json;print(json.load(open('/root/autodl-tmp/datasets/concept_bank_v3/split.json'))['holdout'][0])") && \
P=$(python3 -c "import json;print(' '.join(sorted(set(n for n in json.load(open('/root/autodl-tmp/datasets/pv/$O/names.json')) if n.strip()))))") && \
HF_HUB_OFFLINE=1 $PY sam3_to_2dmap.py --image /root/autodl-tmp/datasets/pv/$O/views/az0/render.png --prompts $P \
  --concept_bank /root/autodl-tmp/datasets/concept_bank_v3/bank.pt --threshold 0.4 --allow_missing \
  --rank_model /root/autodl-tmp/runs/mask_rank_v1/rank.pt --out /tmp/rank_map.png --legend /tmp/rank_legend.json && \
python3 -c "import json;[print(r['part'], r['pixels'], r['color']) for r in json.load(open('/tmp/rank_legend.json'))]"
```

Expected: a `map.png` and legend with one row per painted name plus `<unassigned>`; colours distinct. (Check `--prompts` / `--legend` argument names against `main()`; `HANDOVER_cloud_segvigen.md` §6 uses exactly this form.)

- [ ] **Step 6: Commit**

`sam3_to_2dmap.py` already has unrelated uncommitted edits (`git status`). Run `git diff sam3_to_2dmap.py`; stage only the hunks from this task with `git add -p sam3_to_2dmap.py`, or ask the user how to handle the pre-existing diff.

```bash
cd /root/autodl-tmp/sam3seggen && git add tests/test_rank_colorize.py && git add -p sam3_to_2dmap.py && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "sam3_to_2dmap: optional --rank_model to let Mask RankGNN decide candidate ownership"
```

---

### Task 7: Record results

**Files:**
- Create: `finetune/REPORT_mask_rankgnn_v1.md`

- [ ] **Step 1: Write the report** in the style of `finetune/REPORT_concept_bank_v4_eval.md`: §0 one-page verdict (gate result; pass/fail against spec §3.5 and §4.7), §1 protocol, §2 oracle tables (holdout / hard / mw × {v3, none}: baseline sweep, oracle-select grid, oracle-pixel, decomposition, best-score-below-0.5, top confusions), §3 ranker tables (tau × mode, shuffle delta, two ablations), §4 next step (downstream hard-20 deployment run on the 3D machine per `HANDOVER_cloud_segvigen.md` §6 step 2 with `--rank_model`). Every number copied from the JSON files; no estimates. If the gate failed, §3 is a single line saying so and §4 points to E2 / multi-view per the spec.

- [ ] **Step 2: Commit**

```bash
cd /root/autodl-tmp/sam3seggen && git add finetune/REPORT_mask_rankgnn_v1.md && \
GIT_AUTHOR_NAME=GzaIcebreak GIT_AUTHOR_EMAIL=GzaIcebreak@users.noreply.github.com GIT_COMMITTER_NAME=GzaIcebreak GIT_COMMITTER_EMAIL=GzaIcebreak@users.noreply.github.com \
git commit -m "Record Mask RankGNN v1: oracle gate and ranker results"
```
