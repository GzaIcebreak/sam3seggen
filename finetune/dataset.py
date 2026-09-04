"""Dataset over (object, variant) pairs produced by make_samples_a/b.

v3 additions (all optional, off by default so v1/v2 runs reproduce):
  * legend: per colour group the majority part name's text vector (Stage B text_cache.pt) and
    its RGB, plus the object name's vector -> the legend tokens of finetune/model.py
  * pair:   the partner view of a paired variant (same object, same meta["pair"], other view,
            identical 3D target) so the model can be conditioned on two maps at once
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset

import common

TEXT_DIM = 256


class TextCache:
    """name -> [256] SAM3 text vector (with concept-bank offsets), from concept_bank.py."""

    def __init__(self, path: str):
        blob = torch.load(path, map_location="cpu")
        self.vectors: dict[str, torch.Tensor] = blob["vectors"]
        self.dim = int(blob.get("dim", TEXT_DIM))
        self.missing: Counter = Counter()

    def get(self, name: str | None) -> torch.Tensor | None:
        if not name:
            return None
        v = self.vectors.get(name.strip())
        if v is None:
            self.missing[name.strip()] += 1
        return v


def majority_name(members: list[int], names: list[str]) -> str | None:
    c = Counter((names[p] or "").strip() for p in members if p < len(names))
    c.pop("", None)
    return c.most_common(1)[0][0] if c else None


def object_name(obj: common.ObjectDir) -> str | None:
    p = os.path.join(obj.path, "names_meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return (json.load(f).get("object") or "").strip() or None
    return None


class VariantDataset(Dataset):
    def __init__(self, roots: list[str], kinds: list[str] | None = None, objects: list[str] | None = None,
                 exclude: list[str] | None = None, text_cache: str | None = None, pair: bool = False,
                 legend_shuffle: bool = False, seed: int = 0):
        self.items = []
        objects = set(objects) if objects else None
        exclude = set(exclude) if exclude else set()
        partners: dict[tuple[str, str], list[str]] = defaultdict(list)
        for root in roots:
            for obj, vname, meta in common.list_variants(root):
                name = os.path.basename(obj.path)
                if (objects is not None and name not in objects) or name in exclude:
                    continue
                if meta.get("pair"):
                    partners[(obj.path, meta["pair"])].append(vname)
                if kinds and meta["kind"] not in kinds:
                    continue
                self.items.append((obj, vname, meta))
        if not self.items:
            raise ValueError(f"no variants found under {roots} (kinds={kinds})")
        self.norm = common.load_normalization()
        self.text = TextCache(text_cache) if text_cache else None
        self.pair = pair
        self.partners = partners
        self.legend_shuffle = legend_shuffle
        self.rng = torch.Generator().manual_seed(seed)
        self._names: dict[str, list[str]] = {}
        self._obj_name: dict[str, str | None] = {}

    def __len__(self):
        return len(self.items)

    def kinds(self) -> list[str]:
        return [meta["kind"] for _, _, meta in self.items]

    def n_paired(self) -> int:
        return sum(1 for obj, vname, meta in self.items if self.partner_of(obj, vname, meta) is not None)

    def partner_of(self, obj, vname, meta) -> str | None:
        if not meta.get("pair"):
            return None
        others = [v for v in self.partners.get((obj.path, meta["pair"]), []) if v != vname]
        return others[0] if others else None

    def names_of(self, obj) -> list[str]:
        if obj.path not in self._names:
            self._names[obj.path] = obj.names() or []
            self._obj_name[obj.path] = object_name(obj)
        return self._names[obj.path]

    def legend(self, obj, meta) -> dict:
        """Text vectors + colours of the coloured groups (grey parts carry no legend entry)."""
        names = self.names_of(obj)
        group_names = [majority_name(g, names) for g in meta["groups"]]
        if self.legend_shuffle and len(group_names) > 1:
            # control run: colours keep their groups, names are permuted between groups
            perm = torch.randperm(len(group_names), generator=self.rng).tolist()
            group_names = [group_names[i] for i in perm]
        text, rgb = [], []
        for gname, color in zip(group_names, meta["colors"]):
            v = self.text.get(gname) if self.text else None
            if v is None:
                continue
            text.append(v)
            rgb.append(torch.tensor(color, dtype=torch.float32) / 255.0)
        obj_vec = self.text.get(self._obj_name.get(obj.path)) if self.text else None
        dim = self.text.dim if self.text else TEXT_DIM
        return {
            "legend_text": torch.stack(text) if text else torch.zeros(0, dim),
            "legend_rgb": torch.stack(rgb) if rgb else torch.zeros(0, 3),
            "obj_text": obj_vec,
            "n_groups": len(meta["groups"]),
        }

    def __getitem__(self, index: int) -> dict:
        obj, vname, meta = self.items[index]
        shape = common.load_slat(obj.shape_slat)
        tex_in = common.load_slat(obj.input_tex_slat)
        tex_out = common.load_slat(os.path.join(obj.variant_dir(vname), "output_tex_slat.pth"))
        cond = torch.load(os.path.join(obj.variant_dir(vname), "cond.pth"), map_location="cpu")["cond"]
        coords = shape["coords"].clone()
        if not (torch.equal(coords, tex_in["coords"]) and torch.equal(coords, tex_out["coords"])):
            raise RuntimeError(f"{obj.path}/{vname}: latent coords differ between shape/input/output")
        n = self.norm
        item = {
            "coords": coords,
            "shape": (shape["feats"] - n["shape_mean"]) / n["shape_std"],
            "tex_in": (tex_in["feats"] - n["tex_mean"]) / n["tex_std"],
            "tex_out": (tex_out["feats"] - n["tex_mean"]) / n["tex_std"],
            "cond": cond[0],  # [T, 1024]
            "cond_partner": None,
            "kind": meta["kind"],
            "name": f"{os.path.basename(obj.path)}/{vname}",
        }
        if self.pair:
            partner = self.partner_of(obj, vname, meta)
            if partner is not None:
                item["cond_partner"] = torch.load(os.path.join(obj.variant_dir(partner), "cond.pth"),
                                                  map_location="cpu")["cond"][0]
        if self.text is not None:
            item.update(self.legend(obj, meta))
        return item


def collate(batch: list[dict]) -> dict:
    coords, shape, tex_in, tex_out, lens = [], [], [], [], []
    for i, item in enumerate(batch):
        c = item["coords"].clone()
        c[:, 0] = i
        coords.append(c)
        shape.append(item["shape"])
        tex_in.append(item["tex_in"])
        tex_out.append(item["tex_out"])
        lens.append(int(c.shape[0]))
    out = {
        "coords": torch.cat(coords),
        "shape": torch.cat(shape),
        "tex_in": torch.cat(tex_in),
        "tex_out": torch.cat(tex_out),
        "cond": torch.stack([item["cond"] for item in batch]),  # [B, T, 1024]
        "cond_partner": [item.get("cond_partner") for item in batch],
        "coords_len_list": lens,
        "kinds": [item["kind"] for item in batch],
        "names": [item["name"] for item in batch],
    }
    if "legend_text" in batch[0]:
        out["legend_text"] = [item["legend_text"] for item in batch]
        out["legend_rgb"] = [item["legend_rgb"] for item in batch]
        out["obj_text"] = [item["obj_text"] for item in batch]
        out["n_groups"] = [item["n_groups"] for item in batch]
    return out
