"""Dataset over (object, variant) pairs produced by make_samples_a/b."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset

import common


class VariantDataset(Dataset):
    def __init__(self, roots: list[str], kinds: list[str] | None = None, objects: list[str] | None = None,
                 exclude: list[str] | None = None):
        self.items = []
        objects = set(objects) if objects else None
        exclude = set(exclude) if exclude else set()
        for root in roots:
            for obj, vname, meta in common.list_variants(root):
                name = os.path.basename(obj.path)
                if kinds and meta["kind"] not in kinds:
                    continue
                if (objects is not None and name not in objects) or name in exclude:
                    continue
                self.items.append((obj, vname, meta))
        if not self.items:
            raise ValueError(f"no variants found under {roots} (kinds={kinds})")
        self.norm = common.load_normalization()

    def __len__(self):
        return len(self.items)

    def kinds(self) -> list[str]:
        return [meta["kind"] for _, _, meta in self.items]

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
        return {
            "coords": coords,
            "shape": (shape["feats"] - n["shape_mean"]) / n["shape_std"],
            "tex_in": (tex_in["feats"] - n["tex_mean"]) / n["tex_std"],
            "tex_out": (tex_out["feats"] - n["tex_mean"]) / n["tex_std"],
            "cond": cond[0],  # [T, 1024]
            "kind": meta["kind"],
            "name": f"{os.path.basename(obj.path)}/{vname}",
        }


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
    return {
        "coords": torch.cat(coords),
        "shape": torch.cat(shape),
        "tex_in": torch.cat(tex_in),
        "tex_out": torch.cat(tex_out),
        "cond": torch.stack([item["cond"] for item in batch]),  # [B, T, 1024]
        "coords_len_list": lens,
        "kinds": [item["kind"] for item in batch],
        "names": [item["name"] for item in batch],
    }
