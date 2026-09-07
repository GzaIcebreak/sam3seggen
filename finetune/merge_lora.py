"""Fold a trained LoRA into the base checkpoint so inference_full.py can load it unchanged.

    finetune\run_ft.bat merge_lora.py --lora finetune/runs/x/lora_last.pt --out ckpt/full_seg_w_2d_map_sam3.ckpt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import common
from lora import inject_lora, load_lora_state_dict, merge_lora
from model import (load_gen3dseg, save_gen3dseg_ckpt, DEFAULT_CKPT, LegendCrossAttention,
                   inject_legend_attention, load_legend_attn_state, legend_attn_state_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = parser.parse_args()

    os.chdir(common.ROOT)
    payload = torch.load(args.lora, map_location="cpu")
    cfg = payload["args"]
    model = load_gen3dseg(args.ckpt, device="cpu")
    inject_lora(model.flow_model, r=cfg["lora_r"], alpha=cfg["lora_alpha"], targets=cfg["lora_targets"].split(","))
    legend_attn = None
    if payload.get("legend_attn"):
        # v4: the wrappers must exist so the LoRA keys (…cross_attn.inner.…) resolve
        inject_legend_attention(model.flow_model)
        load_legend_attn_state(model, payload["legend_attn"])
        legend_attn = legend_attn_state_dict(model)
    load_lora_state_dict(model, payload["lora"])
    n = merge_lora(model.flow_model)
    if legend_attn is not None:
        # unwrap so the .ckpt keeps the plain Gen3DSeg layout; the wrapper state travels in the side file
        for block in model.flow_model.blocks:
            if isinstance(block.cross_attn, LegendCrossAttention):
                block.cross_attn = block.cross_attn.inner
    save_gen3dseg_ckpt(model, args.out)
    print(f"merged {n} LoRA layers from step {payload.get('step')} -> {args.out}")
    if payload.get("legend"):
        # v3/v4: legend encoder (+ v4 legend attention) are not part of the DiT; keep them next to
        # the merged ckpt for inference_full.py --legend_ckpt
        side = os.path.splitext(args.out)[0] + "_legend.pt"
        torch.save({"legend": payload["legend"], "legend_attn": legend_attn,
                    "step": payload.get("step"), "args": cfg}, side)
        print(f"legend encoder{' + legend attention' if legend_attn else ''} -> {side}")


if __name__ == "__main__":
    main()
