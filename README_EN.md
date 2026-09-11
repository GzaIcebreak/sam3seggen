# SAM3-SegviGen（[中文](README.md)）

A text-prompted 3D part segmentation pipeline based on [SegviGen](https://github.com/Nelipot-Lee/SegviGen)
+ [SAM3](https://huggingface.co/facebook/sam3): feed it a GLB and a list of semantic prompts,
get back one GLB with one named mesh per part — real textures included.

Upstream SegviGen: [Project Page](https://fenghora.github.io/SegviGen-Page/) |
[Paper](https://arxiv.org/abs/2603.16869) |
[Online Demo](https://huggingface.co/spaces/fenghora/SegviGen) |
[Weights](https://huggingface.co/fenghora/SegviGen)

## 🌟 What this fork adds

**Recognition — semantic prompts instead of hand-painted maps.**
Upstream's 2D-guided mode needs a manually painted color map. Here SAM3 segments the
conditioning render from plain text prompts (`"helmet"`, `"body=head+face+hand"`), and the
masks are colorized into the 2D map SegviGen consumes. Prompt *groups* merge several
concepts into one output part, and `--unassigned_to` folds whatever no prompt claimed into
a named part, so the output has exactly as many parts as requested.

**Merging — every semantic object comes out as ONE mesh.**
Naive 2D guidance shatters complex shells into dozens of fragments. `segment_parts.py`
takes the other route: several prompt-free full segmentations are intersected into
deliberately over-segmented atoms, then multi-view renders are masked by SAM3 and every
atom *component* takes the prompt that covers it most specifically. Parts sharing a name
are merged into a single mesh with their textures packed into an atlas — lossless UV
remapping, no rebake. Language only picks names; every boundary comes from geometry, so a
mislabelled pixel can no longer tear one open.

**Automatic front-view selection.**
The 2D-guided mode is sensitive to which side of the model gets rendered. `--front_view`
picks the conditioning view automatically:

| mode | how it decides | needs |
|---|---|---|
| `metric` | silhouette symmetry + coverage + centeredness | nothing, offline |
| `auto` | metric top-3, then SAM3 prompt confidence | SAM3 env |
| `vlm` | metric top-4 grid, a VLM (Kimi/Moonshot) picks the semantic front | `MOONSHOT_API_KEY` |

**Texture baking as an option (default on).**
Each output part is re-UV'd in Blender and the source model's albedo is baked back onto it
(`--no_texture` skips Blender entirely and emits flat placeholder colors for speed).

**Robustness fixes over upstream.**
Strict legend/manifest validation with a `--sam3_only` audit mode, off-body fragment
cleanup, correct glTF V-flip and metallic-factor handling in atlas merging, and camera
conventions verified against the renderer (IoU 0.987).

## 📷 Results

Auto-selected front view → SAM3 semantic 2D map → final textured split (mushroom + chair,
exactly two meshes):

<p>
  <img src="docs/images/front_render.png" width="30%"/>
  <img src="docs/images/sam3_2d_map.png" width="30%"/>
  <img src="docs/images/vote_result_0.png" width="30%"/>
</p>
The two extracted parts — the mushroom alone / the chair with the mushroom removed:

<p>
  <img src="docs/images/extracted_mushroom.png" width="30%"/>
  <img src="docs/images/chair_only.png" width="30%"/>
  <img src="docs/images/vote_result_90.png" width="30%"/>
</p>

## 🔨 Deployment

Developed and tested on Windows 11 + RTX 5090D (32 GB); upstream targets Linux with ≥24 GB
VRAM — both work. Two Python environments are required because SAM3 (transformers 5.x) and
SegviGen (transformers 4.57.6) conflict:

1. SegviGen env (the one that runs this repo): [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) dependencies first
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

    pip install mathutils
    pip install transformers==4.57.6   # pinned: TRELLIS.2 issue #101
    pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
    pip install --upgrade Pillow trimesh
    # Linux only: sudo apt-get install -y libsm6 libxrender1 libxext6
    ```

2. SAM3 env (a separate venv): transformers 5.x + the `facebook/sam3` weights
    ```sh
    pip install "transformers>=5"   # plus torch matching your CUDA
    ```

3. Weights (see below)

### Where the weights live

The repo itself ships **no** weights (`weights/` is gitignored). Deployment needs three
kinds of files: upstream public checkpoints, gated models, and the ones we trained.
Direct access to huggingface.co is often blocked in CN; the download scripts default to
[`hf-mirror.com`](https://hf-mirror.com) via `HF_ENDPOINT` (`env.sh` / `download_weights.sh`).

| what | remote | lands at | how |
|---|---|---|---|
| three SegviGen ckpts (~7.3 GB each) | [`fenghora/SegviGen`](https://huggingface.co/fenghora/SegviGen) | `ckpt/full_seg.ckpt`, `full_seg_w_2d_map.ckpt`, `interactive_seg.ckpt` | `python download_ckpts.py` |
| TRELLIS.2-4B (voxel / texture codecs) | [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) | `microsoft/TRELLIS.2-4B/` | `./download_weights.sh` |
| matting RMBG / BiRefNet | [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) or [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) | `weights/...`, pointed to by `SEGVIGEN_RMBG` | `./download_weights.sh` |
| SAM3 (gated — accept the license on HF first) | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | `weights/facebook/sam3` (`SEGVIGEN_SAM3`) | `export HF_TOKEN=… && ./download_weights.sh --gated` |
| DINOv3 (gated) | [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `weights/facebook/dinov3-vitl16-pretrain-lvd1689m` (`SEGVIGEN_DINOV3`) | same |
| **SAM3 concept bank v3 (the deployed default)** | [`Zaun1996/sam3-concept-bank`](https://huggingface.co/Zaun1996/sam3-concept-bank) root `bank.pt` | any path, `--concept_bank` at inference | `huggingface-cli download Zaun1996/sam3-concept-bank bank.pt` |
| later concept-bank experiments (**not deployed**) | [`v5/`](https://huggingface.co/Zaun1996/sam3-concept-bank/tree/main/v5) and [`mask_rank_v3/`](https://huggingface.co/Zaun1996/sam3-concept-bank/tree/main/mask_rank_v3) on the same repo | ablation / reproduction | pull the subdirectory |
| SegviGen LoRA (v6, …) | [`Zaun1996/segvigen-lora`](https://huggingface.co/Zaun1996/segvigen-lora) | download `v6/lora_last.pt`, then `finetune/merge_lora.py` into `ckpt/full_seg_w_2d_map.ckpt` | private; needs an HF token |

```sh
# upstream SegviGen ckpts → ckpt/
python download_ckpts.py

# plus TRELLIS.2-4B + matting; add --gated for SAM3 / DINOv3 (license + HF_TOKEN)
./download_weights.sh
# ./download_weights.sh --gated

# the deployed SAM3 concept bank (287 KB)
huggingface-cli download Zaun1996/sam3-concept-bank bank.pt --local-dir datasets/concept_bank_v3
```

Hook the bank in with `python sam3_to_2dmap.py --concept_bank datasets/concept_bank_v3/bank.pt --threshold 0.4 …`.
`v5/` (per-pixel CE + decoder LoRA) and `mask_rank_v3/` (candidate ranker) look better on
holdout numbers but worse on external assets, so **deployment stays on the root `bank.pt`
+ overlay @0.4**. Each subdirectory README has the metrics and the why.

Runtime configuration:

- `SEGVIGEN_PY_SAM3` — Python of the SAM3 venv (default `../.venv_holo/Scripts/python.exe`).
- GPU backends (verified on Blackwell): `ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm FLEX_GEMM_ALGO=explicit_gemm`.
- VLM front-view mode: `MOONSHOT_API_KEY` (env var or a gitignored `.env` at the repo
  root); `SEGVIGEN_VLM_BASE_URL` / `SEGVIGEN_VLM_MODEL` override the defaults
  (`https://api.moonshot.cn/v1`, `kimi-latest`).

## 📒 The interface

### `segment_parts.py` — the current pipeline: over-segment, then name

```sh
python segment_parts.py \
  --glb model.glb \
  --prompts head torso arm hand leg foot \
  --unassigned_to torso \
  --out out/parts.glb --work_dir out/work
```

Geometry decides every boundary, language only picks names. The pipeline is five fixed
stages:

1. **paint.** The source model is rendered over a fixed view grid; if those renders carry
   no colour of their own (an untextured model), one `full_seg` sample's part colouring is
   painted onto them flat (`flat_paint.py`). Textured models skip this.
2. **guidance.** SAM3 masks every view and the overlays a human reviews are written to
   `work/guidance/`. This deliberately comes before the samples: a bad prompt set shows up
   here, and the split has not been paid for yet.
3. **split.** `--samples` prompt-free `full_seg` runs, the conditioning camera jittered by
   `--azimuth_jitter` around `--azimuth`, their partitions intersected — two faces share an
   atom only if *every* sample coloured them alike, so every cut any sample drew survives
   (`data_toolkit/meet_samples.py`).
4. **units.** Atoms are cut into connected components and each remesh inner wall is fused
   into the shell it lines. This has to happen before anything is named: the two are not
   edge-connected, so a separately-voted inner wall inherits whichever part sits nearest it
   (`unit_vote.split_units`).
5. **merge**, gated by `--merge`: each unit takes the name whose stage-2 masks cover it,
   the most specific one winning; unseen faces inherit the nearest visible one. Faces are
   exported per name with the source albedo baked back on.
6. **complete**, gated by `--complete`: our parts are open where they were cut, so X-Part
   regenerates each as a closed solid (`xpart_complete.py`).

Stages 3 and 6 cost GPU minutes; the rest is seconds once the renders are cached.

### What more samples buy is a less lucky split, not a finer one

One draw of 5 samples left Mickey with 13 atoms, the largest covering 48% of the surface
and the ears never cut from the head; a different draw of 5 gave 34 atoms and all six
parts. **That spread between draws is wider than the gap between 5 and 9**, and closing it
is what the extra samples are for: 7 and 9 land on the same six parts, so the default is 7.
9 only adds atoms (40 against 34), not parts.

Reading each sample more finely is no substitute: dropping `--color_tol` from 20 to 3
multiplied the labels per sample twentyfold and still gave 13 atoms, because the extra
labels are speckle. Nor is widening `--azimuth_jitter`, which can cost parts: at 60 degrees
several samples came back with 2-6 labels, and those near-blank partitions fragment the
meet along boundaries that are not real.

The knobs to coarsen with are the size floors (`--min_atom_faces` 300, `--min_unit_faces`
600). An atom below that is a sliver of disagreement between samples, not a part boundary.
Raising them from 150/300 took the robot from 66 atoms to 59 and Mickey from 40 to 34
without either model losing a named part, and **the share of surface named by an actual
vote rather than by the nearest-neighbour fallback went up** (robot 75.0% to 76.1%).
Coarser still keeps the parts but starts making units that span two of them, and the vote
coverage falls away again (73.3% at 1000).

### Telling X-Part what a part looks like (`--condition`)

A box is a lossy prompt, because a box also contains whatever else passes through it.
X-Part conditions each part on the source surface it finds *inside the box*, so the robot's
torso box handed over the tops of both legs and what came back was a torso with legs.

We are not limited to a box: the split already decided, per face, which part each triangle
belongs to. The default `--condition surface` samples the conditioning points from exactly
those faces and passes them as `part_surface_inbbox` -- the same tensor X-Part would have
built by cropping, only built from the assignment. The box still goes along; it is what
sizes the token budget.

Measured as how far a point on the generated solid strays from that part's own surface
(counted beyond 2% of the model diagonal):

| | box | surface |
|---|---|---|
| robot torso | 65.4% (volume 0.0530) | **5.0%** (volume 0.0183) |
| robot legs | 9.2% / 13.2% | **2.3% / 2.3%** |
| Mickey, mean over 11 parts | 24.1% | **18.7%** |

It comes with one new failure mode: on each model exactly one very small part blew up
instead, 15 to 59 times its volume. The box crop incidentally gave a small part some
surrounding context to anchor its scale against, and its own surface alone does not. A
solid that overruns its prompt box by half the box's width is therefore reported, so the
failure is at least visible. `--condition box` keeps the old behaviour to compare against.

The default grid is `45,225 × 10`, a barely-raised 3/4 pair. A level azimuth-90 misses
`torso` on the chest, but height costs more than it buys: at 35 degrees the camera looks
down far enough that the torso hides the legs and base. `--flat_paint on|off` forces or
disables stage 1.

Why over-segment first: `full_seg` has no granularity knob and a single sample fuses
neighbouring parts often enough to matter — on the robot test model, shoulder armour and
both arms came out as one 24k-face atom in *every* same-view sample, and only a jittered
conditioning view broke it apart. Over-segmentation costs the naming step nothing (it can
only merge), while under-segmentation is unrecoverable.

`work/atoms.glb` shows the atoms the vote merged, one colour each; `work/vote_report.json`
has the per-unit vote table. When a part comes out wrong, look at `atoms.glb` first: if
the boundary is not there, no amount of prompt tuning will produce it.

### `merge_parts.py` — the naming half, on its own

Splitting is expensive and prompt-independent; naming is cheap and is what you re-run
while deciding what the parts should be called. `--merge off` stops after stage 4, still
writing the guidance overlays if prompts were given, so you can look before merging:

```sh
python segment_parts.py --glb model.glb --merge off --out split/units.glb \
  --prompts head torso arm hand leg foot --unassigned_to torso
# after reviewing split/work/guidance/*.png
python merge_parts.py --glb model.glb --split split/work \
  --prompts head torso arm hand leg foot --unassigned_to torso \
  --out named/parts.glb
```

Renders and masks are cached in the split directory, keyed by the prompts that produced
them, so re-running with the same prompts costs seconds and re-running with new prompts
only re-runs SAM3. `--merge unit` writes one node per voted unit instead of fusing them,
which is how you see *which* unit took a wrong name.

Every unit is voted on **per view**: each camera that sees enough of it picks the most
specific mask covering it there, and the unit takes the name most cameras agree on.
Pooling pixels across views instead lets whichever camera happens to see the unit head-on
decide alone — and that is exactly the camera where a part half hidden behind another one
gets its neighbour's name. On the robot, switching to per-view voting moved ~9.7k faces of
shoulder armour from `torso` to `arm`.

### `segment_api.py` — deprecated: prompts in, one named-parts GLB out

Superseded by `segment_parts.py`. A 2D map painted from one view steers the generative
model here, so a mislabelled pixel becomes a torn 3D boundary. Kept as the reference
implementation of the 2D-map route; its front-view selection and SAM3 painter options are
still used elsewhere.

```sh
python segment_api.py \
  --glb model.glb \
  --prompts "mushroom=small mushroom" chair \
  --unassigned_to chair \
  --front_view auto \
  --out out/parts.glb --work_dir out/work
```

- `--prompts`: one entry per output part; join concepts with `+` to merge them
  (`body=head+face+hand`), optionally under an explicit `name=` prefix. Fully user-defined,
  nothing hard-coded.
- `--front_view metric|auto|vlm`: pick the conditioning view automatically; `--azimuth`
  (degrees) remains available to pin a fixed view.
- `--parts_output combined|separate`: `combined` (default) writes only `--out`, one node per
  part. `separate` additionally exports each part on its own into a `parts/` directory beside
  it and adds a `file` key to every `parts.json` row. The single files are carved out of the
  combined result, so node names and baked textures are identical either way.
- `--split_mode stain|weld|refine`: how SegviGen's colouring becomes part boundaries. `stain`
  (default) cuts exactly along the predicted colours and only hands fragments under 100 faces
  to their neighbour — SegviGen's own `split.py` rule — so the parts match the coloured mesh.
  `weld` keeps the same cuts but looks at the 2D guide map one same-colour piece at a time:
  when enough of a piece faces the camera and its visible faces clearly vote for another
  part, the whole piece is renamed (never cut inside, hidden pieces never touched). `refine`
  is the older pipeline: it overwrites visible faces pixel by pixel, votes seam bands and
  gives detached islands to the part surrounding them — closer to the guide map from the
  front, at 3–5x the number of pieces. `finetune/split_bench.py` scores the three on the
  ext_bench assets without ground truth.
- `--no_v6`: fall back to the base 2D-map checkpoint. The default is
  `ckpt/full_seg_v6.ckpt` (trajectory-supervision LoRA merged in); `--no_sam` or an explicit
  `--ckpt` bypasses it anyway.
- `--no_sam`: drop SAM3 entirely and run the prompt-free full_seg checkpoint on a plain
  render (unnamed, color-clustered parts).
- `--no_texture`: skip the Blender re-UV + bake step (fast); default keeps real textures.
- `--sam3_only`: stop after render + SAM3 and keep `render.png` / `sam3_2d_map.png` /
  legend for auditing.

The 2D map is painted by concept bank v3's text offsets and the score-0.4 smallest-first
overlay.

- `--assign rank`: let the EASE Mask RankGNN edit that overlay set (drop keep<0.1, add
  keep>=0.9). It wins in-distribution but can delete a prompt outright, so it is off by default.
- `--assign auto`: paint both maps from the one forward pass and keep the ranker's edit only
  where it costs no prompt; the decision lands in `<map>_auto.json`. That is a second
  painting, not a second SAM3 run.
- `--assign argmax`: v5's per-pixel competition.
- `--no_concept_bank`: use stock SAM3 embeddings instead of concept bank v3.
- `--concept_bank` / `--rank_model`: point at other weights (or set `SEGVIGEN_CONCEPT_BANK` /
  `SEGVIGEN_RANK_MODEL`). A default that is not on this box downgrades with a printed note
  rather than failing; a path you name explicitly is an error if it is missing.
- `--sam3_threshold`: defaults to the calibrated value for the painter in use — 0.4 with the
  concept bank, 0.3 without.

When any prompt ends up with no mask, strict validation (the default) reports `SAM3 produced
no mask for requested component(s)`; relax it with `--allow_partial`.

Python:

```python
from segment_api import segment

manifest = segment(
    "model.glb", ["mushroom=small mushroom", "chair"], "out/parts.glb",
    with_texture=True,            # texture baking is optional, default on
    front_view="auto",            # metric | auto | vlm | None
    use_v6=True,                  # default; False falls back to the base 2D-map checkpoint
    assign="paint",               # default; rank = EASE, auto = score both and keep one
    parts_output="combined",      # default; separate also writes one glb per part
    split_mode="stain",           # default; weld = rename whole pieces from the map, refine = per-pixel overwrite
    unassigned_to="chair",
    work_dir="out/work",          # keep intermediates for inspection
)
# manifest: [{"label": 0, "name": "mushroom", "node": "part_00_mushroom", "faces": ..., ...}]
```

### `serve_api.py` — the same pipeline over HTTP

```sh
./run_serve.sh --port 8020          # interactive docs at /docs

curl -X POST http://127.0.0.1:8020/segment \
  -F "glb=@model.glb" \
  -F "prompts=leaves" -F "prompts=fruit" \
  -F "unassigned_to=fruit" -F "samples=5"
```

`POST /segment` runs `segment_parts.py`; `POST /segment_legacy` is the old 2D-map route
with its `azimuth` / `front_view` / `assign` / `split_mode` options.

The response carries the `parts` manifest and download links: `GET /jobs/{id}/download`
for the result, `GET /jobs/{id}/parts/{node}.glb` for one part, `GET /jobs/{id}/atoms` for
the atoms the vote merged and `GET /jobs/{id}/report` for the vote table (legacy jobs have
`/map` and `/render` instead). `GET /health` reports the resolved default weights and
whether the GPU is busy.

Repeat `prompts` once per part rather than space-separating them — a concept may itself
contain spaces (`small mushroom`).

Every stage still runs as a subprocess that loads its own model, so a request costs a couple
of minutes, and the box has one GPU: jobs take a lock and a second request gets 409 instead
of queueing invisibly. This is a test harness, not a throughput service.

### `segment_vote.py` — deprecated: one sample, coverage voting

Superseded by `segment_parts.py`, which intersects several samples instead of trusting
one and breaks ties by IoU instead of coverage (coverage hands a unit to whichever mask is
largest, so feet became legs and arms became torso).

```sh
python segment_vote.py \
  --glb model.glb \
  --prompts "mushroom=small mushroom" chair \
  --unassigned_to chair --sam3_threshold 0.65 \
  --out out/merged.glb --work_dir out/vote_work
```

Pipeline: full segmentation → multi-view renders → SAM3 masks per view → per-part voting
(face-level votes pooled per part; `min_cover` guards against mask bleed) → same-name parts
merged into one mesh with a texture atlas. Output: exactly one mesh node per prompt name.

### Upstream inference scripts

The original entry points still work unchanged — interactive segmentation
(`inference_interactive.py`), full segmentation and 2D-map-guided full segmentation
(`inference_full.py`, with `--two_d_map`).

### Tests

```sh
python -m unittest discover tests
```

## 🧪 Fine-tuning (fixing "SAM3 colours don't make it into SegviGen")

Full documentation lives in [`finetune/README.md`](finetune/README.md) (Chinese). Every
script runs from the repo root via `finetune\run_ft.bat <script> <args>` (same environment
variables as the inference .bat files, `.venv`); only `sam3_masks.py` runs in `.venv_holo`
and is spawned automatically as a subprocess by path A.

**Why.** The upstream 2D-guidance model was trained on pixel-perfect maps rasterised from the
3D ground truth. SAM3 maps have ragged edges, missed parts, grey unassigned regions and
semantic merges — that domain gap is the root cause of colours not being honoured. The
`finetune/` pipeline adapts the `full_seg_w_2d_map` model with LoRA so that colours follow the
2D map with boundaries snapped to geometry, and grey means *unassigned* (a grey part stays grey
in 3D instead of receiving a guessed colour; a half-covered part is completed in 3D).

**Two sample-generation paths.** Each object is voxelised / encoded once; a variant only
recolours the voxels and re-runs the texture encoder, so dozens of variants per object are cheap:

| Path | Source of the 2D condition map | Scripts |
|---|---|---|
| A | Blender render of the textured mesh → SAM3 masks → bound to GT parts (coverage / precision rules, unbound parts turn grey) | `make_samples_a.py` |
| B | Pixel-perfect map + synthetic corruption (boundary jitter, speckle, whole-part grey, partial erase, grey holes, neighbour merge) | `make_samples_b.py` + `corrupt.py` |

```bat
REM Data: PartVerse (resumable download -> split by anno_infos face labels -> prompts from captions)
finetune\run_ft.bat download_partverse.py --out E:\data\partverse
finetune\run_ft.bat import_partverse.py --partverse E:\data\partverse --out E:\data\pv --limit 2500

REM Samples: chunked subprocess driver (object lists for path B / path A), isolates native o_voxel crashes, resumable
finetune\run_ft.bat run_batch.py --dataset_root E:\data\pv --objects_b E:\data\pv_list_b.txt --objects_a E:\data\pv_list_a.txt

REM Train / export / evaluate
finetune\run_ft.bat train.py --dataset_root E:\data\pv --out_dir finetune\runs\v1 --max_steps 4000
finetune\run_ft.bat merge_lora.py --lora finetune\runs\v1\lora_last.pt --out ckpt\full_seg_w_2d_map_ft.ckpt
finetune\run_ft.bat eval_fidelity.py --object E:\data\pv\<obj> --variant <name> --run_inference --ckpt ckpt\full_seg_w_2d_map_ft.ckpt
```

**Modules.** `common.py` (directory layout, ID palette, camera reproduction, voxel recolouring,
SLAT encoding, DINOv3 conditioning), `dataset.py` / `lora.py` / `model.py` / `train.py`
(v-prediction flow-matching LoRA training with step-0 per-kind loss checks and holdout),
`merge_lora.py` (folds LoRA into a checkpoint usable directly by `inference_full.py`),
`eval_fidelity.py` (fidelity / purity of an output GLB against the colours the 2D map asked for,
with automatic Y-up frame alignment).

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).
However, please note that the code in **`trellis2`** originates from the [TRELLIS.2](https://github.com/Microsoft/TRELLIS.2) project and remains subject to its original license terms.
Users must comply with the licensing requirements of TRELLIS.2 when using or redistributing that portion of the code.

## Citation

```
@article{li2026segvigen,
      title = {SegviGen: Repurposing 3D Generative Model for Part Segmentation}, 
      author = {Lin Li and Haoran Feng and Zehuan Huang and Haohua Chen and Wenbo Nie and Shaohua Hou and Keqing Fan and Pan Hu and Sheng Wang and Buyu Li and Lu Sheng},
      journal = {arXiv preprint arXiv:2603.16869},
      year = {2026}
}
``` 
