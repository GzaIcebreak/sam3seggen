# SAM3-SegviGen（[中文](README.md)）

A text-prompted 3D part segmentation pipeline based on [SegviGen](https://github.com/Nelipot-Lee/SegviGen)
+ [SAM3](https://huggingface.co/facebook/sam3): feed it a GLB and a list of semantic prompts,
get back one GLB with one named mesh per part — real textures included.

Upstream SegviGen: [Project Page](https://fenghora.github.io/SegviGen-Page/) |
[Paper](https://arxiv.org/abs/2603.16869) |
[Online Demo](https://huggingface.co/spaces/fenghora/SegviGen) |
[Weights](https://huggingface.co/fenghora/SegviGen)

## 📣 What's new

The main line is no longer "steer generation with one 2D map". It over-segments by
geometry, names the pieces, then optionally closes them and bakes the source albedo back
on. The CLI, the Python call and HTTP share **one** config object
(`pipeline.PipelineOptions`), so a knob cannot drift between entry points.

- **Six stages:** `paint → guidance → split → units → [merge] → [complete → bake]`.
  Geometry decides every boundary; language only names. `--merge off` stops after units;
  `--complete off` (the default) stops at the open `parts.glb`.
- **X-Part completion** (`--complete full`): regenerates each open cut as a closed solid.
  Default `--condition surface` conditions on the faces the split already assigned, not on
  whatever falls inside a box. Pieces below `--min_area_share 0.005` fold into the nearest
  larger neighbour (they used to be dropped). A solid that overruns its box is redrawn
  `--redraws 2` times from the same prompt and kept only if it sits closer.
- **Texture bake:** both the open parts and the closed solids get the source albedo back.
  The solid uses a looser cage (`0.05 / 0.15`) because a generated surface only
  approximates the source.
- **Hanging scraps:** an unvoted sliver that only touches one named part and is under 10%
  of it joins that part (Mickey's moustache on the head); otherwise `--unassigned_to`.
- **One interface:** the table below is the current default. `GET /health` returns the
  six stages, every switch and this default as-is. `POST /segment` now defaults
  `sam3_threshold` to **0.4** (the concept bank), not the 0.3 the old form used.

Entry points and fields are under [The interface](#-the-interface).

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
atom *component* takes the prompt that covers it most specifically. Faces that share a
name become one node, with the source albedo baked back on. Language only picks names;
every boundary comes from geometry, so a mislabelled pixel can no longer tear one open.

**Completion — an open cut can be closed into a solid.**
A split shell is empty where it was cut. `--complete full` hands each part to X-Part to
regenerate as a watertight solid, then bakes the texture back with the same path. The
default condition is the faces the split assigned, not whatever happened to sit in the box.

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

Over-segment then name. Each row is **original / stain / repaired**: the stain is one
colour per named part, the third panel explodes the closed X-Part solids.

Mickey (coarse, `head` / `torso` / `base`; the moustache hangs on the head):

<p>
  <img src="docs/images/mickey_compare.png" width="100%"/>
</p>

Pineapple (coarse, `leaves` / `fruit`):

<p>
  <img src="docs/images/pineapple_compare.png" width="100%"/>
</p>

Wall-clock of a coarse split (`--complete off`, 7 `full_seg` samples per model) on
an RTX 5090. Each tooth is one `full_seg` child: VRAM peaks at 13–15 GB, then
drops to ~0 when the process exits. Peak VRAM **15.24 GB**, peak pipeline RSS 8.7 GB.

<p>
  <img src="docs/images/perf_curves.png" width="100%"/>
</p>

## 🔨 Deployment

Developed and tested on Windows 11 + RTX 5090D (32 GB); upstream targets Linux with ≥24 GB
VRAM — both work. Two Python environments are required for the split (SAM3's
transformers 5.x conflicts with SegviGen's 4.57.6); `--complete full` needs a third
X-Part env:

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

3. X-Part env (only `--complete full`; `boxes` writes prompts and stays in this interpreter)
    ```sh
    # Hunyuan3D-Part / XPart, its own venv; weights under SEGVIGEN_XPART_WEIGHTS
    # Dispatched via SEGVIGEN_PY_XPART — this repo never imports it
    ```

4. Weights (see below)

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
- `SEGVIGEN_PY_XPART` / `SEGVIGEN_XPART_ROOT` / `SEGVIGEN_XPART_WEIGHTS` — X-Part
  interpreter, source root and weights; used only by `--complete full`.
- GPU backends (verified on Blackwell): `ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm FLEX_GEMM_ALGO=explicit_gemm`.
- VLM front-view mode: `MOONSHOT_API_KEY` (env var or a gitignored `.env` at the repo
  root); `SEGVIGEN_VLM_BASE_URL` / `SEGVIGEN_VLM_MODEL` override the defaults
  (`https://api.moonshot.cn/v1`, `kimi-latest`).

## 📒 The interface

Three entry points read one contract: `pipeline.PipelineOptions`. Paths (`glb` / `out` /
`work_dir`) change every run and stay on the call; every switch that configures *how* the
pipeline runs lives on this object. The CLI attaches them with `add_cli_arguments()`,
`POST /segment` uses the same field names, and `GET /health` returns `public()`.

| entry | file | what it does |
|---|---|---|
| CLI, full run | `segment_parts.py` | all six stages; `--merge off` splits without naming |
| CLI, naming only | `merge_parts.py` | name / export / optionally complete an existing `work/` |
| Python | `from pipeline import PipelineOptions` | `segment_kwargs()` / `merge_kwargs()` |
| HTTP | `serve_api.py` | `POST /segment`; the old 2D route is `/segment_legacy` |

```python
from pipeline import PipelineOptions
from segment_parts import segment_parts

opts = PipelineOptions(merge="name", complete="off", granularity="medium")
segment_parts("model.glb", ["head", "torso"], "out/parts.glb",
              work_dir="out/work", **opts.segment_kwargs())
```

### `segment_parts.py` — the current pipeline: over-segment, then name

```sh
python segment_parts.py \
  --glb model.glb \
  --prompts head torso arm hand leg foot \
  --unassigned_to torso \
  --out out/parts.glb --work_dir out/work
```

Geometry decides every boundary, language only picks names. The pipeline is six fixed
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
   regenerates each as a closed solid (`xpart_complete.py`). `--complete full` then bakes
   the source albedo onto those solids (the same Blender selected-to-active path as the
   split, with a looser cage, because a generated surface only approximates the source).

Stages 3 and 6 cost GPU minutes; the rest is seconds once the renders are cached.

| switch | values | current default | what it does |
|---|---|---|---|
| `--merge` | `name` / `unit` / `off` | `name` | naming; `off` stops after units |
| `--complete` | `off` / `boxes` / `full` | `off` | X-Part; `full` regenerates then bakes |
| `--condition` | `surface` / `box` | `surface` | X-Part prompt: split faces, or crop-in-box |
| `--flat_paint` | `auto` / `on` / `off` | `auto` | temporary flat colour on a grey model |
| `--granularity` | `fine` / `medium` / `coarse` | `medium` | atom/unit floors 150/300, 300/600, 800/1600 |
| `--min_atom_faces` / `--min_unit_faces` | int | from granularity | explicit floor wins |
| `--samples` | int | `7` | meet samples |
| `--azimuth` / `--azimuth_jitter` | float | `0` / `30` | conditioning camera |
| `--color_tol` | float | `20` | colour distance inside one sample |
| `--mirror` | `auto` / `none` / `x` / `y` / `z` | `auto` | also intersect the reflection |
| `--min_recall` | float | `0.5` | mask must cover this share of a unit |
| `--view_azimuths` × `--view_elevations` | degrees | `45,225` × `10` | SAM3 voting views |
| `--radius` / `--resolution` | | `2` / `512` | voting renders |
| `--sam3_threshold` | float | `0.4` | concept-bank threshold (0.3 without a bank) |
| `--min_area_share` | float | `0.005` | fold a tiny X-Part piece into its neighbour |
| `--redraws` | int | `2` | redraw a solid that overruns its box |
| `--octree_resolution` / `--seed` | | `512` / `42` | X-Part reconstruction |
| `--texture_size` | int | `2048` | bake resolution |
| `--no_texture` / `--no_reuse` | flag | off | skip bake / ignore cache |
| `--strict_parts` | flag | off | fail if a prompt got no mask or no faces; default skips that word and continues |
| `--unassigned_to` | part name | none | absorb unclaimed units; else they are dropped |

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

`--condition box` keeps the old behaviour to compare against.

### X-Part is not a function of its prompt

`partformer_dit` draws a part-identity embedding with `torch.randperm` on every forward
pass, so the same part from the same prompt does not come back the same. Mickey's small
foot overran its own box by 675%, 174%, 7%, 112%, 1615% and 882% across six runs. **A part
that occasionally comes back many times its own size is a bad draw, not a bad prompt.**

The box is what makes that checkable: it says how big the part was, and nothing whose job
is to close a cut should need half the box again. Anything past that is drawn again from
the same conditioning (`--redraws`, default 2), and a redraw is **kept only if it really
does sit closer to the box** -- that foot's two redraws came back at 1615% and 882% and
were both refused.

### Fold a component too small to generate, rather than separating it (`--min_area_share`)

X-Part is not reliable on a sliver, and a sliver is usually not a part anyway but a
leftover of where the split cut. A connected component below this share of the surface is
merged into the nearest bigger one by surface distance, taking its host's name.

This also closes a hole in the old behaviour: such components used to be **dropped**, so
their surface went into no prompt at all and nothing X-Part returned covered it. Folding
is what dropping should always have been.

The share is measured after the remesh's inner walls are removed -- those are dropped
rather than folded, since their normals face inward and conditioning on them would
describe a shape that is inside out in half its points -- so the numbers are about twice
the raw share on a double-shelled mesh.

Mickey's three feet sit at 1.24-1.30%, right at the edge of what X-Part can do, and
neither conditioning is stable on them. That is the call this knob leaves to you: the
default `0.005` keeps them and you get a foot that sometimes overruns, while `0.015` folds
them into the base for eight clean solids with no overruns and no separate feet.

### Split granularity (`--granularity`)

The two size floors only make sense together -- a floor on the atoms that the units then
undo is no floor at all -- so they share one name: `fine` 150/300, `medium` 300/600
(default), `coarse` 800/1600. Passing `--min_atom_faces` or `--min_unit_faces` overrides
the corresponding half.

Go finer to keep a part the size of a bolt head or a button; go coarser when the parts are
large and the split is shattering flat surfaces into panels.

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

When any prompt ends up with no mask, the current pipeline skips that word and
continues. The old 2D route still reports `SAM3 produced no mask for requested
component(s)` unless you pass `--allow_partial`. Pass `--strict_parts` to make
the current pipeline fail the same way.

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
  -F "unassigned_to=fruit" \
  -F "granularity=coarse" -F "complete=full"
```

`POST /segment` runs `segment_parts.py`. Multipart field names match `PipelineOptions`;
anything omitted takes the table above. Repeat `prompts` once per part (a concept may
contain spaces); they may be empty only with `merge=off`. `sam3_threshold` defaults to
**0.4**. `POST /segment_legacy` is the old 2D-map route.

A successful response carries the `parts` manifest, the `options` that actually ran, and
these links:

| method | path | what |
|---|---|---|
| `GET` | `/health` | six stages, switches, current defaults, GPU busy flag |
| `POST` | `/segment` | upload a GLB + options → manifest and download links |
| `POST` | `/segment_legacy` | old 2D route (`front_view` / `assign` / `split_mode`) |
| `GET` | `/jobs/{id}/download` | `parts.glb` (open, named parts) |
| `GET` | `/jobs/{id}/atoms` | over-segmented atoms before the vote |
| `GET` | `/jobs/{id}/report` | per-unit vote table (absent when `merge=off`) |
| `GET` | `/jobs/{id}/complete` | textured X-Part solids (`complete=full`) |
| `GET` | `/jobs/{id}/complete_raw` | generated solids before the bake |
| `GET` | `/jobs/{id}/guidance/{name}` | a review overlay from `work/guidance/` |
| `GET` | `/jobs/{id}/map`, `/render` | legacy jobs only |

Every stage still runs as a subprocess that loads its own model, so a request costs several
minutes, and the box has one GPU: jobs take a lock and a second request gets 409 instead
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
