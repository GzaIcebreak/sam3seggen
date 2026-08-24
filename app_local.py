"""SegviGen 本地 Gradio demo (适配 Windows 本地环境).

与官方 HuggingFace Space 版面的区别:
- 模型/权重全部走本机路径 (ckpt/, microsoft/TRELLIS.2-4B, weights/...)
- "Generate 2D map" 模式用本机 SAM3 (.venv_holo 子进程) 替代 FLUX.2
- 推理复用本地已验证的 inference_full.py
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("FLEX_GEMM_ALGO", "explicit_gemm")
os.environ.setdefault(
    "SEGVIGEN_DINOV3",
    r"E:\AI_New\ModelGen\weights\facebook\dinov3-vitl16-pretrain-lvd1689m",
)
os.environ.setdefault(
    "SEGVIGEN_RMBG",
    r"E:\AI_New\ModelGen\weights\briaai\RMBG-2.0",
)

import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import gradio as gr

import inference_full as inf
import split as splitter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)  # inference_full 用相对路径读 microsoft/TRELLIS.2-4B/pipeline.json

CKPT_FULL_SEG = os.path.join(ROOT_DIR, "ckpt", "full_seg.ckpt")
CKPT_W_2D_MAP = os.path.join(ROOT_DIR, "ckpt", "full_seg_w_2d_map.ckpt")
TRANSFORMS_JSON = os.path.join(ROOT_DIR, "data_toolkit", "transforms.json")
SAM3_SCRIPT = os.path.join(ROOT_DIR, "sam3_to_2dmap.py")
PY_SAM3 = r"E:\AI_New\ModelGen\.venv_holo\Scripts\python.exe"

TMP_DIR = os.path.join(ROOT_DIR, "_tmp_gradio_seg")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "demo_examples")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = TMP_DIR
os.environ["GRADIO_EXAMPLES_CACHE"] = os.path.join(TMP_DIR, "examples_cache")
os.makedirs(os.environ["GRADIO_EXAMPLES_CACHE"], exist_ok=True)


def _normalize_path(x):
    """兼容不同 Gradio 版本: File/Model3D 可能是 str / dict / object."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name") or x.get("path") or x.get("data")
    return getattr(x, "name", None) or getattr(x, "path", None) or None


def _raise_user_error(msg: str):
    if hasattr(gr, "Error"):
        raise gr.Error(msg)
    raise RuntimeError(msg)


def _collect_examples(example_dir: str) -> List[List[str]]:
    d = Path(example_dir)
    if not d.is_dir():
        return []
    examples: List[List[str]] = []
    for glb_path in sorted(d.rglob("*.glb")):
        png_path = glb_path.with_suffix(".png")
        if png_path.is_file():
            examples.append([str(glb_path), str(png_path)])
    return examples


FULL_SEG_EXAMPLES = _collect_examples(EXAMPLES_DIR)


def _update_img_box(mode: str):
    is_generate = str(mode).startswith("Generate")
    if is_generate:
        return gr.update(
            interactive=False,
            label="2D Segmentation Map (auto-generated)",
            value=None,
        )
    return gr.update(
        interactive=True,
        label="2D Segmentation Map",
        value=None,
    )


def _generate_map_with_sam3(render_img: str, out_map: str, prompts: str):
    tokens = [t for t in (prompts or "").strip().split() if t]
    if not tokens:
        raise gr.Error("Generate 模式需要提供 SAM3 prompts (部件名,空格分隔)。")
    print(f"[SAM3] prompts: {tokens}")
    proc = subprocess.run(
        [
            PY_SAM3,
            SAM3_SCRIPT,
            "--image",
            render_img,
            "--out",
            out_map,
            "--prompts",
            *tokens,
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        if "no mask for requested component" in proc.stderr:
            raise gr.Error(
                f"SAM3 没有在模型上找到这些部件: {', '.join(tokens)}。\n"
                "请按模型实际组成修改 prompts,例如蘑菇可填 'cap stem gills'、"
                "建筑可填 'roof door window wall'。"
            )
        raise gr.Error(f"SAM3 生成 2D map 失败 (exit {proc.returncode}): {proc.stderr[-500:]}")
    if not os.path.isfile(out_map):
        raise gr.Error("SAM3 生成 2D map 失败: 输出文件未找到。")


def run_seg(glb_in, map_mode, img_in, sam3_prompts):
    """Segment 按钮: 生成整体分割 GLB 并显示在第二个框。

    Upload 模式  -> 有图则用 full_seg_w_2d_map.ckpt
    Auto 模式    -> 无图则渲染条件视图,用 full_seg.ckpt
    Generate 模式-> 先渲染,再 SAM3 子进程生成 2D map,用 full_seg_w_2d_map.ckpt
    """
    try:
        glb_path = _normalize_path(glb_in)
        img_path = _normalize_path(img_in)

        if glb_path is None or (not os.path.isfile(glb_path)):
            _raise_user_error("Please upload a valid .glb file.")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        workdir = os.path.join(TMP_DIR, run_id)
        os.makedirs(workdir, exist_ok=True)

        in_glb = os.path.join(workdir, "input.glb")
        shutil.copy(glb_path, in_glb)

        out_glb = os.path.join(workdir, "segmented.glb")
        in_vxz = os.path.join(workdir, "input.vxz")

        is_generate = str(map_mode).startswith("Generate")
        effective_img_path = None

        if is_generate:
            render_img = os.path.join(workdir, "render.png")
            generated_img = os.path.join(workdir, "2d_map_generated.png")
            inf.render_from_transforms(in_glb, TRANSFORMS_JSON, render_img)
            _generate_map_with_sam3(render_img, generated_img, sam3_prompts)
            effective_img_path = generated_img
        elif img_path is not None and os.path.isfile(img_path):
            copied_img = os.path.join(workdir, "2d_map.png")
            shutil.copy(img_path, copied_img)
            effective_img_path = copied_img

        if effective_img_path is not None and os.path.isfile(effective_img_path):
            ckpt = CKPT_W_2D_MAP
            item = {
                "2d_map": True,
                "glb": in_glb,
                "input_vxz": in_vxz,
                "img": effective_img_path,
                "export_glb": out_glb,
            }
            preview_img = effective_img_path
        else:
            ckpt = CKPT_FULL_SEG
            render_img = os.path.join(workdir, "render.png")
            item = {
                "2d_map": False,
                "glb": in_glb,
                "input_vxz": in_vxz,
                "transforms": TRANSFORMS_JSON,
                "img": render_img,
                "export_glb": out_glb,
            }
            preview_img = None

        inf.inference(ckpt, item)

        if not os.path.isfile(out_glb):
            _raise_user_error("Export failed: output glb not found.")

        return out_glb, out_glb, preview_img

    except gr.Error:
        raise
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(err)
        raise


def run_refine_segmentation(
    seg_glb_path_state,
    color_quant_step,
    palette_sample_pixels,
    palette_min_pixels,
    palette_max_colors,
    palette_merge_dist,
    samples_per_face,
    flip_v,
    uv_wrap_repeat,
    transition_conf_thresh,
    transition_prop_iters,
    transition_neighbor_min,
    small_component_action,
    small_component_min_faces,
    postprocess_iters,
    min_faces_per_part,
    bake_transforms,
):
    """Segment 按钮: 把分割 GLB 按纹理调色板拆成 parts GLB,显示在第四个框。"""
    try:
        seg_glb_path = seg_glb_path_state if isinstance(seg_glb_path_state, str) else None
        if (seg_glb_path is None) or (not os.path.isfile(seg_glb_path)):
            _raise_user_error("Please run Segmentation first (the segmented GLB is missing).")

        out_dir = os.path.dirname(seg_glb_path)
        out_parts_glb = os.path.join(out_dir, "segmented_parts.glb")

        splitter.split_glb_by_texture_palette_rgb(
            in_glb_path=seg_glb_path,
            out_glb_path=out_parts_glb,
            min_faces_per_part=min_faces_per_part,
            bake_transforms=bool(bake_transforms),
            color_quant_step=color_quant_step,
            palette_sample_pixels=palette_sample_pixels,
            palette_min_pixels=palette_min_pixels,
            palette_max_colors=palette_max_colors,
            palette_merge_dist=palette_merge_dist,
            samples_per_face=samples_per_face,
            flip_v=flip_v,
            uv_wrap_repeat=uv_wrap_repeat,
            transition_conf_thresh=transition_conf_thresh,
            transition_prop_iters=transition_prop_iters,
            transition_neighbor_min=transition_neighbor_min,
            small_component_action=small_component_action,
            small_component_min_faces=small_component_min_faces,
            postprocess_iters=postprocess_iters,
            debug_print=True,
        )

        if not os.path.isfile(out_parts_glb):
            _raise_user_error("Split failed: output parts glb not found.")

        return out_parts_glb

    except gr.Error:
        raise
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(err)
        raise


CSS_TEXT = """
<style>
#in_glb  { height: 520px !important; }
#seg_glb { height: 520px !important; }
#part_glb{ height: 520px !important; }
#img     { height: 520px !important; }
</style>
"""

with gr.Blocks(title="SegviGen Local Demo") as demo:
    gr.HTML(CSS_TEXT)
    gr.Markdown(
        """
# SegviGen: Repurposing 3D Generative Model for Part Segmentation
"""
    )

    # ---------------- 2x2 Layout ----------------
    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            in_glb = gr.Model3D(label="Input GLB", elem_id="in_glb")
        with gr.Column(scale=1, min_width=260):
            seg_glb = gr.Model3D(label="Processed GLB", elem_id="seg_glb")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            with gr.Accordion("2D Segmentation Map (Optional)", open=True):
                map_mode = gr.Radio(
                    choices=["Upload", "Generate (SAM3)"],
                    value="Upload",
                    label="2D Map Mode",
                )
                in_img = gr.Image(
                    label="2D Segmentation Map",
                    type="filepath",
                    elem_id="img",
                    interactive=True,
                )
                sam3_prompts = gr.Textbox(
                    label="SAM3 prompts (Generate 模式必填,按模型部件用空格分隔,如蘑菇 'cap stem gills')",
                    value="cap stem gills",
                )

            seg_btn = gr.Button("Process", variant="primary")

            if FULL_SEG_EXAMPLES:
                gr.Examples(
                    examples=FULL_SEG_EXAMPLES,
                    inputs=[in_glb, in_img],
                    label="Examples",
                    examples_per_page=3,
                    cache_examples=False,
                )
            else:
                gr.Markdown(f"**No examples found** in: `{EXAMPLES_DIR}` (expected: `*.glb` + same-name `*.png`).")

            with gr.Accordion("Advanced segmentation options", open=False):
                def _g(name, default):
                    return getattr(splitter, name, default)

                color_quant_step = gr.Slider(
                    1, 64, value=_g("COLOR_QUANT_STEP", 16), step=1, label="COLOR_QUANT_STEP"
                )
                gr.Markdown(
                    "*COLOR_QUANT_STEP controls the RGB quantization step, where a larger value merges similar colors more aggressively and a smaller value preserves finer color differences.*"
                )

                palette_sample_pixels = gr.Number(
                    value=_g("PALETTE_SAMPLE_PIXELS", 2_000_000), precision=0, label="PALETTE_SAMPLE_PIXELS"
                )
                gr.Markdown(
                    "*PALETTE_SAMPLE_PIXELS sets the maximum number of sampled pixels used to estimate the palette, where more samples improve stability but increase runtime.*"
                )

                palette_min_pixels = gr.Number(
                    value=_g("PALETTE_MIN_PIXELS", 500), precision=0, label="PALETTE_MIN_PIXELS"
                )
                gr.Markdown(
                    "*PALETTE_MIN_PIXELS specifies the minimum pixel count required to keep a color in the palette, where a higher threshold suppresses noise but may discard small parts.*"
                )

                palette_max_colors = gr.Number(
                    value=_g("PALETTE_MAX_COLORS", 256), precision=0, label="PALETTE_MAX_COLORS"
                )
                gr.Markdown(
                    "*PALETTE_MAX_COLORS limits the maximum number of colors retained in the palette, where a larger limit yields finer partitions and a smaller limit enforces stronger merging.*"
                )

                palette_merge_dist = gr.Number(
                    value=_g("PALETTE_MERGE_DIST", 32), precision=0, label="PALETTE_MERGE_DIST"
                )
                gr.Markdown(
                    "*PALETTE_MERGE_DIST defines the distance threshold for merging nearby palette colors in RGB space, where a larger threshold merges near duplicates more often and a smaller threshold keeps colors distinct.*"
                )

                samples_per_face = gr.Dropdown(
                    choices=[1, 4], value=_g("SAMPLES_PER_FACE", 4), label="SAMPLES_PER_FACE"
                )
                gr.Markdown(
                    "*SAMPLES_PER_FACE sets the number of UV samples per triangle used for label voting, where more samples improve robustness near boundaries but increase computation.*"
                )

                flip_v = gr.Checkbox(value=_g("FLIP_V", True), label="FLIP_V")
                gr.Markdown(
                    "*FLIP_V toggles whether the V coordinate is flipped to match common glTF texture conventions, and you should disable it only if the texture appears vertically inverted.*"
                )

                uv_wrap_repeat = gr.Checkbox(value=_g("UV_WRAP_REPEAT", True), label="UV_WRAP_REPEAT")
                gr.Markdown(
                    "*UV_WRAP_REPEAT selects how out of range UVs are handled by either repeating via modulo or clamping to the unit interval, and repeating is typically preferred for tiled textures.*"
                )

                transition_conf_thresh = gr.Slider(
                    0.25, 1.0, value=float(_g("TRANSITION_CONF_THRESH", 1.0)), step=0.25, label="TRANSITION_CONF_THRESH"
                )
                gr.Markdown(
                    "*TRANSITION_CONF_THRESH sets the confidence threshold for transition handling, where a higher value makes refinement more conservative and a lower value enables more aggressive smoothing.*"
                )

                transition_prop_iters = gr.Number(
                    value=_g("TRANSITION_PROP_ITERS", 6), precision=0, label="TRANSITION_PROP_ITERS"
                )
                gr.Markdown(
                    "*TRANSITION_PROP_ITERS specifies the number of propagation iterations used in transition refinement, where more iterations strengthen diffusion effects but increase runtime.*"
                )

                transition_neighbor_min = gr.Number(
                    value=_g("TRANSITION_NEIGHBOR_MIN", 1), precision=0, label="TRANSITION_NEIGHBOR_MIN"
                )
                gr.Markdown(
                    "*TRANSITION_NEIGHBOR_MIN requires a minimum number of supporting neighbors to propagate a label, where a higher requirement is more conservative and a lower requirement is more permissive.*"
                )

                small_component_action = gr.Dropdown(
                    choices=["reassign", "drop"], value=_g("SMALL_COMPONENT_ACTION", "reassign"), label="SMALL_COMPONENT_ACTION"
                )
                gr.Markdown(
                    "*SMALL_COMPONENT_ACTION determines how small connected components are handled by either reassigning them to neighboring labels or dropping them entirely.*"
                )

                small_component_min_faces = gr.Number(
                    value=_g("SMALL_COMPONENT_MIN_FACES", 50), precision=0, label="SMALL_COMPONENT_MIN_FACES"
                )
                gr.Markdown(
                    "*SMALL_COMPONENT_MIN_FACES defines the face count threshold used to classify a component as small, where a higher threshold merges or removes more fragments and a lower threshold preserves more small parts.*"
                )

                postprocess_iters = gr.Number(
                    value=_g("POSTPROCESS_ITERS", 3), precision=0, label="POSTPROCESS_ITERS"
                )
                gr.Markdown(
                    "*POSTPROCESS_ITERS sets the number of post processing iterations, where more iterations produce stronger cleanup at the cost of additional computation.*"
                )

                min_faces_per_part = gr.Number(
                    value=_g("MIN_FACES_PER_PART", 1), precision=0, label="MIN_FACES_PER_PART"
                )
                gr.Markdown(
                    "*MIN_FACES_PER_PART enforces a minimum number of faces per exported part, where a larger value filters tiny outputs and a smaller value retains fine components.*"
                )

                bake_transforms = gr.Checkbox(value=_g("BAKE_TRANSFORMS", True), label="BAKE_TRANSFORMS")
                gr.Markdown(
                    "*BAKE_TRANSFORMS controls whether scene graph transforms are baked into geometry before splitting, where enabling it improves consistency in world space and disabling it preserves node transforms.*"
                )

        with gr.Column(scale=1, min_width=260):
            refine_btn = gr.Button("Segment", variant="secondary")
            part_glb = gr.Model3D(label="Segmented GLB", elem_id="part_glb")

    seg_glb_state = gr.State(None)

    map_mode.change(
        fn=_update_img_box,
        inputs=[map_mode],
        outputs=[in_img],
    )

    seg_btn.click(
        fn=run_seg,
        inputs=[in_glb, map_mode, in_img, sam3_prompts],
        outputs=[seg_glb, seg_glb_state, in_img],
    )

    refine_btn.click(
        fn=run_refine_segmentation,
        inputs=[
            seg_glb_state,
            color_quant_step,
            palette_sample_pixels,
            palette_min_pixels,
            palette_max_colors,
            palette_merge_dist,
            samples_per_face,
            flip_v,
            uv_wrap_repeat,
            transition_conf_thresh,
            transition_prop_iters,
            transition_neighbor_min,
            small_component_action,
            small_component_min_faces,
            postprocess_iters,
            min_faces_per_part,
            bake_transforms,
        ],
        outputs=[part_glb],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860)
