"""Speed-ups of the split stage: batched samples, label-only exports, shared voxels."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.parts_rebake import face_base_colors  # noqa: E402
from pipeline import PipelineOptions, DEFAULT_SAMPLE_EXPORT, SAMPLE_EXPORT_MODES  # noqa: E402


def _flat_coloured_mesh(colors):
    """Two triangles with un-shared vertices, one colour each, round-tripped through glb."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.arange(6).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate([np.asarray(colors, dtype=np.uint8), np.full((2, 1), 255, np.uint8)], axis=1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=rgba)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seg.glb")
        mesh.export(path)
        scene = trimesh.load(path, force="scene")
        return next(g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh))


class FaceBaseColors(unittest.TestCase):
    def test_reads_face_colours_from_a_label_only_glb(self):
        mesh = _flat_coloured_mesh([[220, 40, 40], [40, 200, 60]])
        colors = face_base_colors(mesh)
        self.assertEqual(colors.shape, (2, 3))
        self.assertEqual(colors.tolist(), [[220, 40, 40], [40, 200, 60]])
        self.assertEqual(colors.dtype, np.int16)


class SampleExportOption(unittest.TestCase):
    def test_default_is_labels_and_reaches_segment_parts_only(self):
        options = PipelineOptions()
        self.assertEqual(DEFAULT_SAMPLE_EXPORT, "labels")
        self.assertIn(options.sample_export, SAMPLE_EXPORT_MODES)
        self.assertIn("sample_export", options.segment_kwargs())
        self.assertNotIn("sample_export", options.merge_kwargs())

    def test_from_mapping_validates_the_mode(self):
        self.assertEqual(PipelineOptions.from_mapping({"sample_export": "textured"}).sample_export, "textured")
        with self.assertRaises(ValueError):
            PipelineOptions.from_mapping({"sample_export": "vertex"})


class InferenceItems(unittest.TestCase):
    def _args(self, **overrides):
        base = dict(two_d_map=False, glb="model.glb", input_vxz="work/input.vxz", recompute_vxz=False,
                    export="labels", blender_reuv=False, rebake_texture_size=2048, azimuth=0.0,
                    legend_ckpt=None, legend=None, img2=None, transforms="t.json",
                    img="render.png", export_glb="seg.glb", items=None)
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_single_item_cli_is_unchanged(self):
        from inference_full import items_from_args
        items = items_from_args(self._args())
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["img"].endswith("render.png"))
        self.assertTrue(items[0]["export_glb"].endswith("seg.glb"))
        self.assertEqual(items[0]["export"], "labels")
        self.assertTrue(items[0]["reuse_vxz"])

    def test_items_json_shares_the_input_and_sets_per_sample_fields(self):
        from inference_full import items_from_args
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "items.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"img": "a.png", "export_glb": "a.glb", "azimuth": 30},
                           {"img": "b.png", "export_glb": "b.glb"}], handle)
            items = items_from_args(self._args(items=path, img=None, export_glb=None, azimuth=-15.0))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["azimuth"], 30.0)
        self.assertEqual(items[1]["azimuth"], -15.0)
        self.assertEqual(items[0]["input_vxz"], items[1]["input_vxz"])
        self.assertTrue(items[1]["export_glb"].endswith("b.glb"))


if __name__ == "__main__":
    unittest.main()
