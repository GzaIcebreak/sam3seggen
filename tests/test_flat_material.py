"""Untextured sources skip the Blender bake and carry their own flat material."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.parts_rebake import export_completed_flat, source_flat_material  # noqa: E402


def _box(visual=None):
    mesh = trimesh.creation.box()
    if visual is not None:
        mesh.visual = visual
    return mesh


class SourceFlatMaterial(unittest.TestCase):
    def _export(self, d, mesh, name="src.glb"):
        path = os.path.join(d, name)
        mesh.export(path)
        return path

    def test_untextured_pbr_source_gives_its_material(self):
        with tempfile.TemporaryDirectory() as d:
            mat = PBRMaterial(baseColorFactor=[200, 180, 160, 255], metallicFactor=0.0, roughnessFactor=0.7)
            path = self._export(d, _box(trimesh.visual.TextureVisuals(material=mat)))
            flat = source_flat_material(path)
        self.assertIsNotNone(flat)
        self.assertEqual(flat["base_color"], [200, 180, 160, 255])
        self.assertAlmostEqual(flat["roughness"], 0.7, places=3)

    def test_textured_source_is_baked(self):
        with tempfile.TemporaryDirectory() as d:
            mesh = _box()
            uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
            image = Image.fromarray(np.full((8, 8, 3), 128, np.uint8))
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=image))
            path = self._export(d, mesh)
            self.assertIsNone(source_flat_material(path))

    def test_vertex_coloured_source_is_baked(self):
        with tempfile.TemporaryDirectory() as d:
            mesh = _box()
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile([255, 0, 0, 255], (len(mesh.faces), 1)))
            path = self._export(d, mesh)
            self.assertIsNone(source_flat_material(path))


class CompletedFlat(unittest.TestCase):
    def test_writes_one_node_per_solid_with_the_flat_material(self):
        with tempfile.TemporaryDirectory() as d:
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.box(), node_name="00_part_00_head", geom_name="00_part_00_head")
            scene.add_geometry(trimesh.creation.icosphere(), node_name="01_part_01_torso", geom_name="01_part_01_torso")
            raw = os.path.join(d, "xpart_parts_raw.glb")
            scene.export(raw)
            material = {"base_color": [255, 255, 255, 255], "metallic": 0.0, "roughness": 0.5}
            manifest = export_completed_flat(raw, d, material, combined_name="xpart_parts.glb")
            out = trimesh.load(os.path.join(d, "xpart_parts.glb"), force="scene")
            parts_json = json.load(open(os.path.join(d, "parts.json"), encoding="utf-8"))
        self.assertEqual(len(manifest), 2)
        self.assertEqual(len(out.geometry), 2)
        self.assertEqual([row["texture"] for row in parts_json], ["flat", "flat"])
        for geom in out.geometry.values():
            self.assertIsNotNone(geom.visual.material)


if __name__ == "__main__":
    unittest.main()
