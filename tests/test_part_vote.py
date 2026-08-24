"""Part-level SAM3 voting: assignment rules and merge/export behaviour.

Pure-CPU: vote tables and tiny synthetic meshes stand in for SAM3 and SegviGen.
"""
import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_toolkit.part_vote import assign_parts, merge_parts


def _write_parts_glb(path, count=3):
    scene = trimesh.Scene()
    for index in range(count):
        box = trimesh.creation.box(extent=(0.1, 0.1, 0.1))
        box.apply_translation([index * 0.5, 0.0, 0.0])
        name = f"part_{index:02d}"
        scene.add_geometry(box, node_name=name, geom_name=name)
    scene.export(path)


class AssignPartsTest(unittest.TestCase):
    def test_winner_takes_all(self):
        votes = np.array([[10.0, 2.0], [1.0, 8.0]])
        owned = np.array([10, 10])
        assignment = assign_parts(votes, ["mushroom", "chair"], owned)
        self.assertEqual(assignment, ["mushroom", "chair"])

    def test_bleed_pixels_cannot_name_a_huge_part(self):
        # 25 bleeding mask pixels on a part that owns 200k pixels: noise, not a name.
        votes = np.array([[27539.0, 0.0], [25.0, 0.0]])
        owned = np.array([28000, 200000])
        assignment = assign_parts(votes, ["mushroom", "chair"], owned)
        self.assertEqual(assignment, ["mushroom", None])

    def test_part_with_no_votes_is_unassigned(self):
        votes = np.array([[10.0, 0.0], [0.0, 0.0]])
        owned = np.array([10, 10])
        assignment = assign_parts(votes, ["mushroom", "chair"], owned)
        self.assertEqual(assignment, ["mushroom", None])

    def test_invisible_part_is_unassigned(self):
        votes = np.array([[0.0, 5.0]])
        owned = np.array([0])
        assignment = assign_parts(votes, ["mushroom", "chair"], owned)
        self.assertEqual(assignment, [None])

    def test_min_cover_is_tunable(self):
        votes = np.array([[4.0, 0.0]])
        owned = np.array([10])
        self.assertEqual(assign_parts(votes, ["a", "b"], owned, min_cover=0.5), [None])
        self.assertEqual(assign_parts(votes, ["a", "b"], owned, min_cover=0.3), ["a"])

    def test_unassigned_to_folds_blanks_into_named_part(self):
        votes = np.array([[10.0, 0.0], [0.0, 0.0]])
        owned = np.array([10, 10])
        assignment = assign_parts(votes, ["mushroom", "chair"], owned, unassigned_to="chair")
        self.assertEqual(assignment, ["mushroom", "chair"])


class MergePartsTest(unittest.TestCase):
    def test_merge_groups_members_into_one_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = os.path.join(tmp, "parts.glb")
            _write_parts_glb(glb, count=3)
            out = os.path.join(tmp, "out", "merged.glb")
            manifest = merge_parts(
                glb, ["mushroom", "chair", "chair"], ["mushroom", "chair"], out,
            )
            names = [row["name"] for row in manifest]
            self.assertEqual(names, ["mushroom", "chair"])
            chair = manifest[1]
            self.assertEqual(chair["members"], ["part_01", "part_02"])
            self.assertEqual(chair["nodes"], ["part_01_chair"])
            scene = trimesh.load(out)
            # two objects total, each exactly one geometry
            self.assertEqual(len(scene.geometry), 2)
            chair_mesh = scene.geometry["part_01_chair"]
            # 2 boxes = 24 vertices / 24 faces worth of geometry in one mesh
            self.assertEqual(len(chair_mesh.faces), 24)
            with open(os.path.join(tmp, "out", "parts.json"), encoding="utf-8") as handle:
                self.assertEqual(len(json.load(handle)), 2)

    def test_atlas_merge_packs_textures_into_one_material(self):
        from PIL import Image

        from data_toolkit.part_vote import _atlas_merge

        meshes = []
        for colour in ((200, 30, 30), (30, 30, 200)):
            box = trimesh.creation.box()
            texture = Image.new("RGB", (64, 64), colour)
            box.visual = trimesh.visual.TextureVisuals(
                uv=np.random.rand(len(box.vertices), 2),
                material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture),
            )
            meshes.append(box)
        merged = _atlas_merge(meshes)
        self.assertEqual(len(merged.faces), 24)
        atlas = merged.visual.material.baseColorTexture
        self.assertEqual(atlas.size, (2 * 64, 1 * 64))
        uv = np.asarray(merged.visual.uv)
        # every uv lands inside its member's atlas tile (a box has 8 corners)
        self.assertTrue((uv[:, 0] < 0.5)[:8].all())
        self.assertTrue((uv[:, 0] >= 0.5)[8:].all())

    def test_strict_rejects_requested_name_without_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = os.path.join(tmp, "parts.glb")
            _write_parts_glb(glb, count=2)
            with self.assertRaises(ValueError):
                merge_parts(glb, ["chair", "chair"], ["mushroom", "chair"],
                            os.path.join(tmp, "out.glb"))

    def test_strict_rejects_leftover_unnamed_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = os.path.join(tmp, "parts.glb")
            _write_parts_glb(glb, count=3)
            with self.assertRaises(ValueError):
                merge_parts(glb, ["chair", "chair", None], ["chair"],
                            os.path.join(tmp, "out.glb"))

    def test_allow_partial_keeps_what_it_can(self):
        with tempfile.TemporaryDirectory() as tmp:
            glb = os.path.join(tmp, "parts.glb")
            _write_parts_glb(glb, count=2)
            manifest = merge_parts(glb, ["chair", None], ["chair"],
                                   os.path.join(tmp, "out.glb"), strict=False)
            self.assertEqual([row["name"] for row in manifest], ["chair"])


if __name__ == "__main__":
    unittest.main()
