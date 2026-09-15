import json
import os
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image

from refine_units import relabel_patches
from source_export import export_from_source, fit_frame, smooth_labels, transfer_labels


def fine_box():
    mesh = trimesh.creation.box()
    for _ in range(4):
        mesh = mesh.subdivide()
    return mesh


class RelabelPatchesTest(unittest.TestCase):
    def test_a_coherent_patch_moves_and_a_speck_stays(self):
        mesh = fine_box()                                   # one unit, voted label 0
        units = np.zeros(len(mesh.faces), dtype=int)
        labels = np.zeros(len(mesh.faces), dtype=int)
        centre = mesh.triangles_center
        per_face = np.where(centre[:, 0] > 0.0, 1, 0)      # masks say: the +x half is part 1
        speck = np.flatnonzero(centre[:, 0] < -0.4)[:3]
        per_face[speck] = 1                                  # ... plus three stray faces
        refined, changes = relabel_patches(mesh, units, labels, per_face, min_share=0.1, min_faces=20)
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes[0]["share_of_unit"], 0.5, places=2)
        self.assertTrue(np.all(refined[centre[:, 0] > 0.0] == 1))
        self.assertTrue(np.all(refined[speck] == 0))

    def test_a_patch_under_the_share_floor_stays_with_its_unit(self):
        mesh = fine_box()
        units = np.zeros(len(mesh.faces), dtype=int)
        labels = np.zeros(len(mesh.faces), dtype=int)
        per_face = np.where(mesh.triangles_center[:, 0] > 0.45, 1, 0)   # one face of six
        refined, changes = relabel_patches(mesh, units, labels, per_face, min_share=0.3)
        self.assertEqual(changes, [])
        self.assertTrue(np.all(refined == 0))


class SourceExportTest(unittest.TestCase):
    def textured(self, mesh):
        uv = (mesh.vertices[:, :2] - mesh.vertices[:, :2].min(axis=0)) / 2.0
        image = Image.new("RGB", (8, 8), (200, 120, 40))
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=trimesh.visual.material.SimpleMaterial(image=image))
        return mesh

    def test_frame_fit_finds_the_axis_swap(self):
        source = trimesh.creation.icosphere(3)
        source.apply_scale([1.0, 2.0, 0.5])
        turned = np.asarray(source.triangles_center) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        name, rotation, scale, shift, distance = fit_frame(turned, np.asarray(source.triangles_center))
        self.assertIn(name, ("y<->z", "y<->z-flip"))
        self.assertLess(distance, 1e-6)
        np.testing.assert_allclose(turned @ rotation.T * scale + shift,
                                   np.asarray(source.triangles_center), atol=1e-6)

    def test_labels_carry_over_and_smooth_out_the_stray_face(self):
        source = trimesh.creation.icosphere(4)
        remesh = trimesh.creation.icosphere(2)
        remesh_labels = (np.asarray(remesh.triangles_center)[:, 2] >= 0).astype(int)
        labels, fit = transfer_labels(remesh, remesh_labels, source)
        z = np.asarray(source.triangles_center)[:, 2]
        self.assertGreater(np.mean(labels[z > 0.1] == 1), 0.99)
        self.assertGreater(np.mean(labels[z < -0.1] == 0), 0.99)
        noisy = labels.copy(); noisy[0] = 1 - noisy[0]
        self.assertEqual(smooth_labels(source, noisy, 1)[0], labels[0])

    def test_parts_are_cut_from_the_source_with_its_uvs(self):
        source = self.textured(trimesh.creation.icosphere(4))
        remesh = trimesh.creation.icosphere(2)
        remesh_labels = (np.asarray(remesh.triangles_center)[:, 2] >= 0).astype(int)
        with tempfile.TemporaryDirectory() as out:
            src_glb = os.path.join(out, "source.glb"); source.export(src_glb)
            remesh_glb = os.path.join(out, "seg.glb"); remesh.export(remesh_glb)
            np.save(os.path.join(out, "labels.npy"), remesh_labels)
            with open(os.path.join(out, "names.json"), "w", encoding="utf-8") as handle:
                json.dump(["bottom", "top"], handle)
            manifest = export_from_source(remesh_glb, src_glb, os.path.join(out, "labels.npy"),
                                          os.path.join(out, "names.json"), os.path.join(out, "parts.glb"))
            self.assertEqual([row["node"] for row in manifest], ["part_00_bottom", "part_01_top"])
            self.assertEqual(sum(row["faces"] for row in manifest), len(source.faces))
            back = trimesh.load(os.path.join(out, "parts.glb"), force="scene")
            self.assertEqual(len(back.geometry), 2)
            for mesh in back.geometry.values():
                self.assertIsNotNone(mesh.visual.uv)
            self.assertTrue(os.path.isfile(os.path.join(out, "parts.json")))

    def test_a_multi_mesh_source_falls_back(self):
        scene = trimesh.Scene([trimesh.creation.box(), trimesh.creation.icosphere(1)])
        with tempfile.TemporaryDirectory() as out:
            src_glb = os.path.join(out, "two.glb"); scene.export(src_glb)
            remesh = trimesh.creation.icosphere(2); remesh_glb = os.path.join(out, "seg.glb"); remesh.export(remesh_glb)
            np.save(os.path.join(out, "labels.npy"), np.zeros(len(remesh.faces), dtype=int))
            with open(os.path.join(out, "names.json"), "w", encoding="utf-8") as handle:
                json.dump(["all"], handle)
            self.assertIsNone(export_from_source(remesh_glb, src_glb, os.path.join(out, "labels.npy"),
                                                 os.path.join(out, "names.json"), os.path.join(out, "parts.glb")))


if __name__ == "__main__":
    unittest.main()
