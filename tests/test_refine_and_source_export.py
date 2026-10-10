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


class RefineEvidenceTest(unittest.TestCase):
    def test_unvoted_faces_lose_their_grown_label(self):
        from data_toolkit.lift_sam3 import drop_unvoted

        labels = np.array([0, 0, 1, 1, 1])
        votes = np.array([[2, 0], [0, 0], [0, 0], [0, 0], [0, 3]], dtype=float)
        self.assertEqual(drop_unvoted(labels, votes).tolist(), [0, -1, -1, -1, 1])

    def test_one_view_is_not_evidence(self):
        from data_toolkit.lift_sam3 import drop_thin_evidence

        labels = np.array([0, 1, 1, -1])
        counts = np.array([[3, 0], [0, 1], [1, 2], [0, 0]])
        self.assertEqual(drop_thin_evidence(labels, counts, 2).tolist(), [0, -1, 1, -1])
        self.assertEqual(drop_thin_evidence(labels, counts, 1).tolist(), [0, 1, 1, -1])

    def test_refine_reads_only_voted_faces(self):
        import data_toolkit.lift_sam3 as lift_sam3
        import refine_units

        mesh = fine_box()
        units = np.zeros(len(mesh.faces), dtype=int)
        labels = np.zeros(len(mesh.faces), dtype=int)           # the vote: all torso
        seen = {}

        def fake_lift(*args, **kwargs):
            seen.update(kwargs)
            # a lift that grew `head` over the whole unit but voted it on no face
            grown = np.ones(len(mesh.faces), dtype=int)
            if kwargs.get("keep_unvoted", True) is False:
                grown[:] = -1
            return grown, ["torso", "head"], None, {}  # no raster: nothing to veto with

        saved = lift_sam3.lift
        lift_sam3.lift = fake_lift
        try:
            new, changes = refine_units.refine_labels_by_masks(
                mesh, units, labels, ["torso", "head"], None, None, 0.7, 512)
        finally:
            lift_sam3.lift = saved
        self.assertIs(seen.get("keep_unvoted"), False)
        self.assertEqual(seen.get("min_views"), 2)
        self.assertEqual(changes, [])
        self.assertTrue((new == 0).all())


class FlankedTest(unittest.TestCase):
    def test_between_two_ears_is_flanked_a_hand_past_a_cuff_is_not(self):
        from refine_units import flanked_share

        patch = np.zeros((10, 10), dtype=bool)
        patch[2:6, 4:6] = True                      # the back of the head
        ears = np.zeros((10, 10), dtype=bool)
        ears[1:7, 1:3] = True
        ears[1:7, 7:9] = True                       # one ear each side
        self.assertEqual(flanked_share(patch, ears), 1.0)
        cuff = np.zeros((10, 10), dtype=bool)
        cuff[2:6, 1:4] = True                       # the old part on one side only
        self.assertEqual(flanked_share(patch, cuff), 0.0)

    def test_the_veto_reads_only_views_that_support_the_move(self):
        from types import SimpleNamespace

        from refine_units import patch_flanked_by

        face_ids = np.zeros((2, 10, 10), dtype=np.int32)
        face_ids[:, 2:6, 4:6] = 1                   # face 0 seen in both views
        head = np.zeros((2, 10, 10), dtype=bool)
        head[:, 1:7, 1:3] = True
        head[:, 1:7, 7:9] = True
        torso = np.zeros((2, 10, 10), dtype=bool)
        torso[0, 0:8, 3:7] = True                   # only view 0 calls the patch torso
        masks = np.stack([head, torso], axis=1)     # [views, concepts, H, W]
        mask_set = SimpleNamespace(masks=masks, scores=np.ones((2, 2)), owners=["head", "torso"])
        self.assertEqual(patch_flanked_by([0], "head", "torso", face_ids, mask_set), 1.0)
        self.assertIsNone(patch_flanked_by([0], "torso", "head", face_ids,
                                           SimpleNamespace(masks=np.zeros_like(masks),
                                                           scores=np.ones((2, 2)),
                                                           owners=["head", "torso"])))


if __name__ == "__main__":
    unittest.main()
