import unittest

import numpy as np
import trimesh

from auto_prompts import drop_whole_words, largest_blob_share, propose_from_masks
from hybrid_complete import (OpenSurfaces, cap_loops, cull_intrusions, quality_score,
                             solid_metrics)


def body_and_tail():
    """An open body box, a tail cylinder sticking out of its +x face, and a body solid
    that X-Part-style grew the tail back."""
    body = trimesh.creation.box(extents=(1, 1, 1))
    tail = trimesh.creation.cylinder(radius=0.08, height=0.8, sections=24)
    tail.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    tail.apply_translation((0.5 + 0.4, 0, 0))
    solid = trimesh.util.concatenate([body.copy(), tail.copy()])
    return body, tail, solid


class CullIntrusionsTest(unittest.TestCase):
    def test_the_regrown_tail_is_cut_off_the_body_and_the_hole_capped(self):
        body, tail, solid = body_and_tail()
        surfaces = OpenSurfaces({0: ("00_body", body), 1: ("01_tail", tail)}, samples=4000)
        box = np.array(body.bounds)
        diag = float(np.linalg.norm(box[1] - box[0]))
        tau = 0.02 * diag
        before = solid_metrics(body, solid, box, samples=4000, surfaces=surfaces, instance=0)
        self.assertGreater(before["intrusion"], 0.04)   # the tail is ~7% of the solid's area
        culled, share = cull_intrusions(solid, 0, surfaces, tau, diag)
        self.assertGreater(share, 0.05)
        self.assertLess(culled.bounds[1][0], 0.6)          # the tail is gone
        after = solid_metrics(body, culled, box, samples=4000, surfaces=surfaces, instance=0)
        self.assertLess(after["intrusion"], 0.01)
        self.assertGreater(quality_score(after), quality_score(before))
        self.assertTrue(culled.is_watertight)

    def test_a_region_with_a_long_rim_is_left_alone(self):
        body, tail, solid = body_and_tail()
        surfaces = OpenSurfaces({0: ("00_body", body), 1: ("01_tail", tail)}, samples=4000)
        diag = float(np.linalg.norm(body.bounds[1] - body.bounds[0]))
        culled, share = cull_intrusions(solid, 0, surfaces, 0.02 * diag, diag, max_rim=0.1)
        self.assertEqual(share, 0.0)
        self.assertIs(culled, solid)

    def test_intrusion_discounts_the_score(self):
        clean = {"cover": 1.0, "fit_p90": 0.0, "extra_p90": 0.0, "escape": 0.0,
                 "largest_shell": 1.0, "intrusion": 0.0}
        self.assertAlmostEqual(quality_score(clean), 1.0)
        self.assertAlmostEqual(quality_score({**clean, "intrusion": 0.05}), 0.5)
        self.assertAlmostEqual(quality_score({**clean, "intrusion": 0.2}), 0.0)


class CapLoopsTest(unittest.TestCase):
    def test_a_box_missing_one_face_is_closed_again(self):
        box = trimesh.creation.box()
        faces = np.asarray(box.faces)
        keep = box.face_normals[:, 2] < 0.5                 # drop the +z face (two triangles)
        opened = trimesh.Trimesh(box.vertices, faces[keep], process=False)
        self.assertFalse(opened.is_watertight)
        edges = opened.edges[trimesh.grouping.group_rows(opened.edges_sorted, require_count=1)]
        from hybrid_complete import _edge_loops
        loops = _edge_loops(edges)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 4)
        closed = cap_loops(opened, loops)
        self.assertTrue(closed.is_watertight)
        self.assertTrue(closed.is_winding_consistent)


class BlobShareTest(unittest.TestCase):
    def test_two_blobs_count_singly(self):
        fg = np.zeros((1, 40, 40), bool)
        fg[:, 0:40, 0:40] = True
        masks = np.zeros((1, 2, 40, 40), bool)
        masks[0, 0, 0:40, 0:14] = True                      # one blob, 35%
        masks[0, 0, 0:40, 26:40] = True                     # second blob, 35%
        masks[0, 1, 0:40, 0:30] = True                      # one blob, 75%
        blob = largest_blob_share(masks, fg)
        self.assertAlmostEqual(blob[0], 0.35, places=2)
        self.assertAlmostEqual(blob[1], 0.75, places=2)

    def test_a_multi_blob_word_over_half_the_object_is_still_a_part_candidate(self):
        size = 64
        fg = np.zeros((2, size, size), bool)
        fg[:, 4:60, 4:60] = True
        masks = np.zeros((2, 2, size, size), bool)
        for row in (8, 24, 40):                             # three shelves, ~60% together
            masks[:, 0, row:row + 12, 6:58] = True
        masks[:, 1, 4:60, 4:60] = True                      # rack: the whole thing
        scores = np.array([[0.8, 0.9], [0.8, 0.9]])
        proposal = propose_from_masks(masks, fg, scores, ["shelf", "rack"])
        self.assertEqual(proposal["main"], "rack")
        self.assertIn("shelf", proposal["parts"])

    def test_vlm_whole_words_are_judged_per_blob(self):
        rows = [{"concept": "shelf", "area": 0.62, "blob": 0.11},
                {"concept": "frame", "area": 0.9, "blob": 0.9}]
        kept, dropped = drop_whole_words(["shelf", "frame"], rows)
        self.assertEqual(kept, ["shelf"])
        self.assertEqual(dropped, ["frame"])


if __name__ == "__main__":
    unittest.main()
