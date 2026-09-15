import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

from hybrid_complete import (apply_scored, largest_shell_share, quality_score,
                             score_candidates, solid_metrics)


def box_mesh(centre=(0.0, 0.0, 0.0), size=1.0):
    mesh = trimesh.creation.box(extents=[size, size, size])
    mesh.apply_translation(centre)
    return mesh


def open_box(centre=(0.0, 0.0, 0.0), size=1.0):
    mesh = box_mesh(centre, size)
    # One triangle cut away: a small cap, like the cut a real part is left with. A whole
    # missing side would be 1/6 of the surface, and closing it is "invented" geometry to
    # the p90 extra term -- a perfect closure of that scores about 0.7.
    return mesh.submesh([np.arange(1, len(mesh.faces))], append=True)


class QualityScoreTest(unittest.TestCase):
    def test_a_perfect_fit_scores_one_and_nothing_scores_zero(self):
        self.assertEqual(quality_score({"cover": 1.0, "extra_p90": 0.0, "escape": 0.0}), 1.0)
        self.assertEqual(quality_score(None), 0.0)

    def test_escaping_the_box_or_inventing_shape_zeroes_the_score(self):
        self.assertEqual(quality_score({"cover": 1.0, "extra_p90": 0.0, "escape": 1.2}), 0.0)
        self.assertEqual(quality_score({"cover": 1.0, "extra_p90": 0.5, "escape": 0.0}), 0.0)

    def test_the_closed_box_fits_its_open_surface_and_a_small_blob_does_not(self):
        surface, box = open_box(), np.array([[-0.5] * 3, [0.5] * 3])
        good = solid_metrics(surface, box_mesh(), box, samples=4000)
        self.assertGreater(good["cover"], 0.95)
        self.assertEqual(good["escape"], 0.0)
        blob = solid_metrics(surface, trimesh.creation.icosphere(radius=0.2), box, samples=4000)
        self.assertLess(blob["cover"], 0.2)
        self.assertGreater(quality_score(good), quality_score(blob))

    def test_a_solid_in_tatters_scores_by_its_biggest_shell(self):
        box = box_mesh()
        self.assertEqual(largest_shell_share(box), 1.0)
        crumbs = trimesh.util.concatenate([box_mesh((float(i) * 3, 0, 0), 0.3) for i in range(4)])
        self.assertAlmostEqual(largest_shell_share(crumbs), 0.25, places=6)
        good = {"cover": 1.0, "extra_p90": 0.0, "escape": 0.0, "largest_shell": 1.0}
        tatters = dict(good, largest_shell=0.05)
        self.assertGreater(quality_score(good), 10 * quality_score(tatters))


class CandidateThresholdTest(unittest.TestCase):
    def test_a_small_instance_is_asked_at_the_lower_threshold(self):
        # One big body, one small far-off piece: both X-Part solids are decent (score ~0.7).
        big, small = box_mesh(size=1.0), box_mesh((4.0, 0.0, 0.0), 0.2)
        rows = [{"name": "body", "instance": 0, "area_share": 0.96, "faces": 12,
                 "box": [[-0.5] * 3, [0.5] * 3]},
                {"name": "body", "instance": 1, "area_share": 0.04, "faces": 12,
                 "box": [[3.9, -0.1, -0.1], [4.1, 0.1, 0.1]]}]
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "boxes.json"), "w", encoding="utf-8") as handle:
                json.dump(rows, handle)
            opened = trimesh.Scene(); solids = trimesh.Scene()
            for row, mesh in zip(rows, (big, small)):
                node = f"{row['instance']:02d}_{row['name']}"
                lo, hi = row["box"][0][0], row["box"][1][0]
                opened.add_geometry(open_box(((lo + hi) / 2, 0.0, 0.0), hi - lo), geom_name=node)
                solids.add_geometry(mesh, geom_name=node)
            opened.export(os.path.join(out, "open_instances.glb"))
            solids.export(os.path.join(out, "xpart_instances.glb"))
            decisions = score_candidates(out, candidate=0.99, candidate_small=0.01)
            self.assertEqual([d["large"] for d in decisions], [True, False])
            self.assertEqual([d["candidate"] for d in decisions], [True, False])


class ApplyScoredTest(unittest.TestCase):
    def test_each_instance_keeps_its_best_solid_or_falls_back_to_the_open_surface(self):
        centres = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
        boxes = [{"name": n, "instance": i, "area_share": 1 / 3, "faces": 10,
                  "box": [[c[0] - 0.5, -0.5, -0.5], [c[0] + 0.5, 0.5, 0.5]]}
                 for i, (n, c) in enumerate(zip(["good", "fixable", "hopeless"], centres))]
        far = lambda c: box_mesh((c[0] + 5.0, 5.0, 5.0), 0.3)          # nowhere near its box
        xpart = [box_mesh(centres[0]), far(centres[1]), far(centres[2])]
        holo = [box_mesh(centres[1]), far(centres[2])]
        with tempfile.TemporaryDirectory() as out:
            def scene(meshes, names):
                s = trimesh.Scene()
                for mesh, name in zip(meshes, names):
                    s.add_geometry(mesh, geom_name=name)
                return s
            nodes = [f"{r['instance']:02d}_{r['name']}" for r in boxes]
            with open(os.path.join(out, "boxes.json"), "w", encoding="utf-8") as handle:
                json.dump(boxes, handle)
            scene([open_box(c) for c in centres], nodes).export(os.path.join(out, "open_instances.glb"))
            scene(xpart, nodes).export(os.path.join(out, "xpart_instances.glb"))
            holo_glb = os.path.join(out, "holopart_instances.glb")
            scene(holo, nodes[1:]).export(holo_glb)
            decisions = score_candidates(out)
            self.assertEqual([d["candidate"] for d in decisions], [False, True, True])
            decisions = apply_scored(out, decisions, holo_glb)
            self.assertEqual([d["backend"] for d in decisions], ["xpart", "holopart", "open"])
            self.assertTrue(os.path.isfile(os.path.join(out, "xpart_parts.glb")))


if __name__ == "__main__":
    unittest.main()
