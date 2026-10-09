import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

from p3sam_units import SOURCE_TO_SEG, accept, atom_colour, p3sam_atoms, seg_glb_from_ids
from pipeline import DEFAULT_UNITS, UNITS_MODES, PipelineOptions


class AcceptTest(unittest.TestCase):
    def test_one_big_part_is_no_split_and_several_real_parts_are(self):
        ok, reason = accept({"shares": [0.988, 0.006, 0.003, 0.003], "largest_share": 0.988})
        self.assertFalse(ok); self.assertIn("part(s)", reason)
        ok, reason = accept({"shares": [0.43, 0.15, 0.09, 0.09, 0.06, 0.06, 0.06, 0.06], "largest_share": 0.43})
        self.assertTrue(ok)
        ok, reason = accept({"shares": [0.9, 0.05, 0.05], "largest_share": 0.9})
        self.assertFalse(ok); self.assertIn("limit", reason)


class SegGlbTest(unittest.TestCase):
    def test_pieces_become_atoms_with_flat_colours_in_the_seg_frame(self):
        box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
        box.apply_translation((0.0, 0.0, 3.0))
        ids = np.where(box.triangles_center[:, 0] > 0, 7, 3).astype(np.int64)
        ids[0] = -1                                     # one face P3-SAM left unlabelled
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "mesh.glb"); box.export(src)
            out = os.path.join(tmp, "seg.glb")
            atoms = seg_glb_from_ids(src, ids, out)
            self.assertEqual(sorted(set(atoms.tolist())), [0, 1])     # renumbered, no -1
            seg = trimesh.load(out, force="mesh", process=False)
            self.assertEqual(len(seg.vertices), 3 * len(box.faces))     # un-shared vertices
            colours = {tuple(c[:3]) for c in np.asarray(seg.visual.face_colors)}
            self.assertEqual(len(colours), 2)
            # rotated into the seg frame: the +Z offset of the source turns into +Y
            self.assertGreater(seg.vertices[:, 1].mean(), 2.0)
            np.testing.assert_allclose(SOURCE_TO_SEG @ np.array([0.0, 0.0, 3.0]), [0.0, 3.0, 0.0])

    def test_colours_are_distinct_for_many_atoms(self):
        colours = [tuple(atom_colour(i)[:3]) for i in range(30)]
        self.assertEqual(len(set(colours)), 30)


class P3samAtomsTest(unittest.TestCase):
    def fake_runner(self, shares):
        def runner(glb, out_dir, py_xpart, *args, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
            box.export(os.path.join(out_dir, "mesh.glb"))
            n = len(box.faces)
            ids = np.zeros(n, dtype=np.int64)
            if len(shares) > 1:
                ids[box.triangles_center[:, 0] > 0] = 1
                ids[box.triangles_center[:, 1] > 0.3] = 2
            np.save(os.path.join(out_dir, "face_ids.npy"), ids)
            stats = {"parts": len(shares), "shares": shares, "largest_share": max(shares), "seconds": 1.0}
            json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"))
            return stats
        return runner

    def test_a_real_split_becomes_the_reference_and_atoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = p3sam_atoms("model.glb", tmp, "python", runner=self.fake_runner([0.5, 0.3, 0.2]))
            self.assertIsNotNone(result)
            self.assertTrue(os.path.isfile(result["seg_glb"]))
            self.assertTrue(os.path.isfile(result["atoms_npy"]))
            self.assertEqual(result["report"]["source"], "p3sam")
            self.assertEqual(result["report"]["atoms"], 3)

    def test_no_split_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(p3sam_atoms("model.glb", tmp, "python", runner=self.fake_runner([0.99])))

    def test_a_failed_run_falls_back(self):
        def boom(*args, **kwargs):
            raise OSError("no such venv")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(p3sam_atoms("model.glb", tmp, "python", runner=boom))


class OptionTest(unittest.TestCase):
    def test_units_is_a_published_switch(self):
        self.assertEqual(UNITS_MODES, ("auto", "p3sam", "segvigen"))
        self.assertIn(DEFAULT_UNITS, UNITS_MODES)
        options = PipelineOptions()
        self.assertEqual(options.segment_kwargs()["units"], DEFAULT_UNITS)
        self.assertNotIn("units", options.merge_kwargs())
        self.assertEqual(options.public()["switches"]["units"], list(UNITS_MODES))


if __name__ == "__main__":
    unittest.main()
