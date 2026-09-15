import unittest

import numpy as np

import trimesh

from hybrid_complete import decide_backend, drop_duplicate_faces, instance_node_name


class LargePolicyTest(unittest.TestCase):
    row = {"name": "tree", "instance": 3, "area_share": 0.63,
           "box": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}
    inside = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    src_ext = np.array([1.0, 1.0, 1.0])

    def test_the_default_keeps_a_large_solid_that_stayed_in_its_box(self):
        self.assertEqual(decide_backend(self.row, self.inside, self.src_ext)["backend"], "xpart")

    def test_always_sends_every_large_instance_to_holopart(self):
        decision = decide_backend(self.row, self.inside, self.src_ext, large_policy="always")
        self.assertEqual(decision["backend"], "holopart")

    def test_always_leaves_small_instances_with_xpart(self):
        small = dict(self.row, area_share=0.01, box=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        decision = decide_backend(small, np.array(small["box"]), self.src_ext,
                                  large_policy="always")
        self.assertEqual(decision["backend"], "xpart")

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError):
            decide_backend(self.row, self.inside, self.src_ext, large_policy="sometimes")

    def test_node_names_match_open_instances(self):
        self.assertEqual(instance_node_name(self.row), "03_tree")


class DuplicateFacesTest(unittest.TestCase):
    def test_a_repeated_face_is_dropped_whatever_its_winding(self):
        box = trimesh.creation.box()
        faces = np.vstack([box.faces, box.faces[:1][:, ::-1], box.faces[1:2]])
        noisy = trimesh.Trimesh(box.vertices, faces, process=False)
        clean = drop_duplicate_faces(noisy)
        self.assertEqual(len(clean.faces), len(box.faces))
        self.assertAlmostEqual(clean.area, box.area, places=6)

    def test_a_clean_mesh_is_returned_untouched(self):
        box = trimesh.creation.box()
        self.assertIs(drop_duplicate_faces(box), box)


if __name__ == "__main__":
    unittest.main()
