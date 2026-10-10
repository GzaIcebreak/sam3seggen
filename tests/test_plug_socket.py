import unittest

import numpy as np
import trimesh

from hybrid_complete import quality_score
from xpart_complete import cut_loops, intrusion_share, plug_height, plug_metrics


def open_tube(radius=0.5, height=2.0):
    """A cylinder without its top cap: the open part, cut at z = +height/2."""
    tube = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    keep = ~((tube.face_normals[:, 2] > 0.9) & (tube.triangles_center[:, 2] > 0))
    return tube.submesh([np.flatnonzero(keep)], append=True)


def neighbour_points(radius=0.5, z=1.0):
    """The neighbour instance's surface: a disc sitting on the cut."""
    r = np.sqrt(np.random.RandomState(0).rand(2000)) * radius
    a = np.random.RandomState(1).rand(2000) * 2 * np.pi
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(2000, z)], axis=1)


class PlugSocketTest(unittest.TestCase):
    def test_a_plug_is_positive_and_a_socket_negative(self):
        tube = open_tube()
        loops = cut_loops(tube, neighbour_points(), diag=3.0)
        self.assertEqual(len(loops), 1)                     # the bottom cap is closed: one cut
        self.assertGreater(loops[0]["n"][2], 0.9)           # outward = up, towards the neighbour
        plug = trimesh.creation.cylinder(radius=0.5, height=2.6, sections=48).apply_translation((0, 0, 0.3))
        socket = trimesh.creation.cylinder(radius=0.5, height=1.4, sections=48).apply_translation((0, 0, -0.3))
        self.assertGreater(plug_height(plug, loops[0]), 0.25)    # sticks out 0.3 of a 1.0 diameter
        self.assertLess(plug_height(socket, loops[0]), -0.25)    # sunk 0.3 below the cut
        m = plug_metrics(tube, socket, neighbour_points(), 3.0)
        self.assertEqual(m["sockets"], 1)
        self.assertLess(m["plug_mean"], -0.25)

    def test_a_hole_of_the_original_mesh_is_not_a_cut(self):
        tube = open_tube()
        far = neighbour_points(z=5.0)                       # no neighbour anywhere near the rim
        self.assertEqual(cut_loops(tube, far, diag=3.0), [])

    def test_a_solid_that_regrows_the_neighbour_intrudes(self):
        tube = open_tube()
        # the neighbour: the disc on the cut plus a column of the same radius above it
        a = np.random.RandomState(2).rand(4000) * 2 * np.pi
        column = np.stack([0.5 * np.cos(a), 0.5 * np.sin(a), 1.0 + np.random.RandomState(3).rand(4000)], axis=1)
        others = np.concatenate([neighbour_points(), column])
        plug = trimesh.creation.cylinder(radius=0.5, height=2.2, sections=48).apply_translation((0, 0, 0.1))
        regrown = trimesh.creation.cylinder(radius=0.5, height=3.0, sections=48).apply_translation((0, 0, 0.5))
        self.assertLess(intrusion_share(plug, tube, others, diag=3.0), 0.05)       # a plug stays in the rim band
        self.assertGreater(intrusion_share(regrown, tube, others, diag=3.0), 0.10)  # the column is copied

    def test_the_score_punishes_a_socket_and_spares_a_plug(self):
        base = {"cover": 1.0, "fit_p90": 0.0, "extra_p90": 0.0, "escape": 0.0, "largest_shell": 1.0,
                "intrusion": 0.0, "hollow": 0.0, "hollow_ref": 0.0}
        self.assertAlmostEqual(quality_score({**base, "plug_mean": 0.4}), 1.0)
        self.assertAlmostEqual(quality_score({**base, "plug_mean": -0.19}), 1.0 - 0.29 / 0.4, places=3)
        self.assertAlmostEqual(quality_score(base), 1.0)         # no loop: no opinion


if __name__ == "__main__":
    unittest.main()
