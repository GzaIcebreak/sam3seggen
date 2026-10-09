import unittest

import numpy as np
import trimesh

from xpart_complete import collar_samples, part_surface_condition, rim_vertices


def open_box(extents, offset, drop_face_normal):
    """A box with the faces facing `drop_face_normal` removed: an open part surface."""
    box = trimesh.creation.box(extents=extents)
    keep = np.dot(box.face_normals, drop_face_normal) < 0.5
    box = box.submesh([np.flatnonzero(keep)], append=True)
    box.apply_translation(offset)
    return box


class CollarTest(unittest.TestCase):
    def setUp(self):
        # a "head" open at the bottom sitting on a "torso" open at the top; they touch at z=0
        self.head = open_box((1.0, 1.0, 1.0), (0, 0, 0.5), (0, 0, -1))
        self.torso = open_box((1.4, 1.0, 2.0), (0, 0, -1.0), (0, 0, 1))
        self.far = trimesh.creation.box(extents=(0.5, 0.5, 0.5)).apply_translation((5, 0, 0))

    def test_the_rim_is_the_open_edge(self):
        rim = rim_vertices(self.head)
        self.assertTrue(len(rim) >= 4)
        self.assertTrue(np.allclose(rim[:, 2], 0.0))     # the open bottom edge at z = 0
        self.assertEqual(len(rim_vertices(self.far)), 0)   # a closed piece has no rim

    def test_collar_points_come_from_the_neighbour_near_the_cut_only(self):
        surfaces = [self.head, self.torso, self.far]
        points, normals = collar_samples(0, surfaces, 8000, collar=0.1)
        self.assertTrue(len(points) > 0)
        self.assertEqual(points.shape, normals.shape)
        reach = 0.1 * np.linalg.norm(self.head.bounds[1] - self.head.bounds[0])
        self.assertTrue(np.all(points[:, 2] <= 0.0 + 1e-6))           # on the torso
        self.assertTrue(np.all(points[:, 2] >= -reach - 1e-6))        # just below the cut
        self.assertTrue(np.all(points[:, 0] < 2.0))                    # nothing from the far box
        self.assertLessEqual(len(points), int(0.25 * 8000))

    def test_the_conditioning_keeps_its_point_count_with_and_without_a_collar(self):
        surfaces = [self.head, self.torso]
        plain = part_surface_condition(surfaces, np.zeros(3), 1.0, num_points=4000)
        collared = part_surface_condition(surfaces, np.zeros(3), 1.0, num_points=4000, collar=0.1)
        self.assertEqual(plain.shape, (2, 4000, 7))
        self.assertEqual(collared.shape, (2, 4000, 7))
        # with the collar, some of the head's points lie on the torso (below z = 0)
        self.assertFalse((plain[0, :, 2] < -1e-6).any())
        self.assertTrue((collared[0, :, 2] < -1e-6).any())


if __name__ == "__main__":
    unittest.main()
