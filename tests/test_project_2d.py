import unittest

import numpy as np

from data_toolkit.project_2d import (
    camera_from_transforms,
    label_visible_faces,
    look_at_axes,
    project_points,
    segvigen_to_render_frame,
)


class Project2DTest(unittest.TestCase):
    def test_visible_faces_take_map_colour_not_neighbour_colour(self):
        # Two quads side by side in front of a camera at +Y looking at origin with Z up.
        # From there the viewer's right is -X, so the +X quads are on the left of the
        # image: red on the map. The -X quads are on the right: blue. SegviGen-style
        # starting labels have every quad blue.
        centroids = np.array([
            [0.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
        ], dtype=np.float64)
        palette = np.array([[220, 40, 40], [40, 90, 230]], dtype=np.float64)
        labels = np.array([1, 1, 1, 1], dtype=np.int64)
        canvas = np.full((32, 32, 3), 255, dtype=np.uint8)
        canvas[:, :16] = (220, 40, 40)
        canvas[:, 16:] = (40, 90, 230)
        cam_pos = np.array([0.0, 2.0, 0.0])
        axes = look_at_axes(cam_pos, np.zeros(3), np.array([0.0, 0.0, 1.0]))

        updated = label_visible_faces(
            centroids, labels, canvas, palette, cam_pos, axes, focal=40.0, principal=16.0,
        )

        self.assertEqual(int(updated[0]), 0)
        self.assertEqual(int(updated[1]), 0)
        self.assertEqual(int(updated[2]), 1)
        self.assertEqual(int(updated[3]), 1)

    def test_projection_puts_up_at_the_top_and_hides_faces_behind_the_camera(self):
        cam_pos = np.array([0.0, 2.0, 0.0])
        axes = look_at_axes(cam_pos, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        u, v, depth = project_points(
            np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5], [0.0, 3.0, 0.0]]), cam_pos, axes, 40.0, 16.0)
        self.assertLess(v[0], 16.0)
        self.assertGreater(v[1], 16.0)
        self.assertAlmostEqual(u[0], 16.0)
        self.assertGreater(depth[0], 0.0)
        self.assertLess(depth[2], 0.0)

    def test_segvigen_frame_is_blender_frame_turned_about_x(self):
        np.testing.assert_allclose(
            segvigen_to_render_frame(np.array([[1.0, 2.0, 3.0]])), [[1.0, -2.0, -3.0]])

    def test_camera_from_transforms_orbits_location_around_z(self):
        entry = {
            "camera_angle_x": 0.6981317007977318,
            "transform_matrix": [
                [1, 0, 0, 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        }
        cam_pos, _axes, _focal, _principal = camera_from_transforms(entry, azimuth=90, resolution=512)
        np.testing.assert_allclose(cam_pos, [0.0, 2.0, 0.0], atol=1e-6)
