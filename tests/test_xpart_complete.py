import unittest

import numpy as np
import trimesh

from xpart_complete import (
    box_escape, part_surface_condition, source_frame_transform, to_source_frame,
    xpart_normalization,
)


def bounds(low, high):
    return np.array([low, high], dtype=np.float64)


class SourceFrameTest(unittest.TestCase):
    """The boxes describe the split's output but prompt X-Part about the source mesh."""

    def test_a_model_standing_on_the_ground_is_lifted_back(self):
        # Mickey: the source stands on y=0, the split centres him in a unit cube. Measured
        # from the real pair -- the shift was 0.4833 and the boxes missed half the model.
        parts = bounds([-0.4132, -0.5019, -0.3770], [0.4131, 0.5016, 0.3771])
        source = bounds([-0.4013, 0.0, -0.3686], [0.3936, 0.9662, 0.3564])
        scale, parts_centre, source_centre = source_frame_transform(parts, source)
        self.assertAlmostEqual(scale, 0.9621, places=3)
        # One uniform scale cannot match three slightly different axis ratios exactly;
        # what matters is that the half-unit shift is gone, not the last decimal.
        moved = to_source_frame(parts, scale, parts_centre, source_centre)
        np.testing.assert_allclose(moved, source, atol=1e-3)

    def test_an_already_centred_model_is_left_where_it_is(self):
        # The robot, which is why this bug stayed invisible for so long.
        parts = bounds([-0.3126, -0.5020, -0.1854], [0.3128, 0.5013, 0.1854])
        source = bounds([-0.3112, -0.5, -0.1837], [0.3112, 0.5, 0.1837])
        scale, parts_centre, source_centre = source_frame_transform(parts, source)
        self.assertLess(float(np.abs(source_centre - parts_centre).max()), 1e-3)
        self.assertAlmostEqual(scale, 0.9942, places=3)

    def test_a_part_set_that_misses_the_model_is_refused(self):
        # Parts covering only the lower half: the per-axis scales disagree, and fitting
        # them would place every box confidently in the wrong spot.
        parts = bounds([-0.5, -0.5, -0.5], [0.5, 0.0, 0.5])
        source = bounds([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        with self.assertRaises(SystemExit):
            source_frame_transform(parts, source)

    def test_a_flat_part_set_is_refused_rather_than_divided_by_zero(self):
        parts = bounds([-0.5, 0.0, -0.5], [0.5, 0.0, 0.5])
        source = bounds([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        with self.assertRaises(SystemExit):
            source_frame_transform(parts, source)


class ConditionTest(unittest.TestCase):
    """What X-Part is told each part looks like."""

    def test_normalization_puts_the_longest_axis_inside_the_unit_ball(self):
        # X-Part's normalize_mesh divides the half-extent by 0.8, so the widest axis of
        # any model lands at +-0.8. Our points have to be scaled the same way or the
        # conditioner sees them at a size it was never trained on.
        centre, scale = xpart_normalization(np.array([[-1.0, 0.0, -0.5], [3.0, 1.0, 0.5]]))
        np.testing.assert_allclose(centre, [1.0, 0.5, 0.0])
        corners = (np.array([[-1.0, 0.0, -0.5], [3.0, 1.0, 0.5]]) - centre) / scale
        self.assertAlmostEqual(float(np.abs(corners).max()), 0.8, places=6)

    def test_a_part_is_described_by_its_own_faces_and_not_its_box(self):
        # The failure this replaces: a box also contains whatever else passes through it,
        # so X-Part was told the robot's torso included the tops of the legs. Here the
        # small cube sits entirely inside the big one's bounding box; conditioning on the
        # big cube must not mention it.
        shell = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        inside = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        condition = part_surface_condition([shell, inside], np.zeros(3), 1.0,
                                           num_points=2048)
        self.assertEqual(condition.shape, (2, 2048, 7))
        # Every point of a cube's surface is flush against one of its six faces.
        np.testing.assert_allclose(np.abs(condition[0, :, :3]).max(axis=1), 0.5, atol=1e-6)
        np.testing.assert_allclose(np.abs(condition[1, :, :3]).max(axis=1), 0.1, atol=1e-6)

    def test_the_seventh_channel_is_the_sharp_edge_flag_x_part_leaves_empty(self):
        condition = part_surface_condition(
            [trimesh.creation.box(extents=[1.0, 1.0, 1.0])], np.zeros(3), 1.0,
            num_points=512)
        np.testing.assert_array_equal(condition[:, :, 6], 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(condition[0, :, 3:6], axis=1), 1.0, atol=1e-6)


class BoxEscapeTest(unittest.TestCase):
    def test_a_solid_inside_its_box_has_not_escaped(self):
        box = bounds([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        self.assertEqual(box_escape(bounds([-0.9, -0.9, -0.9], [0.9, 0.9, 0.9]), box), 0.0)

    def test_one_runaway_axis_is_not_averaged_away_by_two_good_ones(self):
        box = bounds([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        # Exactly right in x and z, twice the box's width too tall in y.
        self.assertAlmostEqual(box_escape(bounds([0.0, 0.0, 0.0], [2.0, 6.0, 2.0]), box), 2.0)

    def test_overshoot_at_both_ends_of_an_axis_adds_up(self):
        box = bounds([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(box_escape(bounds([-1.0, 0.0, 0.0], [3.0, 2.0, 2.0]), box), 1.0)


if __name__ == "__main__":
    unittest.main()
