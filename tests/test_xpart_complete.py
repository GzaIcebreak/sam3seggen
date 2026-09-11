import unittest

import numpy as np

from xpart_complete import source_frame_transform, to_source_frame


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


if __name__ == "__main__":
    unittest.main()
