import unittest

import numpy as np
import trimesh

from hybrid_complete import hollow_share, quality_score


def hollow_sphere(radius=1.0, wall=0.04):
    outer = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    inner = trimesh.creation.icosphere(subdivisions=4, radius=radius - wall)
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


class HollowShareTest(unittest.TestCase):
    def test_a_filled_sphere_is_not_hollow_and_a_thin_double_wall_is(self):
        filled = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        box = filled.bounds
        self.assertLess(hollow_share(filled, box), 0.05)
        self.assertGreater(hollow_share(hollow_sphere(), box), 0.8)

    def test_a_thin_plate_is_judged_against_its_own_thickness(self):
        # a blade: the box's smallest extent is the thickness itself, so it is not "hollow"
        plate = trimesh.creation.box(extents=(2.0, 0.5, 0.05))
        self.assertLess(hollow_share(plate, plate.bounds), 0.1)

    def test_hollowness_sinks_the_score(self):
        good = {"cover": 1.0, "fit_p90": 0.0, "extra_p90": 0.0, "escape": 0.0,
                "largest_shell": 1.0, "intrusion": 0.0, "hollow": 0.05}
        bowl = dict(good, hollow=0.9)
        self.assertAlmostEqual(quality_score(good), 1.0, places=3)
        self.assertLess(quality_score(bowl), 0.05)
        self.assertAlmostEqual(quality_score(dict(good, hollow=0.15)), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
