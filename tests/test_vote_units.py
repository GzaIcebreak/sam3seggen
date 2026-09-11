import unittest
from unittest import mock

import numpy as np

from data_toolkit import unit_vote
from data_toolkit.lift_sam3 import MaskSet


def two_boxes():
    """Two disjoint boxes carrying one atom label, so split_units would find two units."""
    import trimesh

    mesh = trimesh.util.concatenate(
        trimesh.creation.box(), trimesh.creation.box().apply_translation([5, 0, 0]))
    return mesh, np.zeros(len(mesh.faces), dtype=np.int64)


def one_view_masks(resolution, concept="head"):
    """A single view whose whole silhouette is claimed by one concept."""
    covered = np.ones((1, 1, resolution, resolution), dtype=bool)
    return MaskSet(
        masks=covered,
        foreground=covered[:, 0].copy(),
        scores=np.ones((1, 1)),
        concepts=[concept], owners=[concept], views=["v0"],
        part_order=[concept], unassigned_to=None,
    )


class VoteUnitsTest(unittest.TestCase):
    """The vote must name the partition the split exported, not one it recomputed."""

    def setUp(self):
        self.mesh, self.atoms = two_boxes()
        self.resolution = 8
        # Every pixel belongs to face 1, so the raster never needs a GPU here.
        self.raster = np.ones((1, self.resolution, self.resolution), dtype=np.int32)

    def run_vote(self, units):
        with mock.patch.object(unit_vote, "rasterize_face_ids", return_value=self.raster), \
             mock.patch.object(unit_vote, "split_units",
                               side_effect=AssertionError("split_units was recomputed")) as split:
            if units is None:
                split.side_effect = None
                split.return_value = np.zeros(len(self.mesh.faces), dtype=np.int64)
            _, _, out = unit_vote.vote(
                self.mesh, self.atoms, one_view_masks(self.resolution),
                cameras=[None], camera_angle_x=0.7, resolution=self.resolution,
                part_order=["head"], units=units)
        return out, split

    def test_given_units_are_used_verbatim(self):
        units = np.arange(len(self.mesh.faces)) // 6
        out, split = self.run_vote(units)
        np.testing.assert_array_equal(out, units)
        split.assert_not_called()

    def test_units_are_still_computed_when_none_are_given(self):
        out, split = self.run_vote(None)
        split.assert_called_once()
        self.assertEqual(len(np.unique(out)), 1)


if __name__ == "__main__":
    unittest.main()
