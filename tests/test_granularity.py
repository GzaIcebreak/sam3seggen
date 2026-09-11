import unittest

import numpy as np
import trimesh

from data_toolkit.meet_samples import DEFAULT_MIN_FACES
from data_toolkit.unit_vote import DEFAULT_MIN_UNIT_FACES
from segment_parts import DEFAULT_GRANULARITY, GRANULARITY
from xpart_complete import drop_inner_shells, fold_small_pieces


class GranularityTest(unittest.TestCase):
    def test_the_default_preset_is_what_the_libraries_do_on_their_own(self):
        # segment_parts resolves the floors from the preset, the toolkit modules from
        # their own constants. If those drift apart, running a stage on its own quietly
        # splits at a different granularity than running the pipeline.
        self.assertEqual(GRANULARITY[DEFAULT_GRANULARITY],
                         (DEFAULT_MIN_FACES, DEFAULT_MIN_UNIT_FACES))

    def test_the_presets_are_ordered_and_the_unit_floor_never_undoes_the_atom_floor(self):
        floors = [GRANULARITY[name] for name in ("fine", "medium", "coarse")]
        self.assertEqual(floors, sorted(floors))
        for atoms, units in floors:
            self.assertGreaterEqual(units, atoms)


def box_at(centre, size):
    mesh = trimesh.creation.box(extents=[size, size, size])
    mesh.apply_translation(centre)
    return mesh


class FoldSmallPiecesTest(unittest.TestCase):
    """A component too small to generate should ride along, not be thrown away."""

    def test_a_sliver_joins_the_piece_it_is_touching_and_keeps_its_surface(self):
        big = box_at([0.0, 0.0, 0.0], 1.0)
        far = box_at([10.0, 0.0, 0.0], 1.0)
        sliver = box_at([10.6, 0.0, 0.0], 0.05)
        folded = fold_small_pieces([("torso", big), ("leg", far), ("foot", sliver)],
                                   min_area_share=0.01)
        self.assertEqual([name for name, _ in folded], ["torso", "leg"])
        # The old behaviour dropped the sliver, leaving surface in no prompt at all.
        self.assertAlmostEqual(sum(float(m.area) for _, m in folded),
                               big.area + far.area + sliver.area, places=6)
        # And it joined the near piece, not the big one on the other side of the model.
        self.assertGreater(folded[1][1].area, far.area)

    def test_nothing_is_folded_when_every_piece_carries_its_weight(self):
        pieces = [("a", box_at([0.0, 0.0, 0.0], 1.0)), ("b", box_at([3.0, 0.0, 0.0], 1.0))]
        folded = fold_small_pieces(pieces, min_area_share=0.01)
        self.assertEqual([name for name, _ in folded], ["a", "b"])
        self.assertEqual(len(folded[0][1].faces), len(pieces[0][1].faces))

    def test_a_threshold_that_swallows_everything_is_refused(self):
        with self.assertRaises(SystemExit):
            fold_small_pieces([("a", box_at([0.0, 0.0, 0.0], 1.0))], min_area_share=1.1)


class InnerShellTest(unittest.TestCase):
    def test_the_remesh_wall_inside_a_part_is_dropped_rather_than_folded_in(self):
        # Its normals face the other way, so conditioning on both would describe a shape
        # that is inside out in half its points -- unlike a sliver, it adds nothing.
        outer = box_at([0.0, 0.0, 0.0], 1.0)
        wall = box_at([0.0, 0.0, 0.0], 0.9)
        kept = drop_inner_shells([("torso", outer), ("torso", wall)])
        self.assertEqual(len(kept), 1)
        np.testing.assert_allclose(kept[0][1].bounds, outer.bounds)

    def test_a_hand_inside_the_arms_box_is_a_different_part_and_survives(self):
        arm = box_at([0.0, 0.0, 0.0], 1.0)
        hand = box_at([0.0, 0.0, 0.0], 0.5)
        self.assertEqual(len(drop_inner_shells([("arm", arm), ("hand", hand)])), 2)


if __name__ == "__main__":
    unittest.main()
