import unittest

import numpy as np
import trimesh

from data_toolkit.meet_samples import DEFAULT_MIN_FACES
from data_toolkit.unit_vote import DEFAULT_MIN_UNIT_FACES
from segment_parts import DEFAULT_GRANULARITY, GRANULARITY
from xpart_complete import drop_inner_shells, fold_small_pieces, merge_split_fragments


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

    def test_within_part_a_sliver_stays_with_its_own_part(self):
        # Ornaments on a tree: the sliver touches the branch but belongs to the ornament.
        tree = box_at([0.0, 0.0, 0.0], 1.0)
        ornament = box_at([5.0, 0.0, 0.0], 0.3)
        sliver = box_at([0.56, 0.0, 0.0], 0.1)
        pieces = [("tree", tree), ("ornament", ornament), ("ornament", sliver)]
        by_distance = fold_small_pieces(pieces, min_area_share=0.01)
        self.assertGreater(by_distance[0][1].area, tree.area)
        folded = fold_small_pieces(pieces, min_area_share=0.01, fold_within_part=True)
        self.assertEqual([name for name, _ in folded], ["tree", "ornament"])
        self.assertAlmostEqual(folded[0][1].area, tree.area, places=6)
        self.assertGreater(folded[1][1].area, ornament.area)

    def test_a_part_floor_keeps_small_things_as_their_own_prompts(self):
        pieces = [("tree", box_at([0.0, 0.0, 0.0], 1.0)),
                  ("ornament", box_at([3.0, 0.0, 0.0], 0.05))]
        self.assertEqual(len(fold_small_pieces(pieces, min_area_share=0.01)), 1)
        kept = fold_small_pieces(pieces, min_area_share=0.01,
                                 part_min_area_share={"ornament": 0.0001})
        self.assertEqual([name for name, _ in kept], ["tree", "ornament"])


class MergeSplitFragmentsTest(unittest.TestCase):
    """Two halves of a hand a staff-width apart are one hand again."""

    def test_close_small_pieces_of_the_same_part_join_and_others_stay(self):
        body = box_at([0.0, 0.0, 0.0], 1.0)
        half_a = box_at([3.0, 0.0, 0.0], 0.2)
        half_b = box_at([3.22, 0.0, 0.0], 0.2)       # 0.02 gap, ~0.5% of the diagonal
        far = box_at([3.0, 0.0, 2.0], 0.2)           # same name, but nowhere near
        staff = box_at([3.11, 0.0, 0.0], 0.05)       # different part, ignored
        pieces = [("body", body), ("body", half_a), ("body", half_b), ("body", far),
                  ("staff", staff)]
        merged = merge_split_fragments(pieces, gap=0.01)
        self.assertEqual(sorted(name for name, _ in merged), ["body", "body", "body", "staff"])
        areas = sorted(round(float(m.area), 6) for _, m in merged)
        self.assertIn(round(float(half_a.area + half_b.area), 6), areas)

    def test_two_big_pieces_never_join_and_zero_gap_is_off(self):
        head = box_at([0.0, 0.0, 0.0], 1.0)
        legs = box_at([1.01, 0.0, 0.0], 1.0)
        pieces = [("body", head), ("body", legs)]
        self.assertEqual(len(merge_split_fragments(pieces, gap=0.05)), 2)
        # Next to a big piece the two are small; alone each would be half the surface.
        small = [("body", head), ("body", box_at([5.0, 0.0, 0.0], 0.2)),
                 ("body", box_at([5.21, 0.0, 0.0], 0.2))]
        self.assertEqual(len(merge_split_fragments(small, gap=0.0)), 3)
        self.assertEqual(len(merge_split_fragments(small, gap=0.1)), 2)


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

    def test_a_smaller_box_outside_the_bigger_one_is_kept(self):
        big = box_at([0.0, 0.0, 0.0], 1.0)
        beside = box_at([2.0, 0.0, 0.0], 0.5)
        self.assertEqual(len(drop_inner_shells([("torso", big), ("torso", beside)])), 2)


if __name__ == "__main__":
    unittest.main()
