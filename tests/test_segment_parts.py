import unittest

import numpy as np

from data_toolkit.meet_samples import meet_labels
from data_toolkit.unit_vote import assign_units, score_units, split_units, tally_votes
from segment_parts import sample_azimuths


def grid_adjacency(count):
    """A path graph over `count` faces: every face touches the next one."""
    return np.stack([np.arange(count - 1), np.arange(1, count)], axis=1)


class SampleAzimuthsTest(unittest.TestCase):
    def test_base_view_comes_first_and_jitter_alternates(self):
        self.assertEqual(sample_azimuths(5, 0.0, 30.0), [0.0, 30.0, -30.0, 15.0, -15.0])

    def test_jitter_is_relative_to_the_base_azimuth(self):
        self.assertEqual(sample_azimuths(3, 135.0, 30.0), [135.0, 165.0, 105.0])

    def test_zero_jitter_reruns_the_same_view(self):
        self.assertEqual(sample_azimuths(3, 10.0, 0.0), [10.0, 10.0, 10.0])


class MeetLabelsTest(unittest.TestCase):
    def test_a_cut_any_sample_drew_survives_the_meet(self):
        # sample A splits the strip in half, sample B splits it in thirds
        a = np.array([0] * 300 + [1] * 300)
        b = np.array([0] * 200 + [1] * 200 + [2] * 200)
        atoms, _ = meet_labels(np.stack([a, b], axis=1), grid_adjacency(600), min_faces=50)
        self.assertEqual(len(np.unique(atoms)), 4)

    def test_slivers_are_absorbed_rather_than_kept_as_atoms(self):
        a = np.zeros(600, dtype=np.int64)
        b = a.copy()
        b[299:301] = 1  # two faces where the samples disagree
        atoms, raw = meet_labels(np.stack([a, b], axis=1), grid_adjacency(600), min_faces=50)
        self.assertEqual(raw, 2)
        self.assertEqual(len(np.unique(atoms)), 1)

    def test_one_label_in_two_places_becomes_two_atoms(self):
        labels = np.array([0] * 200 + [1] * 200 + [0] * 200)
        atoms, _ = meet_labels(labels[:, None], grid_adjacency(600), min_faces=50)
        self.assertEqual(len(np.unique(atoms)), 3)


class UnitTest(unittest.TestCase):
    def test_one_atom_in_two_pieces_becomes_two_units(self):
        import trimesh

        # two disjoint boxes carrying the same atom label
        mesh = trimesh.util.concatenate(
            trimesh.creation.box(), trimesh.creation.box().apply_translation([5, 0, 0]))
        atoms = np.zeros(len(mesh.faces), dtype=np.int64)
        adjacency = mesh.face_adjacency
        units = split_units(mesh, atoms, adjacency, min_unit_faces=2)
        self.assertEqual(len(np.unique(units)), 2)


class ScoreUnitsTest(unittest.TestCase):
    """Shapes are [view, unit, concept] / [view, unit] / [view, concept]."""

    def setUp(self):
        # one view, one unit; 'leg' covers all of it but is huge, 'foot' covers most of
        # it and fits
        self.owners = ["leg", "foot"]
        self.inter = np.array([[[100.0, 80.0]]])
        self.unit_pixels = np.array([[100.0]])
        self.mask_pixels = np.array([[1000.0, 90.0]])

    def assign(self, inter, unassigned_to, mask_pixels=None):
        names, recall, iou = score_units(inter, self.unit_pixels,
                                         self.mask_pixels if mask_pixels is None else mask_pixels,
                                         self.owners)
        votes, weight = tally_votes(recall, iou, self.unit_pixels, min_recall=0.5,
                                    min_visible_pixels=1)
        seen = (self.unit_pixels >= 1).sum(axis=0)
        assignment, _ = assign_units(None, np.zeros(1, dtype=np.int64), names, votes, weight,
                                     seen, unassigned_to)
        return names, recall, assignment

    def test_the_specific_mask_wins_over_the_containing_one(self):
        names, recall, assignment = self.assign(self.inter, None)
        self.assertGreater(recall[0, 0, names.index("leg")], recall[0, 0, names.index("foot")])
        self.assertEqual(assignment, ["foot"])

    def test_a_unit_no_mask_covers_goes_to_the_catch_all(self):
        _, _, assignment = self.assign(np.array([[[10.0, 5.0]]]), "torso")
        self.assertEqual(assignment, ["torso"])

    def test_the_majority_of_views_wins_over_one_close_up(self):
        # view 0 sees the unit large and fused into 'leg'; views 1 and 2 see it small but
        # separate. Pooling pixels would hand it to view 0.
        owners = ["leg", "foot"]
        inter = np.array([[[900.0, 0.0]], [[0.0, 100.0]], [[0.0, 100.0]]])
        unit_pixels = np.array([[900.0], [100.0], [100.0]])
        mask_pixels = np.array([[2000.0, 0.0], [0.0, 110.0], [0.0, 110.0]])
        names, recall, iou = score_units(inter, unit_pixels, mask_pixels, owners)
        votes, weight = tally_votes(recall, iou, unit_pixels, min_recall=0.5,
                                    min_visible_pixels=1)
        assignment, _ = assign_units(None, np.zeros(1, dtype=np.int64), names, votes, weight,
                                     (unit_pixels >= 1).sum(axis=0), None)
        self.assertEqual(assignment, ["foot"])

    def test_concepts_of_one_part_do_not_add_up(self):
        # 'hand' asked for two overlapping concepts; the best one counts, not their sum
        names, _, _ = score_units(np.array([[[60.0, 55.0]]]), self.unit_pixels,
                                  np.array([[70.0, 65.0]]), ["hand", "hand"])
        self.assertEqual(names, ["hand"])


if __name__ == "__main__":
    unittest.main()
