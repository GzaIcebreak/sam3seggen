import json
import os
import tempfile
import unittest

import numpy as np

from data_toolkit.parts_rebake import (
    SPLIT_MODES,
    _split_labels,
    absorb_small_fragments,
    merge_labels_by_part,
    palette_from_legend,
    reassign_label_islands,
    weld_pieces_to_map,
)


class PartsRebakeGroupingTest(unittest.TestCase):
    def test_palette_reads_part_field_from_legend(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "legend.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump([
                    {"prompt": "head", "part": "body", "color": [10, 20, 30]},
                    {"prompt": "hand", "part": "body", "color": [40, 50, 60]},
                    {"prompt": "staff", "part": "staff", "color": [70, 80, 90]},
                ], file)

            colors, concepts, parts = palette_from_legend(path)

            np.testing.assert_array_equal(colors, [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
            self.assertEqual(concepts, ["head", "hand", "staff"])
            self.assertEqual(parts, ["body", "body", "staff"])

    def test_island_then_merge_keeps_small_disconnected_body_concept(self):
        # Faces: head head head head | hand | staff staff
        # Hand is < 20% of head. If those two body concepts were one label
        # already, island cleanup would give the hand to staff. At concept
        # level the hand is its own largest patch, so it survives, then merge.
        labels = np.array([0, 0, 0, 0, 1, 2, 2], dtype=np.int32)
        areas = np.ones(len(labels), dtype=np.float64)
        adjacency = np.array([
            [0, 1], [1, 2], [2, 3],
            [3, 4],
            [4, 5], [5, 6],
        ], dtype=np.int64)

        after_island = reassign_label_islands(adjacency, labels, areas, min_island_ratio=0.2)
        self.assertEqual(int((after_island == 1).sum()), 1)

        merged, names = merge_labels_by_part(after_island, ["body", "body", "staff"])
        self.assertEqual(names, ["body", "staff"])
        self.assertEqual(int((merged == 0).sum()), 5)
        self.assertEqual(int((merged == 1).sum()), 2)

    def test_stain_split_only_moves_fragments_below_the_face_floor(self):
        # A chain: torso x4 | foot x1 (a speck) | leg x3 | foot x3 (real, detached from
        # the other foot). Island cleanup would fold the 3-face foot into leg because
        # it is small next to leg; the stain rule keeps it and only absorbs the speck.
        self.assertEqual(SPLIT_MODES, ("stain", "weld", "refine"))
        labels = np.array([0, 0, 0, 0, 2, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        adjacency = np.array([[i, i + 1] for i in range(len(labels) - 1)], dtype=np.int64)

        out = absorb_small_fragments(adjacency, labels, min_faces=2)

        self.assertEqual(out.tolist(), [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(labels.tolist(), [0, 0, 0, 0, 2, 1, 1, 1, 2, 2, 2])

        untouched = absorb_small_fragments(adjacency, labels, min_faces=0)
        self.assertEqual(untouched.tolist(), labels.tolist())

    def test_weld_renames_whole_pieces_and_never_cuts_inside_one(self):
        # Two torso pieces (label 0) separated by a leg run (label 1). The map paints
        # the second torso piece as leg on most of its visible faces -> whole piece
        # becomes leg. The first piece has a split vote -> stays. A hidden piece stays.
        labels = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2], dtype=np.int32)
        adjacency = np.array([[i, i + 1] for i in range(len(labels) - 1)], dtype=np.int64)
        seen = np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

        out, moved_faces, moved_pieces = weld_pieces_to_map(
            adjacency, labels, seen, min_visible_share=0.5, min_visible_faces=2, min_agreement=0.6)

        self.assertEqual(out.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        self.assertEqual((moved_faces, moved_pieces), (4, 1))

        # Faces that face away from the camera cannot outvote SegviGen: with only one
        # visible face the piece is below the visibility floor and keeps its colour.
        seen_sparse = np.array([-1] * 6 + [1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
        same, moved_faces, _ = weld_pieces_to_map(
            adjacency, labels, seen_sparse, min_visible_share=0.5, min_visible_faces=2)
        self.assertEqual(same.tolist(), labels.tolist())
        self.assertEqual(moved_faces, 0)
