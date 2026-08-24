import json
import os
import tempfile
import unittest

import numpy as np

from data_toolkit.parts_rebake import (
    merge_labels_by_part,
    palette_from_legend,
    reassign_label_islands,
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
