"""Unit tests for data_toolkit/front_view.py -- synthetic silhouettes, no GPU needed."""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.front_view import (
    build_grid,
    parse_vlm_choice,
    rank_views,
    sam3_legend_score,
    silhouette_metrics,
)


def _write_mask(fg, directory, name):
    """Save a boolean mask as an RGBA png (alpha 255 on foreground) and return its path."""
    rgba = np.zeros((*fg.shape, 4), dtype=np.uint8)
    rgba[..., 3] = np.where(fg, 255, 0)
    rgba[..., :3] = 200
    path = os.path.join(directory, name)
    Image.fromarray(rgba).save(path)
    return path


def _rect_mask(shape, x0, x1, y0, y1):
    fg = np.zeros(shape, dtype=bool)
    fg[y0:y1, x0:x1] = True
    return fg


class SilhouetteMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="frontview_")
        # Centred, symmetric, well-covered: the ideal "front" silhouette.
        self.good = _write_mask(_rect_mask((512, 512), 128, 384, 128, 384), self.tmp, "good.png")
        # Same area pushed hard to the left: decent coverage, poor centeredness/symmetry axis.
        self.offcentre = _write_mask(_rect_mask((512, 512), 10, 266, 128, 384), self.tmp, "off.png")
        # Thin sliver: an edge-on view.
        self.sliver = _write_mask(_rect_mask((512, 512), 240, 272, 64, 448), self.tmp, "sliver.png")
        # Asymmetric blob: rectangle plus a lump on one side.
        blob = _rect_mask((512, 512), 128, 384, 160, 352)
        blob[180:240, 384:440] = True
        self.asym = _write_mask(blob, self.tmp, "asym.png")
        # Empty render.
        self.empty = _write_mask(np.zeros((512, 512), dtype=bool), self.tmp, "empty.png")

    def test_empty_render_scores_zero(self):
        metrics = silhouette_metrics(self.empty)
        self.assertEqual(metrics["score"], 0.0)
        self.assertEqual(metrics["fg_ratio"], 0.0)

    def test_symmetric_front_beats_sliver(self):
        good = silhouette_metrics(self.good)
        sliver = silhouette_metrics(self.sliver)
        self.assertGreater(good["symmetry"], 0.95)
        self.assertGreater(good["score"], sliver["score"])

    def test_symmetry_drops_for_asymmetric_blob(self):
        good = silhouette_metrics(self.good)
        asym = silhouette_metrics(self.asym)
        self.assertGreater(good["symmetry"], asym["symmetry"])

    def test_centeredness_drops_for_offcentre(self):
        good = silhouette_metrics(self.good)
        off = silhouette_metrics(self.offcentre)
        self.assertGreater(good["centeredness"], off["centeredness"])

    def test_rank_views_orders_best_first(self):
        ranked = rank_views([self.sliver, self.good, self.empty, self.offcentre])
        self.assertEqual(ranked[0][0], self.good)
        self.assertEqual(ranked[-1][0], self.empty)
        scores = [m["score"] for _, m in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class Sam3LegendScoreTest(unittest.TestCase):
    def test_mean_over_expected_prompts(self):
        legend = [
            {"part": "mushroom", "score": 0.9},
            {"part": "chair", "score": 0.7},
            {"part": "<unassigned>", "score": None},
        ]
        self.assertAlmostEqual(sam3_legend_score(legend, ["mushroom", "chair"]), 0.8)

    def test_missing_prompt_scores_zero(self):
        legend = [{"part": "mushroom", "score": 0.9}]
        self.assertAlmostEqual(sam3_legend_score(legend, ["mushroom", "chair"]), 0.45)

    def test_empty_expected_names(self):
        self.assertEqual(sam3_legend_score([], []), 0.0)


class VlmHelpersTest(unittest.TestCase):
    def test_parse_plain_number(self):
        self.assertEqual(parse_vlm_choice("3", 8), 2)

    def test_parse_number_in_sentence(self):
        self.assertEqual(parse_vlm_choice("The front view is tile 2.", 4), 1)

    def test_parse_rejects_out_of_range(self):
        self.assertIsNone(parse_vlm_choice("9", 4))
        self.assertIsNone(parse_vlm_choice("zero", 4))
        self.assertIsNone(parse_vlm_choice("", 4))
        self.assertIsNone(parse_vlm_choice(None, 4))

    def test_build_grid_layout_and_labels(self):
        tmp = tempfile.mkdtemp(prefix="frontview_grid_")
        paths = [
            _write_mask(_rect_mask((64, 64), 16, 48, 16, 48), tmp, f"v{i}.png")
            for i in range(4)
        ]
        grid, count = build_grid(paths, cell=64)
        self.assertEqual(count, 4)
        self.assertEqual(grid.size, (128, 128))
        # Label band is drawn top-left of each tile as a white box.
        self.assertEqual(grid.getpixel((10, 10)), (255, 255, 255))
        self.assertEqual(grid.getpixel((64 + 10, 10)), (255, 255, 255))
        self.assertEqual(grid.getpixel((10, 64 + 10)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
