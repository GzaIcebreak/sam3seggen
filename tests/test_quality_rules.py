import unittest

import numpy as np
import trimesh

from auto_prompts import accept_shortlist, resolve_overlaps, word_stats
from hybrid_complete import hollow_share, quality_score, solid_metrics
from smart_prompts import build_shortlist_question


def masks_for(words, size=48, views=2):
    fg = np.zeros((views, size, size), bool)
    fg[:, 4:44, 4:44] = True
    masks = np.zeros((views, len(words), size, size), bool)
    scores = np.zeros((views, len(words)))
    for i, (name, box) in enumerate(words.items()):
        if box is None:
            continue
        r0, r1, c0, c1 = box
        masks[:, i, r0:r1, c0:c1] = True
        scores[:, i] = 0.8
    return masks, fg, scores, list(words)


class ContainmentTest(unittest.TestCase):
    def test_a_word_inside_another_folds_into_it_whatever_the_order(self):
        # cushion sits entirely inside the ear cup; the ear cup is listed AFTER it
        masks, fg, scores, names = masks_for({
            "cushion": (20, 30, 10, 20), "ear cup": (16, 34, 6, 24), "headband": (4, 10, 4, 44), "body": (4, 44, 4, 44)})
        kept, dropped = resolve_overlaps(["cushion", "ear cup", "headband"], {}, masks, fg, names)
        self.assertEqual(kept, ["ear cup", "headband"])
        self.assertEqual(dropped, [("cushion", "ear cup")])
        rows = word_stats(masks, fg, scores, names, min_views=0.25)
        chosen = {"object": "headphones", "main": "body", "parts": ["cushion", "ear cup", "headband"]}
        proposal, reason = accept_shortlist(chosen, rows, masks=masks, foreground=fg, concepts=names)
        self.assertIsNone(reason)
        self.assertEqual(proposal["part_names"], ["ear cup", "headband"])
        self.assertEqual(proposal["kimi"]["dropped_overlap"], {"cushion": "ear cup"})

    def test_a_word_only_partly_inside_stays(self):
        masks, fg, scores, names = masks_for({"paw": (34, 44, 10, 20), "leg": (20, 36, 10, 20), "body": (4, 44, 4, 44)})
        kept, dropped = resolve_overlaps(["leg", "paw"], {}, masks, fg, names)
        self.assertEqual(kept, ["leg", "paw"])


class HollowRefTest(unittest.TestCase):
    def test_a_part_that_is_a_shell_is_not_punished_for_being_one(self):
        # a thin ring: one connected shell whose wall (0.03) is thinner than the inward step
        shell = trimesh.creation.annulus(r_min=0.97, r_max=1.0, height=1.0)
        solid = shell.copy()
        metrics = solid_metrics(shell, solid, shell.bounds, samples=2000)
        self.assertGreater(metrics["hollow"], 0.5)
        self.assertGreater(metrics["hollow_ref"], 0.5)
        self.assertGreater(quality_score(metrics), 0.8)           # hollow_ref cancels the penalty
        filled_ref = trimesh.creation.cylinder(radius=1.0, height=1.0)
        metrics2 = solid_metrics(filled_ref, solid, shell.bounds, samples=2000)
        self.assertLess(metrics2["hollow_ref"], 0.1)
        self.assertLess(quality_score(metrics2), 0.3)             # a filled part came back as a shell


class QuestionTest(unittest.TestCase):
    def test_the_question_asks_for_major_parts_only(self):
        question = build_shortlist_question(["arm", "head"])
        self.assertIn("MAJOR parts", question)
        self.assertIn("NOT accessories", question)
        self.assertIn("1 to 6", question)


if __name__ == "__main__":
    unittest.main()
