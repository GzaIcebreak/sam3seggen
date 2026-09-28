import unittest

import numpy as np

from auto_prompts import GENERIC_WORDS, propose_from_masks


def scene(concept_boxes, views=4, size=64):
    """masks/foreground/scores for concepts drawn as filled boxes (row0,row1,col0,col1)."""
    names = list(concept_boxes)
    foreground = np.zeros((views, size, size), bool)
    foreground[:, 8:56, 8:56] = True                       # the object silhouette
    masks = np.zeros((views, len(names), size, size), bool)
    scores = np.zeros((views, len(names)))
    for k, (name, (box, score, seen)) in enumerate(concept_boxes.items()):
        r0, r1, c0, c1 = box
        for v in range(seen):
            masks[v, k, r0:r1, c0:c1] = True
            scores[v, k] = score
    return masks, foreground, scores, names


class ProposeFromMasksTest(unittest.TestCase):
    def test_whole_object_word_names_the_remainder_and_parts_partition_the_rest(self):
        masks, fg, scores, names = scene({
            "body": ((8, 56, 8, 56), 0.7, 4),           # everything: main, not a part
            "head": ((8, 24, 8, 56), 0.6, 4),
            "leg": ((40, 56, 8, 56), 0.8, 4),
            "ear": ((8, 14, 8, 20), 0.9, 4),            # inside head: swallowed by it
            "boot": ((50, 56, 8, 56), 0.9, 4),          # inside leg: swallowed
            "housing": ((8, 56, 8, 30), 0.5, 4),        # generic word: never
            "legs": ((40, 56, 8, 56), 0.5, 4),          # plural of a present word: never
        })
        out = propose_from_masks(masks, fg, scores, names)
        self.assertEqual(out["main"], "body")
        self.assertEqual(sorted(out["parts"]), ["head", "leg"])
        self.assertGreater(out["covered"], 0.6)

    def test_a_word_seen_from_one_side_only_is_not_trusted(self):
        masks, fg, scores, names = scene({
            "body": ((8, 56, 8, 56), 0.7, 4),
            "wheel": ((40, 56, 8, 56), 0.9, 4),
            "spoiler": ((8, 16, 8, 56), 0.9, 1),        # 1 of 4 views
        })
        out = propose_from_masks(masks, fg, scores, names)
        self.assertEqual(out["parts"], ["wheel"])

    def test_without_a_whole_object_word_there_is_no_main(self):
        masks, fg, scores, names = scene({
            "backrest": ((8, 30, 8, 56), 0.9, 4),
            "seat": ((30, 40, 8, 56), 0.9, 4),
            "leg": ((40, 56, 8, 56), 0.9, 4),
        })
        out = propose_from_masks(masks, fg, scores, names)
        self.assertIsNone(out["main"])
        self.assertEqual(sorted(out["parts"]), ["backrest", "leg", "seat"])

    def test_generic_words_are_not_part_names(self):
        for word in ("housing", "panel", "column", "platform"):
            self.assertIn(word, GENERIC_WORDS)


if __name__ == "__main__":
    unittest.main()
