import unittest

import numpy as np

from auto_prompts import (
    accept_shortlist, object_noun, pick_variants, resolve_overlaps, singular, variant_phrases,
    word_stats,
)
from smart_prompts import build_shortlist_question, parse_reply


def masks_for(words, size=48, views=2):
    """Foreground square; each word drawn as a band (row range) or nothing."""
    fg = np.zeros((views, size, size), bool)
    fg[:, 4:44, 4:44] = True
    masks = np.zeros((views, len(words), size, size), bool)
    scores = np.zeros((views, len(words)))
    for i, (name, band) in enumerate(words.items()):
        if band is None:
            continue
        r0, r1 = band
        masks[:, i, r0:r1, 4:44] = True
        scores[:, i] = 0.8
    return masks, fg, scores, list(words)


class SingularAndNounTest(unittest.TestCase):
    def test_plurals_become_singular_and_irregulars_are_known(self):
        self.assertEqual(singular("arms"), "arm")
        self.assertEqual(singular("robot legs"), "robot leg")
        self.assertEqual(singular("feet"), "foot")
        self.assertEqual(singular("leaves"), "leaf")
        self.assertEqual(singular("glasses"), "glasses")   # eyewear, not a glass
        self.assertEqual(singular("boxes"), "box")
        self.assertEqual(singular("body"), "body")
        self.assertEqual(singular("bus"), "bus")

    def test_the_object_noun_drops_figurine_style_suffixes(self):
        self.assertEqual(object_noun("dog figurine"), "dog")
        self.assertEqual(object_noun("robot"), "robot")
        self.assertEqual(object_noun("lego minifigure"), "minifigure")
        self.assertIsNone(object_noun("a very long object description here"))


class VariantTest(unittest.TestCase):
    def test_variants_are_the_word_its_alternatives_singular_and_object_phrase(self):
        chosen = {"object": "dog figurine", "main": "body", "parts": ["arms", "head"],
                  "alternatives": {"head": ["face"]}}
        phrases = variant_phrases(chosen)
        self.assertEqual(phrases["arms"], ["arms", "arm", "dog arms"])
        self.assertEqual(phrases["head"], ["head", "face", "dog head"])
        self.assertEqual(phrases["body"], ["body", "dog body"])

    def test_the_best_detected_variant_segments_the_part(self):
        # SAM3 ignores "arms" and only saw "head" from one of four sides; "arm" and "dog head" work
        masks, fg, scores, names = masks_for({
            "arms": None, "arm": (4, 12), "dog arms": None,
            "head": (12, 20), "face": None, "dog head": (12, 20), "body": (20, 40), "dog body": None},
            views=4)
        for view in (1, 2, 3):
            scores[view, names.index("head")] = 0.0
            masks[view, names.index("head")] = False
        rows = word_stats(masks, fg, scores, names, min_views=0.25)
        chosen = {"object": "dog figurine", "main": "body", "parts": ["arms", "head"],
                  "alternatives": {"head": ["face"]}}
        picked = pick_variants(chosen, rows)
        self.assertEqual(picked["arms"], "arm")
        self.assertEqual(picked["head"], "dog head")      # seen from all sides, head from one
        self.assertEqual(picked["body"], "body")          # its own word, seen everywhere
        proposal, reason = accept_shortlist(chosen, rows, masks=masks, foreground=fg, concepts=names)
        self.assertIsNone(reason)
        self.assertEqual(proposal["parts"], ["arms=arm", "head=dog head"])
        self.assertEqual(proposal["part_names"], ["arms", "head"])
        self.assertEqual(proposal["kimi"]["variants_used"], {"arms": "arm", "head": "dog head"})
        self.assertNotIn("not_found", proposal["kimi"])

    def test_the_own_word_wins_over_a_bigger_variant_when_sam3_knows_it(self):
        masks, fg, scores, names = masks_for({"hand": (36, 40), "glove": (30, 40), "body": (4, 44)})
        rows = word_stats(masks, fg, scores, names, min_views=0.25)
        chosen = {"object": "figure", "main": "body", "parts": ["hand"], "alternatives": {"hand": ["glove"]}}
        self.assertEqual(pick_variants(chosen, rows)["hand"], "hand")


class OverlapTest(unittest.TestCase):
    def test_a_later_word_with_the_same_mask_is_folded_into_the_earlier_one(self):
        masks, fg, scores, names = masks_for({
            "legs": (24, 44), "jeans": (25, 44), "shoes": (40, 44), "body": (4, 44)})
        rows = word_stats(masks, fg, scores, names, min_views=0.25)
        chosen = {"object": "human figure", "main": "body", "parts": ["legs", "jeans", "shoes"]}
        kept, dropped = resolve_overlaps(["legs", "jeans", "shoes"], {}, masks, fg, names)
        self.assertEqual(kept, ["legs", "shoes"])
        self.assertEqual(dropped, [("jeans", "legs")])
        proposal, reason = accept_shortlist(chosen, rows, masks=masks, foreground=fg, concepts=names)
        self.assertIsNone(reason)
        self.assertEqual(proposal["part_names"], ["legs", "shoes"])
        self.assertEqual(proposal["kimi"]["dropped_overlap"], {"jeans": "legs"})

    def test_without_masks_nothing_is_folded(self):
        masks, fg, scores, names = masks_for({"legs": (24, 44), "jeans": (25, 44), "body": (4, 44)})
        rows = word_stats(masks, fg, scores, names, min_views=0.25)
        chosen = {"object": "figure", "main": "body", "parts": ["legs", "jeans"]}
        proposal, _ = accept_shortlist(chosen, rows)
        self.assertEqual(proposal["part_names"], ["legs", "jeans"])


class QuestionAndReplyTest(unittest.TestCase):
    def test_the_question_asks_for_singulars_and_alternatives_and_the_reply_keeps_both(self):
        question = build_shortlist_question(["arm", "head", "body"])
        self.assertIn("alternatives", question)
        self.assertIn("SINGULAR", question)
        self.assertNotIn("MUST be taken verbatim", question)
        reply = ('{"object": "dog figurine", "main": "body", "parts": ["head", "overalls", "panel"], '
                 '"alternatives": {"head": ["dog head", "face"], "overalls": ["dungarees"]}}')
        parsed = parse_reply(reply, ["head", "body"], generic={"panel"}, keep_unknown=True)
        self.assertEqual(parsed["parts"], ["head", "overalls"])     # overalls: not in the bank, kept
        self.assertEqual(parsed["dropped"], ["panel"])              # generic shape word: still out
        self.assertEqual(parsed["alternatives"], {"head": ["head", "dog head", "face"],
                                                  "overalls": ["overalls", "dungarees"]})
        strict = parse_reply(reply, ["head", "body"], generic={"panel"})
        self.assertEqual(strict["parts"], ["head"])                 # the review path still filters


if __name__ == "__main__":
    unittest.main()
