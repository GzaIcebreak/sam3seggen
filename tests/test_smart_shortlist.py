import unittest

import numpy as np

from auto_prompts import accept_shortlist, covered_share, word_stats
from smart_prompts import build_shortlist_question, parse_reply


def masks_for(words, size=48, views=2):
    """Foreground square; each word drawn as a horizontal band (or nothing)."""
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


class WordStatsTest(unittest.TestCase):
    def test_found_words_are_eligible_and_absent_words_are_not(self):
        masks, fg, scores, names = masks_for({"head": (4, 12), "tail": None, "body": (12, 44)})
        rows = {r["concept"]: r for r in word_stats(masks, fg, scores, names)}
        self.assertTrue(rows["head"]["eligible"])
        self.assertFalse(rows["tail"]["eligible"])
        self.assertAlmostEqual(rows["head"]["area"], 0.2, places=2)
        self.assertAlmostEqual(rows["body"]["blob"], 0.8, places=2)


class AcceptShortlistTest(unittest.TestCase):
    def test_keeps_found_parts_drops_missing_and_whole_words(self):
        masks, fg, scores, names = masks_for({"head": (4, 12), "tail": None, "body": (12, 44)})
        rows = word_stats(masks, fg, scores, names)
        chosen = {"object": "dog", "main": "body", "parts": ["head", "tail", "body"], "dropped": []}
        proposal, reason = accept_shortlist(chosen, rows)
        self.assertIsNone(reason)
        self.assertEqual(proposal["parts"], ["head"])
        self.assertEqual(proposal["main"], "body")
        self.assertEqual(proposal["kimi"]["not_found"], ["tail"])
        self.assertEqual(proposal["kimi"]["dropped_whole"], ["body"])
        self.assertTrue(proposal["shortlist"])
        self.assertAlmostEqual(covered_share(masks, fg, names, proposal["parts"]), 0.2, places=2)

    def test_nothing_found_or_no_main_falls_back(self):
        masks, fg, scores, names = masks_for({"handle": None, "blade": (4, 44)})
        rows = word_stats(masks, fg, scores, names)
        proposal, reason = accept_shortlist(
            {"object": "sword", "main": "blade", "parts": ["handle"], "dropped": []}, rows)
        self.assertIsNone(proposal)
        self.assertIn("not found", reason)
        proposal, reason = accept_shortlist(
            {"object": "sword", "main": None, "parts": ["blade"], "dropped": []}, rows)
        self.assertIsNone(proposal)
        self.assertIn("main", reason)


class ShortlistQuestionTest(unittest.TestCase):
    def test_the_question_carries_the_vocabulary_and_the_reply_is_filtered_to_it(self):
        vocabulary = ["wheel", "door", "hood", "body"]
        question = build_shortlist_question(vocabulary)
        for word in vocabulary:
            self.assertIn(word, question)
        self.assertIn("refer words from this vocabulary", question)
        reply = '{"object": "car", "main": "body", "parts": ["wheel", "door", "spoiler"]}'
        parsed = parse_reply(reply, vocabulary)
        self.assertEqual(parsed["parts"], ["wheel", "door"])
        self.assertEqual(parsed["dropped"], ["spoiler"])


if __name__ == "__main__":
    unittest.main()
