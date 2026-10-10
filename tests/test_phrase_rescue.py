import os
import unittest

from phrase_rescue import (
    choose_phrase, phrase_stats, rescue_enabled, rescue_phrases, weak_concepts)


def cand(phrase, tier, seen, score, area=0.2, views=4):
    return {"phrase": phrase, "tier": tier, "seen": seen, "score": score, "area": area, "views": views}


class RescuePhrasesTest(unittest.TestCase):
    def test_qualified_phrases_first_then_synonyms(self):
        phrases = rescue_phrases("head")
        self.assertEqual(phrases[0], ("animal head", 0))
        self.assertIn(("character head", 0), phrases)
        self.assertEqual(phrases[-1], ("face", 1))

    def test_a_plural_tries_its_singular_first(self):
        self.assertEqual(rescue_phrases("arms")[0], ("arm", 0))
        self.assertIn(("robot arm", 0), rescue_phrases("arms"))

    def test_phrases_already_prompted_are_not_measured_again(self):
        phrases = [p for p, _ in rescue_phrases("torso", taken=["body", "head"])]
        self.assertNotIn("body", phrases)
        self.assertIn("chest", phrases)


class WeakTest(unittest.TestCase):
    def test_unseen_and_borderline_plain_words_are_weak(self):
        raw = {"head": [], "torso": [(0.6, 0.3)] * 4, "tail": [(0.49, 0.05)] * 3}
        weak = weak_concepts(["head", "torso", "tail"], ["head", "torso", "tail"], raw, 4)
        self.assertEqual(weak, ["head", "tail"])

    def test_explicit_phrases_are_left_alone_but_the_catch_all_is_not(self):
        raw = {"dog head": [], "head": []}
        self.assertEqual(weak_concepts(["dog head"], ["head"], raw, 4), [])
        # the puppy run: the catch-all was `head` and SAM3 saw it in 3 of 8 views
        self.assertEqual(weak_concepts(["head", "torso"], ["head", "torso"],
                                       {"head": [(0.5, 0.2)] * 3, "torso": [(0.6, 0.3)] * 8}, 8,
                                       unassigned_to="head"), ["head"])

    def test_stats_use_the_median(self):
        stats = phrase_stats([(0.9, 0.1), (0.5, 0.3), (0.7, 0.2)], 4)
        self.assertEqual((stats["seen"], stats["score"], stats["area"]), (3, 0.7, 0.2))


class ChooseTest(unittest.TestCase):
    def test_the_puppy_head(self):
        own = phrase_stats([], 4)
        pick = choose_phrase(own, [cand("animal head", 0, 4, 0.93), cand("cartoon head", 0, 2, 0.6),
                                   cand("face", 1, 4, 0.97, area=0.1)], 4)
        self.assertEqual(pick["phrase"], "animal head")   # the same word beats a synonym

    def test_a_near_tie_goes_to_the_earlier_phrase(self):
        own = phrase_stats([], 4)
        pick = choose_phrase(own, [cand("animal head", 0, 4, 0.96), cand("toy head", 0, 4, 0.97)], 4)
        self.assertEqual(pick["phrase"], "animal head")
        pick = choose_phrase(own, [cand("animal head", 0, 4, 0.70), cand("toy head", 0, 4, 0.97)], 4)
        self.assertEqual(pick["phrase"], "toy head")

    def test_a_synonym_only_when_no_qualified_phrase_works(self):
        own = phrase_stats([], 4)
        pick = choose_phrase(own, [cand("animal head", 0, 1, 0.9), cand("face", 1, 4, 0.8)], 4)
        self.assertEqual(pick["phrase"], "face")

    def test_the_whole_object_is_not_a_part(self):
        own = phrase_stats([], 4)
        self.assertIsNone(choose_phrase(own, [cand("animal body", 0, 4, 0.9, area=0.95)], 4))

    def test_the_word_stays_unless_clearly_beaten(self):
        own = phrase_stats([(0.45, 0.2)] * 4, 4)
        self.assertIsNone(choose_phrase(own, [cand("animal head", 0, 4, 0.55)], 4))
        self.assertEqual(choose_phrase(own, [cand("animal head", 0, 4, 0.9)], 4)["phrase"], "animal head")

    def test_the_switch(self):
        os.environ["SEGVIGEN_PHRASE_RESCUE"] = "0"
        try:
            self.assertFalse(rescue_enabled())
        finally:
            del os.environ["SEGVIGEN_PHRASE_RESCUE"]
        self.assertTrue(rescue_enabled())
        self.assertFalse(rescue_enabled(disabled=True))


if __name__ == "__main__":
    unittest.main()
