import unittest

from sam3_multiview import unseen_concepts


class UnseenConceptsTest(unittest.TestCase):
    def test_a_word_no_view_saw_is_reported(self):
        self.assertEqual(
            unseen_concepts({"head": 2, "armrest": 0}, ["head", "armrest"],
                            ["head", "armrest"]),
            ["armrest"],
        )

    def test_the_catch_all_part_may_have_no_detections(self):
        self.assertEqual(
            unseen_concepts({"seat": 2, "dump": 0}, ["seat", "dump"],
                            ["seat", "dump"], unassigned_to="dump"),
            [],
        )

    def test_every_detected_concept_is_kept(self):
        self.assertEqual(
            unseen_concepts({"blade": 1, "guard": 2}, ["blade", "guard"],
                            ["blade", "guard"]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
