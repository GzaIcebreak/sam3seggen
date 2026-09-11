import unittest

from merge_parts import (
    DEFAULT_CONCEPT_BANK, DEFAULT_VIEW_AZIMUTHS, DEFAULT_VIEW_ELEVATIONS, masks_name,
)


class MergeDefaultsTest(unittest.TestCase):
    def test_default_cameras_are_the_low_three_quarter_pair(self):
        self.assertEqual(DEFAULT_VIEW_AZIMUTHS, "45,225")
        # High enough to read the 3/4, low enough not to hide the legs and base.
        self.assertEqual(DEFAULT_VIEW_ELEVATIONS, "10")

    def test_mask_cache_changes_when_the_bank_does(self):
        shared = dict(
            prompts=["head", "torso"], unassigned_to="torso",
            threshold=0.4, model="sam3", azimuths="45,225", elevations="10",
        )
        raw = masks_name(**shared, concept_bank="", overlay="raw")
        v3 = masks_name(**shared, concept_bank=DEFAULT_CONCEPT_BANK, overlay="v3")
        self.assertNotEqual(raw, v3)

    def test_mask_cache_changes_when_the_views_are_flat_painted(self):
        # Masks read off a flat-painted render are not the same masks; reusing them
        # across a --flat_paint change would silently hide the switch having any effect.
        shared = dict(
            prompts=["head", "torso"], unassigned_to="torso", threshold=0.4,
            model="sam3", azimuths="45,225", elevations="10",
            concept_bank=DEFAULT_CONCEPT_BANK,
        )
        self.assertNotEqual(masks_name(**shared, flat_paint=False),
                            masks_name(**shared, flat_paint=True))
