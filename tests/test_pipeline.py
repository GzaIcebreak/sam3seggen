"""The CLI, HTTP API and library call must share one set of switches and defaults."""
import argparse
import inspect
import unittest

from data_toolkit.meet_samples import DEFAULT_COLOR_TOL as MEET_COLOR_TOL
from data_toolkit.meet_samples import DEFAULT_MIN_FACES
from data_toolkit.unit_vote import DEFAULT_MIN_RECALL as VOTE_MIN_RECALL
from data_toolkit.unit_vote import DEFAULT_MIN_UNIT_FACES
from pipeline import (
    DEFAULT_COLOR_TOL, DEFAULT_COMPLETE, DEFAULT_CONDITION, DEFAULT_FLAT_PAINT,
    DEFAULT_GRANULARITY, DEFAULT_MERGE, DEFAULT_MIN_AREA_SHARE, DEFAULT_MIN_RECALL,
    DEFAULT_REDRAWS, DEFAULT_SAM3_THRESHOLD, DEFAULT_SAMPLES, GRANULARITY,
    PipelineOptions, add_cli_arguments, check_cli, floors,
)
from sam3_multiview import BANK_THRESHOLD
from xpart_complete import DEFAULT_MIN_AREA_SHARE as XPART_MIN_AREA_SHARE

import merge_parts
import segment_parts


class PipelineDefaultsTest(unittest.TestCase):
    def test_named_presets_match_the_libraries_own_floors(self):
        self.assertEqual(GRANULARITY[DEFAULT_GRANULARITY],
                         (DEFAULT_MIN_FACES, DEFAULT_MIN_UNIT_FACES))
        self.assertEqual(floors(), (DEFAULT_MIN_FACES, DEFAULT_MIN_UNIT_FACES))

    def test_an_explicit_floor_wins_over_the_preset(self):
        self.assertEqual(floors("coarse", min_atom_faces=10), (10, 1600))
        self.assertEqual(floors("fine", min_unit_faces=99), (150, 99))

    def test_unknown_granularity_is_refused(self):
        with self.assertRaises(ValueError):
            floors("huge")

    def test_defaults_match_the_stage_modules(self):
        options = PipelineOptions()
        self.assertEqual(options.samples, DEFAULT_SAMPLES)
        self.assertEqual(options.samples, 7)
        self.assertEqual(options.azimuth_jitter, 30.0)
        self.assertEqual(options.color_tol, None)
        self.assertEqual(DEFAULT_COLOR_TOL, MEET_COLOR_TOL)
        self.assertEqual(DEFAULT_MIN_RECALL, VOTE_MIN_RECALL)
        self.assertEqual(DEFAULT_SAM3_THRESHOLD, BANK_THRESHOLD)
        self.assertEqual(DEFAULT_MIN_AREA_SHARE, XPART_MIN_AREA_SHARE)
        self.assertEqual(options.redraws, DEFAULT_REDRAWS)
        self.assertEqual(options.merge, DEFAULT_MERGE)
        self.assertEqual(options.complete, DEFAULT_COMPLETE)
        self.assertEqual(options.condition, DEFAULT_CONDITION)
        self.assertEqual(options.flat_paint, DEFAULT_FLAT_PAINT)
        self.assertEqual(options.view_azimuths, "45,225")
        self.assertEqual(options.view_elevations, "10")
        snapshot = options.public()["defaults"]
        self.assertEqual(snapshot["sam3_threshold"], 0.4)
        self.assertEqual(snapshot["color_tol"], MEET_COLOR_TOL)
        self.assertEqual(snapshot["min_recall"], VOTE_MIN_RECALL)
        self.assertEqual(snapshot["min_atom_faces"], DEFAULT_MIN_FACES)
        self.assertEqual(snapshot["min_unit_faces"], DEFAULT_MIN_UNIT_FACES)


class PipelineKwargsTest(unittest.TestCase):
    def test_segment_kwargs_are_accepted_by_segment_parts(self):
        parameters = inspect.signature(segment_parts.segment_parts).parameters
        for key in PipelineOptions().segment_kwargs():
            self.assertIn(key, parameters)

    def test_merge_kwargs_are_accepted_by_merge_parts(self):
        parameters = inspect.signature(merge_parts.merge_parts).parameters
        kwargs = PipelineOptions().merge_kwargs()
        self.assertNotIn("samples", kwargs)
        self.assertNotIn("granularity", kwargs)
        for key in kwargs:
            self.assertIn(key, parameters)

    def test_from_namespace_round_trips_the_cli_defaults(self):
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser, split=True, merge_off=True)
        args = parser.parse_args([])
        check_cli(parser, args)
        options = PipelineOptions.from_namespace(args)
        self.assertEqual(options, PipelineOptions())

    def test_from_mapping_overrides_and_ignores_unknown_keys(self):
        options = PipelineOptions.from_mapping({
            "complete": "full",
            "granularity": "coarse",
            "allow_partial": True,
            "no_concept_bank": True,
            "unassigned_to": "",
            "prompts": ["head"],
            "glb": "ignored.glb",
        })
        self.assertEqual(options.complete, "full")
        self.assertEqual(options.granularity, "coarse")
        self.assertEqual(options.resolved_floors(), (800, 1600))
        self.assertFalse(options.strict_parts)
        self.assertEqual(options.concept_bank, "")
        self.assertIsNone(options.unassigned_to)


class PipelineReexportTest(unittest.TestCase):
    def test_segment_parts_still_exposes_the_granularity_table(self):
        self.assertEqual(segment_parts.GRANULARITY, GRANULARITY)
        self.assertEqual(segment_parts.DEFAULT_GRANULARITY, DEFAULT_GRANULARITY)
        self.assertEqual(segment_parts.DEFAULT_SAMPLES, DEFAULT_SAMPLES)
        self.assertEqual(segment_parts.DEFAULT_SAM3_THRESHOLD, 0.4)


if __name__ == "__main__":
    unittest.main()
