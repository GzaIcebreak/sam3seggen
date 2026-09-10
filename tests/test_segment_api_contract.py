import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

import segment_api
from prompt_specs import normalize_part_specs, part_names, validate_named_rows


class SegmentApiContractTest(unittest.TestCase):
    def test_segment_exposes_sam3_only_mode(self):
        parameter = inspect.signature(segment_api.segment).parameters["sam3_only"]
        self.assertIs(parameter.default, False)

    def test_build_audit_result_returns_absolute_paths_and_legend(self):
        legend = [{"prompt": "seat"}]
        result = segment_api._build_sam3_audit_result(
            "render.png",
            "map.png",
            legend,
        )
        self.assertEqual(result, {
            "render": os.path.abspath("render.png"),
            "map": os.path.abspath("map.png"),
            "legend": legend,
        })

    def test_sam3_only_rejects_no_sam_before_running_subprocesses(self):
        with mock.patch.object(segment_api, "_run") as run:
            with self.assertRaisesRegex(ValueError, "requires use_sam3=True"):
                segment_api.segment(
                    "input.glb",
                    [],
                    "unused.glb",
                    use_sam3=False,
                    sam3_only=True,
                    work_dir="audit",
                )
        run.assert_not_called()

    def test_sam3_only_requires_persistent_work_dir_before_running_subprocesses(self):
        with mock.patch.object(segment_api, "_run") as run:
            with self.assertRaisesRegex(ValueError, "requires work_dir"):
                segment_api.segment(
                    "input.glb",
                    ["seat"],
                    "unused.glb",
                    sam3_only=True,
                )
        run.assert_not_called()

    def test_sam3_only_cannot_disable_strict_legend_validation(self):
        with mock.patch.object(segment_api, "_run") as run:
            with self.assertRaisesRegex(ValueError, "requires strict_parts=True"):
                segment_api.segment(
                    "input.glb",
                    ["seat"],
                    "unused.glb",
                    sam3_only=True,
                    strict_parts=False,
                    work_dir="audit",
                )
        run.assert_not_called()

    def test_sam3_only_returns_after_dynamic_legend_validation(self):
        legend = [
            {"prompt": "armor", "part": "armor"},
            {"prompt": "head", "part": "body"},
            {"prompt": "face", "part": "body"},
        ]
        with tempfile.TemporaryDirectory() as work_dir:
            legend_path = os.path.join(work_dir, "sam3_2d_map_legend.json")

            def fake_run(command):
                if "sam3_to_2dmap.py" in command[1]:
                    with open(legend_path, "w", encoding="utf-8") as file:
                        json.dump(legend, file)

            with mock.patch.object(segment_api, "_run", side_effect=fake_run) as run:
                result = segment_api.segment(
                    "input.glb",
                    ["armor", "body=head+face"],
                    "unused.glb",
                    sam3_only=True,
                    work_dir=work_dir,
                    unassigned_to="body",
                )

            self.assertEqual(run.call_count, 2)
            sam3_command = run.call_args_list[1].args[0]
            self.assertIn("--unassigned_to", sam3_command)
            self.assertEqual(
                sam3_command[sam3_command.index("--prompts") + 1:],
                ["armor", "body=head+face"],
            )
            self.assertEqual(result, {
                "render": os.path.abspath(os.path.join(work_dir, "render.png")),
                "map": os.path.abspath(os.path.join(work_dir, "sam3_2d_map.png")),
                "legend": legend,
            })

    def test_default_backend_is_the_deployed_ease_painter(self):
        with mock.patch.object(segment_api.os.path, "exists", return_value=True):
            opts = segment_api._resolve_map_backend(
                None, "rank", None, 0.1, 0.9, "sam3", None)
        command = segment_api._sam3_cmd("py", "r.png", "m.png", "l.json", opts)
        self.assertEqual(command[command.index("--threshold") + 1], 0.4)
        self.assertEqual(command[command.index("--assign") + 1], "rank")
        self.assertEqual(command[command.index("--rank_drop") + 1], 0.1)
        self.assertEqual(command[command.index("--rank_add") + 1], 0.9)
        self.assertIn("--concept_bank", command)

    def test_missing_default_weights_downgrade_instead_of_failing(self):
        with mock.patch.object(segment_api.os.path, "exists", return_value=False):
            opts = segment_api._resolve_map_backend(
                None, "rank", None, 0.1, 0.9, "sam3", None)
        self.assertEqual(opts["assign"], "paint")
        self.assertIsNone(opts["concept_bank"])
        self.assertEqual(opts["threshold"], 0.3)
        self.assertNotIn("--concept_bank",
                         segment_api._sam3_cmd("py", "r.png", "m.png", "l.json", opts))

    def test_explicitly_named_weights_must_exist(self):
        with mock.patch.object(segment_api.os.path, "exists", return_value=False):
            with self.assertRaisesRegex(ValueError, "concept bank not found"):
                segment_api._resolve_map_backend(
                    "bank.pt", "rank", None, 0.1, 0.9, "sam3", None)
            with self.assertRaisesRegex(ValueError, "rank model not found"):
                segment_api._resolve_map_backend(
                    False, "rank", "rank.pt", 0.1, 0.9, "sam3", None)

    def test_v6_is_the_default_checkpoint_for_the_two_d_map_path(self):
        with mock.patch.object(segment_api.os.path, "exists", return_value=True):
            self.assertEqual(segment_api._resolve_ckpt(None, True, True),
                             segment_api.DEFAULT_CKPT_V6)
            self.assertEqual(segment_api._resolve_ckpt(None, True, False),
                             segment_api.DEFAULT_CKPT)
            self.assertEqual(segment_api._resolve_ckpt(None, False, True),
                             segment_api.DEFAULT_CKPT_NO_SAM)
            self.assertEqual(segment_api._resolve_ckpt("mine.ckpt", True, True),
                             os.path.abspath("mine.ckpt"))

    def test_missing_v6_falls_back_to_the_base_checkpoint(self):
        with mock.patch.object(segment_api.os.path, "exists", return_value=False):
            self.assertEqual(segment_api._resolve_ckpt(None, True, True),
                             segment_api.DEFAULT_CKPT)

    def test_ease_is_off_by_default(self):
        self.assertEqual(inspect.signature(segment_api.segment).parameters["assign"].default,
                         "paint")
        with mock.patch.object(segment_api.os.path, "exists", return_value=True):
            opts = segment_api._resolve_map_backend(
                None, "paint", None, 0.1, 0.9, "sam3", None)
        self.assertNotIn("--assign",
                         segment_api._sam3_cmd("py", "r.png", "m.png", "l.json", opts))

    def test_auto_assign_passes_the_ranker_the_sam3_side_needs(self):
        with mock.patch.object(segment_api.os.path, "exists", return_value=True):
            opts = segment_api._resolve_map_backend(
                None, "auto", None, 0.1, 0.9, "sam3", None)
        command = segment_api._sam3_cmd("py", "r.png", "m.png", "l.json", opts)
        self.assertEqual(command[command.index("--assign") + 1], "auto")
        self.assertIn("--rank_model", command)

    def test_parts_output_is_validated_before_running_subprocesses(self):
        self.assertEqual(
            inspect.signature(segment_api.segment).parameters["parts_output"].default,
            "combined")
        with mock.patch.object(segment_api, "_run") as run:
            with self.assertRaisesRegex(ValueError, "parts_output must be"):
                segment_api.segment("in.glb", ["seat"], "out.glb", parts_output="zip")
        run.assert_not_called()

    def test_split_follows_stain_by_default_and_rejects_unknown_modes(self):
        self.assertEqual(
            inspect.signature(segment_api.segment).parameters["split_mode"].default, "stain")
        self.assertEqual(segment_api.SPLIT_MODES, ("stain", "weld", "refine"))
        with mock.patch.object(segment_api, "_run") as run:
            with self.assertRaisesRegex(ValueError, "split_mode must be"):
                segment_api.segment("in.glb", ["seat"], "out.glb", split_mode="islands")
        run.assert_not_called()

    def test_separate_output_records_one_file_per_manifest_row(self):
        manifest = [{"node": "part_00_seat"}, {"node": "part_01_leg"}]
        exported = {}

        class FakeScene:
            geometry = {"part_00_seat": "seat-mesh", "part_01_leg": "leg-mesh"}

            def __init__(self, mapping=None):
                self.mapping = mapping

            def export(self, path):
                exported[os.path.basename(path)] = self.mapping

        fake = mock.Mock(load=mock.Mock(return_value=FakeScene()), Scene=FakeScene)
        with mock.patch.dict("sys.modules", {"trimesh": fake}):
            with tempfile.TemporaryDirectory() as out_dir:
                rows = segment_api._write_separate_parts("parts.glb", manifest, out_dir)
                self.assertEqual([os.path.basename(r["file"]) for r in rows],
                                 ["part_00_seat.glb", "part_01_leg.glb"])
        self.assertEqual(exported, {
            "part_00_seat.glb": {"part_00_seat": "seat-mesh"},
            "part_01_leg.glb": {"part_01_leg": "leg-mesh"},
        })

    def test_separate_output_rejects_a_node_missing_from_the_combined_glb(self):
        class FakeScene:
            geometry = {"part_00_seat": "seat-mesh"}

        fake = mock.Mock(load=mock.Mock(return_value=FakeScene()))
        with mock.patch.dict("sys.modules", {"trimesh": fake}):
            with tempfile.TemporaryDirectory() as out_dir:
                with self.assertRaisesRegex(ValueError, "no geometry named 'part_01_leg'"):
                    segment_api._write_separate_parts(
                        "parts.glb", [{"node": "part_01_leg"}], out_dir)

    def test_segment_defaults_to_strict_part_validation(self):
        strict_parts = inspect.signature(segment_api.segment).parameters["strict_parts"]
        self.assertTrue(strict_parts.default)

    def test_single_string_does_not_split_into_characters(self):
        specs = normalize_part_specs("armor")
        self.assertEqual(part_names(specs), ["armor"])

    def test_arbitrary_names_drive_final_manifest_validation(self):
        expected = part_names(normalize_part_specs([
            "seat",
            "support=leg+frame",
        ]))
        validate_named_rows(
            expected,
            [{"name": "support"}, {"name": "seat"}],
            key="name",
        )

    def test_extra_unassigned_part_is_rejected(self):
        expected = part_names(normalize_part_specs(["seat", "leg"]))
        with self.assertRaisesRegex(ValueError, "<unassigned>"):
            validate_named_rows(
                expected,
                [
                    {"name": "seat"},
                    {"name": "leg"},
                    {"name": "<unassigned>"},
                ],
                key="name",
            )


if __name__ == "__main__":
    unittest.main()
