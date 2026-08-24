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
