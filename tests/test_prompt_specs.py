import unittest

from prompt_specs import (
    normalize_part_specs,
    part_names,
    validate_named_rows,
    validate_part_coverage,
    validate_target_name,
)


class PromptSpecsTest(unittest.TestCase):
    def test_accepts_one_prompt_string(self):
        self.assertEqual(
            normalize_part_specs("armor"),
            [("armor", ["armor"])],
        )

    def test_accepts_arbitrary_grouped_prompt_list(self):
        self.assertEqual(
            normalize_part_specs([
                "roof",
                "opening=door+window",
                "wall=facade+brick",
            ]),
            [
                ("roof", ["roof"]),
                ("opening", ["door", "window"]),
                ("wall", ["facade", "brick"]),
            ],
        )

    def test_rejects_duplicate_output_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate component name"):
            normalize_part_specs(["body=head", "body=hand+leg"])

    def test_rejects_empty_name_or_concept(self):
        for value in ("", "=head", "body=", "body=head++leg"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_part_specs(value)

    def test_unassigned_target_is_dynamic(self):
        validate_target_name("wall", ["roof", "opening", "wall"])
        with self.assertRaisesRegex(ValueError, "unassigned_to"):
            validate_target_name("body", ["roof", "opening", "wall"])

    def test_named_rows_must_match_dynamic_request_exactly(self):
        validate_named_rows(
            ["roof", "opening"],
            [{"prompt": "opening"}, {"prompt": "roof"}],
            key="prompt",
        )
        with self.assertRaisesRegex(ValueError, "missing=.*opening"):
            validate_named_rows(["roof", "opening"], [{"prompt": "roof"}], key="prompt")
        with self.assertRaisesRegex(ValueError, "extra=.*floor"):
            validate_named_rows(
                ["roof"],
                [{"prompt": "roof"}, {"prompt": "floor"}],
                key="prompt",
            )

    def test_rejects_rows_missing_name_field(self):
        with self.assertRaisesRegex(ValueError, "missing field"):
            validate_named_rows(
                ["roof", "opening"],
                [{"prompt": "roof"}, {}],
                key="prompt",
            )

    def test_legend_coverage_allows_multiple_concepts_per_part(self):
        validate_part_coverage(
            ["roof", "opening"],
            [
                {"prompt": "roof", "part": "roof"},
                {"prompt": "door", "part": "opening"},
                {"prompt": "window", "part": "opening"},
            ],
        )

    def test_legend_coverage_falls_back_to_prompt_when_part_omitted(self):
        validate_part_coverage(
            ["roof", "opening"],
            [{"prompt": "roof"}, {"prompt": "opening"}],
        )

    def test_legend_coverage_rejects_missing_part(self):
        with self.assertRaisesRegex(ValueError, "missing=.*opening"):
            validate_part_coverage(
                ["roof", "opening"],
                [{"prompt": "roof", "part": "roof"}],
            )


if __name__ == "__main__":
    unittest.main()
