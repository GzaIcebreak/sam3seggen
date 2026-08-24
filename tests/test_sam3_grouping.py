import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import sam3_to_2dmap as module


class Sam3GroupingTest(unittest.TestCase):
    @patch.object(module, "segment_prompts")
    def test_keeps_grouped_concepts_separate_with_part_name(self, segment_prompts):
        segment_prompts.return_value = [
            {"prompt": "door", "mask": np.array([[1, 0], [0, 0]], bool), "score": 0.8},
            {"prompt": "window", "mask": np.array([[0, 1], [0, 0]], bool), "score": 0.9},
        ]
        parts = module.segment_parts(
            object(), object(), Image.new("RGBA", (2, 2)),
            [("opening", ["door", "window"])], 0.3, "cpu",
        )
        self.assertEqual(
            [(part["prompt"], part["part"]) for part in parts],
            [("door", "opening"), ("window", "opening")],
        )
        np.testing.assert_array_equal(parts[0]["mask"], np.array([[1, 0], [0, 0]], bool))
        np.testing.assert_array_equal(parts[1]["mask"], np.array([[0, 1], [0, 0]], bool))

    @patch.object(module, "segment_prompts")
    def test_rejects_any_requested_component_without_a_mask(self, segment_prompts):
        segment_prompts.return_value = [
            {"prompt": "roof", "mask": np.ones((2, 2), bool), "score": 0.9},
        ]
        with self.assertRaisesRegex(ValueError, "opening"):
            module.segment_parts(
                object(), object(), Image.new("RGBA", (2, 2)),
                [("roof", ["roof"]), ("opening", ["door", "window"])],
                0.3, "cpu",
            )

    def test_unassigned_foreground_is_folded_into_dynamic_target(self):
        image = Image.new("RGBA", (2, 2), (255, 255, 255, 255))
        parts = [{
            "prompt": "roof",
            "part": "roof",
            "mask": np.array([[1, 0], [0, 0]], bool),
            "score": 1.0,
        }]
        colored, legend = module.colorize(
            image, parts, instance=False, unassigned_to="roof",
        )
        self.assertEqual([row["prompt"] for row in legend], ["roof"])
        self.assertEqual(legend[0]["pixels"], 4)
        self.assertEqual(len(np.unique(np.asarray(colored).reshape(-1, 3), axis=0)), 1)

    def test_colorize_gives_grouped_concepts_distinct_colors(self):
        image = Image.new("RGBA", (2, 2), (255, 255, 255, 255))
        parts = [
            {
                "prompt": "head",
                "part": "body",
                "mask": np.array([[1, 0], [0, 0]], bool),
                "score": 1.0,
            },
            {
                "prompt": "hand",
                "part": "body",
                "mask": np.array([[0, 1], [0, 0]], bool),
                "score": 1.0,
            },
            {
                "prompt": "staff",
                "part": "staff",
                "mask": np.array([[0, 0], [1, 0]], bool),
                "score": 1.0,
            },
        ]
        colored, legend = module.colorize(image, parts, instance=False)
        named = [row for row in legend if row["prompt"] != "<unassigned>"]
        colors = {row["prompt"]: tuple(row["color"]) for row in named}
        self.assertEqual(
            [(row["prompt"], row["part"]) for row in named],
            [("head", "body"), ("hand", "body"), ("staff", "staff")],
        )
        self.assertEqual(len(set(colors.values())), 3)
        pixels = np.asarray(colored)
        self.assertEqual(tuple(pixels[0, 0]), colors["head"])
        self.assertEqual(tuple(pixels[0, 1]), colors["hand"])
        self.assertEqual(tuple(pixels[1, 0]), colors["staff"])


if __name__ == "__main__":
    unittest.main()
