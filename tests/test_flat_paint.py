import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from flat_paint import COLORLESS_SATURATION, is_colorless, mean_saturation


def views_of(colors, resolution=64):
    """A temp views dir whose silhouette is a centred square of `colors[i]` per view."""
    directory = tempfile.mkdtemp()
    views = []
    for index, color in enumerate(colors):
        image = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        image[16:48, 16:48, :3] = color
        image[16:48, 16:48, 3] = 255
        name = f"view_{index}.png"
        Image.fromarray(image).save(os.path.join(directory, name))
        views.append({"name": name[:-4], "image": name, "azimuth": float(index * 90),
                      "elevation": 10.0})
    manifest = {"camera_angle_x": 0.7, "radius": 2.0, "resolution": resolution,
                "views": views}
    with open(os.path.join(directory, "cameras.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return directory, manifest


class ColorlessTest(unittest.TestCase):
    def test_a_grey_model_has_no_saturation(self):
        directory, manifest = views_of([[200, 200, 200], [120, 120, 120]])
        self.assertEqual(mean_saturation(directory, manifest), 0.0)
        self.assertTrue(is_colorless(directory, manifest))

    def test_a_textured_model_is_left_alone(self):
        directory, manifest = views_of([[220, 40, 40], [40, 60, 200]])
        self.assertFalse(is_colorless(directory, manifest))

    def test_the_threshold_sits_between_the_two_models_measured(self):
        # The untextured robot renders at 0.25-0.42 mean saturation and Mickey at 43-53,
        # so the gate needs no per-model tuning -- but it does need to stay in the gap.
        self.assertLess(0.42, COLORLESS_SATURATION)
        self.assertLess(COLORLESS_SATURATION, 43.0)

    def test_background_pixels_do_not_dilute_the_measurement(self):
        # The silhouette is a small part of the frame; a mean over the whole image would
        # call everything colourless.
        directory, manifest = views_of([[220, 40, 40]])
        self.assertGreater(mean_saturation(directory, manifest), 100.0)


if __name__ == "__main__":
    unittest.main()
