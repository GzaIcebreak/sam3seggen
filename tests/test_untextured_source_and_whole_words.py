import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

from auto_prompts import drop_whole_words
from source_export import export_from_source


class UntexturedSourceExportTest(unittest.TestCase):
    def test_a_flat_coloured_single_mesh_is_still_cut_from_the_source(self):
        source = trimesh.creation.icosphere(4)                     # no uv, no texture
        remesh = trimesh.creation.icosphere(2)
        labels = (remesh.triangles_center[:, 2] > 0).astype(int)   # top / bottom
        with tempfile.TemporaryDirectory() as out:
            src_glb = os.path.join(out, "src.glb"); source.export(src_glb)
            remesh_glb = os.path.join(out, "seg.glb"); remesh.export(remesh_glb)
            np.save(os.path.join(out, "labels.npy"), labels)
            with open(os.path.join(out, "names.json"), "w", encoding="utf-8") as handle:
                json.dump(["bottom", "top"], handle)
            manifest = export_from_source(remesh_glb, src_glb, os.path.join(out, "labels.npy"),
                                          os.path.join(out, "names.json"),
                                          os.path.join(out, "parts.glb"))
            self.assertIsNotNone(manifest)
            self.assertEqual([row["name"] for row in manifest], ["bottom", "top"])
            self.assertTrue(all(row["source"] == "original" for row in manifest))
            self.assertTrue(all(row["texture_size"] is None for row in manifest))
            self.assertEqual(sum(row["faces"] for row in manifest), len(source.faces))
            scene = trimesh.load(os.path.join(out, "parts.glb"), force="scene")
            self.assertEqual(len(scene.geometry), 2)


class DropWholeWordsTest(unittest.TestCase):
    CANDIDATES = [{"concept": "crossbar", "area": 0.83}, {"concept": "handle", "area": 0.65},
                  {"concept": "wheel", "area": 0.12}, {"concept": "door", "area": 0.2}]

    def test_a_part_covering_half_the_silhouette_is_dropped(self):
        kept, dropped = drop_whole_words(["handle", "crossbar", "wheel"], self.CANDIDATES)
        self.assertEqual(kept, ["wheel"])
        self.assertEqual(dropped, ["handle", "crossbar"])

    def test_small_parts_and_words_without_a_mask_pass(self):
        kept, dropped = drop_whole_words(["wheel", "door", "mirror"], self.CANDIDATES)
        self.assertEqual(kept, ["wheel", "door", "mirror"])
        self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()
