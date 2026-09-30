"""Complete-stage speed-ups: JPEG textures stay JPEG; single-tree sliver folding."""
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from glb_images import glb_image_mimes, keep_jpeg, mesh_materials  # noqa: E402


def _textured_box(fmt):
    mesh = trimesh.creation.box()
    uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 64, 3), np.uint8)).save(buf, format=fmt)
    buf.seek(0)
    image = Image.open(buf)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=image))
    return mesh


class KeepJpeg(unittest.TestCase):
    def _roundtrip(self, fmt):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.glb")
            _textured_box(fmt).export(src)
            self.assertEqual(glb_image_mimes(src), {f"image/{fmt.lower()}"})
            loaded = trimesh.load(src, force="scene")
            # the pipeline exports copies (submesh, apply_transform): PIL copies lose format
            scene = trimesh.Scene([g.copy() for g in loaded.geometry.values()])
            marked = keep_jpeg(mesh_materials(scene), src)
            out = os.path.join(d, "out.glb")
            scene.export(out)
            return marked, glb_image_mimes(out)

    def test_jpeg_source_exports_jpeg(self):
        marked, mimes = self._roundtrip("JPEG")
        self.assertEqual(marked, 1)
        self.assertEqual(mimes, {"image/jpeg"})

    def test_png_source_is_left_alone(self):
        marked, mimes = self._roundtrip("PNG")
        self.assertEqual(marked, 0)
        self.assertEqual(mimes, {"image/png"})

    def test_not_a_glb(self):
        with tempfile.NamedTemporaryFile(suffix=".obj") as f:
            self.assertEqual(glb_image_mimes(f.name), set())


class FoldSmallPieces(unittest.TestCase):
    def setUp(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        from xpart_complete import fold_small_pieces
        self.fold = fold_small_pieces

    def _piece(self, centre, size):
        return trimesh.creation.box(extents=[size] * 3).apply_translation(centre)

    def test_sliver_goes_to_nearest_big_piece(self):
        pieces = [("a", self._piece([0, 0, 0], 1.0)), ("b", self._piece([5, 0, 0], 1.0)),
                  ("s", self._piece([4.2, 0, 0], 0.05))]
        out = dict(self.fold(pieces, min_area_share=0.01))
        self.assertEqual(set(out), {"a", "b"})
        self.assertGreater(len(out["b"].faces), len(out["a"].faces))

    def test_exact_tie_goes_to_first_candidate(self):
        # the sliver touches both big boxes at distance 0: the first kept piece wins
        pieces = [("a", self._piece([0, 0, 0], 1.0)), ("b", self._piece([1.0, 0, 0], 1.0)),
                  ("s", trimesh.Trimesh([[0.5, 0, 0], [0.5, 0.01, 0], [0.5, 0, 0.01]], [[0, 1, 2]]))]
        out = dict(self.fold(pieces, min_area_share=0.01))
        self.assertEqual(len(out["a"].faces), 13)
        self.assertEqual(len(out["b"].faces), 12)

    def test_fold_within_part_prefers_own_part(self):
        pieces = [("tree", self._piece([0, 0, 0], 1.0)), ("ball", self._piece([3, 0, 0], 1.0)),
                  ("ball", self._piece([0.7, 0, 0], 0.05))]
        out = self.fold(pieces, min_area_share=0.01, fold_within_part=True)
        sizes = {name: len(mesh.faces) for name, mesh in out}
        self.assertEqual(sizes, {"tree": 12, "ball": 24})


if __name__ == "__main__":
    unittest.main()
