import io
import os
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

import serve_api
import smart_prompts
from smart_prompts import build_shortlist_question, guide_part_count


def guide_png(colours=((220, 40, 40), (40, 90, 230), (30, 180, 70)), size=96):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    band = size // (len(colours) + 1)
    for i, rgb in enumerate(colours):
        for y in range(8 + i * band, 8 + (i + 1) * band):
            for x in range(8, size - 8):
                image.putpixel((x, y), (*rgb, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class GuideImageTest(unittest.TestCase):
    def test_flat_colours_are_counted_and_the_question_names_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "guide.png")
            with open(path, "wb") as f:
                f.write(guide_png())
            self.assertEqual(guide_part_count(path), 3)
        question = build_shortlist_question(["head", "arm", "body"], guide_colours=3)
        self.assertIn("3 colours", question)
        self.assertIn("LAST tile", question)
        self.assertNotIn("LAST tile", build_shortlist_question(["head"]))


class GuidePromptsEndpointTest(unittest.TestCase):
    def setUp(self):
        self.old_key = serve_api.vlm_key_available
        self.old_shortlist = smart_prompts.kimi_shortlist
        self.old_bank = serve_api.segment_parts.DEFAULT_CONCEPT_BANK
        self.calls = []

        def fake_shortlist(images, vocabulary, guide_image=None, **kwargs):
            self.calls.append((list(images), guide_image))
            return {"object": "lego minifigure", "main": "torso",
                    "parts": ["head", "arm", "leg"], "dropped": ["waist"],
                    "separate": {"leg": 2, "arm": 2}, "model": "qwen3.8-max"}

        serve_api.vlm_key_available = lambda: True
        smart_prompts.kimi_shortlist = fake_shortlist
        import auto_prompts
        self.old_concepts = auto_prompts.bank_concepts
        auto_prompts.bank_concepts = lambda *a, **k: ["head", "arm", "leg", "torso"]
        self.client = TestClient(serve_api.app)

    def tearDown(self):
        serve_api.vlm_key_available = self.old_key
        smart_prompts.kimi_shortlist = self.old_shortlist
        import auto_prompts
        auto_prompts.bank_concepts = self.old_concepts

    def test_the_guide_becomes_prompts_for_the_pipeline(self):
        response = self.client.post("/guide_prompts", files={
            "guide": ("gt.png", io.BytesIO(guide_png()), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["prompts"], "head, arm, leg, torso")
        self.assertEqual(body["unassigned_to"], "torso")
        self.assertEqual(body["colours"], {"leg": 2, "arm": 2})
        self.assertEqual(body["dropped_not_in_bank"], ["waist"])
        images, guide = self.calls[0]
        self.assertEqual(images, [])
        self.assertTrue(guide.endswith(".png"))

    def test_without_a_vlm_key_it_is_refused(self):
        serve_api.vlm_key_available = lambda: False
        response = self.client.post("/guide_prompts", files={
            "guide": ("gt.png", io.BytesIO(guide_png()), "image/png")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("SEGVIGEN_VLM_API_KEY", response.json()["detail"])

    def test_the_old_guided_one_shot_is_gone(self):
        response = self.client.post("/pipeline_guided", files={
            "glb": ("toy.glb", io.BytesIO(b"glb"), "model/gltf-binary"),
            "guide": ("gt.png", io.BytesIO(guide_png()), "image/png")})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
