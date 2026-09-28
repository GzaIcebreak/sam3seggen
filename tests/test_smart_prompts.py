import io
import json
import unittest

from fastapi.testclient import TestClient

import serve_api
from smart_prompts import build_question, parse_reply


class ParseReplyTest(unittest.TestCase):
    allowed = ["body", "wheel", "door", "window", "hood", "wing"]

    def test_json_in_a_code_fence_is_read_and_unknown_words_are_dropped(self):
        reply = '''```json
{"object": "sports car", "main": "body", "parts": ["Wheel", "door", "window", "spoiler", "wheel"]}
```'''
        out = parse_reply(reply, self.allowed)
        self.assertEqual(out["object"], "sports car")
        self.assertEqual(out["main"], "body")
        self.assertEqual(out["parts"], ["wheel", "door", "window"])
        self.assertEqual(out["dropped"], ["spoiler"])

    def test_the_main_word_is_not_also_a_part_and_an_unknown_main_is_dropped(self):
        out = parse_reply('{"object": "car", "main": "chassis", "parts": ["body", "wheel"]}', self.allowed)
        self.assertIsNone(out["main"])
        self.assertEqual(out["parts"], ["body", "wheel"])
        out = parse_reply('{"main": "body", "parts": ["body", "wheel"]}', self.allowed)
        self.assertEqual(out["parts"], ["wheel"])

    def test_reasoning_text_with_stray_braces_before_the_answer_is_fine(self):
        reply = ("We need {parts} for this. The shares {12%} suggest wheels matter. "
                 "Final: {\"object\": \"car\", \"main\": \"body\", \"parts\": [\"wheel\", \"door\"]} done")
        out = parse_reply(reply, self.allowed)
        self.assertEqual(out["parts"], ["wheel", "door"])

    def test_no_json_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_reply("I think it is a car.", self.allowed)

    def test_the_question_lists_the_candidates_with_their_share(self):
        text = build_question([{"concept": "wheel", "area": 0.12, "views": 8}])
        self.assertIn("wheel (12%, seen in 8 views)", text)
        self.assertIn("a car has no wing", text)


class SmartModeFormTest(unittest.TestCase):
    def test_smart_without_a_key_is_refused_before_anything_is_queued(self):
        original = serve_api.vlm_key_available
        serve_api.vlm_key_available = lambda: False
        try:
            client = TestClient(serve_api.app)
            files = {"glb": ("toy.glb", io.BytesIO(b"glb"), "model/gltf-binary")}
            response = client.post("/pipeline", data={"mode": "smart"}, files=files)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("SEGVIGEN_VLM_API_KEY", response.json()["detail"])
            response = client.post("/pipeline", data={"mode": "clever"}, files=files)
            self.assertEqual(response.status_code, 400)
        finally:
            serve_api.vlm_key_available = original

    def test_mode_is_a_published_switch(self):
        client = TestClient(serve_api.app)
        health = client.get("/health").json()
        self.assertEqual(health["switches"]["mode"], ["auto", "smart"])
        self.assertEqual(health["defaults"]["mode"], "auto")
        self.assertIn("base_url", health["smart_mode"])
        self.assertIn("key_configured", health["smart_mode"])


if __name__ == "__main__":
    unittest.main()
