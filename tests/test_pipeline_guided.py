import io
import json
import os
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from PIL import Image

import serve_api
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


class PipelineGuidedEndpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_jobs_dir, serve_api.JOBS_DIR = serve_api.JOBS_DIR, self.tmp.name
        self.old_run = serve_api._run_job
        self.old_key = serve_api.vlm_key_available
        self.seen = []

        def fake_job(job_id, upload, filename, options, legacy=False):
            self.seen.append((job_id, dict(options)))
            serve_api._record_job(job_id, filename)
            with open(os.path.join(serve_api.JOBS_DIR, job_id, "parts.glb"), "wb") as f:
                f.write(b"open")
            serve_api._finish_job(job_id, result={"job_id": job_id, "parts": [], "seconds": 0.01})

        serve_api._run_job = fake_job
        serve_api.vlm_key_available = lambda: True
        self.client = TestClient(serve_api.app)

    def tearDown(self):
        serve_api._run_job = self.old_run
        serve_api.vlm_key_available = self.old_key
        serve_api.JOBS_DIR = self.old_jobs_dir
        self.tmp.cleanup()

    def post(self, with_guide=True, **data):
        files = {"glb": ("toy.glb", io.BytesIO(b"glb"), "model/gltf-binary")}
        if with_guide:
            files["guide"] = ("gt.png", io.BytesIO(guide_png()), "image/png")
        return self.client.post("/pipeline_guided", data=data, files=files)

    def test_the_guide_is_stored_in_the_job_and_handed_to_the_pipeline_in_smart_mode(self):
        response = self.post(options=json.dumps({"merge_gap": 0.01}))
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]
        deadline = time.time() + 5
        while time.time() < deadline and not self.seen:
            time.sleep(0.02)
        self.assertTrue(self.seen, "the worker never ran the job")
        seen_id, options = self.seen[0]
        self.assertEqual(seen_id, job_id)
        self.assertEqual(options["mode"], "smart")
        self.assertEqual(options["prompts"], "")
        self.assertEqual(options["merge_gap"], 0.01)
        self.assertEqual(options["guide_image"], os.path.join(serve_api.JOBS_DIR, job_id, "guide.png"))
        self.assertTrue(os.path.isfile(options["guide_image"]))
        record = json.load(open(os.path.join(serve_api.JOBS_DIR, job_id, "job.json")))
        self.assertEqual(record["guide"], "guide.png")

    def test_missing_guide_is_rejected(self):
        self.assertEqual(self.post(with_guide=False).status_code, 422)

    def test_without_a_vlm_key_it_is_refused_up_front(self):
        serve_api.vlm_key_available = lambda: False
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertIn("SEGVIGEN_VLM_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
