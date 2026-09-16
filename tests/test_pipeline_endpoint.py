import io
import json
import os
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

import serve_api


def wait_for(client, job_id, states=("done", "error"), timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        summary = client.get(f"/jobs/{job_id}").json()
        if summary["state"] in states:
            return summary
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {states}")


class PipelineEndpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_jobs_dir, serve_api.JOBS_DIR = serve_api.JOBS_DIR, self.tmp.name
        self.old_run = serve_api._run_job
        self.seen = []

        def fake_job(job_id, upload, filename, options, legacy=False):
            self.seen.append((job_id, options["prompts"], options.get("merge_gap")))
            job = os.path.join(serve_api.JOBS_DIR, job_id)
            serve_api._record_job(job_id, filename)
            time.sleep(0.05)
            with open(os.path.join(job, "parts.glb"), "wb") as f:
                f.write(b"open")
            if options.get("complete") != "off":
                os.makedirs(os.path.join(job, "complete"), exist_ok=True)
                with open(os.path.join(job, "complete", "xpart_parts.glb"), "wb") as f:
                    f.write(b"closed")
            result = {"job_id": job_id, "parts": [], "seconds": 0.05}
            serve_api._finish_job(job_id, result=result)
            return result

        serve_api._run_job = fake_job
        self.client = TestClient(serve_api.app)

    def tearDown(self):
        serve_api._run_job = self.old_run
        serve_api.JOBS_DIR = self.old_jobs_dir
        self.tmp.cleanup()

    def post(self, **data):
        return self.client.post("/pipeline", data=data,
                                files={"glb": ("toy.glb", io.BytesIO(b"glb"), "model/gltf-binary")})

    def test_one_shot_returns_a_ticket_then_the_repaired_model(self):
        response = self.post(prompts="armor, body", options=json.dumps({"merge_gap": 0.01}))
        self.assertEqual(response.status_code, 202, response.text)
        ticket = response.json()
        self.assertEqual(ticket["state"], "queued")
        self.assertEqual(ticket["result_url"], f"/jobs/{ticket['job_id']}/result")
        summary = wait_for(self.client, ticket["job_id"])
        self.assertEqual(summary["state"], "done")
        self.assertIsNone(summary["position"])
        result = self.client.get(ticket["result_url"])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, b"closed")
        self.assertIn("repaired_parts.glb", result.headers["content-disposition"])
        self.assertEqual(self.seen[-1][1:], ("armor, body", 0.01))

    def test_without_repair_the_result_is_the_open_parts(self):
        ticket = self.post(options=json.dumps({"complete": "off"})).json()
        wait_for(self.client, ticket["job_id"])
        result = self.client.get(ticket["result_url"])
        self.assertEqual(result.content, b"open")

    def test_two_requests_queue_instead_of_409(self):
        first = self.post(prompts="a").json()
        second = self.post(prompts="b").json()
        self.assertEqual(second["state"], "queued")
        self.assertGreaterEqual(second["position"], 1)
        wait_for(self.client, first["job_id"])
        wait_for(self.client, second["job_id"])
        self.assertEqual([row[1] for row in self.seen[-2:]], ["a", "b"])

    def test_result_before_done_is_409_with_the_state(self):
        # nothing has run the job yet: hold the GPU lock so the worker cannot start it
        serve_api._gpu.acquire()
        try:
            ticket = self.post(prompts="a").json()
            response = self.client.get(ticket["result_url"])
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["detail"]["state"], "queued")
        finally:
            serve_api._gpu.release()
        wait_for(self.client, ticket["job_id"])

    def test_bad_options_are_400_before_anything_is_queued(self):
        self.assertEqual(self.post(options="not json").status_code, 400)
        self.assertEqual(self.post(options=json.dumps({"complete": "sometimes"})).status_code, 400)
        self.assertEqual(self.post(options=json.dumps(["a"])).status_code, 400)
        self.assertEqual(self.client.get("/jobs").json()["jobs"], [])


if __name__ == "__main__":
    unittest.main()
