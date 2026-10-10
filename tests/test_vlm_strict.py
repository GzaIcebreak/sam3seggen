import unittest

from auto_prompts import VlmUnavailable, ask_vlm_or_fail


class VlmStrictTest(unittest.TestCase):
    def test_a_hiccup_is_retried(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise OSError("connection refused")
            return {"parts": ["head"]}

        self.assertEqual(ask_vlm_or_fail(flaky, "naming the parts", wait=0), {"parts": ["head"]})
        self.assertEqual(len(calls), 2)

    def test_no_answer_fails_the_job_instead_of_falling_back(self):
        calls = []

        def dead():
            calls.append(1)
            raise OSError("connection refused")

        with self.assertRaises(VlmUnavailable) as caught:
            ask_vlm_or_fail(dead, "naming the parts", attempts=3, wait=0)
        self.assertEqual(len(calls), 3)
        self.assertIn("大模型调用失败", str(caught.exception))
        self.assertIn("connection refused", str(caught.exception))

    def test_propose_prompts_has_no_silent_fallback_left(self):
        import inspect

        import auto_prompts

        source = inspect.getsource(auto_prompts.propose_prompts)
        self.assertNotIn("VLM failed", source)        # the old silent-fallback messages
        self.assertNotIn("except Exception", source)
        self.assertEqual(source.count("ask_vlm_or_fail("), 2)


if __name__ == "__main__":
    unittest.main()
