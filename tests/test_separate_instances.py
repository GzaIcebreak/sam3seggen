import unittest

import numpy as np
import trimesh

from data_toolkit.parts_rebake import welded_face_adjacency
from merge_parts import split_separate_instances
from smart_prompts import build_shortlist_question, parse_reply


def two_legs_and_a_body():
    body = trimesh.creation.box(extents=(2, 1, 1))
    left = trimesh.creation.box(extents=(0.4, 0.4, 1)).apply_translation((-0.6, 0, -1))
    right = trimesh.creation.box(extents=(0.4, 0.4, 1)).apply_translation((0.6, 0, -1))
    mesh = trimesh.util.concatenate([body, left, right])
    labels = np.array([0] * len(body.faces) + [1] * (len(left.faces) + len(right.faces)))
    return mesh, labels


class SplitSeparateInstancesTest(unittest.TestCase):
    def test_two_legs_become_leg_and_leg_2(self):
        mesh, labels = two_legs_and_a_body()
        new_labels, names, report = split_separate_instances(
            labels, ["body", "leg"], welded_face_adjacency(mesh), mesh.area_faces, ["leg"])
        self.assertEqual(names, ["body", "leg", "leg 2"])
        self.assertEqual(report, [("leg", 2)])
        self.assertEqual(int((new_labels == 1).sum()), 12)
        self.assertEqual(int((new_labels == 2).sum()), 12)
        self.assertEqual(int((new_labels == 0).sum()), 12)

    def test_names_not_listed_and_single_pieces_are_left_alone(self):
        mesh, labels = two_legs_and_a_body()
        new_labels, names, report = split_separate_instances(
            labels, ["body", "leg"], welded_face_adjacency(mesh), mesh.area_faces, ["body"])
        self.assertEqual(names, ["body", "leg"])
        self.assertEqual(report, [])
        np.testing.assert_array_equal(new_labels, labels)
        _, names, _ = split_separate_instances(
            labels, ["body", "leg"], welded_face_adjacency(mesh), mesh.area_faces, "all")
        self.assertEqual(names, ["body", "leg", "leg 2"])


class SeparateInTheQuestionAndReplyTest(unittest.TestCase):
    def test_guided_question_asks_for_separate_and_the_reply_keeps_only_parts(self):
        question = build_shortlist_question(["leg", "arm", "torso"], guide_colours=5)
        self.assertIn('"separate"', question)
        reply = ('{"object": "astronaut", "main": "torso", "parts": ["leg", "arm"], '
                 '"separate": ["leg", "torso", "wing"]}')
        parsed = parse_reply(reply, ["leg", "arm", "torso"])
        self.assertEqual(parsed["separate"], ["leg"])
        self.assertEqual(parse_reply('{"object":"x","main":"torso","parts":["leg"]}',
                                     ["leg", "torso"])["separate"], [])


if __name__ == "__main__":
    unittest.main()
