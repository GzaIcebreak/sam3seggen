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


class ClusteredInstancesTest(unittest.TestCase):
    def test_a_boot_in_two_components_stays_one_boot_and_the_count_caps_the_split(self):
        # two boots, each a sole + an upper that do not share vertices (4 components)
        pieces, labels = [], []
        for x in (-0.6, 0.6):
            upper = trimesh.creation.box(extents=(0.4, 0.4, 0.5)).apply_translation((x, 0, 0.3))
            sole = trimesh.creation.box(extents=(0.5, 0.5, 0.08)).apply_translation((x, 0, 0.0))
            pieces += [upper, sole]
        mesh = trimesh.util.concatenate(pieces)
        labels = np.zeros(len(mesh.faces), dtype=int)
        adjacency = welded_face_adjacency(mesh)
        new_labels, names, report = split_separate_instances(
            labels, ["boot"], adjacency, mesh.area_faces, {"boot": 2},
            centroids=mesh.triangles_center)
        self.assertEqual(names, ["boot", "boot 2"])
        self.assertEqual(report, [("boot", 2)])
        # each boot keeps its sole and upper together: both labels cover 24 faces
        self.assertEqual(sorted(np.bincount(new_labels).tolist()), [24, 24])
        # without a count the proximity clustering alone also gives two boots
        _, names2, _ = split_separate_instances(labels, ["boot"], adjacency, mesh.area_faces,
                                                ["boot"], centroids=mesh.triangles_center)
        self.assertEqual(names2, ["boot", "boot 2"])


class JoinedInstancesTest(unittest.TestCase):
    def test_legs_joined_at_the_crotch_are_cut_in_two_by_position(self):
        # legs much taller than the pair is wide: the longest axis is the wrong one to cut
        left = trimesh.creation.box(extents=(0.4, 0.4, 3.0)).apply_translation((-0.5, 0, 0))
        right = trimesh.creation.box(extents=(0.4, 0.4, 3.0)).apply_translation((0.5, 0, 0))
        bridge = trimesh.creation.box(extents=(1.0, 0.3, 0.3)).apply_translation((0, 0, 1.35))
        mesh = trimesh.util.concatenate([left, right, bridge])
        mesh.merge_vertices()
        labels = np.zeros(len(mesh.faces), dtype=int)
        new_labels, names, report = split_separate_instances(
            labels, ["leg"], welded_face_adjacency(mesh), mesh.area_faces, {"leg": 2},
            centroids=mesh.triangles_center)
        self.assertEqual(names, ["leg", "leg 2"])
        self.assertEqual(report, [("leg", 2)])
        xs = mesh.triangles_center[:, 0]
        self.assertTrue((new_labels[xs < -0.3] == new_labels[xs < -0.3][0]).all())
        self.assertTrue((new_labels[xs > 0.3] == new_labels[xs > 0.3][0]).all())
        self.assertNotEqual(new_labels[xs < -0.3][0], new_labels[xs > 0.3][0])


class SeparateInTheQuestionAndReplyTest(unittest.TestCase):
    def test_guided_question_asks_for_separate_and_the_reply_keeps_only_parts(self):
        question = build_shortlist_question(["leg", "arm", "torso"], guide_colours=5)
        self.assertIn('"colours"', question)
        reply = ('{"object": "astronaut", "main": "torso", "parts": ["leg", "arm"], '
                 '"separate": {"leg": 2, "torso": 1, "wing": 2}}')
        parsed = parse_reply(reply, ["leg", "arm", "torso"])
        self.assertEqual(parsed["separate"], {"leg": 2})
        self.assertEqual(parse_reply('{"object":"x","main":"torso","parts":["leg"], '
                                     '"separate": ["leg"]}', ["leg", "torso"])["separate"], {"leg": 2})
        self.assertEqual(parse_reply('{"object":"x","main":"torso","parts":["leg"]}',
                                     ["leg", "torso"])["separate"], {})
        colours = ('{"object":"astronaut","main":"torso","parts":["leg","arm","boot"], '
                   '"colours": {"leg": 2, "arm": 1, "boot": "2", "torso": 1}}')
        self.assertEqual(parse_reply(colours, ["leg", "arm", "boot", "torso"])["separate"],
                         {"leg": 2, "boot": 2})


if __name__ == "__main__":
    unittest.main()
