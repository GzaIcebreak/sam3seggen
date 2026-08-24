import unittest

import numpy as np
import trimesh

from inference_full import drop_offbody_components


def body_with_bridge_chain(tip=-0.4977, segments=33):
    """Open body patch with the same one-face-wide topology as the observed spike."""
    vertices = [
        [-0.001, -0.15, 0.0],
        [0.001, -0.15, 0.0],
        [0.0, 0.2, 0.2],
        [-0.2, 0.2, -0.5],
        [0.2, 0.2, 0.5],
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [1, 4, 2],
    ]

    edge = [0, 1]
    for y in np.linspace(-0.15, tip, segments + 1)[1:]:
        vertex = len(vertices)
        vertices.append([
            -0.001 if vertex % 2 else 0.001,
            float(y),
            0.0,
        ])
        faces.append([edge[0], edge[1], vertex])
        edge = [edge[1], vertex]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class OffbodyCleanupTest(unittest.TestCase):
    def test_drops_aabb_reaching_single_face_bridge_chain(self):
        spike = body_with_bridge_chain()
        body = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
        mesh = trimesh.util.concatenate([spike, body])
        expected_faces = len(mesh.faces) - 33

        cleaned = drop_offbody_components(mesh)

        self.assertEqual(len(cleaned.faces), expected_faces)
        self.assertGreater(cleaned.bounds[0, 1], -0.3)

    def test_preserves_non_aabb_hair_tip_staff_and_thin_plate(self):
        cases = {
            "hair_tip": body_with_bridge_chain(tip=-0.35, segments=18),
            "staff": trimesh.creation.cylinder(radius=0.02, height=0.7, sections=24),
            "thin_plate": trimesh.creation.box(extents=(0.02, 0.695, 1.0)),
        }
        cases["thin_plate"].apply_translation([0.0, -0.15, 0.0])

        for name, mesh in cases.items():
            with self.subTest(name=name):
                expected_faces = len(mesh.faces)
                cleaned = drop_offbody_components(mesh)
                self.assertEqual(len(cleaned.faces), expected_faces)


if __name__ == "__main__":
    unittest.main()
