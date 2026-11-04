import unittest

import numpy as np
from visual_align.composite import composite_locked_frame
from visual_align.geometry import apply_affine_2d, estimate_similarity_2d
from visual_align.masks import make_masks


class AlignmentContracts(unittest.TestCase):
    def test_gloves_and_unedited_background_survive_overlapping_mesh(self):
        raw = np.full((12, 16, 3), 60, dtype=np.uint8)
        background = np.full_like(raw, 200)
        hole = np.zeros(raw.shape[:2], dtype=bool)
        hole[3:10, 4:12] = True
        gloves = np.zeros_like(hole)
        gloves[5:8, 7:9] = True
        alpha = np.zeros((2, *hole.shape), dtype=np.float32)
        alpha[0, 5:10, 6:12] = 0.5
        mesh = np.zeros_like(raw)
        mesh[5:10, 6:12] = 20
        result, _ = composite_locked_frame(raw, background, hole, gloves, mesh, alpha)
        np.testing.assert_array_equal(result[gloves], raw[gloves])
        np.testing.assert_array_equal(result[~hole], raw[~hole])
        np.testing.assert_array_equal(result[3, 4], [200, 200, 200])
        np.testing.assert_array_equal(result[8, 6], [120, 120, 120])

    def test_cuff_margin_covers_device_even_when_tracker_prediction_is_empty(self):
        arm = np.zeros((160, 160), dtype=bool)
        glove = np.zeros_like(arm)
        arm[40:150, 60:100] = True
        glove[40:80, 60:100] = True
        generator, removal = make_masks(arm, glove, np.zeros_like(arm), 8, 32)
        self.assertTrue(generator[90, 120])
        self.assertFalse(generator[10, 10])
        self.assertFalse(removal[glove].any())

    def test_similarity_recovers_rotation_scale_and_translation(self):
        source = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        expected = np.array([[0.0, -1.5, 10.0], [1.5, 0.0, 20.0]])
        target = apply_affine_2d(source, expected)
        actual = estimate_similarity_2d(source[:2], target[:2])
        np.testing.assert_allclose(apply_affine_2d(source, actual), target, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
