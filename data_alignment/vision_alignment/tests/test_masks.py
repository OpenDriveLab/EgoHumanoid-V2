import unittest

import numpy as np
from visual_align.masks import make_masks
from visual_align.segment import binary_union


class MaskContracts(unittest.TestCase):
    def test_gloves_are_removed_for_generation_but_preserved_in_composite(self):
        arm = np.zeros((24, 24), bool)
        arm[10:24, 9:14] = True
        glove = np.zeros_like(arm)
        glove[5:12, 8:16] = True
        tracker = np.zeros_like(arm)
        tracker[12:15, 13:17] = True
        generation, removal = make_masks(arm, glove, tracker, 2)
        self.assertTrue(generation[glove].all())
        self.assertFalse(removal[glove].any())
        self.assertTrue(removal[(arm | tracker) & ~glove].all())
        self.assertFalse(generation[:2].any())

    def test_multiple_instances_and_empty_prediction(self):
        masks = np.zeros((2, 8, 8), bool)
        masks[0, 1:3, 1:3] = True
        masks[1, 5:7, 5:7] = True
        result = binary_union({"out_binary_masks": masks}, 8, 8)
        self.assertEqual(np.count_nonzero(result), 8)
        empty = binary_union({"out_binary_masks": np.empty((0, 8, 8))}, 8, 8)
        self.assertEqual(empty.shape, (8, 8))
        self.assertFalse(empty.any())

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            make_masks(
                np.zeros((2, 2), bool),
                np.zeros((3, 3), bool),
                np.zeros((2, 2), bool),
                0,
            )
        with self.assertRaises(ValueError):
            make_masks(*(np.zeros((2, 2), bool) for _ in range(3)), -1)


if __name__ == "__main__":
    unittest.main()
