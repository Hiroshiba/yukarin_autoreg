import unittest

import numpy as np

from yukarin_autoreg.model import MaskGRU

batch_size = 2
length = 3
hidden_size = 8


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.mask_gru = MaskGRU(out_size=hidden_size)
        self.c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.f_array = np.random.rand(batch_size, length).astype(np.float32)
        self.curr_c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.c_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.f_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.curr_c_one = np.random.rand(batch_size, 1).astype(np.float32)

    def test_init(self):
        pass

    def test_forward(self):
        self.mask_gru(
            self.c_array,
            self.f_array,
            self.curr_c_array,
            hidden_coarse=None,
            hidden_fine=None,
        )

    def test_mask(self):
        dummy = np.zeros_like(self.curr_c_one)

        ca, fa = self.mask_gru(
            self.c_one,
            self.f_one,
            self.curr_c_one,
            hidden_coarse=None,
            hidden_fine=None,
        )

        cb, fb = self.mask_gru(
            self.c_one,
            self.f_one,
            dummy,
            hidden_coarse=None,
            hidden_fine=None,
        )
        self.assertTrue(np.all(ca.data == cb.data))
        self.assertTrue(np.all(fa.data != fb.data))


if __name__ == '__main__':
    unittest.main()
