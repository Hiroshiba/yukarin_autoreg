import unittest

import numpy as np

from yukarin_autoreg.model import MaskGRU

batch_size = 2
length = 3
hidden_size = 8


class TestMaskGRU(unittest.TestCase):
    def setUp(self):
        self.mask_gru = MaskGRU(out_size=hidden_size)
        self.c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.f_array = np.random.rand(batch_size, length).astype(np.float32)
        self.curr_c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.c_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.f_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.curr_c_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.hidden_coarse = np.random.rand(batch_size, hidden_size // 2).astype(np.float32)
        self.hidden_fine = np.random.rand(batch_size, hidden_size // 2).astype(np.float32)

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

    def test_onestep_coarse(self):
        ca, _ = self.mask_gru(
            self.c_one,
            self.f_one,
            self.curr_c_one,
            hidden_coarse=self.hidden_coarse,
            hidden_fine=self.hidden_fine,
        )

        cb = self.mask_gru.onestep_coarse(
            self.c_one,
            self.f_one,
            hidden_coarse=self.hidden_coarse,
            hidden_fine=self.hidden_fine,
        )

        np.testing.assert_equal(ca[:, 0, :].data, cb.data)

    def test_onestep_fine(self):
        _, fa = self.mask_gru(
            self.c_one,
            self.f_one,
            self.curr_c_one,
            hidden_coarse=self.hidden_coarse,
            hidden_fine=self.hidden_fine,
        )

        fb = self.mask_gru.onestep_fine(
            self.c_one,
            self.f_one,
            self.curr_c_one,
            hidden_coarse=self.hidden_coarse,
            hidden_fine=self.hidden_fine,
        )

        np.testing.assert_equal(fa[:, 0, :].data, fb.data)

    def test_same_forward(self):
        hca, hfa = self.mask_gru(
            c_array=self.c_array,
            f_array=self.f_array,
            curr_c_array=self.curr_c_array,
            hidden_coarse=self.hidden_coarse,
            hidden_fine=self.hidden_fine,
        )

        hcb, hfb = self.hidden_coarse, self.hidden_fine
        for i, (c, f, curr_c) in enumerate(zip(
                np.split(self.c_array, length, axis=1),
                np.split(self.f_array, length, axis=1),
                np.split(self.curr_c_array, length, axis=1),
        )):
            bef_hcb = hcb
            hcb = self.mask_gru.onestep_coarse(c[:, 0], f[:, 0], hcb, hfb)
            hfb = self.mask_gru.onestep_fine(c[:, 0], f[:, 0], curr_c[:, 0], bef_hcb, hfb)

            np.testing.assert_equal(hca[:, i, :].data, hcb.data)
            np.testing.assert_equal(hfa[:, i, :].data, hfb.data)


if __name__ == '__main__':
    unittest.main()
