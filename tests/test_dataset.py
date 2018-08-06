import unittest

import numpy as np

from yukarin_autoreg.dataset import WavesDataset, decode_16bit, encode_16bit, normalize, Input

batch_size = 2
length = 3
hidden_size = 8


class TestEncode16Bit(unittest.TestCase):
    def setUp(self):
        self.wave_increase = np.linspace(-1, 1, num=256 ** 2).astype(np.float32)

    def test_init(self):
        pass

    def test_encode(self):
        coarse, fine = encode_16bit(self.wave_increase)

        self.assertTrue(np.all(coarse >= 0))
        self.assertTrue(np.all(coarse < 256))
        self.assertTrue(np.all(fine >= 0))
        self.assertTrue(np.all(fine < 256))

        self.assertTrue(np.all(np.diff(coarse) >= 0))
        self.assertTrue((np.diff(fine) >= 0).sum() == 256 ** 2 - 255 - 1)


class TestDecode16Bit(unittest.TestCase):
    def setUp(self):
        self.wave_increase = np.linspace(-1, 1, num=256 ** 2).astype(np.float32)

    def test_init(self):
        pass

    def test_decode(self):
        c, f = np.meshgrid(np.arange(256), np.arange(256))
        w = decode_16bit(c.ravel(), f.ravel())

        self.assertTrue(np.all(-1 <= w))
        self.assertTrue(np.all(w <= 1))

        self.assertEqual((w < 0).sum(), 2 ** 15)
        self.assertEqual((w >= 0).sum(), 2 ** 15)

    def test_encode_decode(self):
        coarse, fine = encode_16bit(self.wave_increase)
        w = decode_16bit(coarse, fine)

        np.testing.assert_allclose(self.wave_increase, w, atol=1e-4)


class TestNormalize(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(normalize(0), -1)
        self.assertEqual(normalize(255), 1)


class TestWavesDataset(unittest.TestCase):
    num = 256
    sampling_length = 16

    def setUp(self):
        waves = [
            np.linspace(-1, 0, num=self.num // 2, endpoint=False),
            np.linspace(0, 1, num=self.num // 2, endpoint=False),
        ]
        inputs = [
            Input(
                wave=w,
                local=np.empty((len(w), 0)),
            )
            for w in waves
        ]
        self.dataset = WavesDataset(
            inputs,
            sampling_length=self.sampling_length,
        )

    def test_init(self):
        pass

    def test_get(self):
        for d in self.dataset:
            self.assertEqual(len(d['input_coarse']), self.sampling_length)
            self.assertEqual(len(d['input_fine']), self.sampling_length - 1)
            self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
            self.assertEqual(len(d['target_fine']), self.sampling_length - 1)

        for _ in range(10):
            self.assertLess(self.dataset[0]['input_coarse'].sum(), self.dataset[1]['input_coarse'].sum())


if __name__ == '__main__':
    unittest.main()
