import unittest

import numpy as np

from yukarin_autoreg.dataset import decode_16bit
from yukarin_autoreg.dataset import encode_16bit
from yukarin_autoreg.dataset import normalize
from yukarin_autoreg.dataset import denormalize
from yukarin_autoreg.dataset import WavesDataset
from yukarin_autoreg.wave import Wave

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

    def test_encode(self):
        coarse, fine = encode_16bit(self.wave_increase)
        w = decode_16bit(coarse, fine)

        np.testing.assert_allclose(self.wave_increase, w, atol=1e-4)


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.wave_increase = np.linspace(-1, 1, num=256 ** 2).astype(np.float32)

    def test_init(self):
        pass

    def test_encode(self):
        w = denormalize(normalize(self.wave_increase))
        np.testing.assert_allclose(self.wave_increase, w, atol=1e-4)


class TestWavesDataset(unittest.TestCase):
    num = 256
    sampling_length = 16

    def setUp(self):
        waves = [
            np.linspace(-1, 0, num=self.num // 2, endpoint=False),
            np.linspace(0, 1, num=self.num // 2, endpoint=False),
        ]
        self.dataset = WavesDataset(
            waves,
            sampling_length=self.sampling_length,
            top_db=None,
            clipping_range=None,
        )

    def test_init(self):
        pass

    def test_get(self):
        for i in range(self.num // self.sampling_length - 1):
            self.assertEqual(len(self.dataset[i]['input_coarse']), self.sampling_length)
            self.assertEqual(len(self.dataset[i]['input_fine']), self.sampling_length - 1)
            self.assertEqual(len(self.dataset[i]['target_coarse']), self.sampling_length - 1)
            self.assertEqual(len(self.dataset[i]['target_fine']), self.sampling_length - 1)

        for i in range(self.num // self.sampling_length - 2):
            self.assertLess(self.dataset[i]['input_coarse'].sum(), self.dataset[i + 1]['input_coarse'].sum())

    def test_clip(self):
        dataset = WavesDataset(
            [np.linspace(0, 0.5, self.num // 4) for _ in range(4)],
            sampling_length=self.sampling_length,
            top_db=None,
            clipping_range=None,
        )
        self.assertEqual(dataset.wave.min(), 0.0)
        self.assertEqual(dataset.wave.max(), 0.5)

        dataset = WavesDataset(
            [np.linspace(0, 0.5, self.num // 4) for _ in range(4)],
            sampling_length=self.sampling_length,
            top_db=None,
            clipping_range=(0, 0.5),
        )
        self.assertEqual(dataset.wave.min(), 0.0)
        self.assertEqual(dataset.wave.max(), 1.0)


if __name__ == '__main__':
    unittest.main()
