import unittest

import numpy as np

from yukarin_autoreg.dataset import BaseWaveDataset, Input, WavesDataset, decode_16bit, decode_single, encode_16bit, \
    encode_single, normalize
from yukarin_autoreg.sampling_data import SamplingData
from yukarin_autoreg.wave import Wave

batch_size = 2
length = 3
hidden_size = 8


class TestEncode16Bit(unittest.TestCase):
    def setUp(self):
        self.wave_increase = np.linspace(-1, 1, num=2 ** 16).astype(np.float32)

    def test_encode(self):
        coarse, fine = encode_16bit(self.wave_increase)

        self.assertTrue(np.all(coarse >= 0))
        self.assertTrue(np.all(coarse < 256))
        self.assertTrue(np.all(fine >= 0))
        self.assertTrue(np.all(fine < 256))

        self.assertTrue(np.all(np.diff(coarse) >= 0))
        self.assertTrue((np.diff(fine) >= 0).sum() == 256 ** 2 - 255 - 1)


class TestEncodeSingle(unittest.TestCase):
    def setUp(self):
        self.wave_increase = np.linspace(-1, 1, num=2 ** 16).astype(np.float64)

    def test_encode_single(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                coarse = encode_single(self.wave_increase, bit=bit)

                self.assertTrue(np.all(coarse >= 0))
                self.assertTrue(np.all(coarse < 2 ** bit))

                self.assertTrue(np.all(np.diff(coarse) >= 0))

                hist, _ = np.histogram(coarse, bins=2 ** bit, range=[0, 2 ** bit])
                np.testing.assert_equal(hist, np.ones(2 ** bit) * (2 ** 16 / 2 ** bit))

    def test_encode_single_float32(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                coarse = encode_single(self.wave_increase.astype(np.float32), bit=bit)

                self.assertTrue(np.all(coarse >= 0))
                self.assertTrue(np.all(coarse < 2 ** bit))

                self.assertTrue(np.all(np.diff(coarse) >= 0))

                hist, _ = np.histogram(coarse, bins=2 ** bit, range=[0, 2 ** bit])
                all_equal = np.all(hist == np.ones(2 ** bit) * (2 ** 16 / 2 ** bit))
                if bit <= 10:
                    self.assertTrue(all_equal)
                else:
                    self.assertFalse(all_equal)


class TestDecode16Bit(unittest.TestCase):
    def setUp(self):
        self.wave_increase = np.linspace(-1, 1, num=2 ** 17).astype(np.float32)

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

        np.testing.assert_allclose(self.wave_increase, w, atol=2 ** -15)


class TestDecodeSingle(unittest.TestCase):
    def test_decode(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                coarse = np.arange(2 ** bit).astype(np.int32)
                w = decode_single(coarse, bit=bit)
                np.testing.assert_equal(w, np.linspace(-1, 1, num=2 ** bit).astype(np.float32))


class TestNormalize(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(normalize(0), -1)
        self.assertEqual(normalize(255), 1)


class TestBaseWaveDataset(unittest.TestCase):
    sampling_length = 16
    sampling_rate = 8000
    local_sampling_rate = 2000
    scale = sampling_rate // local_sampling_rate

    def setUp(self):
        self.wave = Wave(wave=np.arange(self.sampling_rate * 10), sampling_rate=self.sampling_rate)
        self.silence = SamplingData(array=np.zeros((self.sampling_rate * 10,), dtype=bool), rate=self.sampling_rate)
        self.local = SamplingData(array=np.arange(self.local_sampling_rate * 10), rate=self.local_sampling_rate)

    def test_extract_input(self):
        wave, silence, local = BaseWaveDataset.extract_input(
            self.sampling_length,
            wave_data=self.wave,
            silence_data=self.silence,
            local_data=self.local,
        )

        self.assertEqual(len(wave), self.sampling_length)
        self.assertEqual(len(silence), self.sampling_length)
        self.assertEqual(len(local), self.sampling_length // self.scale)

        self.assertEqual(wave[0] % self.scale, 0)

    def test_convert_to_dict(self):
        wave, silence, local = BaseWaveDataset.extract_input(
            self.sampling_length,
            wave_data=self.wave,
            silence_data=self.silence,
            local_data=self.local,
        )

        dataset = BaseWaveDataset(sampling_length=self.sampling_length, to_double=True, bit=16)
        d = dataset.convert_to_dict(wave, silence, local)
        self.assertEqual(len(d['input_coarse']), self.sampling_length)
        self.assertEqual(len(d['input_fine']), self.sampling_length - 1)
        self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
        self.assertEqual(len(d['target_fine']), self.sampling_length - 1)
        self.assertEqual(len(d['silence']), self.sampling_length - 1)
        self.assertEqual(len(d['local']), self.sampling_length // self.scale)

        dataset = BaseWaveDataset(sampling_length=self.sampling_length, to_double=False, bit=10)
        d = dataset.convert_to_dict(wave, silence, local)
        self.assertEqual(len(d['input_coarse']), self.sampling_length)
        self.assertIsNone(d['input_fine'])
        self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
        self.assertIsNone(d['target_fine'])
        self.assertEqual(len(d['silence']), self.sampling_length - 1)
        self.assertEqual(len(d['local']), self.sampling_length // self.scale)


class TestWavesDataset(unittest.TestCase):
    num = 256
    sampling_length = 16
    sampling_rate = 8000

    def setUp(self):
        waves = [
            np.ones(self.num // 2) * -1,
            np.ones(self.num // 2),
        ]
        self.inputs = [
            Input(
                wave=Wave(wave=w, sampling_rate=self.sampling_rate),
                local=SamplingData(array=np.empty((len(w), 0)), rate=self.sampling_rate),
                silence=SamplingData(array=np.zeros((len(w),), dtype=bool), rate=self.sampling_rate),
            )
            for w in waves
        ]

    def test_get_double(self):
        dataset = WavesDataset(
            self.inputs,
            sampling_length=self.sampling_length,
            gaussian_noise_sigma=0,
            to_double=True,
            bit=16,
        )
        for d in dataset:
            self.assertEqual(len(d['input_coarse']), self.sampling_length)
            self.assertEqual(len(d['input_fine']), self.sampling_length - 1)
            self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
            self.assertEqual(len(d['target_fine']), self.sampling_length - 1)

        self.assertTrue(np.all(dataset[0]['input_coarse'] == -1))
        self.assertTrue(np.all(dataset[1]['input_coarse'] == 1))

    def test_get_single(self):
        dataset = WavesDataset(
            self.inputs,
            sampling_length=self.sampling_length,
            gaussian_noise_sigma=0,
            to_double=False,
            bit=10,
        )
        for d in dataset:
            self.assertEqual(len(d['input_coarse']), self.sampling_length)
            self.assertIsNone(d['input_fine'])
            self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
            self.assertIsNone(d['target_fine'])

        self.assertTrue(np.all(dataset[0]['input_coarse'] == -1))
        self.assertTrue(np.all(dataset[1]['input_coarse'] == 1))
