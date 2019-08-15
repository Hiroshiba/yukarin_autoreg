import unittest
from typing import Optional

import librosa
import numpy as np

from yukarin_autoreg.data import encode_16bit, encode_single, decode_16bit, decode_single, \
    to_log_melspectrogram, encode_mulaw, decode_mulaw


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

    def test_decode_one_value(self):
        self.assertEqual(decode_single(0), -1)
        self.assertEqual(decode_single(255), 1)


class TestMulaw(unittest.TestCase):
    def setUp(self):
        self.dummy_array = np.linspace(-1, 1, num=2 ** 16).astype(np.float32)

    def test_encode_mulaw(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                mu = 2 ** bit
                y = encode_mulaw(self.dummy_array, mu=mu)
                self.assertEqual(y.min(), -1)
                self.assertEqual(y.max(), 1)

    def test_decode_mulaw(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                mu = 2 ** bit
                y = decode_mulaw(self.dummy_array, mu=mu)
                self.assertEqual(y.min(), -1)
                self.assertEqual(y.max(), 1)

    def test_encode_decode_mulaw(self):
        for bit in range(1, 16 + 1):
            with self.subTest(bit=bit):
                mu = 2 ** bit
                x = encode_mulaw(self.dummy_array, mu=mu)
                y = decode_mulaw(x, mu=mu)
                np.testing.assert_allclose(self.dummy_array, y, atol=1 ** -mu)


class TestToLogMelspectrogram(unittest.TestCase):
    @staticmethod
    def _to_log_melspectrogram(
            x: np.ndarray = None,
            sampling_rate: int = 24000,
            preemph: Optional[float] = 0.97,
            n_mels: int = 80,
            n_fft: int = 2048,
            win_length: int = 1024,
            hop_length: int = 256,
            fmin: float = 125,
            fmax: float = 12000,
            min_level_db: float = -100,
            normalize: bool = True,
    ):
        if x is None:
            x, _ = librosa.load(librosa.util.example_audio_file(), sr=sampling_rate)

        return to_log_melspectrogram(
            x=x,
            sampling_rate=sampling_rate,
            preemph=preemph,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            min_level_db=min_level_db,
            normalize=normalize,
        )

    def test_can_convert_to_log_melspectrogram(self):
        self._to_log_melspectrogram()

    def test_can_convert_to_log_melspectrogram_with_preemph(self):
        self._to_log_melspectrogram(preemph=None)
        self._to_log_melspectrogram(preemph=0.97)

    def test_can_convert_to_log_melspectrogram_with_min_level_db(self):
        sp = self._to_log_melspectrogram(min_level_db=-10, normalize=False)
        self.assertGreaterEqual(sp.min(), -10)

        sp = self._to_log_melspectrogram(min_level_db=-100, normalize=False)
        self.assertGreaterEqual(sp.min(), -100)

    def test_can_convert_to_log_melspectrogram_with_normalize(self):
        sp = self._to_log_melspectrogram(normalize=False)
        self.assertGreaterEqual(sp.min(), -100)

        sp = self._to_log_melspectrogram(normalize=True)
        self.assertLessEqual(sp.max(), 1)
        self.assertGreaterEqual(sp.min(), 0)
