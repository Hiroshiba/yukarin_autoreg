import unittest

import numpy as np

from yukarin_autoreg.dataset import BaseWaveDataset, Input, WavesDataset
from yukarin_autoreg.sampling_data import SamplingData
from yukarin_autoreg.wave import Wave

batch_size = 2
length = 3
hidden_size = 8


class TestBaseWaveDataset(unittest.TestCase):
    sampling_length = 16
    sampling_rate = 8000
    local_sampling_rate = 2000
    scale = sampling_rate // local_sampling_rate

    def setUp(self):
        self.wave = Wave(
            wave=np.linspace(0, self.sampling_rate * 10, self.sampling_rate * 10, endpoint=False),
            sampling_rate=self.sampling_rate,
        )
        self.silence = SamplingData(
            array=np.zeros((self.sampling_rate * 10,), dtype=bool),
            rate=self.sampling_rate,
        )
        self.local = SamplingData(
            array=np.linspace(0, self.sampling_rate * 10, self.local_sampling_rate * 10, endpoint=False),
            rate=self.local_sampling_rate,
        )

    def test_extract_input(self):
        for _ in range(10):
            wave, silence, local = BaseWaveDataset.extract_input(
                self.sampling_length,
                wave_data=self.wave,
                silence_data=self.silence,
                local_data=self.local,
            )

            self.assertEqual(len(wave), self.sampling_length)
            self.assertEqual(len(silence), self.sampling_length)
            self.assertEqual(len(local), self.sampling_length // self.scale)

            wave_as_local = wave.reshape(self.scale, -1).min(axis=1)
            self.assertTrue(np.all(wave_as_local == local))

    def test_convert_to_dict(self):
        wave, silence, local = BaseWaveDataset.extract_input(
            self.sampling_length,
            wave_data=self.wave,
            silence_data=self.silence,
            local_data=self.local,
        )

        dataset = BaseWaveDataset(sampling_length=self.sampling_length, to_double=True, bit=16, mulaw=False)
        d = dataset.convert_to_dict(wave, silence, local)
        self.assertEqual(len(d['input_coarse']), self.sampling_length)
        self.assertEqual(len(d['input_fine']), self.sampling_length - 1)
        self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
        self.assertEqual(len(d['target_fine']), self.sampling_length - 1)
        self.assertEqual(len(d['silence']), self.sampling_length - 1)
        self.assertEqual(len(d['local']), self.sampling_length // self.scale)

        dataset = BaseWaveDataset(sampling_length=self.sampling_length, to_double=False, bit=10, mulaw=False)
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
        for mulaw in [True, False]:
            with self.subTest(mulaw=mulaw):
                dataset = WavesDataset(
                    self.inputs,
                    sampling_length=self.sampling_length,
                    gaussian_noise_sigma=0,
                    to_double=True,
                    bit=16,
                    mulaw=mulaw,
                )
                for d in dataset:
                    self.assertEqual(len(d['input_coarse']), self.sampling_length)
                    self.assertEqual(len(d['input_fine']), self.sampling_length - 1)
                    self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
                    self.assertEqual(len(d['target_fine']), self.sampling_length - 1)

                self.assertTrue(np.all(dataset[0]['input_coarse'] == -1))
                self.assertTrue(np.all(dataset[1]['input_coarse'] == 1))

    def test_get_single(self):
        for mulaw in [True, False]:
            with self.subTest(mulaw=mulaw):
                dataset = WavesDataset(
                    self.inputs,
                    sampling_length=self.sampling_length,
                    gaussian_noise_sigma=0,
                    to_double=False,
                    bit=10,
                    mulaw=mulaw,
                )
                for d in dataset:
                    self.assertEqual(len(d['input_coarse']), self.sampling_length)
                    self.assertIsNone(d['input_fine'])
                    self.assertEqual(len(d['target_coarse']), self.sampling_length - 1)
                    self.assertIsNone(d['target_fine'])

                self.assertTrue(np.all(dataset[0]['input_coarse'] == -1))
                self.assertTrue(np.all(dataset[1]['input_coarse'] == 1))
