import unittest

import numpy as np
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave

from yukarin_autoreg.dataset import BaseWaveDataset, Input, WavesDataset

batch_size = 2
length = 3
hidden_size = 8


class TestBaseWaveDataset(unittest.TestCase):
    def test_extract_input(self):
        for sampling_rate, local_sampling_rate, sampling_length, time_length in [
            [800, 200, 16, 10],
            [24000, 24000 / 256, 1024, 100],
        ]:
            with self.subTest(
                    sampling_rate=sampling_rate,
                    local_sampling_rate=local_sampling_rate,
                    sampling_length=sampling_length,
                    time_length=time_length,
            ):
                scale = sampling_rate // local_sampling_rate

                wave_data = Wave(
                    wave=np.linspace(
                        0,
                        int(sampling_rate * time_length),
                        int(sampling_rate * time_length),
                        endpoint=False,
                    ),
                    sampling_rate=sampling_rate,
                )
                silence_data = SamplingData(
                    array=np.zeros((sampling_rate * time_length,), dtype=bool),
                    rate=sampling_rate,
                )
                local_data = SamplingData(
                    array=np.linspace(
                        0,
                        int(sampling_rate * time_length),
                        int(local_sampling_rate * time_length),
                        endpoint=False,
                    ),
                    rate=local_sampling_rate,
                )

                for _ in range(10):
                    wave, silence, local = BaseWaveDataset.extract_input(
                        sampling_length,
                        wave_data=wave_data,
                        silence_data=silence_data,
                        local_data=local_data,
                        local_padding_size=0,
                    )

                    self.assertEqual(len(wave), sampling_length)
                    self.assertEqual(len(silence), sampling_length)
                    self.assertEqual(len(local), sampling_length // scale)

                    wave_as_local = wave.reshape(int(sampling_length // scale), -1).min(axis=1)
                    self.assertTrue(np.all(wave_as_local == local))

    def test_extract_input_with_local_padding(self):
        for sampling_rate, local_sampling_rate, sampling_length, time_length, local_padding_size in [
            [800, 200, 16, 1, 100],
            [24000, 24000 / 256, 1024, 4, 1024],
        ]:
            with self.subTest(
                    sampling_rate=sampling_rate,
                    local_sampling_rate=local_sampling_rate,
                    sampling_length=sampling_length,
                    time_length=time_length,
                    local_padding_size=local_padding_size,
            ):
                scale = sampling_rate // local_sampling_rate

                wave_data = Wave(
                    wave=np.linspace(
                        0,
                        int(sampling_rate * time_length),
                        int(sampling_rate * time_length),
                        endpoint=False,
                    ),
                    sampling_rate=sampling_rate,
                )
                silence_data = SamplingData(
                    array=np.zeros((sampling_rate * time_length,), dtype=bool),
                    rate=sampling_rate,
                )
                local_data = SamplingData(
                    array=np.linspace(
                        0,
                        int(sampling_rate * time_length),
                        int(local_sampling_rate * time_length),
                        endpoint=False,
                    ),
                    rate=local_sampling_rate,
                )
                for _ in range(10000):
                    wave, silence, local = BaseWaveDataset.extract_input(
                        sampling_length,
                        wave_data=wave_data,
                        silence_data=silence_data,
                        local_data=local_data,
                        local_padding_size=local_padding_size,
                        padding_value=np.nan,
                    )

                    self.assertEqual(len(wave), sampling_length)
                    self.assertEqual(len(silence), sampling_length)
                    self.assertEqual(len(local), (sampling_length + local_padding_size * 2) // scale)

                    num_pad = np.isnan(local).sum()
                    self.assertLessEqual(num_pad, local_padding_size)

                    self.assertTrue(not np.isnan(local[0]) or not np.isnan(local[-1]))

                    wave_as_local = wave.reshape(int(sampling_length // scale), -1).min(axis=1)
                    pad = int(local_padding_size // scale)
                    local_wo_pad = local[pad:-pad]
                    self.assertTrue(np.all(wave_as_local == local_wo_pad))

    def test_convert_to_dict(self):
        sampling_rate = 800
        local_sampling_rate = 200
        scale = sampling_rate // local_sampling_rate
        time_length = 10
        sampling_length = 16

        wave_data = Wave(
            wave=np.linspace(0, sampling_rate * time_length, sampling_rate * time_length, endpoint=False),
            sampling_rate=sampling_rate,
        )
        silence_data = SamplingData(
            array=np.zeros((sampling_rate * time_length,), dtype=bool),
            rate=sampling_rate,
        )
        local_data = SamplingData(
            array=np.linspace(0, sampling_rate * time_length, local_sampling_rate * time_length, endpoint=False),
            rate=local_sampling_rate,
        )

        wave, silence, local = BaseWaveDataset.extract_input(
            sampling_length,
            wave_data=wave_data,
            silence_data=silence_data,
            local_data=local_data,
            local_padding_size=0,
        )

        dataset = BaseWaveDataset(
            sampling_length=sampling_length,
            to_double=True,
            bit=16,
            mulaw=False,
            local_padding_size=0,
        )
        d = dataset.convert_to_dict(wave, silence, local)
        self.assertEqual(len(d['coarse']), sampling_length)
        self.assertEqual(len(d['fine']), sampling_length - 1)
        self.assertEqual(len(d['encoded_coarse']), sampling_length)
        self.assertEqual(len(d['encoded_fine']), sampling_length)
        self.assertEqual(len(d['silence']), sampling_length - 1)
        self.assertEqual(len(d['local']), sampling_length // scale)

        dataset = BaseWaveDataset(
            sampling_length=sampling_length,
            to_double=False,
            bit=10,
            mulaw=False,
            local_padding_size=0,
        )
        d = dataset.convert_to_dict(wave, silence, local)
        self.assertEqual(len(d['coarse']), sampling_length)
        self.assertIsNone(d['fine'])
        self.assertEqual(len(d['encoded_coarse']), sampling_length)
        self.assertIsNone(d['encoded_fine'])
        self.assertEqual(len(d['silence']), sampling_length - 1)
        self.assertEqual(len(d['local']), sampling_length // scale)


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
                    local_padding_size=0,
                )
                for d in dataset:
                    self.assertEqual(len(d['coarse']), self.sampling_length)
                    self.assertEqual(len(d['fine']), self.sampling_length - 1)
                    self.assertEqual(len(d['encoded_coarse']), self.sampling_length)
                    self.assertEqual(len(d['encoded_fine']), self.sampling_length)

                self.assertTrue(np.all(dataset[0]['coarse'] == -1))
                self.assertTrue(np.all(dataset[1]['coarse'] == 1))

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
                    local_padding_size=0,
                )
                for d in dataset:
                    self.assertEqual(len(d['coarse']), self.sampling_length)
                    self.assertIsNone(d['fine'])
                    self.assertEqual(len(d['encoded_coarse']), self.sampling_length)
                    self.assertIsNone(d['encoded_fine'])

                self.assertTrue(np.all(dataset[0]['coarse'] == -1))
                self.assertTrue(np.all(dataset[1]['coarse'] == 1))
