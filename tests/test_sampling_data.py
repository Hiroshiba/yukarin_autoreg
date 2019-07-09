import unittest

import numpy as np

from yukarin_autoreg.sampling_data import SamplingData


class TestSamplingData(unittest.TestCase):
    def test_resample_twice(self):
        sample100 = np.random.rand(100, 1)
        sample100_rate100 = SamplingData(array=sample100, rate=100)

        a = sample100_rate100.resample(sampling_rate=200, index=0, length=200)
        b = np.repeat(sample100, 2, axis=0)
        np.testing.assert_equal(a, b)

    def test_resample_half(self):
        sample100 = np.random.rand(100, 1)
        sample100_rate100 = SamplingData(array=sample100, rate=100)

        a = sample100_rate100.resample(sampling_rate=200, index=50, length=100)
        b = np.repeat(sample100, 2, axis=0)[50:150]
        np.testing.assert_equal(a, b)

    def test_resample_random(self):
        for _ in range(1000):
            num = np.random.randint(256 ** 2) + 1
            size = np.random.randint(5) + 1
            rate = np.random.randint(100) + 1
            scale = np.random.randint(100) + 1
            index = np.random.randint(num)
            length = np.random.randint(num - index)

            sample = np.random.rand(num, size)
            data = SamplingData(array=sample, rate=rate)

            a = data.resample(sampling_rate=rate * scale, index=index, length=length)
            b = np.repeat(sample, scale, axis=0)[index:index + length]
            np.testing.assert_equal(a, b)
