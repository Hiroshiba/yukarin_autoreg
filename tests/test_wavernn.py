import unittest

import numpy as np

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.model import WaveRNN

batch_size = 2
length = 3
hidden_size = 8

config = ModelConfig(
    hidden_size=hidden_size,
    bit_size=16,
)


class TestWaveRNN(unittest.TestCase):
    def setUp(self):
        self.wave_rnn = WaveRNN(config=config)
        self.c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.f_array = np.random.rand(batch_size, length).astype(np.float32)

    def test_init(self):
        pass

    def test_call(self):
        self.wave_rnn(
            c_array=self.c_array,
            f_array=self.f_array[:, :-1],
        )


if __name__ == '__main__':
    unittest.main()
