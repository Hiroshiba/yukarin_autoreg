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
        self.curr_c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.c_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.f_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.hidden_coarse = np.random.rand(batch_size, hidden_size // 2).astype(np.float32)
        self.hidden_fine = np.random.rand(batch_size, hidden_size // 2).astype(np.float32)

    def test_init(self):
        pass

    def test_call(self):
        self.wave_rnn(
            c_array=self.c_array,
            f_array=self.f_array[:, :-1],
        )

    def test_forward_one(self):
        self.wave_rnn.forward_one(
            self.c_one[:, 0],
            self.f_one[:, 0],
            hidden_coarse=self.hidden_coarse,
            hidden_fine=self.hidden_fine,
        )

    def test_same_forward(self):
        oca, ofa, hca, hfa = self.wave_rnn.forward(
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
            ocb, ofb, hcb, hfb = self.wave_rnn.forward_one(c[:, 0], f[:, 0], curr_c[:, 0], hcb, hfb)

            np.testing.assert_equal(oca[:, :, i].data, ocb.data)
            np.testing.assert_equal(ofa[:, :, i].data, ofb.data)

        np.testing.assert_equal(hca.data, hcb.data)
        np.testing.assert_equal(hfa.data, hfb.data)


if __name__ == '__main__':
    unittest.main()
