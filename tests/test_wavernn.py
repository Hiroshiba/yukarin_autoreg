import unittest

import numpy as np

from yukarin_autoreg.network import WaveRNN

batch_size = 2
length = 3
hidden_size = 8
local_size = 5


def _make_wave_rnn(dual_softmax: bool):
    wave_rnn = WaveRNN(
        upconv_scales=[],
        upconv_residual=False,
        upconv_channel_ksize=0,
        residual_encoder_channel=None,
        residual_encoder_num_block=None,
        dual_softmax=dual_softmax,
        bit_size=16 if dual_softmax else 10,
        hidden_size=hidden_size,
        local_size=local_size,
        bug_fixed_gru_dimension=True,
    )

    # set 'b'
    for p in wave_rnn.params():
        if 'b' not in p.name:
            continue
        p.data = np.random.rand(*p.shape).astype(p.dtype)

    return wave_rnn


def _make_hidden(dual_softmax: bool):
    if dual_softmax:
        hidden_coarse = np.random.rand(batch_size, hidden_size // 2).astype(np.float32)
        hidden_fine = np.random.rand(batch_size, hidden_size // 2).astype(np.float32)
    else:
        hidden_coarse = np.random.rand(batch_size, hidden_size).astype(np.float32)
        hidden_fine = None
    return hidden_coarse, hidden_fine


class TestWaveRNN(unittest.TestCase):
    def setUp(self):
        self.c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.f_array = np.random.rand(batch_size, length).astype(np.float32)
        self.l_array = np.random.rand(batch_size, length, local_size).astype(np.float32)
        self.curr_c_array = np.random.rand(batch_size, length).astype(np.float32)
        self.c_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.f_one = np.random.rand(batch_size, 1).astype(np.float32)
        self.l_one = np.random.rand(batch_size, 1, local_size).astype(np.float32)
        self.curr_c_one = np.random.rand(batch_size, 1).astype(np.float32)

    def test_make_wave_rnn(self):
        for dual_softmax in [True, False]:
            with self.subTest(dual_softmax=dual_softmax):
                _make_wave_rnn(dual_softmax=dual_softmax)

    def test_call(self):
        for dual_softmax in [True, False]:
            with self.subTest(dual_softmax=dual_softmax):
                wave_rnn = _make_wave_rnn(dual_softmax=dual_softmax)
                wave_rnn(
                    c_array=self.c_array,
                    f_array=self.f_array[:, :-1] if dual_softmax else None,
                    l_array=self.l_array,
                )

    def test_forward_one(self):
        for dual_softmax in [True, False]:
            with self.subTest(dual_softmax=dual_softmax):
                wave_rnn = _make_wave_rnn(dual_softmax=dual_softmax)
                hidden_coarse, hidden_fine = _make_hidden(dual_softmax=dual_softmax)
                wave_rnn.forward_one(
                    self.c_one[:, 0],
                    self.f_one[:, 0] if dual_softmax else None,
                    self.l_one[:, 0],
                    hidden_coarse=hidden_coarse,
                    hidden_fine=hidden_fine if dual_softmax else None,
                )

    def test_batchsize1_forward(self):
        for dual_softmax in [True, False]:
            with self.subTest(dual_softmax=dual_softmax):
                wave_rnn = _make_wave_rnn(dual_softmax=dual_softmax)
                hidden_coarse, hidden_fine = _make_hidden(dual_softmax=dual_softmax)

                oca, ofa, hca, hfa = wave_rnn.forward_rnn(
                    c_array=self.c_array,
                    f_array=self.f_array if dual_softmax else None,
                    l_array=self.l_array,
                    curr_c_array=self.curr_c_array if dual_softmax else None,
                    hidden_coarse=hidden_coarse,
                    hidden_fine=hidden_fine if dual_softmax else None,
                )

                ocb, ofb, hcb, hfb = wave_rnn.forward_rnn(
                    c_array=self.c_array[:1],
                    f_array=self.f_array[:1] if dual_softmax else None,
                    l_array=self.l_array[:1],
                    curr_c_array=self.curr_c_array[:1] if dual_softmax else None,
                    hidden_coarse=hidden_coarse[:1],
                    hidden_fine=hidden_fine[:1] if dual_softmax else None,
                )

                np.testing.assert_allclose(oca.data[:1], ocb.data, atol=1e-6)
                if dual_softmax:
                    np.testing.assert_allclose(ofa.data[:1], ofb.data, atol=1e-6)

                np.testing.assert_allclose(hca.data[:1], hcb.data, atol=1e-6)
                if dual_softmax:
                    np.testing.assert_allclose(hfa.data[:1], hfb.data, atol=1e-6)

    def test_batchsize1_forward_one(self):
        for dual_softmax in [True, False]:
            with self.subTest(dual_softmax=dual_softmax):
                wave_rnn = _make_wave_rnn(dual_softmax=dual_softmax)
                hidden_coarse, hidden_fine = _make_hidden(dual_softmax=dual_softmax)

                oca, ofa, hca, hfa = wave_rnn.forward_one(
                    self.c_one[:, 0],
                    self.f_one[:, 0] if dual_softmax else None,
                    self.l_one[:, 0],
                    self.curr_c_one[:, 0] if dual_softmax else None,
                    hidden_coarse=hidden_coarse,
                    hidden_fine=hidden_fine if dual_softmax else None,
                )

                ocb, ofb, hcb, hfb = wave_rnn.forward_one(
                    self.c_one[:1, 0],
                    self.f_one[:1, 0] if dual_softmax else None,
                    self.l_one[:1, 0],
                    self.curr_c_one[:1, 0] if dual_softmax else None,
                    hidden_coarse=hidden_coarse[:1],
                    hidden_fine=hidden_fine[:1] if dual_softmax else None,
                )

                np.testing.assert_allclose(oca.data[:1], ocb.data, atol=1e-6)
                if dual_softmax:
                    np.testing.assert_allclose(ofa.data[:1], ofb.data, atol=1e-6)

                np.testing.assert_allclose(hca.data[:1], hcb.data, atol=1e-6)
                if dual_softmax:
                    np.testing.assert_allclose(hfa.data[:1], hfb.data, atol=1e-6)

    def test_same_forward(self):
        for dual_softmax in [True, False]:
            with self.subTest(dual_softmax=dual_softmax):
                wave_rnn = _make_wave_rnn(dual_softmax=dual_softmax)
                hidden_coarse, hidden_fine = _make_hidden(dual_softmax=dual_softmax)

                oca, ofa, hca, hfa = wave_rnn.forward_rnn(
                    c_array=self.c_array,
                    f_array=self.f_array if dual_softmax else None,
                    l_array=self.l_array,
                    curr_c_array=self.curr_c_array if dual_softmax else None,
                    hidden_coarse=hidden_coarse,
                    hidden_fine=hidden_fine if dual_softmax else None,
                )

                hcb, hfb = hidden_coarse, hidden_fine
                for i, (c, f, l, curr_c) in enumerate(zip(
                        np.split(self.c_array, length, axis=1),
                        np.split(self.f_array, length, axis=1),
                        np.split(self.l_array, length, axis=1),
                        np.split(self.curr_c_array, length, axis=1),
                )):
                    ocb, ofb, hcb, hfb = wave_rnn.forward_one(
                        c[:, 0],
                        f[:, 0] if dual_softmax else None,
                        l[:, 0],
                        curr_c[:, 0] if dual_softmax else None,
                        hcb,
                        hfb if dual_softmax else None,
                    )

                    np.testing.assert_equal(oca[:, :, i].data, ocb.data)
                    if dual_softmax:
                        np.testing.assert_equal(ofa[:, :, i].data, ofb.data)

                np.testing.assert_equal(hca.data, hcb.data)
                if dual_softmax:
                    np.testing.assert_equal(hfa.data, hfb.data)
