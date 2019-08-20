import unittest

import numpy as np

from yukarin_autoreg.network.univ_wave_rnn import UnivWaveRNN

batch_size = 2
length = 3
hidden_size = 8
loal_size = 5
bit_size = 9


def _make_univ_wave_rnn():
    wave_rnn = UnivWaveRNN(
        dual_softmax=False,
        bit_size=bit_size,
        conditioning_size=7,
        embedding_size=32,
        hidden_size=hidden_size,
        linear_hidden_size=11,
        local_size=loal_size,
        local_scale=1,
    )

    # set 'b'
    for p in wave_rnn.params():
        if 'b' not in p.name:
            continue
        p.data = np.random.rand(*p.shape).astype(p.dtype)

    return wave_rnn


def _make_hidden():
    hidden = np.random.rand(batch_size, hidden_size).astype(np.float32)
    return hidden


class TestUnivWaveRNN(unittest.TestCase):
    def setUp(self):
        self.x_array = np.random.randint(0, bit_size ** 2, size=[batch_size, length]).astype(np.int32)
        self.l_array = np.random.rand(batch_size, length, loal_size).astype(np.float32)
        self.x_one = np.random.randint(0, bit_size ** 2, size=[batch_size, 1]).astype(np.int32)
        self.l_one = np.random.rand(batch_size, 1, loal_size).astype(np.float32)

    def test_make_wave_rnn(self):
        _make_univ_wave_rnn()

    def test_call(self):
        wave_rnn = _make_univ_wave_rnn()
        wave_rnn(
            x_array=self.x_array,
            l_array=self.l_array,
        )

    def test_forward_one(self):
        wave_rnn = _make_univ_wave_rnn()
        hidden = _make_hidden()
        l_one = wave_rnn.forward_encode(self.l_one).data
        wave_rnn.forward_one(
            self.x_one[:, 0],
            l_one[:, 0],
            hidden=hidden,
        )

    def test_batchsize1_forward(self):
        wave_rnn = _make_univ_wave_rnn()
        hidden = _make_hidden()
        l_array = wave_rnn.forward_encode(self.l_array).data

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array,
            l_array=l_array,
            hidden=hidden,
        )

        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:1],
            l_array=l_array[:1],
            hidden=hidden[:1],
        )

        np.testing.assert_allclose(oa.data[:1], ob.data, atol=1e-6)
        np.testing.assert_allclose(ha.data[:1], hb.data, atol=1e-6)

    def test_batchsize1_forward_one(self):
        wave_rnn = _make_univ_wave_rnn()
        hidden = _make_hidden()
        l_one = wave_rnn.forward_encode(self.l_one).data

        oa, ha = wave_rnn.forward_one(
            self.x_one[:, 0],
            l_one[:, 0],
            hidden=hidden,
        )

        ob, hb = wave_rnn.forward_one(
            self.x_one[:1, 0],
            l_one[:1, 0],
            hidden=hidden[:1],
        )

        np.testing.assert_allclose(oa.data[:1], ob.data, atol=1e-6)
        np.testing.assert_allclose(ha.data[:1], hb.data, atol=1e-6)

    def test_same_call_and_forward(self):
        wave_rnn = _make_univ_wave_rnn()
        hidden = _make_hidden()

        oa, ha = wave_rnn(
            x_array=self.x_array,
            l_array=self.l_array,
            hidden=hidden,
        )

        l_array = wave_rnn.forward_encode(self.l_array).data
        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:, :-1],
            l_array=l_array[:, 1:],
            hidden=hidden,
        )

        np.testing.assert_equal(oa.data, ob.data)
        np.testing.assert_equal(ha.data, hb.data)

    def test_same_forward_rnn_and_forward_one(self):
        wave_rnn = _make_univ_wave_rnn()
        hidden = _make_hidden()
        l_array = wave_rnn.forward_encode(self.l_array).data

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array,
            l_array=l_array,
            hidden=hidden,
        )

        hb = hidden
        for i, (x, l) in enumerate(zip(
                np.split(self.x_array, length, axis=1),
                np.split(l_array, length, axis=1),
        )):
            ob, hb = wave_rnn.forward_one(
                x[:, 0],
                l[:, 0],
                hb,
            )

            np.testing.assert_equal(oa[:, :, i].data, ob.data)

        np.testing.assert_equal(ha.data, hb.data)
