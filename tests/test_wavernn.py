import unittest

import numpy as np
from parameterized import parameterized_class

from yukarin_autoreg.network.wave_rnn import WaveRNN
from yukarin_autoreg.utility.chainer_initializer_utility import get_weight_initializer

batch_size = 2
length = 3
hidden_size = 8
loal_size = 5
bit_size = 9
speaker_size = 10


def _make_hidden():
    hidden = np.random.rand(batch_size, hidden_size).astype(np.float32)
    return hidden


@parameterized_class(('gaussian', 'input_categorical', 'with_speaker', 'weight_initializer'), [
    (False, True, False, None),
    (True, False, False, None),
    (False, True, True, None),
    (True, False, True, None),
    (False, True, False, 'GlorotNormal'),
    (False, True, False, 'HeNormal'),
    (False, True, False, 'LeCunNormal'),
    (False, True, False, 'Normal'),
    (False, True, False, 'GlorotUniform'),
    (False, True, False, 'HeUniform'),
    (False, True, False, 'LeCunUniform'),
    (False, True, False, 'Uniform'),
    (False, True, False, 'PossibleOrthogonal'),
    (False, True, True, 'GlorotNormal'),
    (False, True, True, 'HeNormal'),
    (False, True, True, 'LeCunNormal'),
    (False, True, True, 'Normal'),
    (False, True, True, 'GlorotUniform'),
    (False, True, True, 'HeUniform'),
    (False, True, True, 'LeCunUniform'),
    (False, True, True, 'Uniform'),
    (False, True, True, 'PossibleOrthogonal'),
])
class TestWaveRNN(unittest.TestCase):
    def setUp(self):
        gaussian = self.gaussian
        input_categorical = self.input_categorical
        with_speaker = self.with_speaker
        weight_initializer = self.weight_initializer

        if input_categorical:
            self.x_array = np.random.randint(0, bit_size ** 2, size=[batch_size, length]).astype(np.int32)
            self.x_one = np.random.randint(0, bit_size ** 2, size=[batch_size, 1]).astype(np.int32)
        else:
            self.x_array = np.random.rand(batch_size, length).astype(np.float32)
            self.x_one = np.random.rand(batch_size, 1).astype(np.float32)

        if with_speaker:
            self.s_one = np.random.randint(0, speaker_size, size=[batch_size, ]).astype(np.int32)
        else:
            self.s_one = None

        wave_rnn = WaveRNN(
            dual_softmax=False,
            bit_size=bit_size,
            gaussian=gaussian,
            input_categorical=input_categorical,
            conditioning_size=7,
            embedding_size=32,
            hidden_size=hidden_size,
            linear_hidden_size=11,
            local_size=loal_size,
            local_scale=1,
            local_layer_num=2,
            speaker_size=speaker_size if with_speaker else 0,
            speaker_embedding_size=3 if with_speaker else 0,
            weight_initializer=get_weight_initializer(weight_initializer),
        )

        # set 'b'
        for p in wave_rnn.params():
            if 'b' not in p.name:
                continue
            p.data = np.random.rand(*p.shape).astype(p.dtype)

        self.wave_rnn = wave_rnn

    def test_call(self):
        wave_rnn = self.wave_rnn
        wave_rnn(
            x_array=self.x_array,
            l_array=self.l_array,
            s_one=self.s_one,
        )

    def test_call_with_local_padding(self):
        local_padding_size = 5

        wave_rnn = self.wave_rnn
        with self.assertRaises(Exception):
            wave_rnn(
                x_array=self.x_array,
                l_array=self.l_array,
                s_one=self.s_one,
                local_padding_size=local_padding_size,
            )

        l_array = np.pad(
            self.l_array,
            pad_width=((0, 0), (local_padding_size, local_padding_size), (0, 0)),
            mode='constant',
        )
        wave_rnn(
            x_array=self.x_array,
            l_array=l_array,
            s_one=self.s_one,
            local_padding_size=local_padding_size,
        )

    def test_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=self.s_one).data
        wave_rnn.forward_one(
            self.x_one[:, 0],
            l_one[:, 0],
            hidden=hidden,
        )

    def test_batchsize1_forward(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=self.s_one).data

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
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=self.s_one).data

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
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()

        oa, ha = wave_rnn(
            x_array=self.x_array,
            l_array=self.l_array,
            s_one=self.s_one,
            hidden=hidden,
        )

        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=self.s_one).data
        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:, :-1],
            l_array=l_array[:, 1:],
            hidden=hidden,
        )

        np.testing.assert_equal(oa.data, ob.data)
        np.testing.assert_equal(ha.data, hb.data)

    def test_same_forward_rnn_and_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=self.s_one).data

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
