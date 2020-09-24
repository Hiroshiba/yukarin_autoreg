import unittest

import cupy as cp
import numpy as np
from parameterized import parameterized_class
from yukarin_autoreg.network.fast_forward import (
    fast_forward_one,
    get_fast_forward_params,
)
from yukarin_autoreg.network.wave_rnn import WaveRNN
from yukarin_autoreg.utility.chainer_initializer_utility import get_weight_initializer

batch_size = 2
length = 3
hidden_size = 8
loal_size = 5
bit_size = 9
speaker_size = 10

xp = cp


def _make_hidden():
    hidden = xp.random.rand(batch_size, hidden_size).astype(xp.float32)
    return hidden


def get_fast_forward_params_one(wave_rnn):
    fast_forward_params = get_fast_forward_params(wave_rnn)
    dtype = fast_forward_params["gru_xb"].dtype
    fast_forward_params["w_gru_x"] = xp.empty(
        (batch_size, len(fast_forward_params["gru_xb"])), dtype=dtype
    )
    fast_forward_params["w_gru_h"] = xp.empty(
        (batch_size, len(fast_forward_params["gru_hb"])), dtype=dtype
    )
    fast_forward_params["w_out_x1"] = xp.empty(
        (batch_size, len(fast_forward_params["O1_b"])), dtype=dtype
    )
    fast_forward_params["w_out_x2"] = xp.empty(
        (batch_size, len(fast_forward_params["O2_b"])), dtype=dtype
    )
    return fast_forward_params


@parameterized_class(
    ("with_speaker", "weight_initializer"),
    [
        (False, None),
        (True, None),
        (False, "GlorotUniform"),
        (False, "PossibleOrthogonal"),
        (True, "GlorotUniform"),
        (True, "PossibleOrthogonal"),
    ],
)
class TestWaveRNN(unittest.TestCase):
    def setUp(self):
        with_speaker = self.with_speaker
        weight_initializer = self.weight_initializer

        self.x_array = xp.random.randint(
            0, bit_size ** 2, size=[batch_size, length]
        ).astype(xp.int32)
        self.x_one = xp.random.randint(0, bit_size ** 2, size=[batch_size, 1]).astype(
            xp.int32
        )

        self.l_array = xp.random.rand(batch_size, length, loal_size).astype(xp.float32)
        self.l_one = xp.random.rand(batch_size, 1, loal_size).astype(xp.float32)

        if with_speaker:
            self.s_one = xp.random.randint(0, speaker_size, size=[batch_size,]).astype(
                xp.int32
            )
        else:
            self.s_one = None

        wave_rnn = WaveRNN(
            dual_softmax=False,
            bit_size=bit_size,
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

        if xp == cp:
            wave_rnn.to_gpu()

        # set 'b'
        for p in wave_rnn.params():
            if "b" not in p.name:
                continue
            p.data = xp.random.rand(*p.shape).astype(p.dtype)

        self.wave_rnn = wave_rnn

    def test_call(self):
        wave_rnn = self.wave_rnn
        wave_rnn(
            x_array=self.x_array, l_array=self.l_array, s_one=self.s_one,
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

        l_array = xp.pad(
            self.l_array,
            pad_width=((0, 0), (local_padding_size, local_padding_size), (0, 0)),
            mode="constant",
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
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one).data
        wave_rnn.forward_one(
            self.x_one[:, 0], l_one[:, 0], hidden=hidden,
        )

    def test_fast_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one).data
        fast_forward_params = get_fast_forward_params_one(wave_rnn)
        fast_forward_one(
            prev_x=self.x_one[:, 0],
            prev_l=l_one[:, 0],
            hidden=hidden,
            **fast_forward_params,
        )

    def test_batchsize1_forward(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one).data

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array, l_array=l_array, hidden=hidden,
        )

        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:1], l_array=l_array[:1], hidden=hidden[:1],
        )

        xp.testing.assert_allclose(oa.data[:1], ob.data, atol=1e-6)
        xp.testing.assert_allclose(ha.data[:1], hb.data, atol=1e-6)

    def test_batchsize1_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one).data

        oa, ha = wave_rnn.forward_one(self.x_one[:, 0], l_one[:, 0], hidden=hidden,)

        ob, hb = wave_rnn.forward_one(
            self.x_one[:1, 0], l_one[:1, 0], hidden=hidden[:1],
        )

        xp.testing.assert_allclose(oa.data[:1], ob.data, atol=1e-6)
        xp.testing.assert_allclose(ha.data[:1], hb.data, atol=1e-6)

    def test_batchsize1_fast_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_one = wave_rnn.forward_encode(l_array=self.l_one, s_one=s_one).data
        fast_forward_params = get_fast_forward_params_one(wave_rnn)

        oa, ha = fast_forward_one(
            prev_x=self.x_one[:, 0],
            prev_l=l_one[:, 0],
            hidden=hidden,
            **fast_forward_params,
        )

        ob, hb = fast_forward_one(
            prev_x=self.x_one[:1, 0],
            prev_l=l_one[:1, 0],
            hidden=hidden[:1],
            **fast_forward_params,
        )

        xp.testing.assert_allclose(oa.data[:1], ob.data, atol=1e-6)
        xp.testing.assert_allclose(ha.data[:1], hb.data, atol=1e-6)

    def test_same_call_and_forward(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()

        oa, ha = wave_rnn(
            x_array=self.x_array, l_array=self.l_array, s_one=self.s_one, hidden=hidden,
        )

        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one).data
        ob, hb = wave_rnn.forward_rnn(
            x_array=self.x_array[:, :-1], l_array=l_array[:, 1:], hidden=hidden,
        )

        xp.testing.assert_equal(oa.data, ob.data)
        xp.testing.assert_equal(ha.data, hb.data)

    def test_same_forward_rnn_and_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one).data

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array, l_array=l_array, hidden=hidden,
        )

        hb = hidden
        for i, (x, l) in enumerate(
            zip(
                xp.split(self.x_array, length, axis=1),
                xp.split(l_array, length, axis=1),
            )
        ):
            ob, hb = wave_rnn.forward_one(x[:, 0], l[:, 0], hb,)

            xp.testing.assert_allclose(oa[:, :, i].data, ob.data, atol=1e-6)

        xp.testing.assert_allclose(ha.data, hb.data, atol=1e-6)

    def test_same_forward_rnn_and_fast_forward_one(self):
        wave_rnn = self.wave_rnn
        hidden = _make_hidden()
        s_one = (
            self.s_one
            if not wave_rnn.with_speaker
            else wave_rnn.forward_speaker(self.s_one)
        )
        l_array = wave_rnn.forward_encode(l_array=self.l_array, s_one=s_one).data
        fast_forward_params = get_fast_forward_params_one(wave_rnn)

        oa, ha = wave_rnn.forward_rnn(
            x_array=self.x_array, l_array=l_array, hidden=hidden,
        )

        hb = hidden
        for i, (x, l) in enumerate(
            zip(
                xp.split(self.x_array, length, axis=1),
                xp.split(l_array, length, axis=1),
            )
        ):
            ob, hb = fast_forward_one(
                prev_x=x[:, 0], prev_l=l[:, 0], hidden=hb, **fast_forward_params,
            )

            xp.testing.assert_allclose(oa[:, :, i].data, ob, atol=1e-6)

        xp.testing.assert_allclose(ha.data, hb, atol=1e-6)
