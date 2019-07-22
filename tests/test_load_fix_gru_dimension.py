import tempfile
import unittest

import chainer
import numpy as np
from chainer import serializers

from yukarin_autoreg.network.wave_rnn import _call_1layer
from yukarin_autoreg.utility.chainer_network_utility import ModifiedNStepGRU
from yukarin_autoreg.utility.load_fix_gru_dimension import load_fix_gru_dimension

batch_size = 2
length = 3
in_size = 5
out_size = 7


def _set_random_bias(network: chainer.Chain):
    for p in network.params():
        if 'b' not in p.name:
            continue
        p.data = np.random.rand(*p.shape).astype(p.dtype)


class TestLoadFixGruDimension(unittest.TestCase):
    def test_forward(self):
        in_array = np.random.rand(batch_size, length, in_size).astype(np.float32)
        gru = ModifiedNStepGRU(
            n_layers=1,
            in_size=in_size,
            out_size=out_size,
            dropout=0.,
        )

        _call_1layer(gru, hidden=None, input=in_array)

    def test_reduce_forward(self):
        in_array_full = np.random.rand(batch_size, length, in_size * 2).astype(np.float32)
        in_array_full[:, :, in_size:] = 0

        gru_full = ModifiedNStepGRU(
            n_layers=1,
            in_size=in_size * 2,
            out_size=out_size,
            dropout=0.,
        )
        _set_random_bias(gru_full)
        out_full = _call_1layer(gru_full, hidden=None, input=in_array_full)

        in_array_reduce = in_array_full[:, :, :in_size]
        gru_reduce = ModifiedNStepGRU(
            n_layers=1,
            in_size=in_size,
            out_size=out_size,
            dropout=0.,
        )

        for i, (w_full, w_reduce) in enumerate(zip(gru_full.ws[0], gru_reduce.ws[0])):
            if i < 3:
                w_reduce.copydata(w_full[:, :in_size])
            else:
                w_reduce.copydata(w_full)

        for b_full, b_reduce in zip(gru_full.bs[0], gru_reduce.bs[0]):
            b_reduce.copydata(b_full)

        out_reduce = _call_1layer(gru_reduce, hidden=None, input=in_array_reduce)

        np.testing.assert_allclose(out_full.data, out_reduce.data, atol=1e-6)

    def test_reduce_load(self):
        in_array_full = np.random.rand(batch_size, length, in_size * 2).astype(np.float32)
        in_array_full[:, :, in_size:] = 0

        gru_full = ModifiedNStepGRU(
            n_layers=1,
            in_size=in_size * 2,
            out_size=out_size,
            dropout=0.,
        )
        _set_random_bias(gru_full)
        out_full = _call_1layer(gru_full, hidden=None, input=in_array_full)

        with tempfile.NamedTemporaryFile(delete=False) as f_save:
            serializers.save_npz(f_save.name, gru_full)

        in_array_reduce = in_array_full[:, :, :in_size]
        gru_reduce = ModifiedNStepGRU(
            n_layers=1,
            in_size=in_size,
            out_size=out_size,
            dropout=0.,
        )

        load_fix_gru_dimension(f_save.name, gru_reduce)

        out_reduce = _call_1layer(gru_reduce, hidden=None, input=in_array_reduce)

        np.testing.assert_allclose(out_full.data, out_reduce.data, atol=1e-6)
