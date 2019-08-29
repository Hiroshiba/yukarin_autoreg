import unittest
from typing import List

import chainer
import numpy as np
from chainer import serializers
from retry import retry

from tests.utility import DownLocalRandomDataset, LocalRandomDataset, RandomDataset, setup_support, \
    train_support, SignWaveDataset
from yukarin_autoreg.config import LossConfig
from yukarin_autoreg.model import Model
from yukarin_autoreg.network import WaveRNN
from yukarin_autoreg.network.univ_wave_rnn import UnivWaveRNN

sampling_rate = 8000
sampling_length = 880

gpu = 3
batch_size = 16
hidden_size = 896
iteration = 300

if gpu is not None:
    chainer.cuda.get_device_from_id(gpu).use()


def _create_model(
        local_size: int,
        local_scale: int = None,
        dual_softmax=True,
        bit_size=16,
        use_univ=False,
):
    if not use_univ:
        network = WaveRNN(
            upconv_scales=[local_scale] if local_scale is not None else [],
            upconv_residual=local_scale is not None,
            upconv_channel_ksize=3,
            residual_encoder_channel=None,
            residual_encoder_num_block=None,
            dual_softmax=dual_softmax,
            bit_size=bit_size,
            hidden_size=hidden_size,
            local_size=local_size,
            bug_fixed_gru_dimension=True,
        )
    else:
        network = UnivWaveRNN(
            dual_softmax=dual_softmax,
            bit_size=bit_size,
            conditioning_size=128,
            embedding_size=256,
            hidden_size=hidden_size,
            linear_hidden_size=512,
            local_size=local_size,
            local_scale=local_scale if local_scale is not None else 1,
            local_layer_num=2,
        )

    loss_config = LossConfig(
        disable_fine=not dual_softmax,
        eliminate_silence=False,
    )
    model = Model(loss_config=loss_config, predictor=network, local_padding_size=0)
    return model


class TestTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit, mulaw, use_univ):
        model = _create_model(local_size=0, dual_softmax=to_double, bit_size=bit, use_univ=use_univ)
        dataset = SignWaveDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < 1)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data < 5)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

        # save model
        serializers.save_npz(
            f'TestTrainingWaveRNN'
            f'-to_double={to_double}'
            f'-bit={bit}'
            f'-mulaw={mulaw}'
            f'-use_univ={use_univ}'
            f'-iteration={iteration}.npz',
            model.predictor,
        )

    def test_train(self):
        for to_double, bit, mulaw, use_univ in (
                (True, 16, False, False),
                (False, 8, False, False),
                (False, 8, True, False),
                (False, 9, True, True),
        ):
            with self.subTest(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ):
                self._wrapper(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ)


class TestCannotTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit, mulaw, use_univ):
        model = _create_model(local_size=0, dual_softmax=to_double, bit_size=bit, use_univ=use_univ)
        dataset = RandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > 4)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        for to_double, bit, mulaw, use_univ in (
                (True, 16, False, False),
                (False, 8, False, False),
                (False, 8, True, False),
                (False, 9, True, True),
        ):
            with self.subTest(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ):
                self._wrapper(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ)


class TestLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit, mulaw, use_univ):
        model = _create_model(local_size=2, dual_softmax=to_double, bit_size=bit, use_univ=use_univ)
        dataset = LocalRandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < 5)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data < 5)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        for to_double, bit, mulaw, use_univ in (
                (True, 16, False, False),
                (False, 8, False, False),
                (False, 8, True, False),
                (False, 9, True, True),
        ):
            with self.subTest(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ):
                self._wrapper(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ)


class TestDownSampledLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit, mulaw, use_univ):
        scale = 4

        model = _create_model(
            local_size=2 * scale,
            local_scale=scale,
            dual_softmax=to_double,
            bit_size=bit,
            use_univ=use_univ,
        )
        dataset = DownLocalRandomDataset(
            sampling_length=sampling_length,
            scale=scale,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > 4)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > 4)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < 5)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data < 5)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        for to_double, bit, mulaw, use_univ in (
                (True, 16, False, False),
                (False, 8, False, False),
                (False, 8, True, False),
                (False, 9, True, True),
        ):
            with self.subTest(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ):
                self._wrapper(to_double=to_double, bit=bit, mulaw=mulaw, use_univ=use_univ)
