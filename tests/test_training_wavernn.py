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

sampling_rate = 8000
sampling_length = 880

gpu = 3
bit_size = 16
batch_size = 16
hidden_size = 896
iteration = 300

if gpu is not None:
    chainer.cuda.get_device_from_id(gpu).use()


def _create_model(
        local_size: int,
        upconv_scales: List[int] = None,
        dual_softmax=True,
        bit_size=16,
):
    if upconv_scales is None:
        upconv_scales = []

    network = WaveRNN(
        upconv_scales=upconv_scales,
        upconv_residual=len(upconv_scales) > 0,
        upconv_channel_ksize=3,
        residual_encoder_channel=None,
        residual_encoder_num_block=None,
        dual_softmax=dual_softmax,
        bit_size=bit_size,
        hidden_size=hidden_size,
        local_size=local_size,
        bug_fixed_gru_dimension=True,
    )

    loss_config = LossConfig(
        disable_fine=not dual_softmax,
    )
    model = Model(loss_config=loss_config, predictor=network)
    return model


class TestTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit):
        model = _create_model(local_size=0, dual_softmax=to_double, bit_size=bit)
        dataset = SignWaveDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
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
            f'TestTrainingWaveRNN-to_double={to_double}-bit={bit}-iteration={iteration}.npz',
            model.predictor,
        )

    def test_train(self):
        for to_double, bit in zip([True, False], [16, 8]):
            with self.subTest(to_double=to_double, bit=bit):
                self._wrapper(to_double=to_double, bit=bit)


class TestCannotTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit):
        model = _create_model(local_size=0, dual_softmax=to_double, bit_size=bit)
        dataset = RandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
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
        for to_double, bit in zip([True, False], [16, 8]):
            with self.subTest(to_double=to_double, bit=bit):
                self._wrapper(to_double=to_double, bit=bit)


class TestLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit):
        model = _create_model(local_size=2, dual_softmax=to_double, bit_size=bit)
        dataset = LocalRandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
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
        for to_double, bit in zip([True, False], [16, 8]):
            with self.subTest(to_double=to_double, bit=bit):
                self._wrapper(to_double=to_double, bit=bit)


class TestDownSampledLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double, bit):
        scales = [4]
        scale = int(np.prod(scales))

        model = _create_model(
            local_size=2 * scale,
            upconv_scales=scales,
            dual_softmax=to_double,
            bit_size=bit,
        )
        dataset = DownLocalRandomDataset(
            sampling_length=sampling_length,
            scale=scale,
            to_double=to_double,
            bit=bit,
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
        for to_double, bit in zip([True, False], [16, 8]):
            with self.subTest(to_double=to_double, bit=bit):
                self._wrapper(to_double=to_double, bit=bit)
