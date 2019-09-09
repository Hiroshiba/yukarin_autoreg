import unittest

import chainer
from chainer import serializers
from retry import retry

from tests.utility import DownLocalRandomDataset, LocalRandomDataset, RandomDataset, setup_support, \
    train_support, SignWaveDataset
from yukarin_autoreg.config import LossConfig
from yukarin_autoreg.model import Model
from yukarin_autoreg.network.wave_rnn import WaveRNN

sampling_rate = 8000
sampling_length = 880

gpu = 0
batch_size = 16
hidden_size = 896
iteration = 300

if gpu is not None:
    chainer.cuda.get_device_from_id(gpu).use()


def _create_model(
        local_size: int,
        local_scale: int = None,
        dual_softmax=False,
        bit_size=10,
        gaussian=False,
        input_categorical=True,
):
    network = WaveRNN(
        dual_softmax=dual_softmax,
        bit_size=bit_size,
        gaussian=gaussian,
        input_categorical=input_categorical,
        conditioning_size=128,
        embedding_size=256,
        hidden_size=hidden_size,
        linear_hidden_size=512,
        local_size=local_size,
        local_scale=local_scale if local_scale is not None else 1,
        local_layer_num=2,
        speaker_size=0,
        speaker_embedding_size=0,
    )

    loss_config = LossConfig(
        disable_fine=not dual_softmax,
        eliminate_silence=False,
    )
    model = Model(loss_config=loss_config, predictor=network, local_padding_size=0)
    return model


def _get_trained_nll(gaussian):
    if not gaussian:
        return 4
    else:
        return 1


class TestTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, input_categorical, gaussian, to_double=False, bit=10, mulaw=True):
        model = _create_model(local_size=0, input_categorical=input_categorical, gaussian=gaussian)
        dataset = SignWaveDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll(gaussian)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > trained_nll)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

        # save model
        serializers.save_npz(
            f'TestTrainingWaveRNN'
            f'-to_double={to_double}'
            f'-bit={bit}'
            f'-mulaw={mulaw}'
            f'-input_categorical={input_categorical}'
            f'-gaussian={gaussian}'
            f'-iteration={iteration}.npz',
            model.predictor,
        )

    def test_train(self):
        for input_categorical, gaussian in (
                (True, False),
                (False, False),
                (False, True),
        ):
            with self.subTest(input_categorical=input_categorical, gaussian=gaussian):
                self._wrapper(input_categorical=input_categorical, gaussian=gaussian)


class TestCannotTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, input_categorical, gaussian, to_double=False, bit=10, mulaw=True):
        model = _create_model(local_size=0, input_categorical=input_categorical, gaussian=gaussian)
        dataset = RandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=0,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll(gaussian)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > trained_nll)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        for input_categorical, gaussian in (
                (True, False),
                (False, False),
                (False, True),
        ):
            with self.subTest(input_categorical=input_categorical, gaussian=gaussian):
                self._wrapper(input_categorical=input_categorical, gaussian=gaussian)


class TestLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, input_categorical, gaussian, to_double=False, bit=10, mulaw=True):
        model = _create_model(local_size=2, input_categorical=input_categorical, gaussian=gaussian)
        dataset = LocalRandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=0,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll(gaussian)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > trained_nll)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        for input_categorical, gaussian in (
                (True, False),
                (False, False),
                (False, True),
        ):
            with self.subTest(input_categorical=input_categorical, gaussian=gaussian):
                self._wrapper(input_categorical=input_categorical, gaussian=gaussian)


class TestDownSampledLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, input_categorical, gaussian, to_double=False, bit=10, mulaw=True):
        scale = 4

        model = _create_model(
            local_size=2 * scale,
            local_scale=scale,
            input_categorical=input_categorical,
            gaussian=gaussian,
        )
        dataset = DownLocalRandomDataset(
            sampling_length=sampling_length,
            scale=scale,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=0,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll(gaussian)

        def _first_hook(o):
            self.assertTrue(o['main/nll_coarse'].data > trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data > trained_nll)

        def _last_hook(o):
            self.assertTrue(o['main/nll_coarse'].data < trained_nll)
            if to_double:
                self.assertTrue(o['main/nll_fine'].data < trained_nll)

        train_support(iteration, reporter, updater, _first_hook, _last_hook)

    def test_train(self):
        for input_categorical, gaussian in (
                (True, False),
                (False, False),
                (False, True),
        ):
            with self.subTest(input_categorical=input_categorical, gaussian=gaussian):
                self._wrapper(input_categorical=input_categorical, gaussian=gaussian)
