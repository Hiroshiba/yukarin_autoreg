import unittest
from itertools import chain

import chainer
from chainer import serializers
from chainer.datasets import ConcatenatedDataset
from retry import retry

from tests.utility import DownLocalRandomDataset, LocalRandomDataset, RandomDataset, setup_support, \
    train_support, SignWaveDataset
from yukarin_autoreg.config import LossConfig
from yukarin_autoreg.dataset import SpeakerWavesDataset
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
        speaker_size=0,
):
    network = WaveRNN(
        dual_softmax=dual_softmax,
        bit_size=bit_size,
        conditioning_size=128,
        embedding_size=256,
        hidden_size=hidden_size,
        linear_hidden_size=512,
        local_size=local_size,
        local_scale=local_scale if local_scale is not None else 1,
        local_layer_num=2,
        speaker_size=speaker_size,
        speaker_embedding_size=speaker_size // 4,
    )

    loss_config = LossConfig(
        disable_fine=not dual_softmax,
        eliminate_silence=False,
        mean_silence=True,
    )
    model = Model(loss_config=loss_config, predictor=network, local_padding_size=0)
    return model


def _get_trained_nll():
    return 4


class TestTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double=False, bit=10, mulaw=True):
        model = _create_model(local_size=0)
        dataset = SignWaveDataset(
            sampling_rate=sampling_rate,
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

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
            '/tmp/'
            f'test_training_wavernn'
            f'-to_double={to_double}'
            f'-bit={bit}'
            f'-mulaw={mulaw}'
            f'-speaker_size=0'
            f'-iteration={iteration}.npz',
            model.predictor,
        )

    def test_train(self):
        self._wrapper()


class TestCannotTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double=False, bit=10, mulaw=True):
        model = _create_model(local_size=0)
        dataset = RandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=0,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

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
        self._wrapper()


class TestLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double=False, bit=10, mulaw=True):
        model = _create_model(local_size=2)
        dataset = LocalRandomDataset(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=0,
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

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
        self._wrapper()


class TestDownSampledLocalTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double=False, bit=10, mulaw=True):
        scale = 4

        model = _create_model(
            local_size=2 * scale,
            local_scale=scale,
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
        trained_nll = _get_trained_nll()

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
        self._wrapper()


class TestSpeakerTrainingWaveRNN(unittest.TestCase):
    @retry(tries=10)
    def _wrapper(self, to_double=False, bit=10, mulaw=True):
        speaker_size = 4
        model = _create_model(
            local_size=0,
            speaker_size=speaker_size,
        )

        datasets = [
            SignWaveDataset(
                sampling_rate=sampling_rate,
                sampling_length=sampling_length,
                to_double=to_double,
                bit=bit,
                mulaw=mulaw,
                frequency=(i + 1) * 110,
            )
            for i in range(speaker_size)
        ]
        dataset = SpeakerWavesDataset(
            wave_dataset=ConcatenatedDataset(*datasets),
            speaker_nums=list(chain.from_iterable([i] * len(d) for i, d in enumerate(datasets))),
        )

        updater, reporter = setup_support(batch_size, gpu, model, dataset)
        trained_nll = _get_trained_nll()

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
            '/tmp/'
            f'test_training_wavernn'
            f'-to_double={to_double}'
            f'-bit={bit}'
            f'-mulaw={mulaw}'
            f'-speaker_size={speaker_size}'
            f'-iteration={iteration}.npz',
            model.predictor,
        )

    def test_train(self):
        self._wrapper()
