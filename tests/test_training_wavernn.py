import unittest
from functools import partial

import chainer
import numpy as np
from chainer.iterators import MultiprocessIterator

from yukarin_autoreg.config import ModelConfig, LossConfig
from yukarin_autoreg.dataset import encode_16bit
from yukarin_autoreg.dataset import SignWaveDataset
from yukarin_autoreg.dataset import normalize
from yukarin_autoreg.model import WaveRNN
from yukarin_autoreg.updater import Updater

sampling_rate = 8000
sampling_length = 880

gpu = 1
batch_size = 16
hidden_size = 896
iteration = 500

model_config = ModelConfig(
    hidden_size=hidden_size,
    bit_size=16,
)

if gpu is not None:
    chainer.cuda.get_device_from_id(gpu).use()


class RandomDataset(chainer.dataset.DatasetMixin):
    def __len__(self):
        return 100

    def get_example(self, i):
        length = sampling_length
        wave = np.random.rand(length) * 2 - 1

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
        )


def _create_optimizer(model):
    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(model)
    return optimizer


class TestTrainingWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = WaveRNN(config=model_config)
        optimizer = _create_optimizer(wave_rnn)
        dataset = SignWaveDataset(sampling_rate=sampling_rate, sampling_length=sampling_length)
        train_iter = MultiprocessIterator(dataset, batch_size)

        if gpu is not None:
            wave_rnn.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = Updater(
            loss_config=LossConfig(),
            predictor=wave_rnn,
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', wave_rnn)

    def test_init(self):
        pass

    def test_train(self):
        observation = {}
        for i in range(iteration):
            with self.reporter.scope(observation):
                self.updater.update()

            if i % 100 == 0:
                print(observation)

            if i == 0:
                self.assertTrue(observation['main/nll_coarse'].data > 4)
                self.assertTrue(observation['main/nll_fine'].data > 4)

        self.assertTrue(observation['main/nll_coarse'].data < 1)
        self.assertTrue(observation['main/nll_fine'].data < 5)


class TestCannotTrainingWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = WaveRNN(config=model_config)
        optimizer = _create_optimizer(wave_rnn)
        train_iter = MultiprocessIterator(RandomDataset(), batch_size)

        if gpu is not None:
            wave_rnn.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = Updater(
            loss_config=LossConfig(),
            predictor=wave_rnn,
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', wave_rnn)

    def test_init(self):
        pass

    def test_train(self):
        observation = {}
        for i in range(iteration):
            with self.reporter.scope(observation):
                self.updater.update()

            if i % 100 == 0:
                print(observation)

            if i == 0:
                self.assertTrue(observation['main/nll_coarse'].data > 4)
                self.assertTrue(observation['main/nll_fine'].data > 4)

        self.assertTrue(observation['main/nll_coarse'].data > 4)
        self.assertTrue(observation['main/nll_fine'].data > 4)


class TestWOMaskTrainingWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = WaveRNN(config=model_config, disable_mask=True)
        optimizer = _create_optimizer(wave_rnn)
        train_iter = MultiprocessIterator(RandomDataset(), batch_size)

        if gpu is not None:
            wave_rnn.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = Updater(
            loss_config=LossConfig(),
            predictor=wave_rnn,
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', wave_rnn)

    def test_init(self):
        pass

    def test_train(self):
        observation = {}
        for i in range(iteration):
            with self.reporter.scope(observation):
                self.updater.update()

            if i % 100 == 0:
                print(observation)

            if i == 0:
                self.assertTrue(observation['main/nll_coarse'].data > 4)
                self.assertTrue(observation['main/nll_fine'].data > 4)

        self.assertTrue(observation['main/nll_coarse'].data < 5)
        self.assertTrue(observation['main/nll_fine'].data > 4)


if __name__ == '__main__':
    unittest.main()
