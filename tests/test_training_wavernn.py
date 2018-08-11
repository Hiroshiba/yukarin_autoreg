import unittest
from functools import partial

import chainer
import numpy as np
from chainer.iterators import MultiprocessIterator

from yukarin_autoreg.config import LossConfig, ModelConfig
from yukarin_autoreg.dataset import SignWaveDataset, encode_16bit, normalize
from yukarin_autoreg.network import WaveRNN
from yukarin_autoreg.model import Model
from chainer.training.updaters import StandardUpdater

sampling_rate = 8000
sampling_length = 880

gpu = 1
batch_size = 16
hidden_size = 896
iteration = 300

if gpu is not None:
    chainer.cuda.get_device_from_id(gpu).use()


class RandomDataset(chainer.dataset.DatasetMixin):
    def __len__(self):
        return 100

    def get_example(self, i):
        length = sampling_length
        wave = np.random.rand(length) * 2 - 1
        local = np.empty(shape=(length, 0), dtype=np.float32)
        silence = np.zeros(shape=(length,), dtype=np.bool)

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
            local=local[1:],
            silence=silence[1:],
        )


class LocalRandomDataset(RandomDataset):
    def get_example(self, i):
        d = super().get_example(i)
        d['local'] = np.stack((
            d['target_coarse'].astype(np.float32) / 256,
            d['target_fine'].astype(np.float32) / 256,
        ), axis=1)
        return d


def _create_optimizer(model):
    optimizer = chainer.optimizers.Adam(alpha=0.005)
    optimizer.setup(model)
    return optimizer


class TestTrainingWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = WaveRNN(config=ModelConfig(
            hidden_size=hidden_size,
            bit_size=batch_size,
            local_size=0,
        ))
        model = Model(wave_rnn)
        optimizer = _create_optimizer(model)
        dataset = SignWaveDataset(sampling_rate=sampling_rate, sampling_length=sampling_length)
        train_iter = MultiprocessIterator(dataset, batch_size)

        if gpu is not None:
            model.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = StandardUpdater(
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', model)

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
        wave_rnn = WaveRNN(config=ModelConfig(
            hidden_size=hidden_size,
            bit_size=batch_size,
            local_size=0,
        ))
        model = Model(wave_rnn)
        optimizer = _create_optimizer(model)
        train_iter = MultiprocessIterator(RandomDataset(), batch_size)

        if gpu is not None:
            model.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = StandardUpdater(
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', model)

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
        wave_rnn = WaveRNN(config=ModelConfig(
            hidden_size=hidden_size,
            bit_size=batch_size,
            local_size=0,
        ), disable_mask=True)
        model = Model(wave_rnn)
        optimizer = _create_optimizer(model)
        train_iter = MultiprocessIterator(RandomDataset(), batch_size)

        if gpu is not None:
            model.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = StandardUpdater(
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', model)

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


class TestLocalTrainingWaveRNN(unittest.TestCase):
    def setUp(self):
        wave_rnn = WaveRNN(config=ModelConfig(
            hidden_size=hidden_size,
            bit_size=batch_size,
            local_size=2,
        ))
        model = Model(wave_rnn)
        optimizer = _create_optimizer(model)
        train_iter = MultiprocessIterator(LocalRandomDataset(), batch_size)

        if gpu is not None:
            model.to_gpu(gpu)

        converter = partial(chainer.dataset.convert.concat_examples)
        self.updater = StandardUpdater(
            device=gpu,
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter,
        )

        self.reporter = chainer.Reporter()
        self.reporter.add_observer('main', model)

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
        self.assertTrue(observation['main/nll_fine'].data < 5)


if __name__ == '__main__':
    unittest.main()
